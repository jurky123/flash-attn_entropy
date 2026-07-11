# Copyright (c) 2023, Tri Dao.

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import os
import warnings

# isort: off
# We need to import the CUDA kernels after importing torch
USE_TRITON_ROCM = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
if not USE_TRITON_ROCM and getattr(torch.version, 'hip', None) is not None:
    try:
        import flash_attn_2_cuda
    except ImportError:
        warnings.warn("flash_attn_2_cuda (which has ROCm/HIP kernels) not found, falling back to Triton implementation")
        USE_TRITON_ROCM = True

if USE_TRITON_ROCM:
    from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2 as flash_attn_gpu
else:
    import flash_attn_2_cuda as flash_attn_gpu

# isort: on

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _get_block_size_n(device, head_dim, is_dropout, is_causal):
    # This should match the block sizes in the CUDA kernel
    assert head_dim <= 256
    major, minor = torch.cuda.get_device_capability(device)
    is_sm8x = major == 8 and minor > 0  # Only include sm86 and sm89, exclude sm80 (A100)
    is_sm80 = major == 8 and minor == 0
    is_sm90 = major == 9 and minor == 0
    if head_dim <= 32:
        return 128
    if head_dim <= 64:
        return 128 if not is_dropout else 64
    elif head_dim <= 96:
        return 64
    elif head_dim <= 128:
        if is_sm8x:
            return 64 if (not is_dropout and is_causal) else 32
        else:
            return 64 if not is_dropout else 32
    elif head_dim <= 192:
        return 64
    elif head_dim <= 224:
        return 64
    elif head_dim <= 256:
        return 64


def round_multiple(x, m):
    return (x + m - 1) // m * m


# torch.compile() support is only enabled for pytorch >= 2.4
# The reason for this is that we are using the new custom_op and register_fake
# APIs, which support inplace modification of inputs in the function itself
if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:
    def noop_custom_op_wrapper(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    _torch_custom_op_wrapper = noop_custom_op_wrapper
    _torch_register_fake_wrapper = noop_register_fake_wrapper


@_torch_custom_op_wrapper("flash_attn::_flash_attn_forward", mutates_args=(), device_types="cuda")
def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state, _ = flash_attn_gpu.fwd(
        q,
        k,
        v,
        None,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        False,  # return_entropy
        None,
    )
    return out, softmax_lse, S_dmask, rng_state


@_torch_register_fake_wrapper("flash_attn::_flash_attn_forward")
def _flash_attn_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    batch_size, seqlen_q, num_heads, head_size = q.shape
    seqlen_k = k.shape[1]
    out = torch.empty_like(q)
    softmax_lse = torch.empty((batch_size, num_heads, seqlen_q), dtype=torch.float32, device=q.device, layout=q.layout)
    p = torch.empty((0,), dtype=q.dtype, device=q.device, layout=q.layout)
    if return_softmax:
        if torch.cuda.is_available() and torch.version.hip:
            p = torch.empty((batch_size, num_heads, seqlen_q, seqlen_k), dtype=q.dtype, device=q.device, layout=q.layout)
        else:
            p = torch.empty((batch_size, num_heads, round_multiple(seqlen_q, 128), round_multiple(seqlen_k, 128)), dtype=q.dtype, device=q.device, layout=q.layout)
    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)

    return out, softmax_lse, p, rng_state


if torch.__version__ >= "2.4.0":
    _wrapped_flash_attn_forward = torch.ops.flash_attn._flash_attn_forward
else:
    _wrapped_flash_attn_forward = _flash_attn_forward


@_torch_custom_op_wrapper("flash_attn::_flash_attn_varlen_forward", mutates_args=(), device_types="cuda")
def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
    block_table: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
    num_splits: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state, _ = flash_attn_gpu.varlen_fwd(
        q,
        k,
        v,
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        return_softmax,
        None,
        num_splits,
    )
    # if out.isnan().any() or softmax_lse.isnan().any():
    #     breakpoint()
    return out, softmax_lse, S_dmask, rng_state


@_torch_register_fake_wrapper("flash_attn::_flash_attn_varlen_forward")
def _flash_attn_varlen_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
    block_table: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
    num_splits: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    paged_kv = block_table is not None
    batch_size = cu_seqlens_q.numel() - 1
    total_q, num_heads, _ = q.shape

    out = torch.empty_like(q)
    softmax_lse = torch.empty((num_heads, total_q), dtype=torch.float32, device=q.device, layout=q.layout)
    p = torch.empty((0,), dtype=q.dtype, device=q.device, layout=q.layout)
    if return_softmax:
        if torch.cuda.is_available() and torch.version.hip:
            p = torch.empty((batch_size, num_heads, max_seqlen_q, max_seqlen_k), dtype=q.dtype, device=q.device, layout=q.layout)
        else:
            p = torch.empty((batch_size, num_heads, round_multiple(max_seqlen_q, 128), round_multiple(max_seqlen_k, 128)), dtype=q.dtype, device=q.device, layout=q.layout)
    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
    return out, softmax_lse, p, rng_state


if torch.__version__ >= "2.4.0":
    _wrapped_flash_attn_varlen_forward = torch.ops.flash_attn._flash_attn_varlen_forward
else:
    _wrapped_flash_attn_varlen_forward = _flash_attn_varlen_forward


@_torch_custom_op_wrapper("flash_attn::_flash_attn_backward", mutates_args=("dq", "dk", "dv"), device_types="cuda")
def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    (
        dq,
        dk,
        dv,
        softmax_d,
    ) = flash_attn_gpu.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        None,
        rng_state,
    )
    return softmax_d


@_torch_register_fake_wrapper("flash_attn::_flash_attn_backward")
def _flash_attn_backward_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    batch_size, seqlen_q, num_heads, _ = q.shape
    if torch.cuda.is_available() and torch.version.hip:
        softmax_d = torch.empty((batch_size, num_heads, seqlen_q), device=q.device, dtype=torch.float32)
    else:
        softmax_d = torch.empty((batch_size, num_heads, round_multiple(seqlen_q, 128)), device=q.device, dtype=torch.float32)
    
    return softmax_d


if torch.__version__ >= "2.4.0":
    _wrapped_flash_attn_backward = torch.ops.flash_attn._flash_attn_backward
else:
    _wrapped_flash_attn_backward = _flash_attn_backward


@_torch_custom_op_wrapper("flash_attn::_flash_attn_varlen_backward", mutates_args=("dq", "dk", "dv"), device_types="cuda")
def _flash_attn_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> torch.Tensor:
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    (
        dq,
        dk,
        dv,
        softmax_d,
    ) = flash_attn_gpu.varlen_bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        None,
        rng_state,
    )
    # if dk.isnan().any() or dk.isnan().any() or dv.isnan().any() or softmax_d.isnan().any():
    #     breakpoint()
    return softmax_d


@_torch_register_fake_wrapper("flash_attn::_flash_attn_varlen_backward")
def _flash_attn_varlen_backward_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> torch.Tensor:
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    batch_size = cu_seqlens_q.numel() - 1
    total_q, num_heads, _ = q.shape

    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    if torch.cuda.is_available() and torch.version.hip:
        softmax_d = torch.empty((num_heads, total_q), device=q.device, dtype=torch.float32)
    else:
        softmax_d = torch.empty((num_heads, total_q + 128 * batch_size), device=q.device, dtype=torch.float32)
    
    return softmax_d


if torch.__version__ >= "2.4.0":
    _wrapped_flash_attn_varlen_backward = torch.ops.flash_attn._flash_attn_varlen_backward
else:
    _wrapped_flash_attn_varlen_backward = _flash_attn_varlen_backward


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and qkv.requires_grad
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        q, k, v = qkv[:, :, 0].detach(), qkv[:, :, 1].detach(), qkv[:, :, 2].detach()
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state, _ =  _wrapped_flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        head_size_og = dout.size(3)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
        _wrapped_flash_attn_backward(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dqkv[:, :, 0],
            dqkv[:, :, 1],
            dqkv[:, :, 2],
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and qkv.requires_grad
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        q, k, v = qkv[:, 0].detach(), qkv[:, 1].detach(), qkv[:, 2].detach()
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state, _ = _wrapped_flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=None,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens, rng_state)
            ctx.dropout_p = dropout_p
            ctx.max_seqlen = max_seqlen
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        head_size_og = dout.size(2)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
        _wrapped_flash_attn_varlen_backward(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dqkv[:, 0],
            dqkv[:, 1],
            dqkv[:, 2],
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, kv]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        k, v = kv[:, :, 0].detach(), kv[:, :, 1].detach()
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq = torch.empty_like(q)
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        head_size_og = dout.size(3)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
        _wrapped_flash_attn_backward(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dkv[:, :, 0],
            dkv[:, :, 1],
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dkv = dkv[..., : dout.shape[-1]]
        return dq, dkv, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, kv]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        k, v = kv[:, 0].detach(), kv[:, 1].detach()
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state, _ = _wrapped_flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=None,
        )
        if is_grad:
            ctx.save_for_backward(
                q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
            )
            ctx.dropout_p = dropout_p
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq = torch.empty_like(q)
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        head_size_og = dout.size(2)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
        _wrapped_flash_attn_varlen_backward(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dkv[:, 0],
            dkv[:, 1],
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dkv = dkv[..., : dout.shape[-1]]
        return dq, dkv, None, None, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        head_size_og = dout.size(3)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
        _wrapped_flash_attn_backward(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        block_table,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state, _ = _wrapped_flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=block_table,
        )
        if is_grad:
            ctx.save_for_backward(
                q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
            )
            ctx.dropout_p = dropout_p
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic

        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        head_size_og = dout.size(2)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
        _wrapped_flash_attn_varlen_backward(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # <=0.0 means deactivate
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_kvpacked_func and flash_attn_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to
            the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnQKVPackedFunc.apply(
        qkv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of K, V.
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        kv: (batch_size, seqlen, 2, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnKVPackedFunc.apply(
        q,
        kv,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def flash_attn_func_with_entropy(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
):
    """FlashAttention forward pass that also returns per-token base-2 entropy.

    This is a no-grad path: entropy is always computed with torch.no_grad().
    Currently only supports dropout_p=0.0, fixed-length sequences, fp16/bf16.

    Args:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Must be 0.0 when computing entropy.
        softmax_scale: float. Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal mask.
        window_size: (left, right). Sliding window local attention.
        softcap: float. Softcapping value (0.0 = deactivated).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32.

    Returns:
        out: (batch_size, seqlen, nheads, headdim). Attention output.
        entropy: (batch_size, nheads, seqlen), fp32. Base-2 entropy of the
            attention distribution for each query token.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    window_size_left, window_size_right = window_size
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    with torch.no_grad():
        out, softmax_lse, S_dmask, rng_state, entropy_tensor = flash_attn_gpu.fwd(
            q,
            k,
            v,
            None,  # out_
            alibi_slopes,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            softcap,
            False,  # return_softmax
            True,   # return_entropy
            None,   # gen_
        )
    return out, entropy_tensor


def _entropy_to_k_ratio(entropy: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear map from per-token entropy to k_ratio.

    Args:
        entropy: [B, H, Sq] fp32, per-token entropy in nats.
        table: [2, P] fp32. row0 = entropy breakpoints (strictly increasing),
               row1 = corresponding k_ratio values in (0, 1].

    Returns:
        k_ratio: [B, H, Sq] fp32, same device as entropy.
    """
    breakpoints = table[0]  # [P]
    values = table[1]       # [P]
    device = entropy.device
    P = breakpoints.shape[0]

    # Binary-search each entropy into the right bucket
    idx_right = torch.searchsorted(breakpoints, entropy)  # [B, H, Sq]
    idx_left = (idx_right - 1).clamp(min=0)
    idx_right = idx_right.clamp(max=P - 1)

    e_left = breakpoints[idx_left]   # [B, H, Sq]
    e_right = breakpoints[idx_right]
    v_left = values[idx_left]
    v_right = values[idx_right]

    # Linear interpolation, handle degenerate segments
    denom = e_right - e_left
    frac = torch.where(denom > 1e-10, (entropy - e_left) / denom, torch.zeros_like(entropy))
    frac = frac.clamp(0.0, 1.0)
    k_ratio = v_left + frac * (v_right - v_left)
    return k_ratio


def _precise_mask(
    q: torch.Tensor,        # [B, Sq, H, D]
    k: torch.Tensor,        # [B, Sk, H, D]
    k_ratio: torch.Tensor,  # [B, H, Sq] fp32
    scale: float,
    causal: bool,
    cumsum_threshold: float,
    chunk_size: int = 256,  # query chunk size to bound memory
) -> torch.Tensor:
    """Generate mask by top-p (cumulative probability) per query row, with k_ratio hard cap.

    For each query row, keeps keys satisfying BOTH:
      - cumulative probability < cumsum_threshold (top-p)
      - count ≤ ceil(k_ratio * Sk) (hard cap)

    Processes queries in chunks to avoid materializing the full [B,H,Sq,Sk]
    score/prob tensor.

    Returns:
        mask: [B, H, Sq, Sk] bool.
    """
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    device = q.device

    q_ = q.float().permute(0, 2, 1, 3)                   # [B, H, Sq, D]
    k_ = k.float().permute(0, 2, 1, 3)                   # [B, H, Sk, D]
    k_t = k_.transpose(-2, -1)                            # [B, H, D, Sk]

    k_cap = (k_ratio * Sk).ceil().long().clamp(min=1, max=Sk)  # [B, H, Sq]

    mask = torch.empty(B, H, Sq, Sk, dtype=torch.bool, device=device)

    for q_start in range(0, Sq, chunk_size):
        q_end = min(q_start + chunk_size, Sq)
        chunk_len = q_end - q_start

        q_chunk = q_[:, :, q_start:q_end]                # [B, H, chunk, D]
        scores_chunk = torch.matmul(q_chunk, k_t) * scale  # [B, H, chunk, Sk]

        if causal:
            row_idx = torch.arange(q_start, q_end, device=device).view(-1, 1)  # [chunk, 1]
            col_idx = torch.arange(Sk, device=device).view(1, -1)              # [1, Sk]
            scores_chunk.masked_fill_(col_idx > row_idx, float('-inf'))

        probs_chunk = torch.softmax(scores_chunk, dim=-1)  # [B, H, chunk, Sk]

        # Sort probs descending, cumsum, find cut point per row
        sorted_probs, sorted_idx = torch.sort(probs_chunk, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        k_cut = (cumsum < cumsum_threshold).sum(dim=-1).clamp(min=1)  # [B, H, chunk]

        k_cap_chunk = k_cap[:, :, q_start:q_end]           # [B, H, chunk]
        k_final = torch.minimum(k_cut, k_cap_chunk)         # [B, H, chunk]
        k_max = k_final.max().item()

        _, topk_idx = torch.topk(probs_chunk, k=min(k_max, Sk), dim=-1)  # [B, H, chunk, k_max]

        pos = torch.arange(k_max, device=device).view(1, 1, 1, -1)
        keep = pos < k_final.unsqueeze(-1)

        mask_chunk = torch.zeros(B, H, chunk_len, Sk, dtype=torch.bool, device=device)
        mask_chunk.scatter_(-1, topk_idx, keep.to(mask_chunk.dtype))
        mask[:, :, q_start:q_end] = mask_chunk

    return mask


# ---- Optional: mask generation CUDA kernel ----
try:
    import mask_gen_cuda
    _HAS_MASK_GEN_KERNEL = True
except ImportError:
    _HAS_MASK_GEN_KERNEL = False


def _mask_gen_cuda(
    q: torch.Tensor,        # [B, Sq, H, D], fp16/bf16
    k: torch.Tensor,        # [B, Sk, H, D], fp16/bf16
    k_param: torch.Tensor,  # [B, H, Sq] fp32 (k_keep for quantile, k_ratio for precise)
    softmax_scale: float,
    precise: bool,
    cumsum_threshold: float = 0.9,
) -> torch.Tensor:
    """CUDA kernel for mask generation (2-pass histogram, tiled, no full [Sq,Sk] matrix).

    Returns:
        mask: [B, H, Sq, Sk] bool.
    """
    if not _HAS_MASK_GEN_KERNEL:
        raise RuntimeError("mask_gen_cuda kernel not available; reinstall with setup.py")

    if q.dtype == torch.float16:
        fn = mask_gen_cuda.mask_gen_precise if precise else mask_gen_cuda.mask_gen_quantile
    elif q.dtype == torch.bfloat16:
        fn = mask_gen_cuda.mask_gen_precise_bf16 if precise else mask_gen_cuda.mask_gen_quantile_bf16
    else:
        raise ValueError(f"Unsupported dtype: {q.dtype}")

    # The kernel expects B=1, Sq, H, D (4D). Ensure input is contiguous.
    q_4d = q.contiguous() if q.dim() == 4 else q
    k_4d = k.contiguous() if k.dim() == 4 else k
    k_param_3d = k_param.contiguous() if k_param.dim() == 3 else k_param

    return fn(q_4d, k_4d, k_param_3d, softmax_scale, cumsum_threshold)


def _quantile_mask(
    q: torch.Tensor,        # [B, Sq, H, D]
    k: torch.Tensor,        # [B, Sk, H, D]
    k_ratio: torch.Tensor,  # [B, H, Sq] fp32, per-token retention ratio
    scale: float,
    causal: bool,
    chunk_size: int = 512,  # query chunk size to bound memory
) -> torch.Tensor:
    """Generate mask by score quantile per query row.

    Processes queries in chunks to avoid materializing the full [B,H,Sq,Sk]
    score tensor. Peak memory = O(B * H * chunk_size * Sk) instead of O(B * H * Sq * Sk).

    Returns:
        mask: [B, H, Sq, Sk] bool.
    """
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    device = q.device

    q_ = q.float().permute(0, 2, 1, 3)                   # [B, H, Sq, D]
    k_ = k.float().permute(0, 2, 1, 3)                   # [B, H, Sk, D]
    k_t = k_.transpose(-2, -1)                            # [B, H, D, Sk]
    k_keep = (k_ratio * Sk).ceil().long().clamp(min=1, max=Sk).unsqueeze(-1)  # [B, H, Sq, 1]

    mask = torch.empty(B, H, Sq, Sk, dtype=torch.bool, device=device)

    for q_start in range(0, Sq, chunk_size):
        q_end = min(q_start + chunk_size, Sq)
        q_chunk = q_[:, :, q_start:q_end]                # [B, H, chunk, D]
        scores_chunk = torch.matmul(q_chunk, k_t) * scale  # [B, H, chunk, Sk]

        if causal:
            row_idx = torch.arange(q_start, q_end, device=device).view(-1, 1)  # [chunk, 1]
            col_idx = torch.arange(Sk, device=device).view(1, -1)              # [1, Sk]
            scores_chunk.masked_fill_(col_idx > row_idx, float('-inf'))

        kk = k_keep[:, :, q_start:q_end]                  # [B, H, chunk, 1]
        k_max = kk.max().item()
        topk_vals = torch.topk(scores_chunk, k=min(k_max, Sk), dim=-1, sorted=True).values  # [B, H, chunk, k_max]
        thresholds = torch.gather(topk_vals, -1, (kk - 1).clamp(min=0))  # [B, H, chunk, 1]
        mask[:, :, q_start:q_end] = (scores_chunk >= thresholds)

    return mask


def flash_attn_with_entropy_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    entropy_to_k_ratio: torch.Tensor,
    precise: bool = False,
    cumsum_threshold: float = 0.9,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softmax_scale: Optional[float] = None,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    mask_chunk_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention forward + per-token entropy + entropy-guided mask.

    Block A: flash_attn_func_with_entropy → (out, entropy), no extra cost vs FA2.
    Block B: entropy → k_ratio via piecewise-linear lookup (torch.searchsorted).
    Block C: mask generation:
      - precise=False: score-quantile mask (topk threshold, no softmax).
      - precise=True: top-p cumsum mask with k_ratio as hard cap.

    Mask generation processes queries in chunks of mask_chunk_size to bound
    peak memory at O(B * H * mask_chunk_size * Sk) instead of O(B * H * Sq * Sk).

    Args:
        q: [B, Sq, H, D], fp16/bf16.
        k: [B, Sk, H, D], fp16/bf16.
        v: [B, Sk, H, D], fp16/bf16.
        entropy_to_k_ratio: [2, P] fp32. row0 = entropy breakpoints (strictly
            increasing), row1 = corresponding k_ratio in (0, 1].
        precise: if True, use top-p cumsum mask with k_ratio hard cap.
                 if False, use score-quantile mask (faster, no softmax).
        cumsum_threshold: only used when precise=True, cumulative probability
            threshold for top-p truncation (default 0.9).
        causal: apply causal mask.
        window_size: (left, right) sliding-window attention.
        softmax_scale: default 1/sqrt(D).
        softcap: softcapping value (0 = disabled).
        alibi_slopes: ALiBi slopes.
        mask_chunk_size: query chunk size for mask generation (default 512).
            Smaller values use less memory but more kernel launches.

    Returns:
        out:     [B, Sq, H, D]  attention output.
        entropy: [B, H, Sq]     per-token entropy in nats (fp32).
        mask:    [B, H, Sq, Sk] bool.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    # Block A: flash attention + entropy (already fused in the CUDA kernel)
    out, entropy = flash_attn_func_with_entropy(
        q, k, v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
    )

    # Block B: entropy → k_ratio lookup
    k_ratio = _entropy_to_k_ratio(entropy, entropy_to_k_ratio)

    # Block C: mask generation (CUDA kernel if available, else chunked Python)
    if precise:
        k_param = k_ratio  # [B, H, Sq] fp32
    else:
        k_param = (k_ratio * k.shape[1]).ceil().clamp(min=1, max=k.shape[1])  # [B, H, Sq] k_keep

    if _HAS_MASK_GEN_KERNEL and not causal:
        # CUDA kernel: 2/3-pass histogram, tiled, no [Sq,Sk] materialization
        mask = _mask_gen_cuda(
            q, k, k_param.float(),
            softmax_scale, precise, cumsum_threshold,
        )
    else:
        # Python fallback: chunked along Sq, uses PyTorch CUDA ops
        mask_fn = _precise_mask if precise else _quantile_mask
        mask = mask_fn(q, k, k_ratio, softmax_scale, causal, cumsum_threshold,
                       chunk_size=mask_chunk_size)

    return out, entropy, mask


def _shrink_block_counts(
    block_counts: torch.Tensor,
    block_size: int,
    density_threshold: float = 1.0 / 3.0,
    frac_threshold: float = 0.6,
) -> torch.Tensor:
    """Convert per-key-token query-row hit counts to block mask.

    Matches shrinkMaskStrict semantics exactly.

    Args:
        block_counts: [B, H, num_qb, num_kb, block_size] int32.
            block_counts[b,h,qb,kb,kt] = number of query rows in q-block qb
            that attend to key token (kb*block_size + kt).
        block_size: the block size used for blockification.
        density_threshold: a key token is "high density" if the fraction
            of query rows attending to it > this value (default 1/3).
        frac_threshold: a block is active if the fraction of high-density
            key tokens > this value among non-zero tokens (default 0.6).

    Returns:
        block_mask: [B, H, num_qb, num_kb] bool.
    """
    col_density = block_counts.float() / block_size
    high = col_density > density_threshold
    non_zero = col_density > 0
    high_sum = high.sum(dim=-1)
    non_zero_sum = non_zero.sum(dim=-1)
    frac_high = high_sum / (non_zero_sum + 1e-9)
    block_mask = frac_high > frac_threshold

    num_qb = block_mask.shape[2]
    num_kb = block_mask.shape[3]
    diag_len = min(num_qb, num_kb)
    if diag_len > 0:
        diag_idx = torch.arange(diag_len, device=block_mask.device)
        block_mask[:, :, diag_idx, diag_idx] = True

    return block_mask


def _block_mask_gen_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    k_param: torch.Tensor,
    softmax_scale: float,
    precise: bool,
    cumsum_threshold: float,
    block_size: int,
) -> torch.Tensor:
    """CUDA kernel for block-level mask generation.

    Returns:
        block_counts: [B, H, num_qb, block_size, num_kb] int32.
    """
    if not _HAS_MASK_GEN_KERNEL:
        raise RuntimeError("mask_gen_cuda kernel not available; reinstall with setup.py")

    if q.dtype == torch.float16:
        fn = mask_gen_cuda.block_mask_gen_precise if precise else mask_gen_cuda.block_mask_gen_quantile
    elif q.dtype == torch.bfloat16:
        fn = mask_gen_cuda.block_mask_gen_precise_bf16 if precise else mask_gen_cuda.block_mask_gen_quantile_bf16
    else:
        raise ValueError(f"Unsupported dtype: {q.dtype}")

    q_4d = q.contiguous() if q.dim() == 4 else q
    k_4d = k.contiguous() if k.dim() == 4 else k
    k_param_3d = k_param.contiguous() if k_param.dim() == 3 else k_param

    return fn(q_4d, k_4d, k_param_3d, softmax_scale, cumsum_threshold, block_size)


def _quantile_block_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    k_ratio: torch.Tensor,
    scale: float,
    causal: bool,
    block_size: int,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Python fallback: chunked quantile mask → block counts.

    Processes queries in chunks, computes token-level topk mask per chunk,
    then accumulates per-key-token hit counts matching shrinkMaskStrict
    semantics. Never materializes the full [B, H, Sq, Sk] token mask.

    Returns:
        block_counts: [B, H, num_qb, num_kb, block_size] int32.
            block_counts[b,h,qb,kb,kt] = number of query rows in q-block qb
            that attend to key token (kb*block_size + kt).
    """
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    device = q.device

    num_qb = (Sq + block_size - 1) // block_size
    num_kb = (Sk + block_size - 1) // block_size
    padded_Sk = num_kb * block_size

    q_ = q.float().permute(0, 2, 1, 3)
    k_ = k.float().permute(0, 2, 1, 3)
    k_t = k_.transpose(-2, -1)
    k_keep = (k_ratio * Sk).ceil().long().clamp(min=1, max=Sk).unsqueeze(-1)

    block_counts = torch.zeros(B, H, num_qb, num_kb, block_size,
                               dtype=torch.int32, device=device)

    for q_start in range(0, Sq, chunk_size):
        q_end = min(q_start + chunk_size, Sq)
        chunk_len = q_end - q_start

        q_chunk = q_[:, :, q_start:q_end]
        scores_chunk = torch.matmul(q_chunk, k_t) * scale

        if causal:
            row_idx = torch.arange(q_start, q_end, device=device).view(-1, 1)
            col_idx = torch.arange(Sk, device=device).view(1, -1)
            scores_chunk.masked_fill_(col_idx > row_idx, float('-inf'))

        kk = k_keep[:, :, q_start:q_end]
        kk_max = kk.max().item()
        topk_vals = torch.topk(scores_chunk, k=min(kk_max, Sk),
                               dim=-1, sorted=True).values
        thresholds = torch.gather(topk_vals, -1, (kk - 1).clamp(min=0))
        mask_chunk = (scores_chunk >= thresholds)

        if padded_Sk > Sk:
            mask_padded = torch.zeros(B, H, chunk_len, padded_Sk,
                                      dtype=torch.bool, device=device)
            mask_padded[:, :, :, :Sk] = mask_chunk
            mask_chunk = mask_padded

        mask_blocks = mask_chunk.view(B, H, chunk_len, num_kb, block_size)

        for i in range(chunk_len):
            abs_qr = q_start + i
            macro_qb = abs_qr // block_size
            block_counts[:, :, macro_qb] += mask_blocks[:, :, i].int()

    return block_counts


def flash_attn_with_block_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    entropy_to_k_ratio: torch.Tensor,
    block_size: int = 64,
    density_threshold: float = 1.0 / 3.0,
    frac_threshold: float = 0.6,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softmax_scale: Optional[float] = None,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    mask_chunk_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention forward + entropy + blockified sparse mask.

    Combines flash_attn_func_with_entropy (Block A), entropy-to-k_ratio
    lookup (Block B), and block-level mask generation (Block C) into a
    single call. The mask is returned at block granularity — no full
    [B, H, Sq, Sk] tensor is ever materialized.

    Args:
        q: [B, Sq, H, D], fp16/bf16.
        k: [B, Sk, H, D], fp16/bf16.
        v: [B, Sk, H, D], fp16/bf16.
        entropy_to_k_ratio: [2, P] fp32. row0 = entropy breakpoints
            (strictly increasing), row1 = corresponding k_ratio in (0, 1].
        block_size: block size for mask blockification (default 64).
        density_threshold: col_density > this → "high density" (default 1/3).
        frac_threshold: fraction of high-density rows > this → block active
            (default 0.6).
        causal: apply causal mask.
        window_size: (left, right) sliding-window attention.
        softmax_scale: default 1/sqrt(D).
        softcap: softcapping value (0 = disabled).
        alibi_slopes: ALiBi slopes.
        mask_chunk_size: query chunk size for Python fallback (default 512).

    Returns:
        out:        [B, Sq, H, D]  attention output.
        entropy:    [B, H, Sq]     per-token entropy (fp32).
        block_mask: [B, H, num_qb, num_kb] bool, where
                    num_qb = ceil(Sq / block_size),
                    num_kb = ceil(Sk / block_size).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    # Block A: flash attention + entropy
    out, entropy = flash_attn_func_with_entropy(
        q, k, v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
    )

    # Block B: entropy → k_ratio
    k_ratio = _entropy_to_k_ratio(entropy, entropy_to_k_ratio)

    # Block C: block mask generation
    k_param = (k_ratio * k.shape[1]).ceil().clamp(min=1, max=k.shape[1])

    if _HAS_MASK_GEN_KERNEL and not causal:
        block_counts = _block_mask_gen_cuda(
            q, k, k_param.float(), softmax_scale,
            precise=False, cumsum_threshold=0.9,
            block_size=block_size,
        )
    else:
        block_counts = _quantile_block_mask(
            q, k, k_ratio, softmax_scale, causal,
            block_size=block_size, chunk_size=mask_chunk_size,
        )

    block_mask = _shrink_block_counts(
        block_counts, block_size,
        density_threshold=density_threshold,
        frac_threshold=frac_threshold,
    )

    return out, entropy, block_mask


def flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_varlen_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_varlen_kvpacked_func and flash_attn_varlen_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (total, 3, nheads, headdim), where total = total number of tokens in the batch.
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into qkv.
        max_seqlen: int. Maximum sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnVarlenQKVPackedFunc.apply(
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of K, V.
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        kv: (total_k, 2, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnVarlenKVPackedFunc.apply(
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_table,
        torch.is_grad_enabled(),
    )


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (q.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)
    out, softmax_lse = flash_attn_gpu.fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        alibi_slopes,
        None,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        num_splits,
    )
    return (out, softmax_lse) if return_softmax_lse else out
