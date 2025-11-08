import torch
import torch.nn.functional as F

def flash_attn_func(q, k, v, dropout_p=0.0, causal=False, softmax_scale=None, **kwargs):
    """
    Drop-in replacement using PyTorch 2.x scaled_dot_product_attention.
    Signature matches the common FlashAttention wrapper.
    """
    return F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale
    )
