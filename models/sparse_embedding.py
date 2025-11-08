import torch
from torch import nn

class CastedSparseEmbedding(nn.Module):
    """
    Safe drop-in: wraps nn.Embedding and exposes .weights for callers
    that expect a 'weights' attribute.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        padding_idx: int | None = None,
        sparse: bool = False,
    ):
        super().__init__()
        self.emb = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            sparse=sparse,
            device=device,
            dtype=dtype,
        )

    @property
    def weights(self) -> torch.Tensor:
        return self.emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(input_ids)


class CastedSparseEmbeddingSignSGD_Distributed(CastedSparseEmbedding):
    """Compatibility shim.

    The original repo updated embedding *buffers* with a custom distributed
    SignSGD routine. We refactored to a standard nn.Parameter-backed embedding,
    so a regular optimizer (AdamW/SGD) + DDP handles updates and sync.

    Any legacy kwargs related to SignSGD are ignored safely.
    """
    def __init__(self, *args, **kwargs):
        # Drop legacy optimizer-specific kwargs if passed by old configs
        for k in ("lr", "weight_decay", "world_size", "rank", "num_replicas", "signsgd", "shard"):
            kwargs.pop(k, None)
        super().__init__(*args, **kwargs)
