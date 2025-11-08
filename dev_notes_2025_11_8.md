# Dev Notes — 2025-11-08

- Environment: Torch 2.4.1+cu121, CUDA 12.1, A100.
- Fixed dataset loader expectations: converted split `dataset.json` from list → metadata mapping; added required fields.
- Hydra config: switched identifiers to `module@Class` and removed extra `models.` prefix where the loader preprends it; added `project_name`, `run_name`, `dataset_path`.
- AdamATan2: replaced with AdamW shim.
- FlashAttention: added local shim & fallback to PyTorch SDPA.
- Sparse embedding: replaced `nn.Buffer` with `nn.Parameter`; created `CastedSparseEmbedding` and compatibility shim `CastedSparseEmbeddingSignSGD_Distributed`.
- Loss: added `SoftmaxCrossEntropyLoss` wrapper over `softmax_cross_entropy`.
- Next: run training, watch for shape/dtype issues; commit patches.

(See ChatGPT canvas “dev_notes_2025_11_8” for the full detailed log.)
