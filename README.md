# TinyLlama From Scratch

This repo is a from-scratch, step-by-step reproduction of the core ideas behind
[TinyLlama](https://arxiv.org/abs/2401.02385), built as a learning project rather
than a faithful full-scale replication. Instead of writing one big model file, each
architectural technique (RMSNorm, RoPE, SwiGLU, GQA) gets its own notebook that
loads the previous checkpoint, adds exactly one new idea, and saves a new
checkpoint — so you can see, cell by cell, what each change actually does to the
model.

**Scale note:** the original TinyLlama is a 1.1B-parameter model pretrained on
~1 trillion tokens across many GPUs. This reproduction trains a much smaller
(~50M param) model on the [TinyStories](https://arxiv.org/abs/2305.07759) dataset,
on a single free-tier Google Colab T4 GPU. The goal is understanding the
architecture and training pipeline end to end, not matching TinyLlama's actual
benchmark numbers.

## Resources / Credits

- **TinyLlama paper**: Zhang, Zeng, Wang, Lu — [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385) (arXiv:2401.02385)
- **TinyLlama official code**: [github.com/jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama)
- **TinyStories paper**: Eldan, Li — [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) (arXiv:2305.07759)
- **TinyStories dataset**: [huggingface.co/datasets/roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- Architecture also draws on **LLaMA** / **LLaMA 2** (RMSNorm, RoPE, SwiGLU, GQA are all inherited from that lineage, which TinyLlama itself builds on)

## Repo Structure

```
TinyLlama-Reproduction/
├── notebooks/          # 12 notebooks, run in order (see below)
├── src/                # final model code exported by each notebook (committed)
├── config/             # tokenizer/model config json (committed, small)
├── assets/tokenizer/   # trained tokenizer files (committed, small)
├── outputs/            # eval metrics, loss curve, sample generations (committed)
├── logs/                # train_log.csv (committed, small)
├── data/               # tokenized dataset chunks -- NOT committed (see below)
├── checkpoints/         # model .pt files -- NOT committed (see below)
├── notes/
├── paper/
└── requirements.txt
```

**`data/` and `checkpoints/` are intentionally empty in this repo** — tokenized
data (GBs of Arrow files) and model checkpoints (each 50MB–few hundred MB) live
on Google Drive, not in git. Everything in the pipeline is built to save its
outputs there. See the "Running This" section below for how to get set up.

## Running This

Each notebook mounts Google Drive and reads/writes under one shared project
folder (`MyDrive/TinyLlama-repo/`). Open the notebooks in Google Colab, run them
**in order** (01 → 12), and each one automatically picks up what the previous
one produced. Every notebook is independently re-runnable — checkpoints are
skipped/regenerated based on whether they already match the current code,
so nothing gets silently duplicated or gets stale.

Notebook 9 (pretraining) is the one long-running step: it self-limits by
wall-clock time (not a fixed step count) and auto-resumes from the latest
checkpoint if a Colab session disconnects — just re-run it.

## Notebooks

| #   | Notebook                  | What it does                                                                                                                                                                                                        |
| --- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 01  | `dataset_preparation`     | Downloads TinyStories, trains a custom BPE tokenizer (vocab=4096), tokenizes the dataset into resumable chunks.                                                                                                     |
| 02  | `model_architecture`      | Builds a plain GPT-style baseline (learned positional embeddings, LayerNorm, standard MHA, GELU FFN) from scratch.                                                                                                  |
| 03  | `rmsnorm`                 | Replaces LayerNorm with RMSNorm throughout the model.                                                                                                                                                               |
| 04  | `rope`                    | Replaces learned positional embeddings with Rotary Position Embeddings (RoPE) applied inside attention.                                                                                                             |
| 05  | `swiglu`                  | Replaces the plain FFN with a SwiGLU (gated) feedforward block.                                                                                                                                                     |
| 06  | `grouped_query_attention` | Replaces standard multi-head attention with Grouped Query Attention (shared K/V across head groups).                                                                                                                |
| 07  | `tinyllama_assembly`      | Consolidates all four techniques into the final, frozen `TinyLlama` model class.                                                                                                                                    |
| 08  | `training_pipeline`       | Builds reusable training utilities: dataset/collate, AdamW with weight-decay groups, cosine LR schedule, AMP, gradient accumulation, checkpoint/resume logic. Smoke-tests everything on a small slice of real data. |
| 09  | `pretraining_run`         | The actual training run. Self-limits by wall-clock time, fully resumable across Colab sessions.                                                                                                                     |
| 10  | `evaluation`              | Computes validation loss/perplexity and plots the training loss curve.                                                                                                                                              |
| 11  | `inference_generation`    | Text generation demo: greedy, temperature, top-k, and top-p (nucleus) sampling.                                                                                                                                     |
| 12  | `export_quantize`         | Packages the final model as fp16 and a dynamically-quantized int8 version for smaller/CPU deployment.                                                                                                               |

## Architecture Summary

- **Tokenizer**: custom-trained byte-level BPE, vocab size 4096
- **Normalization**: RMSNorm (pre-norm)
- **Position encoding**: RoPE (no learned position embeddings)
- **Attention**: Grouped Query Attention (8 query heads, 2 KV heads)
- **Feedforward**: SwiGLU
- **Precision**: trained with AMP (fp16 autocast + GradScaler), checkpoints stored fp16

## Honest Limitations

- Not trained on anywhere near TinyLlama's 1T-token scale — expect rough,
  sometimes repetitive generations, not fluent long-form text.
- The validation set used in NB10 was carved out of the training pool after
  the fact (no held-out split existed before training started) — treat those
  numbers as a sanity check, not a rigorous benchmark.
- Educational focus: code favors clarity and step-by-step comparison over
  training-throughput optimizations like FlashAttention or KV-caching during
  generation.
