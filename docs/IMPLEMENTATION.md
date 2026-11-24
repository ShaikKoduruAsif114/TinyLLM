# Implementation audit

References use repository-relative source paths and one-based notebook cell numbers. Line references describe the preserved source; no model or notebook was executed during recovery.

## Tokenizer and data path

`tokeniser/bpe_tokeniser.py:12–17, 80–105` assigns PAD/BOS/EOS/UNK IDs 0/1/2/3. Training splits on whitespace, represents each word as characters plus `</w>`, greedily merges the most frequent adjacent pair, and allocates sorted vocabulary IDs (`:150–183`). Encoding selects the earliest learned merge rank and caches word segmentations (`:107–146,188–207`). This is word-based character BPE, not byte-level BPE; original whitespace is not preserved. Decode joins tokens, replaces end-of-word markers with spaces, skips PAD/BOS/EOS and renders UNK as `?` (`:212–235`).

`tokeniser/train_tokeniser.py:4–12` trains a target vocabulary of 4096 on the first 50,000 TinyStories train stories. `prepare_data.py:23–27,52–59,74–105,119–120` selects 100,000 train stories, encodes and appends EOS, converts to uint16, then splits the concatenated stream 99:1 by token index. The initial Git revision selected the full dataset instead; a later preparation notebook still saves partial full-dataset progress, not a completed preparation/training run. Tokenizer fitting precedes splitting, so training-only tokenizer fit on the final token split is not established. No official TinyStories validation/test benchmark is implemented by this script.

The root and recovered encoded-dataset tokenizer files both have vocabulary size 4096. Although raw hashes differ, fresh standard-library JSON comparison confirms equal token-to-ID dictionaries and equal ordered merge lists. These copies are semantically the same tokenizer artifacts; formatting/line-ending differences must not be promoted into separate tokenizer experiments. Keep tokenizer/array/checkpoint provenance together because this equality alone does not prove which preprocessing run produced a checkpoint. Array headers inspected without loading payloads identify recovered train/validation streams as little-endian uint16 with shapes `(19877768,)` and `(200786,)`. Header inspection establishes storage shape, not tokenizer lineage or dataset contents.

## Sampling and training

The initial Git `dataset.py` uses deterministic stride-one overlapping windows. Current `dataset.py:23–45,51–73` ignores the requested index and draws a Python `random.randint` start, with epoch length approximately token-count/context. Despite the comment, samples are not guaranteed non-overlapping. Validation uses the same random sampler through `dataloader.py:29–37,48–55`, even with `shuffle=False`.

`train.py:38–59` configures vocabulary 4096, context 128, hidden width 384, six heads, six blocks, dropout 0.1, RoPE, batch 32, 10 epochs, AdamW learning rate 3e-4, weight decay 0.01 and seed 42. The effective loader batch size comes from `dataloader.py:16–17`, not the unused standalone training batch variable. `train.py:62–67` seeds Torch/CUDA but not Python `random` or NumPy; no explicit per-loader seed setup is present. Validation loss is the unweighted mean of batch means (`:153–178`), not an explicitly token-weighted aggregate. Gradient clipping, optional AMP and epoch-wise cosine scheduling are implemented (`:128–150,220–233,279–289`).

Checkpoint dictionaries include epoch, model, optimizer, scheduler, config and best validation loss (`train.py:70–105`). The code uses generic best/latest names, while recovered historical files use six encoding-specific best/latest names; the complete training lineage/renaming procedure is not proved by that source. Scheduler state is saved before `scheduler.step()`, so exact resume scheduling deserves review. Restoring a checkpoint also deserializes it; this audit did not do so.

## Decoder, attention and positions

`model/gpt.py:41–117` ties token embeddings to the no-bias vocabulary projection. `model/block.py:19–37` uses two LayerNorms, residual self-attention and residual FFN. `model/feedforward.py:13–25` expands to 4× width, applies GELU, projects back and drops out. `model/attention.py:49–84` computes joint QKV, splits six 64-dimensional heads, applies optional Q/K RoPE, scaled dot-product causal attention and output projection. It explicitly materializes attention matrices and returns them for visualization; no FlashAttention or KV cache is implemented.

Current attention makes a lower-triangular boolean mask at input length (`model/attention.py:65–69`); the source archive and initial Git version used a persistent fixed-size `bias` mask. This fixes a mask-size constraint, not every limit to extrapolation. Learned positions are clamped to the last table entry (`model/positional_encoding.py:28–39`), so beyond-table positions lose distinct learned identities. Sinusoidal tables are prebuilt at twice configured context (`model/gpt.py:52–56`); RoPE computes frequencies at runtime and rotates half-channel pairs (`model/positional_encoding.py:5–7,42–58`).

`model/gpt.py:194–198` computes shifted-token cross entropy ignoring PAD ID 0. The scripts' token weighting uses all target elements, not explicitly non-PAD counts; no PAD values were counted during this read-only audit. `model/gpt.py:218–242` generates with temperature/top-k sampling and context cropping to configured `block_size`; no server, instruction tuning, chat interface or production API is present.

### Parameter-count check without importing PyTorch

For tied vocabulary size V=4096, width C=384 and N=6 blocks, the source gives `V*C + N*(12*C*C + 13*C) + 2*C = 12,220,416`. Learned positions add `context*C`: 12,269,568 at context 128 and 12,318,720 at context 256. RoPE buffers and sinusoidal tables are not trainable parameters. The notebook's larger learned count is a construction detail, not evidence that a compatible 256-position model was trained.

## Notebook and script boundaries

- `01_forward_pass.ipynb` is a forward-pass teaching trace with saved historical outputs, not a training run. Cells 19/22 say nine tokens while cell 20 outputs six. Cell 48 saves `Allclose: False` with a nonzero manual/module attention discrepancy; later FFN comparison succeeds. Do not call every check passed.
- `tinyllm.ipynb` restores an external backup, modifies source files in cells 12/14, reloads modules, changes evaluation context and produces historical figures/text. These side effects and nonsequential execution history matter.
- `notebook_source.py` is a notebook export containing IPython magics and shell commands, not a directly runnable plain Python module. `notebook_outputs.txt` is retained privately because it is an unsanitized output dump.
- The auxiliary attention comparison script still reads `first_attn.bias` (`scripts/test_attention_compare.py:43`), which current dynamic-mask attention no longer defines. It is not a validated current smoke test.
