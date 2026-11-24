# TinyLLM — decoder-only Transformer and positional-encoding experiments

TinyLLM is an educational PyTorch GPT implementation for next-token prediction on TinyStories. It includes a custom word-based BPE tokenizer, six pre-LayerNorm Transformer blocks, six attention heads, 384-dimensional hidden states, tied token/output embeddings, and learned, sinusoidal, or rotary positional encodings.

## What is here

The repository root is the existing implementation baseline, not a newly retrained model. Recovered implementations are preserved separately rather than silently replacing the root code:

- `recovered/local-tinyllm/`: local source tree, forward-pass notebook, evaluation notebook, and the separate clean-evaluation script.
- `recovered/extra-notebooks/`: early Kaggle setup/preparation notebooks, a limited fixed workflow notebook, and an unexecuted end-to-end template.
- `recovered/archive-tinyllm-main/`: earlier source archive, including fixed-size attention masks and unclamped learned positions.
- `recovered/git-history/`: changed source/notebook versions from the five existing commits; these are partial historical changes, not standalone complete installations.
- `recovered/repository-baseline/`: preserved baseline snapshot.
- `recovered/reports/perplexity_vs_context_length.csv`: historical aggregate results, with important protocol and checkpoint-loading limitations below.

Public notebooks contain **cell source only**: outputs, execution counts, and execution metadata are stripped. Full original notebooks, datasets, checkpoints, archives, and private context remain local and are not bundled. Empty YAML configs are historical placeholders, not usable experiment configurations. The zero-byte workflow notebook is invalid and is not published as an implementation.

## Historical results — not a clean reproduced benchmark

| Evaluation context | RoPE perplexity | Learned perplexity | Sinusoidal perplexity |
|---:|---:|---:|---:|
| 64 | 6.916048 | 7.018327 | 8.921249 |
| 128 | 7.002447 | 7.063384 | 9.140389 |
| 192 | 7.524043 | 10.011307 | 67.364614 |
| 256 | 9.441891 | 14.510801 | 204.373088 |

These are saved validation outputs from `tinyllm.ipynb` cells 21, 24, 27–28 and the recovered CSV, **not fresh inference or training**. The reported training configuration uses context 128. Evaluation uses 784 deterministic windows at each length, so token coverage changes with length. The loader's `endswith("attn.bias")` test also removes trainable `c_attn.bias` values, and saved missing-key output confirms this. Learned-model construction at length 256 does not match the saved 128-position checkpoint shape in the visible code. Execution state/edits are therefore incompletely recoverable, and these numbers must not be presented as a validated, fair positional-encoding ranking. See `docs/EXPERIMENTS.md`.

## Architecture and execution boundaries

The tied-weight RoPE/sinusoidal architecture has **12,220,416 trainable parameters**, verified by static arithmetic, not model loading. Learned positions add 49,152 parameters at training length 128; the historical notebook prints a larger 256-position construction. Dynamic masking means a causal mask built to the current input length; it is not masked-language-model pretraining. Standard generation crops inputs to configured `block_size`, so 256-token evaluation does not automatically enable 256-token generation from a 128-context configuration.

### Setup and data placement

The historical root dependencies are listed in `requirements.txt` (`torch`, `numpy`, `pandas`, `matplotlib`, `datasets`, `tqdm`); versions are not pinned. No dependency install or runtime compatibility test was performed during recovery.

For a deliberately chosen, trusted implementation variant, work from that variant's project root, not a mixture of folders:

1. Use a separate Python environment; install that variant's dependencies if you choose to run it.
2. Use the supplied matched `vocab.json` and `merges.json`. The root and recovered encoded-data copies have different raw hashes but compare equal after JSON parsing (4096 vocabulary entries and the same ordered merge list); byte differences here do not establish a different tokenizer.
3. Provide encoded `data/train.npy` and `data/val.npy`, or intentionally run tokenizer/data preparation with appropriate dataset access and review of the split policy. `tokeniser/train_tokeniser.py` learns BPE from the first 50,000 training stories. `prepare_data.py` encodes 100,000 stories in the current source, appends EOS, writes uint16 token streams, and takes a final 1% token slice for validation. This is not the dataset's official validation split, and the split is not story-boundary aware.
4. `python train.py` starts/resumes the configured RoPE run, uses context 128 and writes generic `checkpoints/best.pt` and `latest.pt`. It does not automatically train three named variants. Review configuration before running: execution can overwrite artifacts.
5. `python analysis/perplexity_vs_length.py` expects three trusted named checkpoints at `checkpoints/{rope,learned,sinusoidal}_best.pt` and `data/val.npy`; it writes into `results/`. The recovered `run_clean_eval.py` instead expects `backup/checkpoints/` and `backup/encoded_data/`, and writes `clean_eval_results.json`. Those local model/data files are intentionally absent from this publication. Both loaders retain the broad suffix-filter issue; these commands are documented interfaces, not verified reproduction commands.

Never load an untrusted checkpoint merely to inspect it: historical scripts call `torch.load`. No checkpoint was deserialized during recovery. The Kaggle notebooks contain clone/install/write/patch/train cells and should be reviewed before execution. The end-to-end template still has a repository placeholder and a defective patch cell; its title does not establish an end-to-end working run.

## Documentation

- `docs/IMPLEMENTATION.md`: actual tokenizer, data, model and training behavior.
- `docs/EXPERIMENTS.md`: metric provenance, evaluation variants and limitations.
- `docs/RECOVERY.md`: recovered variants, publication boundary and missing artifacts.

This is a recovered research/learning project, not a production-ready language model, a fresh benchmark, or a claim that all historical experiments can currently be reproduced.


## Archive safety

Raw assets and original notebooks remain local under `private/`. Credentials are not copied. See [archive safety](docs/ARCHIVE-SAFETY.md) for verification limits and the existing-history warning.
