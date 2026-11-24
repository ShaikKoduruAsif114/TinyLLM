# Recovery and publication boundary

## What was recovered

The existing repository baseline was at `5d40a46` before publication. Its root implementation already includes dynamic attention masks, learned-position clamping and analysis utilities. A separate local tree has additional untracked evaluation/export artifacts and private model/data backups; recovery preserves these separately, without rewriting canonical ML code.

The original repository contains five commits. Changed source/notebook versions are preserved by short commit under `recovered/git-history/`:

| Revision | Distinction |
|---|---|
|`6c91dc9`|Initial implementation, stride-one dataset, fixed attention mask, unclamped learned positions; includes empty analysis/config placeholders.|
|`3352ac4`|BPE merge-rank caching and tokenizer test updates.|
|`2190cf7`|Preparation subset changed to 100,000 stories.|
|`e5d40d9`|Random sampling and configurable samples per epoch.|
|`5d40a46`|Dynamic-mask/learned-clamp changes, evaluation utilities, README and evaluation notebook.|

These history folders contain files changed in each commit, not complete runnable snapshots. Preserve the root or full baseline when assembling a historical version; do not import arbitrary individual files from different variants.

### Notebook variants

- `local-tinyllm/01_forward_pass.ipynb`: 57-cell educational forward-pass notebook with historical outputs retained privately.
- `local-tinyllm/tinyllm.ipynb`: 59-cell checkpoint-restoration and positional-encoding evaluation notebook; source and output integrity issues are documented in `EXPERIMENTS.md`.
- `extra-notebooks/notebook6402d029ab.ipynb`: 14-cell Kaggle setup and partial TinyStories preparation; saved output stops around 2% of full-dataset encoding, not completed training.
- `extra-notebooks/notebookea2a3eb80e.ipynb`: 8-cell Kaggle environment/clone setup, not training.
- `extra-notebooks/TinyLLM_Complete_Kaggle_Workflow_fixed.ipynb`: 5 cells; clone/inspection/GPU check and markdown next steps, no saved completed tokenizer/preparation/training outputs.
- `extra-notebooks/TinyLLM_End_to_End_Kaggle.ipynb`: 6-cell unexecuted template; retains `<YOUR_TINYLLM_REPO_URL>`. Its patch prepends parser usage before importing argparse and inserts a literal escaped newline in a replacement; do not describe it as ready to run. It contains deletion/copy/training steps for named checkpoints, but no evidence those steps ran.
- A similarly named complete workflow original is zero bytes. It is preserved privately as invalid, not replaced by a fictitious implementation.
- The source archive also contains the forward-pass notebook. Physically distinct source copies are preserved even where identical.

### Archives and local-only artifacts

A source archive was inspected and its 25 file members recovered separately under `recovered/archive-tinyllm-main/`. It retains fixed-mask/unclamped-position source and empty analysis/config files; it is not interchangeable with the current root. The raw archive itself remains local.

Model/data archives were only listed, not deserialized: a full backup contains six encoding-specific best/latest checkpoints plus encoded data/tokenizer files; a RoPE weights archive contains the two RoPE checkpoints; an encoded-dataset archive contains train/validation arrays and its tokenizer pair. Six unpacked checkpoint files and a separate encoded-data folder were also inventoried. Recovered array storage headers were read without loading arrays. Large model/archive files are metadata-only in the inventory; small files were streaming-SHA256 hashed. These backups are not public release assets.

## Historical context and gaps

Existing Git history, notebook text/source/output evidence, and project-specific review notes were inspected read-only. Later project/resume-review context corroborates where claims were repeated; it is not proof of original training runs. Private notes and raw chats are not published. Original development conversations beyond those accessible records remain incomplete, and absence from a partial full-text index is not evidence that no history exists.

Saved notebook text mentions `perplexity_vs_context_length.csv`, plot images, attention heatmaps and embedding-similarity figures. The aggregate CSV was recovered separately; standalone named figure files and `clean_eval_results.json` were not found in the bounded discovery inventory. Embedded notebook MIME outputs are retained only in private originals, not republished as newly rendered figures.

## Verification actually performed

- Read-only source inspection, Git status/history inspection, ZIP member listing, JSON notebook parsing, storage-header inspection and streaming hashes.
- A static parameter-count calculation and arithmetic consistency check of saved CSV perplexities.
- Preservation of separate physical variants and historical changed-file versions, with publication classifications in a private flat inventory.
- **Not performed:** model deserialization, notebook execution, training, inference, package installation, downloads, benchmark repair or deployment validation.

Public notebooks are source-only, with all outputs, execution counts and execution metadata stripped. Private checkpoints, datasets, original archives, invalid zero-byte notebooks, raw output dumps, personal paths and conversational transcripts remain outside the public tree. Follow the final publication manifest for exactly which safe files were copied; this document does not claim that private artifacts are downloadable.
