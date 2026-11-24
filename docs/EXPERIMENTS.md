# Experiment evidence and limits

## Historical result record

`recovered/reports/perplexity_vs_context_length.csv` has twelve aggregate rows. Its values agree with the saved tables in `recovered/local-tinyllm/tinyllm.ipynb`, one-based cells 21 (RoPE), 24 (learned), 27 (sinusoidal) and 28 (combined). Every row's perplexity agrees with `exp(validation_loss)` to floating-point precision in a new arithmetic-only consistency check. This validates transcription/arithmetic, not predictions.

| Context | RoPE loss / PPL | Learned loss / PPL | Sinusoidal loss / PPL |
|---:|---:|---:|---:|
|64|1.933844 / 6.916048|1.948525 / 7.018327|2.188436 / 8.921249|
|128|1.946260 / 7.002447|1.954924 / 7.063384|2.212703 / 9.140389|
|192|2.018104 / 7.524043|2.303715 / 10.011307|4.210120 / 67.364614|
|256|2.245156 / 9.441891|2.674893 / 14.510801|5.319947 / 204.373088|

The resume's rounded 9.44 / 14.51 / 204.37 figures have historical output provenance. They are validation—not independently held-out test—results and have unresolved evaluation integrity problems.

## Distinct evaluation variants

1. **Training validation:** `train.py:153–178` averages batch losses from randomly sampled windows. Cell 4 of `tinyllm.ipynb` saves a RoPE checkpoint config with context 128 and best validation loss 2.0371786064031174. That checkpoint's internal fields were not read in this audit; this value comes from historical text output.
2. **Notebook fixed-count windows:** notebook cell 5 and `analysis/eval_utils.py:20–37` choose starts from zero at stride L, capped at 784. The four lengths cover 50,176 / 100,352 / 150,528 / 200,704 target tokens respectively. Models at the same L share windows, but comparisons across L also change token coverage and reset positions at different boundaries. Cell 7's saved RoPE PPL at 128 is 7.002447066503277, distinct from saved training-validation loss.
3. **Notebook expanded model:** cell 19, lines 17–20, replaces checkpoint `block_size` with evaluation length before constructing a new model. It deletes matching state keys and loads with `strict=False` (lines 27–39). Cell 20 saves missing-key diagnostics; later outputs suggest additional unrecorded/edited execution state.
4. **Root analysis utility:** `analysis/eval_utils.py:50–80` expands construction context too, but replaces a learned table with its trained-size table before load. It preserves learned table dimensions and uses current clamping. It still silently removes some trained biases due to the suffix condition, so it is not a clean correction of all notebook issues.
5. **Recovered clean-eval script:** `recovered/local-tinyllm/run_clean_eval.py:42–68,89–118` keeps original checkpoint config, selects a common token prefix divisible by 768, and evaluates four contexts with differing sequence counts but equal total coverage. It attempts dynamic-mask/learned-clamp monkey patches. No `clean_eval_results.json` was found in the searched inventory; no fresh run is claimed. Its loader retains the same trainable-bias deletion bug. Qualified versus bare module imports can also mean that monkey patches address a different class object from the GPT block's imported class.

## Blocking interpretation issues

**Trainable QKV bias deletion:** Python's literal suffix test returns true for both `blocks.0.attn.bias` and `blocks.0.attn.c_attn.bias`. The notebook cell 19 condition `k.endswith("attn.bias")` therefore deletes the fixed mask and the QKV projection bias. Cell 20 explicitly lists all six `c_attn.bias` keys as missing. New linear layers initialize those biases to zero (`model/gpt.py:127–135`), so evaluation can differ from trained weights. The same condition appears in `analysis/eval_utils.py:64–66` and `run_clean_eval.py:65–68`. An audit must not describe this as dropping only causal-mask buffers.

**Learned table shape inconsistency:** cell 19 constructs a learned embedding table of length 256, while a context-128 checkpoint would contain length 128. The visible loader does not resize/copy that table. `strict=False` does not by itself resolve incompatible tensor sizes. Yet saved cells 23/24 show success and scores, and cells 41/54 print 12,318,720 parameters and shape `(256,384)`. This does not establish how those learned weights were actually produced. Do not silently attribute the learned result to trained-table clamping or to a corrected loader.

**State/provenance gaps:** notebook patches, module reloads, nonsequential execution counts and stale outputs prevent reconstruction of the exact effective code at each historical metric. Generic training checkpoint names differ from restored named files. No complete three-run training logs, seed sweep, matched training budgets, uncertainty intervals or clean held-out test rerun were recovered. No checkpoint was loaded to verify optimizer state or exact configuration.

**No universal ranking:** these saved numbers do not establish that sinusoidal positions generally fail or that RoPE is universally superior. The experiment depends on this data stream, trained checkpoints, coverage policy and loader state. Private revision notes contain conflicting lower-context values; those notes are not accepted metric evidence or republished.

The safe statement is: *the recovered notebook and CSV record these validation perplexities, but source/provenance issues prevent treating them as a clean reproduced ablation.*
