# TinyLLM

GitHub description: From-scratch GPT on TinyStories comparing RoPE, learned, and sinusoidal positional encodings for length extrapolation.

TinyLLM is a from-scratch 12.2M-parameter GPT trained on TinyStories. It is a controlled experiment isolating the effect of positional encoding on length extrapolation across RoPE, learned, and sinusoidal encodings.

## Reproduce

Install the dependencies and prepare the encoded data:

```bash
pip install -r requirements.txt
python prepare_data.py
```

Train each encoding by changing `config["pos_encoding"]` in `train.py` to `rope`, `learned`, and `sinusoidal`, running `python train.py` for each configuration, and preserving the resulting checkpoints as `checkpoints/rope_best.pt`, `checkpoints/learned_best.pt`, and `checkpoints/sinusoidal_best.pt`.

Run the analysis from the repository root:

```bash
python analysis/perplexity_vs_length.py
python analysis/embedding_similarity.py
python analysis/embedding_similarity.py --include-sinusoidal
python analysis/attention_viz.py
```

The scripts write CSV and PNG outputs under `results/`. Perplexity evaluation uses the same deterministic, non-overlapping validation windows and fixed maximum of 784 samples at every context length.

## Results

The table below reproduces the saved outputs from the uploaded evaluation notebook. The checkpoint artifacts are not included in this checkout, so run `analysis/perplexity_vs_length.py` after restoring them to regenerate `results/perplexity_vs_length.csv` and the plot.

| Context length | RoPE perplexity | Learned perplexity | Sinusoidal perplexity |
|---:|---:|---:|---:|
| 64 | 6.916048 | 7.018327 | 8.921249 |
| 128 | 7.002447 | 7.063384 | 9.140389 |
| 192 | 7.524043 | 10.011307 | 67.364614 |
| 256 | 9.441891 | 14.510801 | 204.373088 |

![Perplexity versus context length](results/perplexity_vs_length.png)

## Key finding

The intended comparison is that RoPE degrades gracefully as context length grows. Learned positional encoding collapses beyond the trained range because every position after 127 is deliberately clamped to the final learned embedding. This is a simulated-extrapolation methodology, not a bug: it makes the model's inability to represent unseen positions explicit. Sinusoidal encoding is mathematically defined at any length, yet the observed experiment shows catastrophic, abrupt degradation between 128 and 192 tokens. That behavior is the most interesting open question in this project; it has not been root-caused, so no stronger explanation is claimed here.