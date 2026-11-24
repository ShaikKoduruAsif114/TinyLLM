"""Measure perplexity as context length increases for all positional encodings."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.eval_utils import MODEL_NAMES, create_loader, evaluate, load_eval_model


CONTEXT_LENGTHS = (64, 128, 192, 256)
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main():
	RESULTS_DIR.mkdir(exist_ok=True)
	rows = []
	for model_name in MODEL_NAMES:
		for context_length in CONTEXT_LENGTHS:
			model = load_eval_model(model_name, context_length)
			average_loss, perplexity = evaluate(
				model, create_loader(context_length)
			)
			rows.append(
				{
					"Model": model_name.title() if model_name != "rope" else "RoPE",
					"Context Length": context_length,
					"Validation Loss": average_loss,
					"Perplexity": perplexity,
				}
			)
			del model

	results = pd.DataFrame(rows)
	results.to_csv(RESULTS_DIR / "perplexity_vs_length.csv", index=False)

	plt.figure(figsize=(8, 6))
	for model_name in ("RoPE", "Learned", "Sinusoidal"):
		data = results[results["Model"] == model_name]
		plt.plot(
			data["Context Length"],
			data["Perplexity"],
			marker="o",
			linewidth=2,
			label=model_name,
		)
	plt.xlabel("Context Length (tokens)")
	plt.ylabel("Perplexity")
	plt.title("Perplexity vs Context Length")
	plt.xticks(CONTEXT_LENGTHS)
	plt.yscale("log")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(RESULTS_DIR / "perplexity_vs_length.png", dpi=300)
	print(results.to_string(index=False))


if __name__ == "__main__":
	main()
