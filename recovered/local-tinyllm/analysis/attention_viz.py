"""Visualize all-layer, all-head attention for one shared input sentence."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.eval_utils import DEVICE, load_eval_model
from tokeniser.bpe_tokeniser import BPETokenizer


def plot_attention(model_name, attention, output_path):
	layer_count = len(attention)
	head_count = attention[0].shape[1]
	figure, axes = plt.subplots(
		layer_count,
		head_count,
		figsize=(3 * head_count, 3 * layer_count),
		squeeze=False,
	)
	for layer_index in range(layer_count):
		for head_index in range(head_count):
			axes[layer_index, head_index].imshow(
				attention[layer_index][0, head_index].cpu().numpy(),
				cmap="viridis",
				origin="lower",
				interpolation="nearest",
			)
			axes[layer_index, head_index].set_xticks([])
			axes[layer_index, head_index].set_yticks([])
			if layer_index == 0:
				axes[layer_index, head_index].set_title(f"Head {head_index + 1}")
			if head_index == 0:
				axes[layer_index, head_index].set_ylabel(f"Layer {layer_index + 1}")
	figure.suptitle(f"{model_name} attention patterns")
	figure.tight_layout()
	figure.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(figure)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--sentence",
		default="Once upon a time there was a",
		help="Sentence encoded and passed to every model.",
	)
	args = parser.parse_args()

	root = Path(__file__).resolve().parents[1]
	results_dir = root / "results"
	results_dir.mkdir(exist_ok=True)
	tokenizer = BPETokenizer()
	tokenizer.load(root / "vocab.json", root / "merges.json")
	token_ids = tokenizer.encode(args.sentence)
	if not token_ids or len(token_ids) > 256:
		raise ValueError("The sentence must encode to between 1 and 256 tokens.")
	inputs = torch.tensor(token_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

	for model_name in ("rope", "learned", "sinusoidal"):
		model = load_eval_model(model_name, 256)
		with torch.no_grad():
			_, _, attention = model(inputs)
		output_path = results_dir / f"{model_name}_attention_heatmaps.png"
		plot_attention(model_name.title() if model_name != "rope" else "RoPE", attention, output_path)
		print(f"Saved {output_path}")
		del model


if __name__ == "__main__":
	main()
