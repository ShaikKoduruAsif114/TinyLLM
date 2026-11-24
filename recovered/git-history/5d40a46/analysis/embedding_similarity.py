"""Plot cosine similarity between positional vectors."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.eval_utils import load_eval_model


def similarity_matrix(embeddings):
	normalized = F.normalize(embeddings, dim=1)
	return normalized @ normalized.T


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--include-sinusoidal",
		action="store_true",
		help="Also plot the first learned-context sinusoidal vectors.",
	)
	args = parser.parse_args()

	results_dir = Path(__file__).resolve().parents[1] / "results"
	results_dir.mkdir(exist_ok=True)
	matrices = {}

	learned_model = load_eval_model("learned", 128)
	matrices["Learned"] = similarity_matrix(
		learned_model.pos_encoding.pos_embed.weight.detach().cpu()
	)
	del learned_model

	if args.include_sinusoidal:
		sinusoidal_model = load_eval_model("sinusoidal", 128)
		matrices["Sinusoidal"] = similarity_matrix(
			sinusoidal_model.pos_encoding.pe[0, :128].detach().cpu()
		)
		del sinusoidal_model

	figure, axes = plt.subplots(
		1, len(matrices), figsize=(7 * len(matrices), 6), squeeze=False
	)
	for axis, (name, matrix) in zip(axes[0], matrices.items()):
		image = axis.imshow(matrix.numpy(), cmap="viridis", vmin=-1, vmax=1)
		axis.set_title(f"{name} positional similarity")
		axis.set_xlabel("Position")
		axis.set_ylabel("Position")
		figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
	figure.tight_layout()
	figure.savefig(results_dir / "embedding_similarity.png", dpi=300)
	print(f"Saved {results_dir / 'embedding_similarity.png'}")


if __name__ == "__main__":
	main()
