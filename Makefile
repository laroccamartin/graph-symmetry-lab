# Convenience commands. Use a virtualenv before running.
.PHONY: setup data train-torch train-jax test

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

data:
	python -m src.data.make_dataset --out data/synth-graphs.npz --n_graphs 300 --min_n 5 --max_n 7 --seed 0

train-torch:
	python -m src.training.torch_train --data data/synth-graphs.npz --epochs 10 --hidden_dim 64 --layers 2

train-jax:
	python -m src.training.jax_train --data data/synth-graphs.npz --epochs 10 --hidden_dim 64 --layers 2

test:
	pytest -q
