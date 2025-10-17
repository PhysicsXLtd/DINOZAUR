# DINOZAUR

**NeurPS 2025 spotlight [[Arxiv](https://arxiv.org/abs/2508.00643)]**

This is an official implementation of the paper _Albert Matveev, Sanmitra Ghosh, Aamal Hussain, James-Michael Leahy, Michalis Michaelides. Light-Weight Diffusion Multiplier and Uncertainty Quantification for Fourier Neural Operators_

![Teaser](assets/bayesian_pred.png)

## Getting started

### Requirements

- Python >=3.10, <3.13.

- Pytorch >=2.6.0, <2.8.0.

- The code is tested with CUDA 12.8 toolkit with `python=3.11`, `torch==2.7.1` on NVIDIA A100 with 80GB memory.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/PhysicsXLtd/DINOZAUR.git
cd DINOZAUR
```

2. Create conda environment:
```bash
conda create -n dinozaur python=3.11
conda activate dinozaur
```

3. Install the package
```bash
pip install .
```
Installation will build [`torch-scatter`](https://github.com/rusty1s/pytorch_scatter) from source, so it may take a few minutes.

## Getting the data

The data is published at [HuggingFace data repository](https://huggingface.co/datasets/PhysicsX/DINOZAUR/tree/main). The repository contains pre-processed `.h5` data files, one per sample. See the HuggingFace README file for the original sources of data.

To download the data, run:
```bash
git clone https://huggingface.co/datasets/PhysicsX/DINOZAUR data
```

The dataset requires about 22GB of space. Files will be saved in `data/` folder.

## Running the code

To train a model, run:
```bash
python train.py path/to/config.yml
```
We provide configs to replicate experiments presented in the paper in the folder `config/`. The training script will print metric values at the end of the run and save the model checkpoint in the folder `logs/`.

To fit the Laplace approximation, you will need a deterministic model trained beforehand. Once the trained model is checkpointed, run:
```bash
python fit_laplace_approximation.py logs/path/to/model.pt
```

### Linting and formatting

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and code formatting. Here is an example of usage:

```bash
# Run linter
ruff check .

# Run linter with auto-fix
ruff check --fix .

# Format code
ruff format .
```

Alternatively, you may install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) for automatic linting and formatting in VS Code.

## Acknowledgments

Many people contributed at both the conceptual and implementation levels to the PyTorch models in this repository. We would like to explicitly acknowledge the contributors who collaborated on the codebase: [Greg Bellchambers](https://github.com/gregb-px), [Bachir Djermani](https://github.com/bachdj-px), [Axen Georget](https://github.com/axen-px), [Sanmitra Ghosh](https://github.com/sanmitrapx), Aamal Hussain, [James-Michael Leahy](https://github.com/j-mleahy), [Albert Matveev](https://github.com/albertmatveev), [Pavel Shmakov](https://github.com/pavel-shmakov), and [Phoenix Tse](https://github.com/phoenix-tse-px).

## Citation

If you find this work useful, please consider citing:
```
@article{matveev2025light,
  title={Light-Weight Diffusion Multiplier and Uncertainty Quantification for Fourier Neural Operators},
  author={Matveev, Albert and Ghosh, Sanmitra and Hussain, Aamal and Leahy, James-Michael and Michaelides, Michalis},
  journal={arXiv preprint arXiv:2508.00643},
  year={2025}
}
```

## License
[GPL-3.0](http://choosealicense.com/licenses/gpl-3.0/)

[![Logo](assets/physicsx.svg)](https://www.physicsx.ai/)
