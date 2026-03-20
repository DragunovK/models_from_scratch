# Models From Scratch

Deep learning fundamentals exploration through implementation of a deep learning framework from scratch based on `numpy`,
including `autograd` engine inspired by Andrej Karpathy's YouTube video.

## Structure

`src` folder contains autograd and deep learning framework source code.

`notebooks` directory contains Jupyter notebooks with experiments and demostrations.

`notes` directory is a Quarto project with notes gathered along the way. 

## Install and Run

> Ideally, use `uv`

To initialize the environment and install the packages, run:

```{bash}
uv sync
```

To use the module nicely in the notebooks, we need to install `mini-dl` module from local sources by running the following command:

```{bash}
python -m pip install -e .
```

`Alternatively`, to avoid uncertainties about kernel used in the notebook and terminal (might occur when using conda and uv), run this in a notebook cell:
```{Jupyter Notebook}
%pip install -e .
```

## Previewing Quarto notes

Make sure Quarto is installed.

From `./notes` directory, run:

```{bash}
quarto preview .
```
