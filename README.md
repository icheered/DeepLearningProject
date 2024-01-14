# Deep learning project
## Installation
Makes sure you install all dependencies. This project uses [Poetry](https://python-poetry.org/) for dependency management and virtualization. If you don't have poetry installed (and don't want to install it), check `pyproject.toml` for the dependencies and install them manually using pip.
```bash
poetry install
```

The `inception-v3` model is included but in a zip file (otherwise it would be >100mb which is too large for github). Unzip it and keep it in the root folder.
```bash
unzip inception-v3.ckpt.zip
```

You can also download the inception-v3 model directly from kaggle, and the other dataset and dev toolkit and stuff (large file).
Download the [inception-v3](https://www.kaggle.com/datasets/google-brain/inception-v3?resource=download) model.
Download the [other dataset and dev toolkit and stuff](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack/data) (large file).