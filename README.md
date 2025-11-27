# FRAME-FM

1. Create & activate the PyTorch Lightning environment
```bash
# Using mamba (recommended)
mamba env create -f envs/Lightning_environment.yml

# Or with conda
conda env create -f envs/Lightning_environment.yml

# Activate
conda activate FRAMETL
```

## Running the framework
```bash
conda activate FRAMETL

python -m src.TorchLightning.train --mnist
# Optional overrides:
# python -m src.TorchLightning.train --mnist --epochs 20 --batch-size 256
```

OR
```
conda activate FRAMETL

python -m src.TorchLightning.train --ocean
# Optional overrides:
# python -m src.TorchLightning.train --ocean --epochs 30 --batch-size 128
```

## For Keras Projects Build the Keras Environment
```bash
# Using mamba (recommended)
mamba env create -f envs/keras_environment.yml
# Or with conda
conda env create -f envs/keras_environment.yml
# Activate
conda activate FRAMEKER

```
## Running the Keras framework
```bash
conda activate FRAMEKER
python -m src.Keras.train --mnist
# Optional overrides:
# python -m src.Keras.train --mnist --epochs 20 --batch-size
```
OR
```bash
conda activate FRAMEKER
python -m src.Keras.train --ocean
# Optional overrides:
# python -m src.Keras.train --ocean --epochs 30 --batch-size
```


