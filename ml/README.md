# Neural Surrogate Pricer

A PyTorch model that learns the option pricing map and prices a whole book in a
single forward pass. This is the deep learning side of the project and it walks
the full model lifecycle. Data generation, training, evaluation, and inference
each live in their own script.

## Why a surrogate

Monte Carlo gives an exact price but costs milliseconds per option. A trained
network gives an approximate price in microseconds and prices thousands of
options at once. A real time risk desk that needs to revalue a large book many
times a second cannot wait for Monte Carlo on every position, so it uses a
surrogate for the fast path and Monte Carlo for validation. This directory
builds that surrogate end to end.

## The lifecycle

### 1. Data

`dataset.py` samples option parameters uniformly across realistic ranges and
labels each one with the exact Black Scholes call price. Prices are computed per
unit strike so the model learns a scale invariant map. Because the label is
analytical the data is noise free, so any error the model shows is genuine
approximation error.

### 2. Training

`train.py` standardizes the features on the training split, trains the MLP with
Adam and mean squared error, tracks validation error every epoch, and saves the
best checkpoint. It writes a training curve to `loss_curve.png` and a metrics
file to `metrics.json`.

```bash
python3 ml/train.py                 # uses ml/configs/default.yaml
python3 ml/train.py --epochs 20     # quick run
```

### 3. Evaluation

`evaluate.py` reloads the checkpoint, regenerates the held out test split, and
reports MAE, RMSE, and MAPE broken out by moneyness bucket. Approximation error
is usually largest deep out of the money where the price is nearly zero, so the
bucket breakdown matters.

```bash
python3 ml/evaluate.py
```

### 4. Inference

`infer.py` measures how fast the trained network prices a large book and checks
the accuracy against Black Scholes on the same book. This is the number that
justifies the surrogate.

```bash
python3 ml/infer.py --n 200000
```

## Committed results

On the development machine using Apple MPS the surrogate reaches a validation
RMSE of about 0.0026 per unit strike after 60 epochs in roughly 7 seconds, and
at inference it prices around 545,000 options per second at under 2 microseconds
each. See `metrics.json`, `eval_report.json`, and `infer_report.json` for the
full numbers, and `loss_curve.png` for the training curve.

## Configuration

Everything is driven by `configs/default.yaml`. The data ranges, model width,
and training hyperparameters all live there so a run is reproducible and easy to
sweep. The device field is set to auto, which selects CUDA, then MPS, then CPU.
