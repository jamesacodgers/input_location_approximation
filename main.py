import os

from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger
from time import strftime

from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import NLL, BNN_VI_ELBO_Regression
from lightning_uq_box.viz_utils import (
    plot_calibration_uq_toolbox,
    plot_predictions_regression,
    plot_toy_regression_data,
)

from src.plot_utils import plot_training_metrics

plt.rcParams["figure.figsize"] = [14, 5]


seed_everything(0)  # seed everything for reproducibility

dataset_name = "test"

experiment_dir = f"experiments/{dataset_name}/{strftime('%Y%m%d_%H%M%S')}"

batch_size = 32
n_points = 512
temperature = 1.0

n_train_epochs = 5

dm = ToyHeteroscedasticDatamodule(batch_size=batch_size, n_points=n_points)

X_train, Y_train, train_loader, X_test, Y_test, test_loader, X_gtext, Y_gtext = (
    dm.X_train,
    dm.Y_train,
    dm.train_dataloader(),
    dm.X_test,
    dm.Y_test,
    dm.test_dataloader(),
    dm.X_gtext,
    dm.Y_gtext,
)

n_train_points = X_train.shape[0]

fig = plot_toy_regression_data(X_train, Y_train, X_test, Y_test)

network = MLP(n_inputs=1, n_hidden=[50, 50], n_outputs=2, activation_fn=nn.ReLU())
network

bbp_model = BNN_VI_ELBO_Regression(
    network,
    optimizer=partial(torch.optim.Adam, lr=3e-3),
    criterion=NLL(),
    stochastic_module_names=[-1],
    num_mc_samples_train=10,
    num_mc_samples_test=25,
    burnin_epochs=20,
    beta=batch_size/(n_train_points * temperature)
)

logger = CSVLogger(experiment_dir,version="")
trainer = Trainer(
    max_epochs=n_train_epochs,  # number of epochs we want to train
    logger=logger,  # log training metrics for later evaluation
    log_every_n_steps=1,
    enable_checkpointing=True,
    enable_progress_bar=True,
    default_root_dir=experiment_dir,
    enable_model_summary=True
)

trainer.fit(bbp_model, dm)

os.mkdir(experiment_dir + "/figs")
fig = plot_training_metrics(experiment_dir, metrics=["train_loss", "val_loss"])
plt.savefig(experiment_dir + "/figs/bbp_training.png")

preds = bbp_model.predict_step(X_gtext)

fig = plot_predictions_regression(
    X_train,
    Y_train,
    X_gtext,
    Y_gtext,
    preds["pred"].squeeze(-1),
    preds["pred_uct"],
    epistemic=preds["epistemic_uct"],
    aleatoric=preds["aleatoric_uct"],
    title="Bayes By Backprop MFVI",
    show_bands=False,
)

plt.savefig(experiment_dir + "/figs/bbp_predictions.png")

preds = bbp_model.predict_step(X_test)
fig = plot_calibration_uq_toolbox(
    preds["pred"].cpu().numpy(),
    preds["pred_uct"].cpu().numpy(),
    Y_test.cpu().numpy(),
    X_test.cpu().numpy(),
)

plt.savefig(experiment_dir + "/figs/bbp_calibration.png")