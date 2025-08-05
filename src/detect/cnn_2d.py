# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from loguru import logger
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.constants import CNN_DTYPE, DEVICE
from .config import DetectConfig


def get_n_layers_from_model_name(model_size: str) -> int:
    if "12b" in model_size:
        return 36
    if "6.9b" in model_size:
        return 32
    if "2.8b" in model_size:
        return 32
    if "1.4b" in model_size:
        return 24
    if "1b" in model_size:
        return 16
    if "410m" in model_size:
        return 24
    if "160m" in model_size:
        return 12
    if "70m" in model_size:
        return 6

    raise ValueError(f"Unknown n_layers for model_size={model_size}")


def get_fc_size(
    n_layers: int, n_feat_cnn: int, kernel_size: int, pooling_kernel: int
) -> int:

    conv1 = nn.Conv2d(
        in_channels=n_layers, out_channels=n_feat_cnn, kernel_size=kernel_size
    )
    conv2 = nn.Conv2d(
        in_channels=n_feat_cnn, out_channels=n_feat_cnn, kernel_size=kernel_size
    )

    with torch.no_grad():
        bs = 1
        x = torch.zeros(bs, n_layers, 64, 64).to(
            device=conv1.weight.device, dtype=conv1.weight.dtype
        )

        # First convolution
        x = conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=pooling_kernel)

        # Second convolution
        x = conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=pooling_kernel)

        return x.numel()


class Cnn2D(nn.Module):
    def __init__(
        self,
        model_size: str,
        n_class: int,
        n_feat_cnn: int = DetectConfig.n_feat_cnn,
        kernel_size: int = DetectConfig.kernel_size,
        pooling_kernel: int = DetectConfig.pooling_kernel,
        n_feat_fc: int = DetectConfig.n_feat_fc,
        dropout: float = 0.5,
        head_pooling: str = DetectConfig.head_pooling,
    ):
        # Init
        super().__init__()

        # Saving params
        self.n_layers = get_n_layers_from_model_name(model_size)
        self.n_feat_cnn = n_feat_cnn
        self.kernel_size = kernel_size
        self.n_feat_fc = n_feat_fc
        self.pooling_kernel = pooling_kernel
        self.n_class = n_class
        if head_pooling == "max":
            self.head_pooling_max = True
        elif head_pooling == "mean":
            self.head_pooling_max = False
        else:
            raise ValueError(f"head_pooling={head_pooling} not in ['mean', 'max'].")

        # Init parameters
        self.dropout = nn.Dropout(p=dropout)
        self._dropout = nn.Dropout(p=dropout)
        self.conv1 = nn.Conv2d(
            in_channels=self.n_layers,
            out_channels=n_feat_cnn,
            kernel_size=self.kernel_size,
        )
        self.conv2 = nn.Conv2d(
            in_channels=n_feat_cnn,
            out_channels=n_feat_cnn,
            kernel_size=self.kernel_size,
        )
        self.fc1 = nn.Linear(
            in_features=get_fc_size(
                self.n_layers, n_feat_cnn, self.kernel_size, pooling_kernel
            ),
            out_features=self.n_feat_fc,
        )
        self.fc2 = nn.Linear(n_feat_fc, n_class)

    def deactivate_dropout(self):
        self.dropout = nn.Dropout(p=0)

    def activate_dropout(self):
        self.dropout = self._dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: shape (bs, 36, 40, 64, 64)
        if self.head_pooling_max:
            x = torch.max(x, dim=2).values  # Shape (bs, 36, 64, 64)
        else:
            x = torch.mean(x, dim=2)  # Shape (bs, 36, 64, 64)

        bs = x.size(0)
        x = x.view(bs, self.n_layers, 64, 64)

        # First convolution
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = F.max_pool2d(x, kernel_size=self.pooling_kernel)

        # Second convolution
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = F.max_pool2d(x, kernel_size=self.pooling_kernel)

        # Fully connected 1
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Fully connected 2
        x = self.fc2(x)

        return x


def eval_cnn(
    model: nn.Module,
    eval_dataloader: DataLoader,
    criterion: nn.Module,
    silent: bool = False,
):
    """Evaluates the model"""
    model.eval()
    num_correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)
    overall_correct = 0
    total_item = 0
    loss = 0

    to_save_keys = []
    to_save_outputs_class = []
    to_save_labels = []

    with torch.no_grad():
        for keys, patterns, labels in tqdm(eval_dataloader, leave=(not silent)):
            patterns, labels = (
                patterns.to(dtype=CNN_DTYPE, device=DEVICE),
                labels.to(DEVICE).long(),
            )

            outputs_class = model(patterns)
            _, predictions = torch.max(outputs_class, dim=1)
            loss += criterion(outputs_class, labels).item()

            # Update overall accuracy
            overall_correct += (predictions == labels).sum().item()
            total_item += labels.size(0)

            # Updating to_save tensors
            to_save_keys.append(keys)
            to_save_outputs_class.append(outputs_class)
            to_save_labels.append(labels)

            # Update per-class accuracy
            for label, prediction in zip(labels, predictions):
                total_per_class[label.item()] += 1
                if label == prediction:
                    num_correct_per_class[label.item()] += 1

    # Compiling metrics
    per_class_accuracy = {
        class_idx: num_correct_per_class[class_idx] / total_per_class[class_idx]
        if total_per_class[class_idx] > 0
        else 0.0
        for class_idx in total_per_class
    }
    overall_accuracy = overall_correct / total_item
    mean_loss = loss / total_item

    # Compiling savable objects
    to_save_keys = torch.cat(to_save_keys)
    to_save_outputs_class = torch.cat(to_save_outputs_class, dim=0)
    to_save_labels = torch.cat(to_save_labels)
    ready_to_save = (
        to_save_keys,
        to_save_outputs_class,
        to_save_labels,
    )

    return (
        mean_loss,
        overall_accuracy,
        per_class_accuracy,
        ready_to_save,
    )


def train_cnn(
    model: nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    detect_cfg: DetectConfig,
    silent: bool = False,
    do_eval: bool = True,
):
    """Trains the models"""

    # Saving model - epoch 0
    base_save_dir: Path = detect_cfg.get_output_dir() / "cnn_2d"
    base_save_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), base_save_dir / f"model_0.pt")

    # Pre-training eval
    if do_eval:
        (eval_loss, overall_accuracy, per_class_accuracy, ready_to_save,) = eval_cnn(
            model,
            eval_dataloader,
            criterion=criterion,
            silent=silent,
        )
        if not silent:
            logger.debug(f"Epoch 0: Eval loss : {eval_loss:.5f}")
            logger.debug(f"Eval accuracy: {overall_accuracy:.2%}")
            logger.debug(
                " | ".join(
                    [
                        f"{per_class_accuracy[key]:.2%}"
                        for key in sorted(per_class_accuracy)
                    ]
                )
            )

            # Wandb
            wandb.log(
                {
                    "epoch": 0,
                    "train_loss": eval_loss,
                    "eval_loss": eval_loss,
                    "overall_accuracy": overall_accuracy,
                    "per_class_accuracy": per_class_accuracy,
                }
            )

            # Saving
            torch.save(ready_to_save, base_save_dir / f"forward_values_0.pt")

    # Training loop
    for epoch in range(detect_cfg.n_epochs):
        if not silent:
            logger.info(f"Training epoch {epoch+1} / {detect_cfg.n_epochs}")
        model.train()

        running_loss = 0.0
        total_item = 0
        for _, patterns, labels in tqdm(train_dataloader, leave=(not silent)):
            patterns, labels = (
                patterns.to(dtype=CNN_DTYPE, device=DEVICE),
                labels.to(DEVICE).long(),
            )

            # Forward
            outputs_class = model(patterns)
            loss = criterion(outputs_class, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Running loss
            running_loss += loss.item()
            total_item += labels.size(0)

        # Evaluation
        if do_eval:
            train_loss = running_loss / total_item
            (
                eval_loss,
                eval_accuracy,
                per_class_accuracy,
                ready_to_save,
            ) = eval_cnn(model, eval_dataloader, criterion=criterion, silent=silent)
            if not silent:
                logger.debug(f"Epoch {epoch + 1}: Train Loss: {train_loss:.5f}")
                logger.debug(f"Epoch {epoch + 1}: Eval loss : {eval_loss:.5f}")
                logger.debug(f"Eval  accuracy: {eval_accuracy:.2%}")
                logger.debug(
                    "Eval  "
                    + " | ".join(
                        [
                            f"{per_class_accuracy[key]:.2%}"
                            for key in sorted(per_class_accuracy)
                        ]
                    )
                )

                # Wandb
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "eval_loss": eval_loss,
                        "overall_accuracy": eval_accuracy,
                        "per_class_accuracy": per_class_accuracy,
                    }
                )

                # Saving
                torch.save(ready_to_save, base_save_dir / f"forward_values_{epoch}.pt")

        # Saving
        torch.save(model.state_dict(), base_save_dir / f"model_{epoch}.pt")
