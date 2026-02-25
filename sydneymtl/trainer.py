import math
import torch
import mlflow
import logging
from abc import ABC
from copy import deepcopy
from typing import Tuple, Literal, Dict
from torch.utils.data import DataLoader

from sydneymtl.metrics import (
    LenientMultiTaskMetricsMeter,
    MultiTaskAverageMeter,
)
from sydneymtl.log_ops import save_and_log_figure, log_object
from sydneymtl.misc import ProgressBar


class SydneyMultiTaskTrainer(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        losses: Dict[str, torch.nn.modules.loss._Loss],
        optimizer: torch.optim.Optimizer = None,
        logger: logging.Logger = None,
        device: str = "cuda",
    ):
        self.model = model
        self.losses = losses
        self.optimizer = optimizer
        self.logger = logging.Logger("MultiTaskTrainer") if logger is None else logger
        self.device = device

        self.include_atrophy9 = False
        self.lenient_map = {0: {0}, 1: {1, 2}, 2: {1, 2, 3}, 3: {2, 3}}
        self.use_hierarchical_loss = False
        self.args = None

    @classmethod
    def from_args(cls, args, model, loss_fns, optimizer):
        model = model.to(args.device)
        lenient_map = {0: {0}, 1: {1, 2}, 2: {1, 2, 3}, 3: {2, 3}, 4: {4}}

        trainer = cls(
            model=model,
            losses=loss_fns,
            optimizer=optimizer,
            device=args.device,
        )
        trainer.args = args
        trainer.lenient_map = lenient_map

        return trainer

    def run_epoch(
        self,
        dataloader: DataLoader,
        phase: Literal["train", "val", "test"],
        current_epoch: int,
        accumulation_steps: int = 1,
    ) -> Tuple[MultiTaskAverageMeter, LenientMultiTaskMetricsMeter]:

        is_train = phase == "train"
        self.model.train(mode=is_train)

        total_step = len(dataloader)
        bar = ProgressBar(max=total_step, check_tty=False)

        loss_meters = MultiTaskAverageMeter(phase, list(self.losses.keys()))

        n_classes = 5 if self.include_atrophy9 else 4
        metrics_meters = LenientMultiTaskMetricsMeter(
            phase,
            n_classes,
            list(self.losses.keys()),
            lenient_map=self.lenient_map,
        )

        with torch.set_grad_enabled(is_train):
            for step, batch in enumerate(dataloader, start=1):
                xs, ys = batch
                xs = xs.to(self.device, non_blocking=True)
                outputs = self.model(xs)

                total_loss = torch.tensor(0.0, device=self.device)

                for taskname, criterion in self.losses.items():
                    task_output = outputs[taskname]
                    task_target = ys[taskname].to(self.device, non_blocking=True)

                    loss = criterion(task_output, task_target)
                    total_loss += loss / accumulation_steps

                    model_confidence = torch.softmax(task_output, dim=-1)

                    loss_meters[taskname].update(loss.detach().item())
                    metrics_meters[taskname].update(
                        model_confidence.detach().cpu().numpy().tolist(),
                        task_target.detach().cpu().numpy().tolist(),
                    )

                loss_meters["total_loss"].update(
                    total_loss.detach().item() * accumulation_steps
                )

                if is_train:
                    total_loss.backward()
                    if (step % accumulation_steps == 0) or (step == total_step):
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                bar.suffix = (
                    f"{phase} | EPOCH {current_epoch}: "
                    f"[{step}/{total_step}] | eta:{bar.eta} | "
                    + " | ".join(
                        [f"{k}: {v}" for k, v in metrics_meters.to_dict().items()]
                    )
                )
                bar.next()

            bar.finish()

        return loss_meters, metrics_meters

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader = None,
        n_epochs: int = 100,
        max_patiences: int = 10,
        accumulation_steps: int = 1,
        use_mlflow: bool = True,
    ):
        """Execute multi-task training with optional early stopping."""
        best_params = deepcopy(self.model.state_dict())
        best_loss = math.inf
        patience = 0

        for current_epoch in range(1, n_epochs + 1):

            train_losses, train_metrics = self.run_epoch(
                train_dataloader,
                phase="train",
                current_epoch=current_epoch,
                accumulation_steps=accumulation_steps,
            )

            if use_mlflow:
                mlflow.log_metrics(train_losses.to_dict(), step=current_epoch)
                mlflow.log_metrics(train_metrics.to_dict(), step=current_epoch)

            if val_dataloader is None:
                continue

            val_losses, val_metrics = self.run_epoch(
                val_dataloader,
                phase="val",
                current_epoch=current_epoch,
            )

            if use_mlflow:
                mlflow.log_metrics(val_losses.to_dict(), step=current_epoch)
                mlflow.log_metrics(val_metrics.to_dict(), step=current_epoch)

            val_total_loss = val_losses["total_loss"].avg

            if val_total_loss < best_loss:
                best_loss = val_total_loss
                patience = 0
                best_params = deepcopy(self.model.state_dict())
            else:
                patience += 1
                if patience == max_patiences:
                    break

        self.model.load_state_dict(best_params)

    def test(
        self,
        test_loader: DataLoader,
        use_mlflow: bool = True,
        labels=[0, 1, 2, 3],
    ):
        losses, metrics = self.run_epoch(
            test_loader,
            phase="test",
            current_epoch=0,
        )

        if use_mlflow:
            log_object(metrics, "test_metric_meters.pickle")
            mlflow.log_metrics(losses.to_dict(), step=0)
            mlflow.log_metrics(metrics.to_dict(), step=0)

            for taskname, meter in metrics.meters.items():
                current_labels = labels
                if self.include_atrophy9 and taskname == "atrophy":
                    current_labels = [0, 1, 2, 3, "N/A"]

                meter.plot_confusion_matrix(labels=current_labels)
                save_and_log_figure(f"{taskname}_confusion_matrix.png")

        return
