import os
import sys
import argparse

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sydneymtl.datasets import SydneyBatch, SydneyDataset
from sydneymtl.networks import MODEL_REGISTRY
from sydneymtl.misc import KeyValueAction
from sydneymtl.trainer import SydneyMultiTaskTrainer
from sydneymtl.log_ops import TRACKING_URI, get_experiment, EXP_NAME


EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXP_DIR)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training script for SydneyMTL multi-task classification.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--data_dir", type=str, help="Directory containing feature files."
    )
    parser.add_argument(
        "--label_csv", type=str, required=True, help="CSV file with labels."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
    )
    parser.add_argument("--model_opt", nargs="*", action=KeyValueAction, default={})
    parser.add_argument(
        "--loss",
        type=str,
        required=True,
        choices=["CrossEntropyLoss", "LogitAdjustedCE"],
    )

    parser.add_argument("--tasknames", nargs="+", type=str)
    parser.add_argument("--nclasses", nargs="+", type=int)

    parser.add_argument("-r", "--run_name", type=str, default="train_sydney_mtl")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_patiences", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("-a", "--accumulation_steps", type=int, default=8)

    parser.add_argument("--include_atrophy9", action="store_true")
    parser.add_argument("--logit_adjustment_tau", type=float, default=0.75)

    args = parser.parse_args()

    if args.tasknames and args.nclasses:
        if len(args.tasknames) != len(args.nclasses):
            raise ValueError("Length of tasknames and nclasses must match.")
        args.model_opt["task_classes"] = dict(zip(args.tasknames, args.nclasses))

    if args.dry_run:
        args.epochs = 1

    return args


def get_golden_set_dataset(args: argparse.Namespace) -> SydneyDataset:
    base_path = "../data/ground_truth"
    csv_path = os.path.join(base_path, "sydney_golden_set.csv")
    data_dir = os.path.join(base_path, "feature_hibouL")

    golden_batch = SydneyBatch.from_csv(
        feature_dir=data_dir,
        csv_path=csv_path,
        tasknames=args.tasknames,
        dry_run=args.dry_run,
    )

    return SydneyDataset(golden_batch)


def main_multi_task(args: argparse.Namespace, sydney_batch: SydneyBatch):

    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = get_experiment(EXP_NAME)

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=args.run_name,
    ):
        mlflow.log_params(vars(args))
        mlflow.set_tag("command", " ".join(sys.argv))
        mlflow.set_tag("kfold", args.kfold)

        mlflow.log_artifact(os.path.abspath(__file__))
        mlflow.log_artifact(os.path.join(ROOT_DIR, "sydneymtl", "losses.py"))
        mlflow.log_artifact(os.path.join(ROOT_DIR, "sydneymtl", "trainer.py"))

        for fold_idx, (train_batch, val_batch, test_batch) in enumerate(
            sydney_batch.kfold_generator(), start=1
        ):

            with mlflow.start_run(
                experiment_id=experiment.experiment_id,
                run_name=f"fold_{fold_idx}",
                nested=True,
            ):
                mlflow.log_param("fold", fold_idx)
                mlflow.log_param("random_state", args.random_state)

                train_dataset = SydneyDataset(train_batch)
                val_dataset = SydneyDataset(val_batch)
                test_dataset = SydneyDataset(test_batch)

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                )

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                )

                model = MODEL_REGISTRY[args.model_name](**args.model_opt).to(
                    args.device
                )

                loss_fns = {}
                for task in args.tasknames:

                    if args.loss == "CrossEntropyLoss":
                        loss_fn = nn.CrossEntropyLoss()

                    elif args.loss == "LogitAdjustedCE":
                        from sydneymtl.losses import LogitAdjustedCE

                        num_classes = (
                            5 if (task == "atrophy" and args.include_atrophy9) else 4
                        )

                        labels = getattr(train_dataset.batch, task)

                        loss_fn = LogitAdjustedCE(
                            num_classes=num_classes,
                            class_labels=labels,
                            tau=args.logit_adjustment_tau,
                        )

                    else:
                        loss_fn = getattr(nn, args.loss)()

                    loss_fns[task] = loss_fn

                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

                trainer = SydneyMultiTaskTrainer.from_args(
                    args, model, loss_fns, optimizer
                )

                trainer.train(
                    train_dataloader=train_loader,
                    val_dataloader=val_loader,
                    n_epochs=args.epochs,
                    max_patiences=args.max_patiences,
                    accumulation_steps=args.accumulation_steps,
                    use_mlflow=True,
                )

                mlflow.pytorch.log_model(model, artifact_path="model")
                trainer.test(test_loader, use_mlflow=True)

                golden_dataset = get_golden_set_dataset(args)
                golden_loader = DataLoader(
                    golden_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                )

                trainer.gt_test(golden_loader, use_mlflow=True)


if __name__ == "__main__":

    args = get_args()

    sydney_batch = SydneyBatch.from_csv(
        feature_dir=args.data_dir,
        csv_path=args.label_csv,
        tasknames=args.tasknames,
        dry_run=args.dry_run,
    )

    main_multi_task(args, sydney_batch)
