import os
import sys
import argparse

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from seedp.datasets import SydneyBatch, SydneyDataset

from seedp.networks import MODEL_REGISTRY
from seedp.misc import KeyValueAction
from seedp.trainer import SydneyMultiTaskTrainer
from seedp.tracking import TRACKING_URI, get_experiment, ExpNames

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXP_DIR)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sydney classification paper training script",
        epilog=(
            """
            $ python3 experiments/train_sydney.py \\
                --data_dir h5_folder_path \\
                --label_csv /vast/AI_team/dataset/SMF/slide/stomach/aperio_gt_450.csv \\
                --model_name multitask_attention_feature_mil \\
                --tasknames hp neut mono atrophy im \\
                --nclasses 4 4 4 5 4 \\
                --model_opt encoder_dim:1024 adaptor_dim:256 \\
                --run_name "train" \\
                --loss "CrossEntropyLoss" \\
                --experiment_name sydney_paper
        """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--data_dir", type=str, help="path with feature files")
    parser.add_argument(
        "--label_csv", type=str, required=True, help="sydney.csv 파일 경로"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, choices=list(MODEL_REGISTRY.keys())
    )
    parser.add_argument("--model_opt", nargs="*", action=KeyValueAction, default={})
    parser.add_argument(
        "--loss",
        type=str,
        required=True,
        choices=[
            "CrossEntropyLoss",
            "LogitAdjustedCE",
        ],
    )
    parser.add_argument("--experiment_name", type=str, required=True)

    # Multi-task specific arguments
    parser.add_argument(
        "--tasknames", nargs="+", type=str, help="태스크 이름들 (예: HP AC CI AT IM)"
    )
    parser.add_argument("--nclasses", nargs="+", type=int, help="각 태스크별 클래스 수")
    parser.add_argument("-r", "--run_name", type=str, default="train_sydney_paper")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1, help="배치 크기")
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

    # task_classes 딕셔너리 생성
    if args.tasknames and args.nclasses:
        if len(args.tasknames) != len(args.nclasses):
            raise ValueError("tasknames와 nclasses의 길이가 일치해야 합니다")
        args.model_opt["task_classes"] = dict(zip(args.tasknames, args.nclasses))

    if args.dry_run:
        args.epochs = 1
    return args


def get_golden_set_dataset(args, taskname=None):
    """
    Golden data set 데이터셋을 생성하고,
    single/multi hibou/resnet 인지에 따라 분기 구분.
    고정 되어 있기 때문에 하드 코딩으로 설정

    """
    base_path = "/vast/AI_team/dataset/SMF/feature/stomach/aperio_gt_450/sydney_paper/ground_truth"
    csv_path = os.path.join(base_path, "sydney_golden_set.csv")

    data_dir = os.path.join(base_path, "feature/hibouL")
    if "resnet" in args.data_dir:
        data_dir = os.path.join(base_path, "feature/resnet50")

    golden_batch = SydneyBatch.from_csv(
        feature_dir=data_dir,
        csv_path=csv_path,
        tasknames=args.tasknames,
        dry_run=args.dry_run,
    )

    if not args.multi_task:
        return SydneyDataset(golden_batch, single_task=taskname)
    return SydneyDataset(golden_batch)


def main_multi_task(args, sydney_batch: SydneyBatch):
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment_name = ExpNames[args.experiment_name].value
    exp = get_experiment(experiment_name)

    with mlflow.start_run(experiment_id=exp.experiment_id, run_name=args.run_name):
        mlflow.log_params(vars(args))
        mlflow.set_tag("command", " ".join(sys.argv))
        mlflow.set_tag("kfold", args.kfold)
        mlflow.log_artifact(os.path.abspath(__file__))
        mlflow.log_artifact(os.path.join(ROOT_DIR, "seedp", "losses.py"))
        mlflow.log_artifact(os.path.join(ROOT_DIR, "seedp", "trainer.py"))

        for fold_idx, (train_batch, val_batch, test_batch) in enumerate(
            sydney_batch.kfold_generator(), start=1
        ):
            with mlflow.start_run(
                experiment_id=exp.experiment_id,
                run_name=f"fold_{fold_idx}",
                nested=True,
            ):
                mlflow.log_param("fold", fold_idx)
                mlflow.log_param("random_state", args.random_state)

                train_dataset = SydneyDataset(train_batch)
                val_dataset = SydneyDataset(val_batch)
                test_dataset = SydneyDataset(test_batch)

                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                )
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                )
                test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                )

                # 각 fold별로 새로운 모델 생성
                model = MODEL_REGISTRY[args.model_name](**args.model_opt).to(
                    args.device
                )
                loss_fns = {}
                for task in args.tasknames:
                    if args.loss == "CrossEntropyLoss":
                        loss_fn = nn.CrossEntropyLoss()
                    elif args.loss == "LogitAdjustedCE":
                        from seedp.losses import LogitAdjustedCE

                        num_classes = (
                            5 if (task == "atrophy" and args.include_atrophy9) else 4
                        )
                        labels = train_dataset.batch.__getattribute__(task)
                        loss_fn = LogitAdjustedCE(
                            num_classes=num_classes,
                            class_labels=labels,
                            tau=args.logit_adjustment_tau,
                        )
                    elif hasattr(nn, args.loss):
                        loss_fn = getattr(nn, args.loss)()

                    loss_fns[task] = loss_fn

                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

                trainer = SydneyMultiTaskTrainer.from_args(
                    args, model, loss_fns, optimizer
                )

                trainer.train(
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    n_epochs=args.epochs,
                    max_patiences=args.max_patiences,
                    accumulation_steps=args.accumulation_steps,
                    use_mlflow=True,
                )

                mlflow.pytorch.log_model(model, artifact_path="model")
                trainer.test(test_dataloader, use_mlflow=True)

                golden_dataset = get_golden_set_dataset(args)
                golden_dataloader = DataLoader(
                    golden_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=args.num_workers,
                    prefetch_factor=args.prefetch_factor,
                )
                trainer.gt_test(golden_dataloader, use_mlflow=True)


def main_single_task(args, sydney_batch):
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment_name = ExpNames[args.experiment_name].value
    exp = get_experiment(experiment_name)

    with mlflow.start_run(experiment_id=exp.experiment_id, run_name=args.run_name):
        mlflow.log_params(vars(args))
        mlflow.set_tag("command", " ".join(sys.argv))
        mlflow.set_tag("n_folds", args.kfold)

        for fold_idx, (train_batch, val_batch, test_batch) in enumerate(
            sydney_batch.kfold_generator(), start=1
        ):
            with mlflow.start_run(
                experiment_id=exp.experiment_id,
                run_name=f"fold_{fold_idx}",
                nested=True,
            ):
                for task, n_classes in zip(args.tasknames, args.nclasses):
                    with mlflow.start_run(
                        experiment_id=exp.experiment_id,
                        run_name=f"fold_{fold_idx}_{task}",
                        nested=True,
                    ):
                        mlflow.log_param("task", task)
                        mlflow.log_param("fold", fold_idx)

                        train_dataset = SydneyDataset(train_batch, single_task=task)
                        val_dataset = SydneyDataset(val_batch, single_task=task)
                        test_dataset = SydneyDataset(test_batch, single_task=task)

                        train_dataloader = DataLoader(
                            train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch_factor,
                        )

                        val_dataloader = DataLoader(
                            val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch_factor,
                        )

                        test_dataloader = DataLoader(
                            test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch_factor,
                        )

                        # 각 fold별로  -Atrophy 9가 있어서 num_classes 다름 model_opt
                        model_opt = args.model_opt.copy()
                        model_opt["num_classes"] = n_classes

                        model = MODEL_REGISTRY[args.model_name](**model_opt).to(
                            args.device
                        )
                        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

                        # TODO: 필요시 수정
                        if args.loss == "CrossEntropyLoss":
                            weight = train_batch.compute_class_weight(task).to(
                                args.device
                            )
                            loss = nn.CrossEntropyLoss(weight=weight)
                        elif args.loss == "LogitAdjustedCE":
                            from seedp.losses import LogitAdjustedCE

                            num_classes = (
                                5
                                if (task == "atrophy" and args.include_atrophy9)
                                else 4
                            )
                            labels = train_dataset.batch.__getattribute__(task)
                            loss = LogitAdjustedCE(
                                num_classes=num_classes,
                                class_labels=labels,
                                tau=args.logit_adjustment_tau,
                            )
                        else:
                            loss = getattr(nn, args.loss)()

                        trainer = SydneySingleTaskTrainer.from_args(
                            args, model, loss, optimizer
                        )

                        trainer.train(
                            train_dataloader,
                            val_dataloader,
                            n_epochs=args.epochs,
                            max_patiences=args.max_patiences,
                            accumulation_steps=args.accumulation_steps,
                        )
                        mlflow.pytorch.log_model(model, artifact_path=f"model")
                        trainer.test(test_dataloader, use_mlflow=True)

                        golden_dataset = get_golden_set_dataset(args, taskname=task)
                        golden_dataloader = DataLoader(
                            golden_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch_factor,
                        )
                        trainer.gt_test(golden_dataloader, use_mlflow=True)


if __name__ == "__main__":
    args = get_args()

    experiment_name = ExpNames[args.experiment_name].value
    exp = get_experiment(experiment_name)

    sydney_batch = SydneyBatch.from_csv(
        feature_dir=args.data_dir,
        csv_path=args.label_csv,
        tasknames=args.tasknames,
        dry_run=args.dry_run,
    )

    if args.multi_task:
        main_multi_task(args, sydney_batch)
    else:
        main_single_task(args, sydney_batch)
