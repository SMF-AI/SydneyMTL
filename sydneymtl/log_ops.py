import os
import mlflow
import pickle
from matplotlib import pyplot as plt
from mlflow.entities.experiment import Experiment
from typing import Any, Optional


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT_DIR, "logs")

TRACKING_URI = "http://XXX.XXX.XXX.XXX:5000/"
EXP_NAME = "sydneymtl"


def get_experiment(experiment_name: str) -> Experiment:
    """Retrieve an MLflow experiment by name. Create it if it does not exist."""
    mlflow.set_tracking_uri(TRACKING_URI)

    client = mlflow.tracking.MlflowClient(TRACKING_URI)
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        client.create_experiment(experiment_name)
        experiment = client.get_experiment_by_name(experiment_name)

    return experiment


def save_and_log_figure(filename: str) -> None:
    """Save current matplotlib figure, log it to MLflow, and remove the local file."""
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    os.remove(filename)
    plt.clf()


def serialize_obj(obj: Any, path: str) -> None:
    """Serialize an object to disk using pickle."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def log_object(obj: Any, artifact_path: Optional[str] = None) -> None:
    """
    Serialize an object, log it as an MLflow artifact, and remove the local file.

    Parameters
    ----------
    obj : Any
        Object to be serialized.
    artifact_path : Optional[str]
        Destination artifact path in MLflow.
        If None, the default artifact root is used.
    """
    local_path = os.path.basename(artifact_path) if artifact_path else "artifact.pkl"
    serialize_obj(obj, local_path)

    if artifact_path:
        dst_dir = os.path.basename(os.path.dirname(artifact_path))
    else:
        dst_dir = None

    mlflow.log_artifact(local_path, dst_dir)
    os.remove(local_path)
