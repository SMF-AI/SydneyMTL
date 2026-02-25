import torch
import torch.nn as nn
from typing import Tuple, Literal, Union, Dict


class AttentionLayer(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alignment = self.linear(x)  # (..., N, 1)
        attention_weights = torch.softmax(alignment.squeeze(-1), dim=-1)
        return attention_weights


class AttentionFeatureMIL(nn.Module):
    """
    Attention-based MIL model operating on pre-extracted instance features.

    Args:
        encoder_dim (int): Input feature dimension.
        adaptor_dim (int): Hidden dimension for adaptor layer.
        num_classes (int): Number of output classes.
    """

    def __init__(
        self,
        encoder_dim: int,
        adaptor_dim: int,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.adaptor = nn.Sequential(
            nn.Linear(encoder_dim, adaptor_dim),
            nn.ReLU(),
            nn.Linear(adaptor_dim, adaptor_dim),
        )

        self.attention_layer = AttentionLayer(adaptor_dim)
        self.classifier = nn.Linear(adaptor_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_with: Literal["attention_weights", "contributions"] | None = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x (Tensor): Shape (B, N, D).
            return_with (str, optional):
                - "attention_weights": return attention weights per instance.
                - "contributions": not implemented.

        Returns:
            logits (Tensor): Shape (B, num_classes).
            Optional attention weights (Tensor): Shape (B, N).
        """

        B, N, _ = x.shape

        instance_features = self.adaptor(x)
        attention_weights = self.attention_layer(instance_features).view(B, N)

        weighted_features = torch.einsum(
            "ijk,ij->ijk", instance_features, attention_weights
        )
        context_vector = weighted_features.sum(dim=1)

        logits = self.classifier(context_vector)

        if return_with == "attention_weights":
            return logits, attention_weights

        if return_with == "contributions":
            raise NotImplementedError('"contributions" option is not implemented yet.')

        return logits


class MultiTaskAttentionFeatureMIL(nn.Module):
    """
    Multi-task Attention-based MIL model.

    Args:
        encoder_dim (int): Input feature dimension.
        adaptor_dim (int): Hidden dimension for adaptor layer.
        task_classes (Dict[str, int]): Dictionary mapping task names to number of classes.

    Returns:
        Dict[str, Tensor]: Task-specific logits.
        Optionally, attention weights per task.
    """

    def __init__(
        self,
        encoder_dim: int,
        adaptor_dim: int,
        task_classes: Dict[str, int],
    ):
        super().__init__()
        self.task_classes = task_classes

        self.adaptor = nn.Sequential(
            nn.Linear(encoder_dim, adaptor_dim),
            nn.GELU(),
            nn.Linear(adaptor_dim, adaptor_dim),
        )

        self.attention_layers = nn.ModuleDict(
            {task: AttentionLayer(adaptor_dim) for task in task_classes}
        )

        self.classifiers = nn.ModuleDict(
            {
                task: nn.Linear(adaptor_dim, num_classes)
                for task, num_classes in task_classes.items()
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        return_with: str = "attention_weights",
    ) -> Union[
        Dict[str, torch.Tensor],
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    ]:
        """
        Args:
            x (Tensor): Shape (B, N, D).
            return_with (str):
                - "attention_weights": return attention weights per task.

        Returns:
            Dict[str, Tensor]: Task-specific logits.
            Optional Dict[str, Tensor]: Task-specific attention weights.
        """

        B, _, _ = x.shape

        instance_features = self.adaptor(x)

        logits_dict: Dict[str, torch.Tensor] = {}
        attn_dict: Dict[str, torch.Tensor] = {}

        for task, attn_layer in self.attention_layers.items():
            attention_weights = attn_layer(instance_features)

            weighted_features = torch.einsum(
                "ij,ijk->ijk", attention_weights, instance_features
            )

            context_vector = weighted_features.sum(dim=1)
            logits = self.classifiers[task](context_vector)

            logits_dict[task] = logits

            if return_with == "attention_weights":
                attn_dict[task] = attention_weights

        if return_with == "attention_weights":
            return logits_dict, attn_dict

        return logits_dict
