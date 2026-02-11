import torch
import torch.nn as nn
from typing import Tuple, Literal, Union, Dict

from seedp.networks.attention_ops import AttentionLayer


class AttentionFeatureMIL(nn.Module):
    """Feature로부터 forward하는 Attention MIL 모델"""

    def __init__(
        self, encoder_dim: int, adaptor_dim: int, num_classes: int = 2, **kwargs
    ):
        super(AttentionFeatureMIL, self).__init__()
        self.encoder_dim = encoder_dim
        self.adaptor_dim = adaptor_dim
        self.num_classes = num_classes

        # Adding an adaptor layer
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
            x (torch.Tensor): (K, N, D)
            return_with (str): default None.
                - "attention_weights": attention weight for each instance
                - "contributions": (미구현) attention_weight*W(linear)@instance_feature

        Returns:
            logits (torch.Tensor): (K, n_classess) shaped tensor. logits for bag prediction
            interpretable components (torch.Tensor): attention weight or contribution
                shape (1, N). This return will be returned when return_with is not None.

        Examples:

        """

        B, N, D = x.shape

        instance_features = self.adaptor(x)  # (B, n, d')
        attention_weights = self.attention_layer(instance_features).view(B, N)
        weighted_features = torch.einsum(
            "ijk,ij->ijk", instance_features, attention_weights
        )

        context_vector = weighted_features.sum(axis=1)  # (B, d')
        logits = self.classifier(context_vector).view(B, self.num_classes)

        if return_with == "attention_weights":
            return logits, attention_weights
        if return_with == "contributions":
            raise NotImplementedError(
                'return_with="contributions" 는 아직 지원하지 않습니다.'
            )

        return logits


class MultiTaskAttentionFeatureMIL(nn.Module):
    """
    Multi-Task Attention Feature MIL

    Args:
        encoder_dim (int): Input feature dimension
        adaptor_dim (int): Adaptor dimension
        task_classes (Dict[str, int]): Task classes

    Returns:
        Dict[str, torch.Tensor]: Task-specific logits

    Examples:
        >>> model = MultiTaskAttentionFeatureMIL(
            encoder_dim=1024,
            adaptor_dim=512,
            task_classes={"task1": 3, "task2": 3}
        )
        >>> logits = model(torch.rand(1, 7, 1024))
        >>> print(logits["task1"].shape)
        torch.Size([1, 3])
        >>> print(logits["task2"].shape)
        torch.Size([1, 3])
    """

    def __init__(
        self, encoder_dim: int, adaptor_dim: int, task_classes: Dict[str, int], **kwargs
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.adaptor_dim = adaptor_dim
        self.task_classes = task_classes

        self.adaptor = nn.Sequential(
            nn.Linear(encoder_dim, adaptor_dim),
            nn.GELU(),
            nn.Linear(adaptor_dim, adaptor_dim),
        )
        self.attention_layers = nn.ModuleDict(
            {task: AttentionLayer(adaptor_dim) for task, _ in task_classes.items()}
        )
        self.classifiers = nn.ModuleDict(
            {t: nn.Linear(adaptor_dim, c) for t, c in task_classes.items()}
        )

    def forward(
        self,
        x: torch.Tensor,
        return_with: str = "attention_weights",
    ) -> Dict[str, torch.Tensor]:

        B, N, D = x.shape
        res = dict()
        res_attn = dict()
        instance_features = self.adaptor(x)  # (B, N, adaptor_dim)
        for task, attn_layer in self.attention_layers.items():
            attention_weights = attn_layer(instance_features)  # (B, D)
            weighted_features = torch.einsum(
                "ij,ijk->ijk", attention_weights, instance_features
            )
            context_vector = weighted_features.sum(axis=1)  # (B, D')
            logits = self.classifiers[task](context_vector)  # (B, n_cls)

            res[task] = logits
            if return_with == "attention_weights":
                res_attn[task] = attention_weights

        if return_with == "attention_weights":
            return res, res_attn

        return res
