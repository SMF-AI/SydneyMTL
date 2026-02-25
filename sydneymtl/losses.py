import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitAdjustedCE(nn.Module):
    """
    Menon & Williamson (2020) - Logit Adjusted CE

    Args:
        class_priors (Tensor): (C,)
        tau (float): τ (temperature)

    Reference:
        - https://arxiv.org/abs/2007.07314
    """

    def __init__(
        self,
        num_classes: int,
        class_labels: torch.Tensor,
        tau: float = 0.75,
    ):
        super().__init__()

        labels = class_labels.to(torch.long).cpu()
        counts = torch.bincount(labels, minlength=num_classes).float()  # (C,)
        class_priors = counts / counts.sum()  # (C,)

        self.register_buffer("log_pi", torch.log(class_priors))  # (C,)
        self.tau = tau

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_pi = self.log_pi.to(logits.device)

        # Eq: f_y(x) + τ log π_y
        adjusted_logits = logits + self.tau * log_pi  # (B, C)

        return F.cross_entropy(adjusted_logits, targets)
