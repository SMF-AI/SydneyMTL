from .mil import (
    AttentionFeatureMIL,
    MultiTaskAttentionFeatureMIL,
)

__all__ = [
    "AttentionFeatureMIL",
    "MultiTaskAttentionFeatureMIL",
]

MODEL_REGISTRY = {
    "attention_feature_mil": AttentionFeatureMIL,
    "multitask_attention_feature_mil": MultiTaskAttentionFeatureMIL,
}
