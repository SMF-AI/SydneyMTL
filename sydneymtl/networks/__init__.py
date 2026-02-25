from .mil import (
    AttentionFeatureMIL,
    MultiTaskAttentionFeatureMIL,
)

# 반드시 위에서 임포트후에 아래의 호출가능한 목록에 추가해주세요
__all__ = [
    "AttentionFeatureMIL",
    "MultiTaskAttentionFeatureMIL",
]

MODEL_REGISTRY = {
    "attention_feature_mil": AttentionFeatureMIL,
    "multitask_attention_feature_mil": MultiTaskAttentionFeatureMIL,
}
