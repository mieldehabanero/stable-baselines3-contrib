from sb3_contrib.common.maskable.policies import (
    MaskableDQNCnnPolicy,
    MaskableDQNPolicy,
    MaskableMultiInputDQNPolicy,
)

MlpPolicy = MaskableDQNPolicy
CnnPolicy = MaskableDQNCnnPolicy
MultiInputPolicy = MaskableMultiInputDQNPolicy
