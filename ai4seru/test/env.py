from stable_baselines3.common.vec_env import SubprocVecEnv

# 不同seed的环境
env_fns = [lambda: gym.make("CartPole-v1", seed=i) for i in range(4)]

# 不同任务参数的环境（假设环境接受difficulty参数）
env_fns = [lambda: CustomEnv(difficulty=i*0.2) for i in range(4)]

# 混合不同类型但空间兼容的环境
env_fns = [
    lambda: gym.make("Ant-v4"),
    lambda: gym.make("Humanoid-v4"),
    # 需确保所有环境有相同的obs/action空间
]

vec_env = SubprocVecEnv(env_fns)