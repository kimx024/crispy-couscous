import os
import gymnasium as gym

os.environ['NO_RESTORE'] = '1'
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

for _ in range(100):
    env.render()
    action = env.action_space.sample()  # Take random actions
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
