from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from file2 import Environment

input_file = "/environment/output.txt"

# Create the environment
env0 = Environment(input_file)
env = DummyVecEnv([lambda: env0])

# Define the model path (replace with the actual path to your files)
policy_path = "/runs/ppo_goalkeeper/policy.pth"
optimizer_path = "/runs/ppo_goalkeeper/policy.optimizer.pth"
pytorch_variables_path = ("/Users/kim.lichtenberg/PycharmProjects/pythonProject/runs/ppo_goalkeeper/pytorch_variables"
                          ".pth")

# Create a PPO model instance
model = PPO('MlpPolicy', env, verbose=1)

# Load the policy parameters
policy_state_dict = torch.load(policy_path)
model.policy.load_state_dict(policy_state_dict)

# Load the optimizer parameters
optimizer_state_dict = torch.load(optimizer_path)
model.policy.optimizer.load_state_dict(optimizer_state_dict)

# Optionally, load other variables
pytorch_variables = torch.load(pytorch_variables_path)
# Here you might want to load these variables into the appropriate model components if necessary

# Now you can use the model for further training or evaluation
# Test the loaded model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
