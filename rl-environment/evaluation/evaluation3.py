import torch
from stable_baselines3 import PPO  # Example using PPO, adjust according to your model
from sklearn.metrics import precision_score, recall_score
import gym

# Step 1: Load the trained model
policy_path = "path/to/your/policy.pth"
optimizer_path = "path/to/your/policy_optimizer.pth"
variables_path = "path/to/your/policy_variables.pth"

model = PPO.load(policy_path)  # Adjust this if using a different algorithm

# Step 2: Set up the environment
env = gym.make('YourEnvironment-v0')  # Replace with your environment

# Step 3: Generate predictions and collect true labels
true_labels = []
predicted_labels = []

num_episodes = 100  # Define the number of episodes for evaluation

for _ in range(num_episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        true_labels.append(env.get_true_label(obs))  # Assuming the environment can provide true labels
        predicted_labels.append(action)
        obs, rewards, done, info = env.step(action)

# Step 4: Calculate precision and recall
precision = precision_score(true_labels, predicted_labels, average='macro')  # Adjust the average parameter as needed
recall = recall_score(true_labels, predicted_labels, average='macro')  # Adjust the average parameter as needed

print(f"Precision: {precision}")
print(f"Recall: {recall}")
