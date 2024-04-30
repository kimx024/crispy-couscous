import gym
import numpy as np
from tensorflow import keras
from keras import layers

env = gym.make("CartPole-v1")
model = keras.Sequential([
    layers.Dense(24, activation='relu', input_shape=(env.observation_space.shape[0],)),
    layers.Dense(24, activation='relu'),
    layers.Dense(env.action_space.n, activation='linear')
])
model.compile(optimizer='adam', loss='mse')

gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration-exploitation trade-off
epsilon_min = 0.01  # Minimum exploration probability
epsilon_decay = 0.995  # Decay rate for exploration probability

# Training loop
for episode in range(10):
    state = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])

    total_reward = 0
    done = False
    while not done:
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            q_values = model.predict(state)
            action = np.argmax(q_values[0])

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])

        target = reward + gamma * np.amax(model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target

        model.fit(state, target_f, epochs=1, verbose=0)

        state = next_state
        total_reward += reward
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    if episode % 50 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}")

env.close()
