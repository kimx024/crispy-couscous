import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
matplotlib.use('TkAgg')


class SoccerPenaltyEnv(gym.Env):
    def __init__(self):
        super(SoccerPenaltyEnv, self).__init__()
        self.player = [600, 100]
        self.goalkeeper_position = [600, 1100]
        self.ball = [600, 100]
        self.action_space = spaces.Discrete(1000 * 1000)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([1200, 1200]), dtype=np.int32)
        self.figure, self.ax = plt.subplots()

    def reset(self):
        self.player = [600, 100]  # Reset player to initial position
        self.goalkeeper_position = [600, 1100]  # Reset goalkeeper to center of the goal line
        self.ball = [600, 100]  # Reset ball to starting position with the player
        # Optionally, you could randomize the starting position slightly or keep it static
        return np.array(self.player + self.goalkeeper_position + self.ball)


    def step(self, action):
        x_goal = action % 1000 + 100  # Calculating the x position where the ball should end up
        y_goal = 1100  # y position at goal line
        section = np.random.choice(['Top-left', 'Top-right', 'Middle-up', 'Middle-down', 'Bottom-left', 'Bottom-right'])
        self.update_goalkeeper_position(section)

        # Simulate the ball moving
        self.ball_trajectory(x_goal, y_goal)

        # Simulate goalkeeper's dive trajectory
        section = self.decide_goalkeeper_dive(x_goal)

        # Check if goal is scored and assign reward
        goal_scored = self._is_goal(x_goal, y_goal)
        reward = 1 if goal_scored else 0
        done = True
        info = {"goalkeeper_section": section, "goolkeeper_position": self.goalkeeper_position}

        return np.array(self.player + self.goalkeeper_position + self.ball), reward, done, info


    def decide_goalkeeper_dive(self, x_goal):
        # Placeholder logic to select the section based on where the ball is going
        if x_goal <= 400:
            return 'Bottom-left' if x_goal <= 250 else 'Top-left'
        elif x_goal <= 700:
            return 'Middle-down' if x_goal <= 550 else 'Middle-up'
        else:
            return 'Bottom-right' if x_goal <= 850 else 'Top-right'


    def update_goalkeeper_position(self, section):
        sections = {
            'Top-left': [250, 750],  # x = 100 to 400, mid x = 250, y = 500 to 1000, mid y = 750
            'Top-right': [850, 750],  # x = 700 to 1000, mid x = 850, y = 500 to 1000, mid y = 750
            'Middle-up': [550, 750],  # x = 400 to 700, mid x = 550, y = 500 to 1000, mid y = 750
            'Middle-down': [550, 300],  # x = 400 to 700, mid x = 550, y = 100 to 500, mid y = 300
            'Bottom-left': [250, 300],  # x = 100 to 400, mid x = 250, y = 100 to 500, mid y = 300
            'Bottom-right': [850, 300]  # x = 700 to 1000, mid x = 850, y = 100 to 500, mid y = 300
        }
        self.goalkeeper_position = sections[section]

    def _is_goal(self, x_goal, y_goal):
        goalie_x, goalie_y = self.goalkeeper_position
        return not ((goalie_x - 150 <= x_goal <= goalie_x + 15)  and (goalie_y - 250 <= y_goal <= goalie_y + 250))


    def ball_trajectory(self, x_goal, y_goal):
        # Move the ball from the player position to the goal position
        x_start, y_start = self.ball
        x_step = (x_goal - x_start) / 10.0  # Dividing the path into 10 steps
        y_step = (y_goal - y_start) / 10.0
        for i in range(5):  # Animate half-way
            self.ball = [x_start + i * x_step, y_start + i * y_step]
            self.render()
        # Delay for goalkeeper's reaction
        for i in range(5, 10):  # Complete the animation
            self.ball = [x_start + i * x_step, y_start + i * y_step]
            self.render()


    def render(self, mode='human'):
        self.ax.clear()
        self.ax.set_xlim(0, 1200)
        self.ax.set_ylim(0, 1200)

        # Draw the goal
        goal = patches.Rectangle((100, 100), 1000, 1000, linewidth=1, edgecolor='r', facecolor='none')
        self.ax.add_patch(goal)

        # Draw the sections
        # self.ax.add_patch(patches.Rectangle((100, 500), 300, 500, linewidth=1, edgecolor='r', facecolor='none'))  # Top-left
        # self.ax.add_patch(patches.Rectangle((400, 500), 300, 500, linewidth=1, edgecolor='r', facecolor='none'))  # Middle-up
        # self.ax.add_patch(patches.Rectangle((700, 500), 300, 500, linewidth=1, edgecolor='r', facecolor='none'))  # Top-right
        # self.ax.add_patch(patches.Rectangle((100, 100), 300, 400, linewidth=1, edgecolor='r', facecolor='none'))  # Bottom-left
        # self.ax.add_patch(patches.Rectangle((400, 100), 300, 400, linewidth=1, edgecolor='r', facecolor='none'))  # Middle-down
        # self.ax.add_patch(patches.Rectangle((700, 100), 300, 400, linewidth=1, edgecolor='r', facecolor='none'))  # Bottom-right

        # Draw the goalkeeper
        goalkeeper = patches.Circle((self.goalkeeper_position[0], self.goalkeeper_position[1]), 80, color='green')
        self.ax.add_patch(goalkeeper)

        # Draw the player
        player = patches.Circle((self.player[0], self.player[1]), 50, color='blue')
        self.ax.add_patch(player)

        # Draw the ball
        ball = patches.Circle((self.ball[0], self.ball[1]), 10, color='black')  # Adjust size as needed
        self.ax.add_patch(ball)

        plt.pause(0.05)  # Shorter pause for smoother animation


    def close(self):
        plt.close(self.figure)

# Create and run the environment
env = SoccerPenaltyEnv()
for i in range(50):  # Let's run it for 2 episodes for simplicity
    obs = env.reset()
    print("")
    print(f"Episode {i+1}")
    action = env.action_space.sample()  # Take a random action

    # Decode the action to understand the direction
    x_goal = action % 1000 + 100  # Calculating the x position where the ball should end up
    y_goal = 1100  # y position at goal line
    direction = ""
    if x_goal <= 433:  # Dividing the goal into three horizontal sections
        direction += "left"
    elif x_goal <= 766:
        direction += "middle"
    else:
        direction += "right"

    new_obs, reward, done, info = env.step(action)
    env.render()  # Render the environment

    # Print detailed information
    print(f"Action taken: Shot towards {direction}")
    print(f"Final ball position: x={env.ball[0]}, y={env.ball[1]}")
    print(f"Goalkeeper's final position: x={env.goalkeeper_position[0]}, y={env.goalkeeper_position[1]}")
    print(f"Goalkeeper dove to: {info['goalkeeper_section']}, Reward: {reward}")
    print("---" * 10)

env.close()
