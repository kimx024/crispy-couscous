import numpy as np
import cv2
import torch
import os
import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class SoccerPenaltyEnv(gym.Env):
    def __init__(self, model_path, weights_path, directory, reference_points):
        super(SoccerPenaltyEnv, self).__init__()
        self.player = [600, 100]
        self.goalkeeper_position = [600, 1100]
        self.ball = [600, 100]
        self.model_path = model_path
        self.weights_path = weights_path
        self.directory = directory
        self.reference_points = reference_points
        self.model = torch.hub.load(self.model_path, 'custom', path=self.weights_path, source='local',
                                    force_reload=True)
        self.files = self.read_folder(self.directory)
        self.current_file_index = 0

        self.action_space = spaces.Discrete(1000 * 1000)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([1200, 1200]), dtype=np.int32)
        self.figure, self.ax = plt.subplots()

    def read_folder(self, directory):
        directory_files = [os.path.join(directory, f) for f in os.listdir(directory) if
                           f.endswith(('png', 'jpg', 'jpeg'))]
        directory_files.sort()
        return directory_files

    def reset(self):
        self.player = [600, 100]
        self.goalkeeper_position = [600, 1100]
        self.ball = [600, 100]
        self.current_file_index = 0
        return np.array(self.player + self.goalkeeper_position + self.ball)

    def step(self, action):
        if self.current_file_index >= len(self.files):
            self.current_file_index = 0

        frame = cv2.imread(self.files[self.current_file_index])
        info = self.model(self.files[self.current_file_index])
        goal_points, goal_sections, ball_position = self.establish_goal(info, self.model)

        if not goal_points:
            reward = 0
            done = True
            info = {"goalkeeper_section": None, "goalkeeper_position": self.goalkeeper_position}
            return np.array(self.player + self.goalkeeper_position + self.ball), reward, done, info

        x_goal = action % 1000 + 100
        y_goal = 1100
        section = np.random.choice(['Top-left', 'Top-right', 'Middle-up', 'Middle-down', 'Bottom-left', 'Bottom-right'])
        self.update_goalkeeper_position(section)
        self.ball_trajectory(x_goal, y_goal)
        section = self.decide_goalkeeper_dive(x_goal)
        goal_scored = self._is_goal(x_goal, y_goal)
        reward = 1 if goal_scored else 0
        done = True
        info = {"goalkeeper_section": section, "goalkeeper_position": self.goalkeeper_position}

        self.current_file_index += 1
        observation = frame
        return np.array(self.player + self.goalkeeper_position + self.ball), reward, done, info

    def decide_goalkeeper_dive(self, x_goal):
        if x_goal <= 400:
            return 'Bottom-left' if x_goal <= 250 else 'Top-left'
        elif x_goal <= 700:
            return 'Middle-down' if x_goal <= 550 else 'Middle-up'
        else:
            return 'Bottom-right' if x_goal <= 850 else 'Top-right'

    def update_goalkeeper_position(self, section):
        sections = {
            'Top-left': [250, 750],
            'Top-right': [850, 750],
            'Middle-up': [550, 750],
            'Middle-down': [550, 300],
            'Bottom-left': [250, 300],
            'Bottom-right': [850, 300]
        }
        self.goalkeeper_position = sections[section]

    def _is_goal(self, x_goal, y_goal):
        goalie_x, goalie_y = self.goalkeeper_position
        return not ((goalie_x - 150 <= x_goal <= goalie_x + 15) and (goalie_y - 250 <= y_goal <= goalie_y + 250))

    def ball_trajectory(self, x_goal, y_goal):
        x_start, y_start = self.ball
        x_step = (x_goal - x_start) / 10.0
        y_step = (y_goal - y_start) / 10.0
        for i in range(5):
            self.ball = [x_start + i * x_step, y_start + i * y_step]
            self.render()
        for i in range(5, 10):
            self.ball = [x_start + i * x_step, y_start + i * y_step]
            self.render()

    def render(self, mode='human'):
        self.ax.clear()
        self.ax.set_xlim(0, 1200)
        self.ax.set_ylim(0, 1200)
        goal = patches.Rectangle((100, 100), 1000, 1000, linewidth=1, edgecolor='r', facecolor='none')
        self.ax.add_patch(goal)
        goalkeeper = patches.Circle((self.goalkeeper_position[0], self.goalkeeper_position[1]), 80, color='green')
        self.ax.add_patch(goalkeeper)
        player = patches.Circle((self.player[0], self.player[1]), 50, color='blue')
        self.ax.add_patch(player)
        ball = patches.Circle((self.ball[0], self.ball[1]), 10, color='black')
        self.ax.add_patch(ball)
        plt.pause(0.05)

    def close(self):
        plt.close(self.figure)

    def establish_goal(self, frame, model):
        goal_sections = {
            "goal-top-left": None,
            "goal-middle-down": None,
            "goal-top-right": None,
            "goal-bottom-left": None,
            "goal-middle-up": None,
            "goal-bottom-right": None
        }
        ball_position = None

        detection = frame.xyxy[0]

        for *box, confidence, class_labels in detection:
            if confidence >= 0.3:
                class_name = model.names[int(class_labels)]
                if class_name in goal_sections:
                    x1, y1, x2, y2 = map(int, box)
                    goal_sections[class_name] = (x1, y1, x2, y2)
                elif class_name == "football":
                    x1, y1, x2, y2 = map(int, box)
                    ball_position = (x1, y1, x2, y2)

        goal_points = []
        if all(goal_sections[section] is not None for section in
               ["goal-bottom-left", "goal-bottom-right", "goal-top-left", "goal-top-right"]):
            goal_points = [
                (goal_sections["goal-bottom-left"][0], goal_sections["goal-bottom-left"][3]),
                (goal_sections["goal-bottom-right"][2], goal_sections["goal-bottom-right"][3]),
                (goal_sections["goal-top-left"][0], goal_sections["goal-top-left"][1]),
                (goal_sections["goal-top-right"][2], goal_sections["goal-top-right"][1]),
            ]
        return goal_points, goal_sections, ball_position


# Example usage
model_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/.venv/lib/python3.10/site-packages/yolov5'
weights_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/model/best.pt'
file_path = "/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/homography-mapping/footage"

reference_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="float32")

env = SoccerPenaltyEnv(model_path, weights_path, file_path, reference_points)
for i in range(100):
    obs = env.reset()
    print("")
    print(f"Episode {i + 1}")
    action = env.action_space.sample()
    x_goal = action % 1000 + 100
    y_goal = 1100
    direction = ""
    if x_goal <= 433:
        direction += "left"
    elif x_goal <= 766:
        direction += "middle"
    else:
        direction += "right"

    new_obs, reward, done, info = env.step(action)
    env.render()
    print(f"Action taken: Shot towards {direction}")
    print(f"Final ball position: x={env.ball[0]}, y={env.ball[1]}")
    print(f"Goalkeeper's final position: x={env.goalkeeper_position[0]}, y={env.goalkeeper_position[1]}")
    print(f"Goalkeeper dove to: {info['goalkeeper_section']}, Reward: {reward}")
    print("---" * 10)

env.close()
