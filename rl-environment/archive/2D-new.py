import gym
from gym import spaces
import numpy as np
import cv2
import torch
import os


def read_folder(directory):
    directory_files = [os.path.join(directory, f) for f in os.listdir(directory) if
                       f.endswith(('png', 'jpg', 'jpeg'))]
    directory_files.sort()
    return directory_files


class FootballEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, directory, model_path, weights_path, reference_points):
        super(FootballEnv, self).__init__()

        self.directory = directory
        self.model_path = model_path
        self.weights_path = weights_path
        self.reference_points = reference_points
        self.model = torch.hub.load(self.model_path, 'custom', path=self.weights_path, source='local',
                                    force_reload=True)

        self.files = read_folder(self.directory)
        self.current_file_index = 0

        # Define action and observation space
        # Example: action_space = spaces.Discrete(2) for a binary action space
        self.action_space = spaces.Discrete(4)  # For examples, 4 discrete actions
        self.observation_space = spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)  # Image shape

    def step(self, action):
        if self.current_file_index >= len(self.files):
            self.current_file_index = 0  # Loop over the files

        frame = cv2.imread(self.files[self.current_file_index])
        info = self.model(self.files[self.current_file_index])
        goal_points, goal_sections, ball_position = self.establish_goal(info, self.model)

        reward = 0
        done = False

        # Example reward logic (needs to be adapted to your specific task)
        if ball_position:
            reward += 1

        self.current_file_index += 1

        observation = frame
        return observation, reward, done, {}

    def reset(self):
        self.current_file_index = 0
        return self.step(0)[0]

    def render(self, mode='human'):
        frame = cv2.imread(self.files[self.current_file_index - 1])
        cv2.imshow('FootballEnv', frame)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

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

        # Extract detected goal sections and ball
        for *box, confidence, class_labels in detection:
            if confidence >= 0.3:
                class_name = model.names[int(class_labels)]
                if class_name in goal_sections:
                    x1, y1, x2, y2 = map(int, box)
                    goal_sections[class_name] = (x1, y1, x2, y2)
                elif class_name == "football":  # Assuming the class name for the ball in the model is 'football'
                    x1, y1, x2, y2 = map(int, box)
                    ball_position = (x1, y1, x2, y2)

        goal_points = []
        if all(goal_sections[section] is not None for section in
               ["goal-bottom-left", "goal-bottom-right", "goal-top-left", "goal-top-right"]):
            goal_points = [
                (goal_sections["goal-bottom-left"][0], goal_sections["goal-bottom-left"][3]),  # Bottom-left
                (goal_sections["goal-bottom-right"][2], goal_sections["goal-bottom-right"][3]),  # Bottom-right
                (goal_sections["goal-top-left"][0], goal_sections["goal-top-left"][1]),  # Top-left
                (goal_sections["goal-top-right"][2], goal_sections["goal-top-right"][1]),  # Top-right
            ]
        return goal_points, goal_sections, ball_position


# Define your model path, weights path, and file path
model_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/.venv/lib/python3.10/site-packages/yolov5'
weights_path = '/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/yolov5model-training/model/best.pt'
file_path = "/Users/kim.lichtenberg/Desktop/kim-fifa/crispy-couscous/homography-mapping/footage"

# Example reference points for homography
reference_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="float32")

# Create the environment
env = FootballEnv(file_path, model_path, weights_path, reference_points)

# Reset the environment
observation = env.reset()

# Example loop to interact with the environment
for _ in range(100):
    action = env.action_space.sample()  # Sample random action
    observation, reward, done, info = env.step(action)
    env.render()
    if done:
        break

env.close()
