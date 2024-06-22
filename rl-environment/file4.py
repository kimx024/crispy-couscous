"""
This file works in a sense that it:
- correctly models the pulsating circle under the ball
- checks if the circle is very small before shooting
- shoots to a random section in the goal
- resets after each shot taken
- shoots in a straight line and produces the correct output for the viewer

- moves the goalkeeper based on the output generated
- enhancement: generates a score
- add a reward
- add a cumulative reward

- move the goalkeeper in a straight line
- check if the lines of ball and goalkeeper intercept

- add the stable baseline model
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random
import ast
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env


class Environment(gym.Env):
    def __init__(self, file_path):
        super(Environment, self).__init__()

        # To use the Processing class, defined down below.
        self.p = Processing(file_path)

        # Define the observation space: a 2D coordinate grid.
        # In pygame, the (0,0) coordinate is starting in the upper left instead of the bottom left.
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([1280, 960]), dtype=np.float32)

        # Define the action space: the agent can randomly select six different areas to shoot to.
        self.action_space = spaces.Discrete(6)

        # Define the agents and objects involved.
        self.player = None
        self.goalkeeper = None
        self.move_goalkeeper_flag = False
        self.ball = None
        self.ellipse = None

        # Define specific, reoccurred places
        self.middle = np.array((640, 680))
        self.left_leg = np.array((590, 680))
        self.right_leg = np.array((740, 420))

        # Specifics for the circle
        self.amplitude = 40
        self.frequency = 0.01
        self.time = 0
        self.update_counter = 0

        # Specifics for the ball
        self.ball_radius = 20
        self.ellipse_locked = False
        self.ball_animation = False
        self.ball_start_pos = np.array(self.middle)
        self.ball_target_pos = np.array(self.middle)  # First target position is always the start position
        self.ball_progress = 0

        # Define the current state. This variable holds all the information before and after each action.
        self.state = np.array([])
        self.episode_step = 0
        self.current_action = None

        # Define the goal sections
        self.goal_sections = [
            ((50, 450), (150, 290)),  # width 400, height 140 - top-left
            ((450, 830), (150, 290)),  # width 380, height 140 - center-up
            ((830, 1230), (150, 290)),  # width 400, height 140 - top-right
            ((50, 450), (290, 430)),  # width 400, height 140 - bottom-left
            ((450, 830), (290, 430)),  # width 380, height 140 - center-down
            ((830, 1230), (290, 430))  # width 400, height 140 - bottom-right
        ]

        # Define the meaning of the actions that can be taken in the action space
        self.action_dictionary = {
            0: "top-left",
            1: "center-up",
            2: "top-right",
            3: "bottom-left",
            4: "center-down",
            5: "bottom-right"
        }

        # Define variables for the input of the Processing class
        self.input = self.p.read_file()
        self.p.formatted_list = self.input
        self.formatted_input = self.p.format_output()
        self.mapped_input = self.p.map_section()

        # Score variables
        # self.player_score = 0
        # self.goalkeeper_score = 0

        # Reward variables
        self.reward_win = 1.0
        self.reward_lose = -1.0
        self.current_reward = 0.0
        self.cumulative_reward = 0.0

        # Initialize Pygame
        pygame.init()
        self.screen_size_x = 1280
        self.screen_size_y = 960
        self.screen = pygame.display.set_mode((self.screen_size_x, self.screen_size_y))
        pygame.display.set_caption('Reinforcement learning environment')
        self.clock = pygame.time.Clock()

    def reset(self, *, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        # Reset the agents and objects involve
        self.player = np.array([600, 750])
        self.ball = np.array([640, 680])
        self.goalkeeper = np.array([640, 240])

        self.ellipse = pygame.Rect(0, 0, 30, 20)
        self.ellipse.center = self.middle

        # Reset all the booleans
        self.ellipse_locked = False
        self.ball_animation = False
        self.move_goalkeeper_flag = False
        self.calculation_flag = False

        # Reset all the counters
        self.update_counter = 0
        self.ball_progress = 0
        self.current_reward = 0.0
        self.cumulative_reward = 0.0

        # Parse all the information in the current state and return the array
        self.state = np.array([self.player, self.ball, self.goalkeeper])
        reset_info = {
            "Episode: ": self.episode_step,
            "Player reset position: ": self.player,
            "Ball reset position: ": self.ball,
            "Goalkeeper reset position ": self.goalkeeper,
            "Ellipse size: ": self.ellipse
        }
        return np.array(self.state), reset_info

    def reset_objects(self):
        self.player = np.array([600, 750])
        self.ball = np.array([640, 680])
        self.goalkeeper = np.array([640, 280], dtype=np.float32)

        self.ellipse = pygame.Rect(0, 0, 30, 20)
        self.ellipse.center = self.middle

        self.ellipse_locked = False
        self.ball_animation = False
        self.move_goalkeeper_flag = False

        self.update_counter = 0
        self.ball_progress = 0

        self.state = np.concatenate([self.player, self.ball, self.goalkeeper])
        reset_info = {
            "Episode: ": self.episode_step,
            "Player reset position: ": self.player,
            "Ball reset position: ": self.ball,
            "Goalkeeper reset position ": self.goalkeeper,
            "Ellipse size: ": self.ellipse
        }
        return self.state, reset_info

    def reset_reward(self):
        self.current_reward = 0.0
        return self.current_reward

    def reset_cumulative_reward(self):
        self.cumulative_reward = 0.0
        return self.cumulative_reward

    def step(self, action):
        self.episode_step += 1
        done_step = False
        truncated_step = False

        # Check if the circle is locked
        # If the circle is locked, the player decides to shoot and the ball can be animated
        if self.ellipse_locked and not self.ball_animation:
            self.shoot_ball(action)

        self.animate_ball()

        # Calculate the cumulative reward and specify when the agent is done with one episode
        if not self.ball_animation and self.ellipse_locked:
            self.calculate_cumulative_reward()
            done_step = True
            print("The action has been truncated and the environment is closed.")

        # print(f'Current reward: {self.current_reward}')
        # print(f'Cumulative reward: {self.cumulative_reward}')

        # Specify when the action gets truncated, for example when the ball, goalkeeper and ellipse get stuck
        if (self.episode_step >= 5000 and not self.ball_animation and not self.move_goalkeeper_flag
                and self.ellipse_locked):
            truncated_step = True
            print(f"The action has been truncated and the environment is closed.\n"
                  f"The number of episodes: {self.episode_step} \n"
                  f"The ball is at: {self.ball}, the goalkeeper at: {self.goalkeeper}")

        self.reset_reward() # Reset rewards after calculation and usage

        # Provide the information to be returned
        self.state = np.concatenate([self.player, self.ball, self.goalkeeper])
        info_step = {"Episode: ": self.episode_step,
                     "Player position: ": self.player,
                     "Ball position: ": self.ball,
                     "Goalkeeper position ": self.goalkeeper,
                     "Reward: ": self.cumulative_reward,
                     }
        # print(info_step)
        return self.state, self.cumulative_reward, done_step, truncated_step, info_step

    def calculate_cumulative_reward(self):
        self.cumulative_reward += self.current_reward
        return self.cumulative_reward

    def check_circle(self):
        """
        Check the size of the pulsating circle under the ball.
        If the circle is within 10% of the ball's size, lock the circle's size and proceed with shooting.
        :return:
        """
        ball_diameter = self.ball_radius * 2
        ellipse_width, ellipse_height = self.ellipse.size
        if (abs(ellipse_width - ball_diameter) <= 0.2 * ball_diameter
                or abs(ellipse_height - ball_diameter) <= 0.2 * ball_diameter):
            self.ellipse_locked = True
            # print(f"Is the circle locked?: {self.ellipse_locked}")
            return self.ellipse

        return self.ellipse

    def shoot_ball(self, action):
        """
         After the player is in range, shoot the ball to a random position in the goal sections of the goal
         The mapping of the goal sections happens as written in the dictionary in the def __init__() function.
         :return:
         """
        # Map the sections of the goal to the actions
        section = self.goal_sections[action]
        (x1, x2), (y1, y2) = section

        # From the extracted coordinates of the tuples, pick a random place
        new_x = random.randint(x1, x2)
        new_y = random.randint(y1, y2)

        self.ball_start_pos = self.ball.copy()
        self.ball_target_pos = np.array([new_x, new_y])

        # Animate the ball to the target position
        self.ball_animation = True
        self.ball_progress = 0
        self.move_goalkeeper_flag = False
        self.current_action = action
        return self.ball

    def animate_ball(self):
        """
        Animate the position of the ball in a straight line.
        :return:
        """
        # If the agent decides to shoot, put the animation in motion.
        # The ball_progress can be increased for smoother or faster processing
        if self.ball_animation:
            self.ball_progress += 0.07

            if not self.move_goalkeeper_flag:
                self.move_goalkeeper()
                self.move_goalkeeper_flag = True

            # The animation is finished when the progress is 1 (0 is not started, 1 is finished animation)
            if self.ball_progress >= 1:
                self.ball_progress = 1
                self.ball_animation = False

                # Store the action by referencing back to the mapping of the dictionary
                action_name = self.action_dictionary[self.current_action]
                print(f"I shoot to: {self.current_action}, {action_name}.\n My coordinates are {self.ball}")

                if self.is_goal():
                    self.current_reward = self.reward_win  # Set the current reward
                else:
                    self.current_reward = self.reward_lose  # Set the current reward

                # Optional: Update score based on the outcome
                # if self.is_goal():
                #     self.player_score += 1
                #     print("Goal!")
                # else:
                #     self.goalkeeper_score += 1
                #     print("Saved by the goalkeeper!")

                # Optional: implement some waiting time
                pygame.time.delay(3000)
                self.reset_objects()
            self.ball = self.ball_start_pos + (self.ball_target_pos - self.ball_start_pos) * self.ball_progress
            return self.ball

    def update_ellipse(self):
        """
        Update the pulsating circle under the ball so the agent can decide when to shoot.
        :return:
        """
        if not self.ellipse_locked:
            pulsate = math.sin(self.time * self.frequency) * self.amplitude
            width = 40 + pulsate
            height = 30 + pulsate

            if width > 0 and height > 0:
                self.ellipse = pygame.Rect(0, 0, width, height)
                self.ellipse.center = self.middle

            self.time += 1

    def move_goalkeeper(self):
        if self.mapped_input:
            section_name = random.choice(self.mapped_input)
            if section_name in self.p.environment_dict:
                new_position = self.p.choose_random_position(section_name)
                self.goalkeeper = np.array(new_position, dtype=np.float32)

    def is_goal(self) -> bool:
        goal_x_min, goal_x_max = 50, 1230
        goal_y_min, goal_y_max = 150, 430

        ball_x, ball_y = self.ball

        if not (goal_x_min <= ball_x <= goal_x_max and goal_y_min <= ball_y <= goal_y_max):
            return False

        goalkeeper_x, goalkeeper_y = self.goalkeeper
        goalkeeper_width, goalkeeper_height = 60, 200

        goalkeeper_box = pygame.Rect(goalkeeper_x - goalkeeper_width // 2,
                                     goalkeeper_y - goalkeeper_height // 2,
                                     goalkeeper_width, goalkeeper_height)

        ball_radius = self.ball_radius
        ball_box = pygame.Rect(ball_x - ball_radius, ball_y - ball_radius, ball_radius * 2, ball_radius * 2)

        overlap = pygame.Rect.colliderect(goalkeeper_box, ball_box)

        if overlap:
            return True
        else:
            return True

    def render(self, mode="human"):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            # Draw the background
        self.screen.fill((255, 255, 255))

        # Draw the goal
        for section in self.goal_sections:
            (x1, x2), (y1, y2) = section
            rectangle = pygame.Rect(x1, y1, x2 - x1, y2 - y1)
            pygame.draw.rect(self.screen, "grey", rectangle, width=2)

        # Draw the player position for the agent
        player = pygame.Rect(0, 0, 60, 150)
        player.center = self.left_leg
        pygame.draw.ellipse(self.screen, "red", player)

        # Draw the goalkeeper
        goalkeeper_x, goalkeeper_y = self.goalkeeper
        goalkeeper = pygame.Rect(goalkeeper_x + 0, goalkeeper_y + 0, 60, 200)
        pygame.draw.ellipse(self.screen, "green", goalkeeper)

        # Update and draw the ellipse
        self.update_ellipse()
        self.update_counter += 1

        if self.update_counter >= 100:
            self.check_circle()
            self.update_counter = 0
            # print("------")
            # print(f"Is circle locked?: {self.ellipse_locked}")
        pygame.draw.ellipse(self.screen, "cyan", self.ellipse)

        # Update the ball and draw it
        self.animate_ball()
        pygame.draw.circle(self.screen, "blue", self.ball, self.ball_radius)

        # Display the score font = pygame.font.Font(None, 36) score_text = font.render(f"Player: {self.player_score}
        # | Goalkeeper: {self.goalkeeper_score}", True, (0, 0, 0)) self.screen.blit(score_text, (10, 10))

        # Initialize pygame
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()


class Processing:
    """
    This class handles the processing from the output of the computer vision model and the conversion
    to work with the current RL environment in the previous class.

    This class is activated in `class Environment(gym.Env)` in the initialisation by the variable p
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.goal_output = None
        self.formatted_list: list = []

        # Repeat the environment's coordinates once more, so it is easier to map the output through this class
        self.environment_dict = {
            "top-left": ((50, 450), (150, 290)),
            "center-up": ((450, 830), (150, 290)),
            "top-right": ((830, 1230), (150, 290)),
            "bottom-left": ((50, 450), (290, 430)),
            "center-down": ((450, 830), (290, 430)),
            "bottom-right": ((830, 1230), (290, 430))
        }

    def read_file(self):
        with open(self.file_path, "r") as file:
            content = file.read()

        self.goal_output = ast.literal_eval(content)
        print("File read successfully")
        return self.goal_output

    def format_output(self) -> list:
        """
        In this function we format the output of the computer vision model to work with the RL-environment
        Because the environment is twice as big, the points in each tuple pair are multiplied by 2
        Then the x coordinates are swapped and individually assigned to coordinates that work with this environment
        :return: A list containing the formatted output
        """
        formatted_list = []
        for pair in self.formatted_list:
            (p1, p2) = pair
            point1 = (p1[0] * 2, p1[1] * 2)
            point2 = (p2[0] * 2, p2[1] * 2)
            x_coordinate = (point2[0], point1[0])
            y_coordinate = (point2[1], point1[1])

            formatted_list.append((x_coordinate, y_coordinate))
        self.formatted_list = formatted_list
        return self.formatted_list

    def map_section(self) -> list:
        """
        In this function, the coordinates of the input of the computer vision is paired
        to a goal section in the RL environment.
        :return: A list containing the mapped output
        """
        section_mapping = []
        # loop over the formatted list with tuples from the previous function
        for pair in self.formatted_list:
            (x_coordinate, y_coordinate) = pair
            # Unpack each individual coordinates in the tuple
            x1, x2 = x_coordinate
            y1, y2 = y_coordinate

            section_found = False
            for section, ((x_min, x_max), (y_min, y_max)) in self.environment_dict.items():
                if (x_min <= x1 <= x_max or x_min <= x2 <= x_max) and (y_min <= y1 <= y_max or y_min <= y2 <= y_max):
                    section_mapping.append(section)
                    section_found = True
                    break

            if not section_found:
                section_mapping.append(random.choice(list(self.environment_dict.keys())))

        # print(section_mapping)
        return section_mapping

    def choose_random_position(self, section_name):
        (x_min, x_max), (y_min, y_max) = self.environment_dict[section_name]
        random_x = random.randint(x_min, x_max)
        random_y = random.randint(y_min, y_max)
        return random_x, random_y


# This first executive is to purely run the environment without the model
if __name__ == "__main__":
    file_path = "/Users/kim.lichtenberg/Desktop/rl-env/shifted-output.txt"

    env = Environment(file_path)
    state, info = env.reset()
    done, truncated = False, False

    # Initialize a counter to declare an ending to break to while loop.
    # Could also be formatted with an if-loop
    counter = 0

    while not done and not truncated:
        env.render()
        if not env.ball_animation:
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
        counter += 1
        # if counter >= 5000:
        #     done, truncated = True, True

    env.close()

# This second executive is to run the RL model. For more information check the Stable baseline guidelines
if __name__ == "__main__":
    file_path = "/Users/kim.lichtenberg/PycharmProjects/pythonProject/environment/output-m.txt"
    env0 = Environment(file_path)
    env = make_vec_env(lambda: env0, n_envs=1)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    model.save("ppo_soccer_model")

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    file_path = "/Users/kim.lichtenberg/PycharmProjects/pythonProject/environment/output-m.txt"
    env0 = Environment(file_path)

    # Ensure the environment is wrapped correctly for vectorized environments
    env = make_vec_env(lambda: env0, n_envs=1)

    # Initialize the PPO model with MLP policy
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model for a specified number of time steps
    model.learn(total_timesteps=10000)

    # Save the trained model
    model.save("ppo_soccer_model")

    # Load the trained model for evaluation
    model = PPO.load("ppo_soccer_model")

    # Reset the environment for evaluation
    obs = env.reset()

    # Run the trained model in the environment and render
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        # Render the environment
        env.render()

        # Check if the episode is done and reset the environment if necessary
        if done:
            obs = env.reset()
