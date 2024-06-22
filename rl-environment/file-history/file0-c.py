"""
This file works in a sense that it:
- correctly models the pulsating circle under the ball
- checks if the circle is very small before shooting
- shoots to a random section in the goal
- resets after each shot taken
- shoots in a straight line and produces the correct output for the viewer
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random
import ast


class Environment(gym.Env):
    """
    Some documentation, because we need it desperately.
    The goals of the program:
    1. It has to embody a virtual environment within the OpenAI Gymnasium resources
    STRIKE 2. It will be created with pygame instead of matplotlib for more flexibility
    3. The goalkeepers logic is hardcoded as follows:
        A. Get input from the computer vision model about possible positions
        B. Choose a random goal direction based on that section
        C. Within that goal section, choose a random position to dive to
            ^- within more pondering, would this even matter?
        D. Dive to that position.
        E. After the dive, return to a random x-axis position within the center
    4.  The agent's logic should be coded as such:
        A. Look at where the goalkeeper is standing.
        STRIKE B. Choose an offensive goal direction based on that section
        STRIKE C. Shoot the ball to that position
    """

    def __init__(self):
        super(Environment, self).__init__()

        # Define the observation space: a 2D coordinate grid.
        # In pygame, the (0,0) coordinate is starting in the upper left instead of the bottom left.
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([1280, 960]), dtype=np.float32)

        # Define the action space: the agent can randomly select six different areas to shoot to.
        self.action_space = spaces.Discrete(6)

        # Define the agents and objects involved.
        self.player = None
        self.goalkeeper = None
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
        self.current_step = 0
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

        self.action_dictionary = {
            0: "top-left",
            1: "center-up",
            2: "top-right",
            3: "bottom-left",
            4: "center-down",
            5: "bottom-right"
        }

        # Define a placeholder for the input of the mappings
        self.input = read_file(file_path)

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
        self.goalkeeper = np.array([640, 280], dtype=np.float32)

        self.ellipse = pygame.Rect(0, 0, 30, 20)
        self.ellipse.center = self.middle

        # Reset all the booleans
        self.ellipse_locked = False
        self.ball_animation = False

        # Reset all the counters
        self.update_counter = 0
        self.ball_progress = 0

        # Parse all the information in the current state and return the array
        self.state = np.array([self.player, self.ball, self.goalkeeper])
        reset_info = {"Player reset position: ": self.player,
                      "Ball reset position: ": self.ball,
                      "Goalkeeper reset position ": self.goalkeeper,
                      "Episode: ": self.current_step}
        return np.array(self.state), reset_info

    def step(self, action):
        self.current_step += 1
        reward_step = None
        done_step, truncated_step = False, False

        if self.ellipse_locked and not self.ball_animation:
            # Shoot the ball to the section corresponding to the action
            self.shoot_ball(action)

        info_step = {"Player position: ": self.player,
                     "Ball position: ": self.ball,
                     "Goalkeeper position ": self.goalkeeper,
                     "Episode: ": self.current_step}
        return self.state, reward_step, done_step, truncated_step, info_step

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
            print(f"Is the ellipse locked?: {self.ellipse_locked}")
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
            self.ball_progress += 0.01

            # The animation is finished when the progress is 1 (0 is not started, 1 is finished animation)
            if self.ball_progress >= 1:
                self.ball_progress = 1
                self.ball_animation = False

                # Store the action by referencing back to the mapping of the dictionary
                action_name = self.action_dictionary[self.current_action]
                print(f"I shoot to: {self.current_action}, {action_name}.\n My coordinates are {self.ball}")

                # Optional: implement some waiting time
                pygame.time.delay(3000)
                self.reset()
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

    def render(self, mode="human"):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Draw the background for the screen
        self.screen.fill((255, 255, 255))

        # Draw the player
        player = pygame.Rect(0, 0, 60, 150)
        player.center = self.left_leg
        pygame.draw.ellipse(self.screen, "red", player)

        # Draw the goalkeeper
        goalkeeper = pygame.Rect(640, 230, 60, 200)
        pygame.draw.ellipse(self.screen, "green", goalkeeper)

        # Update the ball and draw it
        self.animate_ball()
        pygame.draw.circle(self.screen, "blue", self.ball, self.ball_radius)

        # Update and draw the ellipse
        self.update_ellipse()
        self.update_counter += 1

        # For every i iterations, check the size of the circle and if there is a good opportunity to shoot
        if self.update_counter >= 100:
            self.check_circle()
            self.update_counter = 0
            print("------")
            print(f"Is circle locked?: {self.ellipse_locked}")
        pygame.draw.ellipse(self.screen, "cyan", self.ellipse)

        # Draw the goal sections
        for section in self.goal_sections:
            (x1, x2), (y1, y2) = section
            rectangle = pygame.Rect(x1, y1, x2 - x1, y2 - y1)
            pygame.draw.rect(self.screen, "grey", rectangle, width=2)

        # Display pygame
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()


def read_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    goal_output = ast.literal_eval(content)
    print("File read successfully")
    return goal_output


file_path = "/environment/output.txt"

if __name__ == "__main__":
    env = Environment()
    state, reset_info = env.reset()
    done, truncated = False, False
    counter = 0

    while not done and not truncated:
        env.render()
        if not env.ball_animation:
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
        counter += 1

    env.close()
