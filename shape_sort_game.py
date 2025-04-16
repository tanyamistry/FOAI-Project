"""
shape_sort_game.py

This file implements a two-stage shape and color sorting game using pygame.
In the first stage, you sort a falling object by shape into one of four bins.
In the second stage, the same object is sorted by color.
If you drop the object correctly in both stages, you receive an extra bonus reward of +10.
The environment is also wrapped as a Gym environment so that an AI agent can train on it.
"""

import pygame
import gym
from gym import spaces
import random
import numpy as np
import sys

# ----- Global Constants -----
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors in RGB format
COLOR_MAP = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'gray': (200, 200, 200),
    'light_blue': (173, 216, 230)
}

# Shape and color options
SHAPE_LIST = ['circle', 'triangle', 'square', 'cross']
COLOR_LIST = ['red', 'green', 'blue', 'yellow']

# Bins settings (4 bins spanning full width)
NUM_BINS = 4
BIN_WIDTH = SCREEN_WIDTH // NUM_BINS
# For shape sorting bins (drawn at top)
SHAPE_BIN_Y = 250
SHAPE_BIN_HEIGHT = 50
# For color sorting bins (drawn at bottom)
COLOR_BIN_Y = 500
COLOR_BIN_HEIGHT = 50

# Conveyor belt positions
SHAPE_CONVEYOR_Y = 150
COLOR_CONVEYOR_Y = 350

# Falling object properties
SHAPE_SIZE = 30  # size for drawing shapes
MOVE_STEP = 20   # horizontal move step (pixels)
DROP_SPEED = 10  # speed for drop animations

# ----- Helper Function for Drawing Shapes -----
def draw_shape(surface, shape, color, center, size):
    """Draw a shape on the surface."""
    if shape == 'circle':
        pygame.draw.circle(surface, color, center, size)
    elif shape == 'triangle':
        x, y = center
        point1 = (x, y - size)
        point2 = (x - size, y + size)
        point3 = (x + size, y + size)
        pygame.draw.polygon(surface, color, [point1, point2, point3])
    elif shape == 'square':
        rect = pygame.Rect(0, 0, 2 * size, 2 * size)
        rect.center = center
        pygame.draw.rect(surface, color, rect)
    elif shape == 'cross':
        x, y = center
        offset = size
        pygame.draw.line(surface, color, (x - offset, y - offset), (x + offset, y + offset), 5)
        pygame.draw.line(surface, color, (x + offset, y - offset), (x - offset, y + offset), 5)
    else:
        pygame.draw.circle(surface, color, center, size)

# ----- Game Classes -----
class FallingShape:
    """A falling object with a given shape and color."""
    def __init__(self):
        self.shape = random.choice(SHAPE_LIST)
        self.color_name = random.choice(COLOR_LIST)
        self.color = COLOR_MAP[self.color_name]
        self.x = SCREEN_WIDTH // 2  # Start near the center horizontally
        self.y = 0                  # Will be set by the environment
        self.is_dropping = False    # Flag for drop animation

    def reset(self):
        self.__init__()

class ShapeSortEnv(gym.Env):
    """
    Custom Gym environment for two-stage sorting.
    
    Observation:
      A 10-dimensional vector:
         [0]   Sorting stage (0: shape sorting; 1: color sorting)
         [1-4] One-hot encoding for shape (circle, triangle, square, cross)
         [5-8] One-hot encoding for color (red, green, blue, yellow)
         [9]   Normalized x-position (0 to 1)
    
    Actions:
      Discrete(3):
         0: Move Left
         1: Move Right
         2: Drop the object
    
    Reward:
      Stage 0 (shape): +10 if the shape is correctly dropped; -10 if not.
      Stage 1 (color): +10 if the color is correctly dropped; -10 if not.
         Additionally, if both stages are correct in the trial, you get an extra bonus of +10.
    
    Episode:
      Consists of a fixed number of trials.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render_mode='human', max_trials=5):
        super(ShapeSortEnv, self).__init__()
        self.render_mode = render_mode
        self.animate = (self.render_mode == 'human')
        self.max_trials = max_trials

        # Define the action and observation spaces.
        self.action_space = spaces.Discrete(3)
        # Observation: [stage, 4 shape one-hot, 4 color one-hot, normalized x]
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        if self.animate:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Shape & Color Sorting Game")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None

        self.score = 0
        self.trial_count = 0
        self._set_bins()
        self.sorting_stage = 0  # 0 for shape sorting; 1 for color sorting
        self.falling_shape = FallingShape()
        self.falling_shape.x = SCREEN_WIDTH // 2
        self.falling_shape.y = SHAPE_CONVEYOR_Y
        self.shape_correct = False  # To store result from stage 0

    def _set_bins(self):
        """Define bins for both sorting stages."""
        self.shape_bins = []
        self.color_bins = []
        for i in range(NUM_BINS):
            shape_rect = pygame.Rect(i * BIN_WIDTH, SHAPE_BIN_Y, BIN_WIDTH, SHAPE_BIN_HEIGHT)
            color_rect = pygame.Rect(i * BIN_WIDTH, COLOR_BIN_Y, BIN_WIDTH, COLOR_BIN_HEIGHT)
            shape_label = SHAPE_LIST[i]
            color_label = COLOR_LIST[i]
            self.shape_bins.append({'rect': shape_rect, 'label': shape_label})
            self.color_bins.append({'rect': color_rect, 'label': color_label})

    def _get_obs(self):
        """Return the current observation as a numpy array."""
        stage = np.array([self.sorting_stage], dtype=np.float32)
        shape_one_hot = np.zeros(4, dtype=np.float32)
        shape_one_hot[SHAPE_LIST.index(self.falling_shape.shape)] = 1.0
        color_one_hot = np.zeros(4, dtype=np.float32)
        color_one_hot[COLOR_LIST.index(self.falling_shape.color_name)] = 1.0
        x_norm = np.array([self.falling_shape.x / SCREEN_WIDTH], dtype=np.float32)
        obs = np.concatenate([stage, shape_one_hot, color_one_hot, x_norm])
        return obs

    def reset(self):
        """Reset the environment and return the initial observation."""
        self.score = 0
        self.trial_count = 0
        self.sorting_stage = 0
        self.falling_shape = FallingShape()
        self.falling_shape.x = SCREEN_WIDTH // 2
        self.falling_shape.y = SHAPE_CONVEYOR_Y
        self.shape_correct = False
        return self._get_obs()

    def step(self, action):
        """
        Execute an action:
          - For left/right actions: update x-position.
          - For drop action: if in stage 0 (shape), drop to shape bins, check correctness, and then move the object to the color conveyor.
            In stage 1 (color), drop to color bins, check correctness, and if both stage rewards are positive then add bonus.
        """
        reward = 0
        done = False
        info = {}

        if not self.falling_shape.is_dropping:
            if action == 0:  # Move left
                self.falling_shape.x = max(0, self.falling_shape.x - MOVE_STEP)
            elif action == 1:  # Move right
                self.falling_shape.x = min(SCREEN_WIDTH, self.falling_shape.x + MOVE_STEP)
            elif action == 2:  # Drop action
                self.falling_shape.is_dropping = True

                if self.sorting_stage == 0:
                    # Drop onto shape bins (target_y is the shape bin row)
                    target_y = SHAPE_BIN_Y
                    while self.falling_shape.y < target_y:
                        self.falling_shape.y += DROP_SPEED
                        self.render()
                        if self.animate:
                            self.clock.tick(FPS)
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                self.close()
                                sys.exit()

                    bin_index = int(self.falling_shape.x // BIN_WIDTH)
                    bin_index = min(bin_index, NUM_BINS - 1)
                    chosen_label = self.shape_bins[bin_index]['label']
                    if chosen_label == self.falling_shape.shape:
                        reward_stage = 10
                        self.shape_correct = True
                    else:
                        reward_stage = -10
                        self.shape_correct = False
                    reward = reward_stage

                    # Move object to the color conveyor belt for the next stage.
                    self.sorting_stage = 1
                    self.falling_shape.y = COLOR_CONVEYOR_Y

                elif self.sorting_stage == 1:
                    # Drop onto color bins.
                    target_y = COLOR_BIN_Y
                    while self.falling_shape.y < target_y:
                        self.falling_shape.y += DROP_SPEED
                        self.render()
                        if self.animate:
                            self.clock.tick(FPS)
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                self.close()
                                sys.exit()

                    bin_index = int(self.falling_shape.x // BIN_WIDTH)
                    bin_index = min(bin_index, NUM_BINS - 1)
                    chosen_label = self.color_bins[bin_index]['label']
                    if chosen_label == self.falling_shape.color_name:
                        reward_stage = 10
                    else:
                        reward_stage = -10

                    # Bonus reward if both shape and color sorting are correct.
                    if self.shape_correct and reward_stage == 10:
                        reward_stage += 10

                    reward = reward_stage

                    self.trial_count += 1
                    # Start a new trial or finish the episode.
                    if self.trial_count < self.max_trials:
                        self.sorting_stage = 0
                        self.falling_shape = FallingShape()
                        self.falling_shape.x = SCREEN_WIDTH // 2
                        self.falling_shape.y = SHAPE_CONVEYOR_Y
                    else:
                        done = True

                self.score += reward
                self.falling_shape.is_dropping = False

        obs = self._get_obs()
        return obs, reward, done, info

    def render(self, mode='human'):
        """Render the game interface with two sets of bins and two conveyor belts."""
        if not self.animate:
            return

        self.screen.fill(COLOR_MAP['light_blue'])

        # --- Draw Conveyors ---
        shape_conveyor = pygame.Rect(0, SHAPE_CONVEYOR_Y - 20, SCREEN_WIDTH, 40)
        pygame.draw.rect(self.screen, COLOR_MAP['gray'], shape_conveyor)
        color_conveyor = pygame.Rect(0, COLOR_CONVEYOR_Y - 20, SCREEN_WIDTH, 40)
        pygame.draw.rect(self.screen, COLOR_MAP['gray'], color_conveyor)

        # --- Draw Bins ---
        for bin_data in self.shape_bins:
            pygame.draw.rect(self.screen, COLOR_MAP['white'], bin_data['rect'])
            pygame.draw.rect(self.screen, COLOR_MAP['black'], bin_data['rect'], 3)
            font = pygame.font.SysFont(None, 30)
            text = font.render(bin_data['label'].capitalize(), True, COLOR_MAP['black'])
            text_rect = text.get_rect(center=bin_data['rect'].center)
            self.screen.blit(text, text_rect)

        for bin_data in self.color_bins:
            pygame.draw.rect(self.screen, COLOR_MAP['white'], bin_data['rect'])
            pygame.draw.rect(self.screen, COLOR_MAP['black'], bin_data['rect'], 3)
            font = pygame.font.SysFont(None, 30)
            text = font.render(bin_data['label'].capitalize(), True, COLOR_MAP['black'])
            text_rect = text.get_rect(center=bin_data['rect'].center)
            self.screen.blit(text, text_rect)

        # --- Draw Falling Object ---
        draw_shape(self.screen, self.falling_shape.shape, self.falling_shape.color,
                   (self.falling_shape.x, self.falling_shape.y), SHAPE_SIZE)

        # --- Display Information ---
        font = pygame.font.SysFont(None, 36)
        stage_text = "Shape Sorting" if self.sorting_stage == 0 else "Color Sorting"
        info_text = f"Stage: {stage_text}  |  Score: {self.score}  |  Trial: {self.trial_count + 1}/{self.max_trials}"
        info_surface = font.render(info_text, True, COLOR_MAP['black'])
        self.screen.blit(info_surface, (20, 20))

        pygame.display.flip()

    def close(self):
        if self.animate:
            pygame.quit()

# ----- Human Play Loop -----
def human_play():
    """Allow a human player to play the game."""
    env = ShapeSortEnv(render_mode='human', max_trials=5)
    obs = env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                env.close()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                action = None
                if event.key == pygame.K_LEFT:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_SPACE:
                    action = 2

                if action is not None:
                    obs, reward, done, info = env.step(action)
                    if done:
                        print("Trial complete! Final Score:", env.score)
                        obs = env.reset()
        env.render()
        if env.animate:
            env.clock.tick(FPS)

    env.close()

if __name__ == '__main__':
    human_play()
