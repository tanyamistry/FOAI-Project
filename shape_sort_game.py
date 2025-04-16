# shape_sort_game.py
import pygame
import gym
from gym import spaces
import random
import numpy as np
import sys


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60


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


SHAPE_LIST = ['circle', 'triangle', 'square', 'cross']
COLOR_LIST = ['red', 'green', 'blue', 'yellow']


NUM_BINS = 4
BIN_WIDTH = SCREEN_WIDTH // NUM_BINS

SHAPE_BIN_Y = 250
SHAPE_BIN_HEIGHT = 50

COLOR_BIN_Y = 500
COLOR_BIN_HEIGHT = 50


SHAPE_CONVEYOR_Y = 150
COLOR_CONVEYOR_Y = 350


SHAPE_SIZE = 30 
MOVE_STEP = 20   
DROP_SPEED = 10  


# ----- Helper Functions -----
def draw_shape(surface, shape, color, center, size):

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

# ----- Environment Class -----
class FallingShape:
    """A falling object with a given shape and color."""
    def __init__(self):
        self.shape = random.choice(SHAPE_LIST)
        self.color_name = random.choice(COLOR_LIST)
        self.color = COLOR_MAP[self.color_name]
        self.x = SCREEN_WIDTH // 2 
        self.y = 0                
        self.is_dropping = False   

    def reset(self):
        self.__init__()

class ShapeSortEnv(gym.Env):
  
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, render_mode='human', max_trials=5):
        super(ShapeSortEnv, self).__init__()
        self.render_mode = render_mode
        self.animate = (self.render_mode == 'human')
        self.max_trials = max_trials

       
        self.action_space = spaces.Discrete(3)
       
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
        self.sorting_stage = 0  
        self.falling_shape = FallingShape()
        self.falling_shape.x = SCREEN_WIDTH // 2
        self.falling_shape.y = SHAPE_CONVEYOR_Y
        self.shape_correct = False  

    def _set_bins(self):

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

        stage = np.array([self.sorting_stage], dtype=np.float32)
        shape_one_hot = np.zeros(4, dtype=np.float32)
        shape_one_hot[SHAPE_LIST.index(self.falling_shape.shape)] = 1.0
        color_one_hot = np.zeros(4, dtype=np.float32)
        color_one_hot[COLOR_LIST.index(self.falling_shape.color_name)] = 1.0
        x_norm = np.array([self.falling_shape.x / SCREEN_WIDTH], dtype=np.float32)
        obs = np.concatenate([stage, shape_one_hot, color_one_hot, x_norm])
        return obs

    def reset(self):

        self.score = 0
        self.trial_count = 0
        self.sorting_stage = 0
        self.falling_shape = FallingShape()
        self.falling_shape.x = SCREEN_WIDTH // 2
        self.falling_shape.y = SHAPE_CONVEYOR_Y
        self.shape_correct = False
        return self._get_obs()

    def step(self, action):
    
        reward = 0
        done = False
        info = {}

        if not self.falling_shape.is_dropping:
            if action == 0:  
                self.falling_shape.x = max(0, self.falling_shape.x - MOVE_STEP)
            elif action == 1: 
                self.falling_shape.x = min(SCREEN_WIDTH, self.falling_shape.x + MOVE_STEP)
            elif action == 2: 
                self.falling_shape.is_dropping = True

                if self.sorting_stage == 0:
                 
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


                    self.sorting_stage = 1
                    self.falling_shape.y = COLOR_CONVEYOR_Y

                elif self.sorting_stage == 1:
                  
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

                
                    if self.shape_correct and reward_stage == 10:
                        reward_stage += 10

                    reward = reward_stage

                    self.trial_count += 1
                   
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

        if not self.animate:
            return

        self.screen.fill(COLOR_MAP['light_blue'])


        shape_conveyor = pygame.Rect(0, SHAPE_CONVEYOR_Y - 20, SCREEN_WIDTH, 40)
        pygame.draw.rect(self.screen, COLOR_MAP['gray'], shape_conveyor)
        color_conveyor = pygame.Rect(0, COLOR_CONVEYOR_Y - 20, SCREEN_WIDTH, 40)
        pygame.draw.rect(self.screen, COLOR_MAP['gray'], color_conveyor)

      
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


        draw_shape(self.screen, self.falling_shape.shape, self.falling_shape.color,
                   (self.falling_shape.x, self.falling_shape.y), SHAPE_SIZE)

       
        font = pygame.font.SysFont(None, 36)
        stage_text = "Shape Sorting" if self.sorting_stage == 0 else "Color Sorting"
        info_text = f"Stage: {stage_text}  |  Score: {self.score}  |  Trial: {self.trial_count + 1}/{self.max_trials}"
        info_surface = font.render(info_text, True, COLOR_MAP['black'])
        self.screen.blit(info_surface, (20, 20))

        pygame.display.flip()

    def close(self):
        if self.animate:
            pygame.quit()


def human_play():
    
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
