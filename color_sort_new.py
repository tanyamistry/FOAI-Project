import pygame
import numpy as np
import random
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import matplotlib.pyplot as plt

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
SHAPE_SIZE = 40
BIN_WIDTH = 120
BIN_HEIGHT = 140
BIN_Y = SCREEN_HEIGHT - BIN_HEIGHT - 20

# Colors
COLORS = {
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),
    "RED": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "BLUE": (0, 0, 255),
    "YELLOW": (255, 255, 0),
    "PURPLE": (128, 0, 128),
    "GRAY": (200, 200, 200)
}

# Game parameters
SHAPES = ["circle", "square", "triangle"]
COLOR_NAMES = ["red", "green", "blue", "yellow"]
SHAPE_COLORS = {name: COLORS[name.upper()] for name in COLOR_NAMES}

class Shape:
    def __init__(self, shape_type, color_name):
        self.shape_type = shape_type
        self.color_name = color_name
        self.color = SHAPE_COLORS[color_name]
        self.x = SCREEN_WIDTH // 2
        self.y = 80
        self.size = SHAPE_SIZE
        self.speed = 5
        
    def draw(self, screen):
        if self.shape_type == "circle":
            pygame.draw.circle(screen, self.color, (self.x, self.y), self.size)
        elif self.shape_type == "square":
            rect = pygame.Rect(self.x-self.size, self.y-self.size, self.size*2, self.size*2)
            pygame.draw.rect(screen, self.color, rect)
        elif self.shape_type == "triangle":
            points = [
                (self.x, self.y - self.size),
                (self.x - self.size, self.y + self.size),
                (self.x + self.size, self.y + self.size)
            ]
            pygame.draw.polygon(screen, self.color, points)

class Bin:
    def __init__(self, x, color_name):
        self.x = x
        self.y = BIN_Y
        self.width = BIN_WIDTH
        self.height = BIN_HEIGHT
        self.color_name = color_name
        self.color = SHAPE_COLORS[color_name]
        
    def draw(self, screen):
        pygame.draw.rect(screen, COLORS["GRAY"], 
                        (self.x - self.width//2, self.y, self.width, self.height), 0, 10)
        pygame.draw.rect(screen, self.color, 
                        (self.x - self.width//2, self.y, self.width, 25), 0, 5)
        font = pygame.font.SysFont('Arial', 24, bold=True)
        text = font.render(self.color_name.upper(), True, COLORS["WHITE"])
        screen.blit(text, (self.x - text.get_width()//2, self.y + 35))

class SortingGameEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self):
        super(SortingGameEnv, self).__init__()
        
        # Enhanced observation space (28 dimensions)
        self.observation_space = spaces.Box(
            low=-1, high=1, 
            shape=(28,), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        
        # Pygame initialization
        self.screen = None
        self.clock = None
        self.font = None
        self.bins = []
        self._init_game()
        
    def _init_game(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Shape Sorting")
            
        self.bins = []
        bin_positions = np.linspace(BIN_WIDTH, SCREEN_WIDTH-BIN_WIDTH, len(COLOR_NAMES))
        for i, color in enumerate(COLOR_NAMES):
            self.bins.append(Bin(bin_positions[i], color))
            
        self.font = pygame.font.SysFont('Arial', 24)
        self.clock = pygame.time.Clock()
        
    def reset(self):
        self.score = 0
        self.step_count = 0
        self.episode_reward = 0
        self.current_shape = self._new_shape()
        return self._get_observation()
    
    def _new_shape(self):
        return Shape(
            random.choice(SHAPES),
            random.choice(COLOR_NAMES)
        )
    
    def _get_observation(self):
    # Current shape features
        x_pos = self.current_shape.x / SCREEN_WIDTH
        color_idx = COLOR_NAMES.index(self.current_shape.color_name)
        shape_idx = SHAPES.index(self.current_shape.shape_type)
        
        # Bin features (distance + color)
        bin_features = []
        for bin in self.bins:
            distance = (bin.x - self.current_shape.x) / SCREEN_WIDTH
            bin_color = np.zeros(len(COLOR_NAMES))
            bin_color[COLOR_NAMES.index(bin.color_name)] = 1
            bin_features.extend([distance, *bin_color.tolist()])  # Convert to list
        
        return np.concatenate([
            [x_pos],
            np.zeros(len(COLOR_NAMES)),
            np.zeros(len(SHAPES)),
            np.array(bin_features)  # Keep as single array
        ]).astype(np.float32)
    
    def step(self, action):
        reward = 0
        done = False
        
        # Handle action
        if action == 0:
            self.current_shape.x = max(SHAPE_SIZE, self.current_shape.x - 5)
        elif action == 1:
            self.current_shape.x = min(SCREEN_WIDTH-SHAPE_SIZE, self.current_shape.x + 5)
        elif action == 2:
            reward = self._handle_drop()
            
        # Update state
        self.step_count += 1
        self.episode_reward += reward
        done = self.step_count >= 100
        
        return self._get_observation(), reward, done, {}
    
    def _handle_drop(self):
        reward = -5  # Default penalty for missed bins
        for bin in self.bins:
            if abs(bin.x - self.current_shape.x) < SHAPE_SIZE:
                if bin.color_name == self.current_shape.color_name:
                    reward = 20
                    self.score += 20
                else:
                    reward = -10
                    self.score -= 10
                break
        self.current_shape = self._new_shape()
        return reward
    
    def render(self, mode='human'):
        self.screen.fill(COLORS["BLACK"])
        pygame.draw.rect(self.screen, COLORS["GRAY"], (0, 100, SCREEN_WIDTH, 30))
        for bin in self.bins:
            bin.draw(self.screen)
        self.current_shape.draw(self.screen)
        score_text = self.font.render(f"Score: {self.score}", True, COLORS["WHITE"])
        self.screen.blit(score_text, (20, 20))
        if mode == 'human':
            pygame.display.flip()
            self.clock.tick(FPS)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-4)
        self.memory = deque(maxlen=10000)
        self.batch_size = 128
        self.gamma = 0.95
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.995
    
    def act(self, state):
        if random.random() < self.eps:
            return random.randint(0, 2)
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.policy_net(state).argmax().item()
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = F.mse_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.eps = max(self.eps_min, self.eps*self.eps_decay)
        
    def update_memory(self, experience):
        self.memory.append(experience)
        
    def sample_memory(self):
        return random.sample(self.memory, self.batch_size)

def train():
    env = SortingGameEnv()
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    
    scores = []
    episodes = 2000
    
    for e in range(episodes):
        state = env.reset()
        score = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_memory((state, action, reward, next_state, done))
            state = next_state
            score += reward
            
            if len(agent.memory) >= agent.batch_size:
                agent.learn(agent.sample_memory())
            
            if done:
                scores.append(score)
                if e % 100 == 0:
                    print(f"Episode {e+1}/{episodes} | Score: {score} | Eps: {agent.eps:.2f}")
    
    plt.plot(scores)
    plt.title("Improved Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.show()
    torch.save(agent.policy_net.state_dict(), "improved_model.pth")

if __name__ == "__main__":
    train()