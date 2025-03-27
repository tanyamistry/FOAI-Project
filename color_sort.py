import pygame
import numpy as np
import random
import os
import time
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
        # Bin body
        pygame.draw.rect(screen, COLORS["GRAY"], 
                        (self.x - self.width//2, self.y, self.width, self.height), 0, 10)
        # Color indicator
        pygame.draw.rect(screen, self.color, 
                        (self.x - self.width//2, self.y, self.width, 25), 0, 5)
        # Label
        font = pygame.font.SysFont('Arial', 24, bold=True)
        text = font.render(self.color_name.upper(), True, COLORS["WHITE"])
        screen.blit(text, (self.x - text.get_width()//2, self.y + 35))

    def contains_point(self, x, y):
        return (self.x - self.width//2 <= x <= self.x + self.width//2) and \
               (self.y <= y <= self.y + self.height)

class SortingGameEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self):
        super(SortingGameEnv, self).__init__()
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(3)  # left, right, drop
        self.observation_space = spaces.Box(
            low=0, 
            high=1,
            shape=(  # Tuple format required
                (
                    1  # x_pos
                    + len(COLOR_NAMES)  # color_one_hot
                    + len(SHAPES)  # shape_one_hot
                    + len(COLOR_NAMES)  # bins_x positions
                    + (len(COLOR_NAMES) * len(COLOR_NAMES))  # bins_color_one_hot
                    + 1  # aligned_with_correct_bin flag
                ),  # Comma makes this a tuple
            ),
            dtype=np.float32
        )

        
        # Pygame initialization
        self.screen = None
        self.clock = None
        self.font = None
        self.bins = []
        
        # Game state
        self.current_shape = None
        self.score = 0
        self.step_count = 0
        self.max_steps = 100
        self.move_speed = 5
        self.shape_dropped = False
        self.game_over = False
        self.total_rewards = []
        self.episode_reward = 0
        self.episode_count = 0
        
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
        # print("Resetting environment...")  # Debug print
        self.score = 0
        self.step_count = 0
        self.episode_reward = 0
        self.shape_dropped = False
        self.game_over = False
        self.current_shape = self._new_shape()
        return self._get_observation()
    
    def _new_shape(self):
        shape = Shape(
            random.choice(SHAPES),
            random.choice(COLOR_NAMES)
        )
        # print(f"New shape created: {shape.shape_type} {shape.color_name}")  # Debug print
        return shape
    
    def _get_observation(self):
    # Current shape information
        x_pos = self.current_shape.x / SCREEN_WIDTH  # Normalized x-position of shape
        color_idx = COLOR_NAMES.index(self.current_shape.color_name)
        shape_idx = SHAPES.index(self.current_shape.shape_type)
        

        color_one_hot = np.zeros(len(COLOR_NAMES))
        shape_one_hot = np.zeros(len(SHAPES))
        color_one_hot[color_idx] = 1
        shape_one_hot[shape_idx] = 1


        bins_x = np.array([bin.x / SCREEN_WIDTH for bin in self.bins])

     
        bins_color_one_hot = []
        for bin in self.bins:
            bin_color_idx = COLOR_NAMES.index(bin.color_name)
            bin_color_vector = np.zeros(len(COLOR_NAMES))
            bin_color_vector[bin_color_idx] = 1
            bins_color_one_hot.extend(bin_color_vector)
        bins_color_one_hot = np.array(bins_color_one_hot)

       # Check if shape is aligned with correct bin
        aligned_with_correct_bin = 0
        correct_bin_x = None
        for bin in self.bins:
            if bin.color_name == self.current_shape.color_name:
                correct_bin_x = bin.x
                if abs(self.current_shape.x - correct_bin_x) < 5:  # 5-pixel threshold
                    aligned_with_correct_bin = 1
                break

        observation = np.concatenate((
            [x_pos],                # Shape's x-position
            color_one_hot,           # Shape color
            shape_one_hot,           # Shape type
            bins_x,                  # Bin x-positions (normalized)
            bins_color_one_hot,      # Bin colors (one-hot)
            [aligned_with_correct_bin]  # Alignment flag
        )).astype(np.float32)

        return observation
    
    def step(self, action):
        initial_x = self.current_shape.x
        
        if action == 0:  # Left
            self.current_shape.x = max(SHAPE_SIZE, self.current_shape.x - self.move_speed)
        elif action == 1:  # Right
            self.current_shape.x = min(SCREEN_WIDTH-SHAPE_SIZE, self.current_shape.x + self.move_speed)
        elif action == 2:  # Drop
            reward = self._handle_drop()
            self.step_count += 1
            self.episode_reward += reward
            return self._get_observation(), reward, self.step_count >= self.max_steps, {}
        
        reward = 0
        # Reward shaping for movement
        for bin in self.bins:
            if bin.color_name == self.current_shape.color_name:
                if abs(bin.x - self.current_shape.x) < abs(bin.x - initial_x):
                    reward += 0.1
                elif abs(bin.x - self.current_shape.x) > abs(bin.x - initial_x):
                    reward -= 0.05
        # Time penalty for each step without dropping
        reward -= 0.2  # Increased time penalty
        
        self.step_count += 1
        self.episode_reward += reward
        done = self.step_count >= self.max_steps
        if done:
            self.total_rewards.append(self.episode_reward)
        
        return self._get_observation(), reward, done, {}
    
    def _handle_drop(self):
        reward = -2  # Increased penalty for missing
        closest_bin = None
        min_distance = float('inf')
        for bin in self.bins:
            distance = abs(bin.x - self.current_shape.x)
            if distance < min_distance:
                min_distance = distance
                closest_bin = bin
        
        if min_distance < BIN_WIDTH/2:
            if closest_bin.color_name == self.current_shape.color_name:
                reward = 15  # Increased reward for correct drop
                self.score += 15
                print(f"✅ Correct drop! +15 points")
            else:
                reward = -5  # Increased penalty for wrong bin
                self.score -= 5
                print(f"❌ Wrong bin! -5 points")
        
        self.current_shape = self._new_shape()
        return reward
    
    def render(self, mode='human'):
        self.screen.fill(COLORS["BLACK"])
        
        # Conveyor belt
        pygame.draw.rect(self.screen, COLORS["GRAY"], (0, 100, SCREEN_WIDTH, 30))
        
        # Bins
        for bin in self.bins:
            bin.draw(self.screen)
            
        # Current shape
        self.current_shape.draw(self.screen)
        
        # UI Elements
        score_text = self.font.render(f"Score: {self.score}", True, COLORS["WHITE"])
        self.screen.blit(score_text, (20, 20))
        
        if mode == 'human':
            pygame.display.flip()
            self.clock.tick(FPS)
        elif mode == 'rgb_array':
            return np.transpose(pygame.surfarray.pixels3d(self.screen), (1, 0, 2))
    
    def close(self):
        pygame.quit()

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        return self.fc(x)

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, *args):
        self.buffer.append(Experience(*args))
        
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones))
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size):
        # Debug: Print initialization parameters
        # print(f"Initializing DQN Agent with:")
        # print(f"  - State Size: {state_size}")
        # print(f"  - Action Size: {action_size}")

        self.state_size = state_size
        self.action_size = action_size
        
        # Two neural networks for stable learning
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        
        # Synchronize target network with policy network initially
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimization setup
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        
        # Replay buffer for experience replay
        self.memory = ReplayBuffer(10000)
        
        # Hyperparameters
        self.batch_size = 32
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.998
        self.tau = 1e-3
        self.update_every = 4
        self.steps = 0
        
    def act(self, state, training=True):
        # Debug: Print current exploration rate and state
        # print(f"Action Selection - Exploration Rate: {self.eps:.2f}")
        # print(f"State: {state}")

        # Exploration: random action selection
        if training and random.random() < self.eps:
            action = random.randint(0, self.action_size-1)
            # print(f"  - Exploration: Randomly selected action {action}")
            return action
        
        # Exploitation: use policy network to select best action
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
            action = q_values.argmax().item()
            # print(f"  - Exploitation: Selected action {action} with Q-values {q_values}")
            return action
    
    def step(self, state, action, reward, next_state, done):
        # Store experience in replay buffer
        self.memory.push(state, action, reward, next_state, done)
        
        # Debug: Print experience details
        # print(f"Experience Stored:")
        # print(f"  - State: {state}")
        # print(f"  - Action: {action}")
        # print(f"  - Reward: {reward}")
        # print(f"  - Done: {done}")
        
        # Increment steps and potentially trigger learning
        self.steps = (self.steps + 1) % self.update_every
        
        if len(self.memory) >= self.batch_size and self.steps == 0:
            # print("Triggering learning process...")
            self.learn()

    def learn(self):
        # Sample batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Debug: Print batch statistics
        # print(f"Learning Batch:")
        # print(f"  - Batch Size: {len(states)}")
        # print(f"  - Average Reward: {rewards.mean().item():.2f}")
        
        # Compute current Q-values
        Q_current = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        Q_next = self.target_net(next_states).max(1)[0].detach()
        targets = rewards + (1 - dones) * self.gamma * Q_next
        
        # Compute loss and update network
        loss = F.mse_loss(Q_current.squeeze(), targets)
        
        # Debug: Print loss and Q-values
        print(f"  - Loss: {loss.item():.4f}")

        
        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update of target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau*policy_param.data + (1-self.tau)*target_param.data)
        
        # Decay exploration rate
        self.eps = max(self.eps_min, self.eps*self.eps_decay)
        # print(f"Updated Exploration Rate: {self.eps:.2f}")
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train():
    env = SortingGameEnv()
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    
    scores = []
    eps_history = []
    batch_size = 64
    episodes = 100  # Increased from 1 to get more meaningful training
    
    for e in range(episodes):
        state = env.reset()
        score = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                scores.append(score)
                eps_history.append(agent.eps)
                
                # print(f"Episode {e+1}/{episodes} | Score: {score:.1f} | Eps: {agent.eps:.2f}")
                
    # Save model and plot results
    agent.save("shape_sorter.pth")
    plt.plot(scores)
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.show()
    
    env.close()

def play(ai=False, model_path=None):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 24)
    
    env = SortingGameEnv()
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    if ai and model_path:
        agent.load(model_path)
        agent.eps = 0
    
    running = True
    score = 0
    state = env.reset()
    
    while running:
        if not ai:
            action = 0
            for event in pygame.event.get():
                 if event.type == pygame.QUIT:
                     running = False
                 elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT: action = 0
                    elif event.key == pygame.K_RIGHT: action = 1
                    elif event.key == pygame.K_SPACE: action = 2
                    elif event.key == pygame.K_ESCAPE: running = False
        
        if ai:
            action = agent.act(state, training=False)
        
        next_state, reward, done, _ = env.step(action)
        env.render()
        score += reward
        state = next_state
        
        if done:
            state = env.reset()
            score = 0
            
        clock.tick(FPS)
        
    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--play', action='store_true', help='Play manually')
    parser.add_argument('--ai', action='store_true', help='Use AI to play')
    args = parser.parse_args()
    
    if args.train:
        train()
    elif args.ai:
        play(ai=True, model_path="shape_sorter.pth")
    else:
        play()