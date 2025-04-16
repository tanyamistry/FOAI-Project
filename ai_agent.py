"""
ai_agent.py

This updated file implements a Deep Q‑Learning agent with Double DQN improvements 
for the two-stage shape/color sorting game. We have increased the network capacity,
extended the replay memory, and tuned the hyperparameters to help the agent learn a 
correct sorting strategy faster.

Changes made:
  - Increased learning rate (1e-3) and faster epsilon decay (0.99) for quicker learning.
  - Replaced MSELoss with SmoothL1Loss (Huber Loss) for stability.
  - Added checkpoint saving and resume functionality so training continues from the saved state.
  - Integrated command-line arguments to choose between training and evaluation.
  - The GUI is enabled (via render_mode='human') so the game window is visible during both training and evaluation.

Usage:
  - To train:   python ai_agent.py --train --episodes 500
  - To evaluate: python ai_agent.py --eval
  - If no argument is passed, it will evaluate if a checkpoint exists, else start training.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gym
from shape_sort_game import ShapeSortEnv  # Do not change this file
import argparse

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Updated DQN Network with Increased Capacity -----
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# ----- Extended Replay Memory -----
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, transition):
        # transition: (state, action, reward, next_state, done)
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# ----- DQN Agent with Double DQN Improvements and Adjusted Hyperparameters -----
class DQNAgent:
    def __init__(self, state_size, action_size, 
                 lr=1e-3, gamma=0.99, batch_size=64, 
                 memory_capacity=50000, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.99):
        self.state_size = state_size
        self.action_size = action_size

        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size
        self.gamma = gamma

        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        """Select an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return  # Not enough samples yet

        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        states = torch.FloatTensor(np.vstack(batch[0])).to(device)
        actions = torch.LongTensor(batch[1]).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.vstack(batch[3])).to(device)
        dones = torch.FloatTensor(batch[4]).unsqueeze(1).to(device)

        # Compute Q(s, a) using the policy network
        state_action_values = self.policy_net(states).gather(1, actions)

        # Double DQN: use the policy network to choose next actions,
        # then evaluate them using the target network.
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_state_values = self.target_net(next_states).gather(1, next_actions)
            expected_state_action_values = rewards + self.gamma * next_state_values * (1 - dones)

        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        """Decay epsilon after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def update_target_network(self):
        """Periodically update the target network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ----- Training Loop with Checkpoint Resume Capability -----
def train_agent(num_episodes=500, target_update=10):
    """
    Train the DQN agent with the environment showing the GUI.
    Training resumes from a saved checkpoint if it exists.
    """
    # Create environment with GUI visible
    env = ShapeSortEnv(render_mode='human', max_trials=20)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    checkpoint_path = "dqn_shape_sort_model.pth"
    start_episode = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # If checkpoint is a dictionary (from our resume training saves)
        if isinstance(checkpoint, dict) and "policy_net_state" in checkpoint:
            agent.policy_net.load_state_dict(checkpoint["policy_net_state"])
            agent.target_net.load_state_dict(checkpoint["target_net_state"])
            agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
            agent.epsilon = checkpoint["epsilon"]
            start_episode = checkpoint["episode"] + 1
            print(f"Resuming training from episode {start_episode} with epsilon {agent.epsilon:.3f}.")
        else:
            # Fallback if only state_dict is available
            agent.policy_net.load_state_dict(checkpoint)
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print("Loaded model checkpoint (without resume data).")
    else:
        print("Starting new training.")

    for episode in range(start_episode, num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            done_flag = 1 if done else 0
            agent.memory.push((state, action, reward, next_state, done_flag))
            state = next_state

            agent.optimize_model()

        agent.update_epsilon()
        if (episode + 1) % target_update == 0:
            agent.update_target_network()

        print(f"Episode {episode+1:4d}  Total Reward: {total_reward:6.2f}  Epsilon: {agent.epsilon:6.3f}")

        # Save checkpoint every 50 episodes
        if (episode + 1) % 50 == 0:
            checkpoint = {
                "policy_net_state": agent.policy_net.state_dict(),
                "target_net_state": agent.target_net.state_dict(),
                "optimizer_state": agent.optimizer.state_dict(),
                "epsilon": agent.epsilon,
                "episode": episode
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at episode {episode+1}")

    # Save final checkpoint
    checkpoint = {
        "policy_net_state": agent.policy_net.state_dict(),
        "target_net_state": agent.target_net.state_dict(),
        "optimizer_state": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "episode": episode
    }
    torch.save(checkpoint, checkpoint_path)
    env.close()
    print("Training complete.")

# ----- Evaluation / Watch Trained Agent -----
def evaluate_agent(model_path="dqn_shape_sort_model.pth", episodes=5):
    env = ShapeSortEnv(render_mode='human', max_trials=20)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy_net = DQN(state_size, action_size).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    # Handle checkpoint format
    if isinstance(checkpoint, dict) and "policy_net_state" in checkpoint:
        policy_net.load_state_dict(checkpoint["policy_net_state"])
    else:
        policy_net.load_state_dict(checkpoint)
    policy_net.eval()

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action = q_values.max(1)[1].item()

            state, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
            env.clock.tick(60)  # Control frame rate

        print(f"Evaluation Episode {ep+1:3d}  Total Reward: {total_reward:6.2f}")
    env.close()

# ----- Main Execution with Command-Line Options -----
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--eval", action="store_true", help="Evaluate the trained agent")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    args = parser.parse_args()

    # If --train is specified, train (resume if checkpoint available)
    if args.train:
        train_agent(num_episodes=args.episodes, target_update=10)
    # If --eval is specified, evaluate the trained agent
    elif args.eval:
        if os.path.exists("dqn_shape_sort_model.pth"):
            evaluate_agent(model_path="dqn_shape_sort_model.pth", episodes=5)
        else:
            print("No checkpoint found. Please run with --train to train the agent first.")
    else:
        # Default behavior: evaluate if a checkpoint exists; otherwise, start training.
        if os.path.exists("dqn_shape_sort_model.pth"):
            print("Checkpoint found. Running evaluation...")
            evaluate_agent(model_path="dqn_shape_sort_model.pth", episodes=5)
        else:
            print("No checkpoint found. Starting training...")
            train_agent(num_episodes=args.episodes, target_update=10)
