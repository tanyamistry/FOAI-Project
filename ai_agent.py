#Ai Agent for Shape Sort Game
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gym
from shape_sort_game import ShapeSortEnv 

USE_GUI = True


CHECKPOINT_PATH = "dqn_shape_sort_checkpoint.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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



class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, transition):
        
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)



class DQNAgent:
    def __init__(self, state_size, action_size, 
                 lr=1e-3,           
                 gamma=0.99, 
                 batch_size=64, 
                 memory_capacity=50000, 
                 epsilon_start=1.0, 
                 epsilon_end=0.05, 
                 epsilon_decay=0.98):  
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
            return 

        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        states = torch.FloatTensor(np.vstack(batch[0])).to(device)
        actions = torch.LongTensor(batch[1]).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.vstack(batch[3])).to(device)
        dones = torch.FloatTensor(batch[4]).unsqueeze(1).to(device)

      
        state_action_values = self.policy_net(states).gather(1, actions)

   
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_state_values = self.target_net(next_states).gather(1, next_actions)
            expected_state_action_values = rewards + self.gamma * next_state_values * (1 - dones)

        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
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



def train_agent(num_episodes=1000, target_update=10):
    """
    Train the DQN agent.
    Note: Increasing the number of trials per episode (set to 20) provides more transitions per episode.
    """
    render_mode = 'human' if USE_GUI else 'rgb_array'
    env = ShapeSortEnv(render_mode=render_mode, max_trials=20)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)


    start_episode = 0
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint['epsilon']
        start_episode = checkpoint['episode'] + 1
        print(f"Resuming training from episode {start_episode}")


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

        checkpoint_data = {
            'episode': episode,
            'policy_net_state_dict': agent.policy_net.state_dict(),
            'target_net_state_dict': agent.target_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon
        }
        torch.save(checkpoint_data, CHECKPOINT_PATH)

 
    torch.save(agent.policy_net.state_dict(), "dqn_shape_sort_model.pth")
    env.close()
    print("Training complete.")



def evaluate_agent(model_path="dqn_shape_sort_model.pth", episodes=5):
    render_mode = 'human' if USE_GUI else 'rgb_array'
    env = ShapeSortEnv(render_mode=render_mode, max_trials=20)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy_net = DQN(state_size, action_size).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
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
            if USE_GUI:
                env.clock.tick(60)

        print(f"Evaluation Episode {ep+1:3d}  Total Reward: {total_reward:6.2f}")
    env.close()



if __name__ == '__main__':
    train_agent(num_episodes=10000, target_update=10)
    


