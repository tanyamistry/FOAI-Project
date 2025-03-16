import numpy as np
from rl_env import ConveyorEnv
from rl_agent import DQNAgent

def train():
    env = ConveyorEnv()
    state_size = 3
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    
    episodes = 500
    print_interval = 50
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.replay()
            
        if (e+1) % print_interval == 0:
            print(f"Episode: {e+1}/{episodes}, Avg Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
            agent.save(f"dqn_model_{e+1}.pth")

if __name__ == "__main__":
    train()