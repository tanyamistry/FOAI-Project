import pygame
import numpy as np
from rl_env import ConveyorEnv
from rl_agent import DQNAgent

class ConveyorUI:
    def __init__(self, model_path):
        pygame.init()
        self.width = 800
        self.height = 400
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("AI Sorting System")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        
        # Initialize environment and agent
        self.env = ConveyorEnv()
        self.agent = DQNAgent(state_size=3, action_size=2)
        self.agent.load(model_path)
        self.agent.epsilon = 0.01  # Minimal exploration
        
        # Colors
        self.colors = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255)
        }
        
    def draw_object(self, obj, x):
        color = self.colors[obj['color']]
        y = self.height//2
        
        if obj['shape'] == 'circle':
            pygame.draw.circle(self.screen, color, (x, y), 20)
        elif obj['shape'] == 'square':
            pygame.draw.rect(self.screen, color, (x-20, y-20, 40, 40))
        else:  # triangle
            points = [(x, y-20), (x-20, y+20), (x+20, y+20)]
            pygame.draw.polygon(self.screen, color, points)
    
    def run(self):
        running = True
        while running:
            self.screen.fill((40, 40, 40))
            
            # Get agent action
            state = self.env._get_state()
            action = self.agent.act(state)
            
            # Environment step
            _, _, done, _ = self.env.step(action)
            if done:
                self.env.reset()
            
            # Draw conveyor
            pygame.draw.rect(self.screen, (80, 80, 80), (0, self.height//2-30, self.width, 60))
            
            # Draw moving object
            obj_x = int(self.env.position * self.width)
            self.draw_object(self.env.current_object, obj_x)
            
            # Draw bins
            bin_positions = [(100, 100), (400, 100), (700, 100)]
            for i, (x, y) in enumerate(bin_positions):
                color = list(self.colors.keys())[i]
                pygame.draw.rect(self.screen, self.colors[color], (x-40, y-40, 80, 80))
            
            # Display info
            text = self.font.render(f"Sorted Correctly: {self.env.sorted_correctly}", True, (255,255,255))
            self.screen.blit(text, (10, 10))
            
            pygame.display.flip()
            self.clock.tick(30)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        pygame.quit()

if __name__ == "__main__":
    # Load the trained model (use latest from training)
    ui = ConveyorUI(model_path="dqn_model_500.pth")
    ui.run()