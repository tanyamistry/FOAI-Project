import numpy as np

class ConveyorEnv:
    def __init__(self):
        self.object_types = {
            'colors': ['red', 'green', 'blue'],
            'shapes': ['circle', 'square', 'triangle']
        }
        self.reset()
        
    def reset(self):
        self.current_object = self._new_random_object()
        self.position = 0.0
        self.sorted_correctly = 0
        self.timestep = 0
        return self._get_state()
    
    def _new_random_object(self):
        return {
            'color': np.random.choice(self.object_types['colors']),
            'shape': np.random.choice(self.object_types['shapes'])
        }
    
    def _get_state(self):
        color_idx = self.object_types['colors'].index(self.current_object['color'])
        shape_idx = self.object_types['shapes'].index(self.current_object['shape'])
        return np.array([
            color_idx / len(self.object_types['colors']),
            shape_idx / len(self.object_types['shapes']),
            self.position
        ])
    
    def step(self, action):
        self.timestep += 1
        self.position = min(1.0, self.position + 0.05)
        
        reward = 0
        done = False
        info = {'sorted': False}
        
        if action == 1:
            done = True
            correct_color = self.current_object['color']
            selected_bin = 'red' if self.position < 0.33 else 'green' if self.position < 0.66 else 'blue'
            
            if selected_bin == correct_color:
                reward = 1.0
                self.sorted_correctly += 1
            else:
                reward = -0.5
            info['sorted'] = True
        
        if self.position >= 1.0 and not done:
            reward = -0.2
            done = True
            
        if done:
            self.current_object = self._new_random_object()
            self.position = 0.0
            
        return self._get_state(), reward, done, info