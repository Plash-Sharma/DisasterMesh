"""
3D Steady-State Random Waypoint Mobility. Ref: EDMBOPR Section 3.2
"""
import numpy as np
import random
from collections import deque

class MobilityModel:
    def __init__(self, node_type: str, area_size=(1000, 1000, 1000)):
        self.area_size = area_size
        self.node_type = node_type # 'survivor', 'rescue', 'trapped'
        
        if node_type == 'survivor':
            self.speed_range = (0.5, 2.0)
        elif node_type == 'rescue':
            self.speed_range = (2.0, 5.0)
        else: # 'trapped'
            self.speed_range = (0.0, 0.1)
            
        self.position = np.array([random.uniform(0, area_size[0]),
                                  random.uniform(0, area_size[1]),
                                  random.uniform(0, area_size[2])])
        self.waypoint = self._get_random_waypoint()
        self.speed = random.uniform(*self.speed_range)
        self.pause_time = 0.0
        self.position_history = deque(maxlen=10) # For avg speed computation
        self.last_time = 0.0
        
    def _get_random_waypoint(self):
        return np.array([random.uniform(0, self.area_size[0]),
                         random.uniform(0, self.area_size[1]),
                         random.uniform(0, self.area_size[2])])
                         
    def step(self, current_time: float, dt: float = 0.1):
        if self.pause_time > 0:
            self.pause_time -= dt
        else:
            direction = self.waypoint - self.position
            distance = np.linalg.norm(direction)
            if distance < 1e-3:
                self.pause_time = random.uniform(0, 5)
                self.waypoint = self._get_random_waypoint()
                self.speed = random.uniform(*self.speed_range)
            else:
                step_size = self.speed * dt
                if step_size > distance:
                    self.position = self.waypoint.copy()
                else:
                    self.position += (direction / distance) * step_size
                    
        if current_time - self.last_time >= 1.0: # Record history every 1s
            self.position_history.append({'pos': self.position.copy(), 'time': current_time})
            self.last_time = current_time
            
    def get_avg_speed(self) -> float:
        """EDMBOPR Eq. 5: Instantaneous average mobility"""
        if len(self.position_history) < 2:
            return 0.0
        
        speeds = []
        recent = list(self.position_history)
        for i in range(1, len(recent)):
            d = np.linalg.norm(recent[i]['pos'] - recent[i-1]['pos'])
            dt = recent[i]['time'] - recent[i-1]['time']
            if dt > 0:
                speeds.append(d / dt)
        return np.mean(speeds) if speeds else 0.0
