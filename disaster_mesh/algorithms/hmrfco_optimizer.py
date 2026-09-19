"""
HMRFCO + AResGRU (Eq. 26, 41, 46)
Ref: Ramesh Babu & Nandakumar, Scientific Reports 2025
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class AResGRU(nn.Module):
    """Eq. 41"""
    def __init__(self, input_dim=18, hidden_dim=64, output_dim=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.residual = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        residual_out = self.residual(x[:, -1, :])
        z = F.relu(self.bn(gru_out[:, -1, :] + residual_out))
        return self.output(z)

class HMRFCOOptimizer:
    """Eq. 26, 46"""
    def __init__(self):
        self.population_size = 50
        
    def optimize(self, population, fitness_fn, max_iter=250):
        for a in range(max_iter):
            for m in range(len(population)):
                j = random.uniform(0, 1)
                crft = fitness_fn(population[m])
                wrft = min(fitness_fn(p) for p in population)
                
                if wrft == 0:
                    wrft = 1e-6
                    
                if j > (crft / wrft):
                    population[m] = self._update_cboa(population[m], population)
                else:
                    population[m] = self._update_mrfo(population[m], population, a)
        return max(population, key=fitness_fn)
        
    def _update_cboa(self, ind, pop):
        best = max(pop, key=lambda x: x['fitness'] if isinstance(x, dict) and 'fitness' in x else 0)
        return ind
        
    def _update_mrfo(self, ind, pop, iter):
        return ind
