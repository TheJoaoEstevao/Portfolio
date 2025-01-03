import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import numpy as np

class SearchSpace:
    """Neural Architecture Search Space Definition"""
    def __init__(self, 
                 input_size: Tuple[int, ...],
                 n_classes: int,
                 max_layers: int = 10,
                 operations: List[str] = ['conv3x3', 'conv1x1', 'maxpool', 'avgpool']):
        self.input_size = input_size
        self.n_classes = n_classes
        self.max_layers = max_layers
        self.operations = operations
        
    def sample_architecture(self) -> Dict:
        """Sample a random architecture from the search space"""
        n_layers = np.random.randint(1, self.max_layers + 1)
        architecture = {
            'n_layers': n_layers,
            'operations': [],
            'connections': []
        }
        
        for i in range(n_layers):
            op = np.random.choice(self.operations)
            architecture['operations'].append(op)
            
            # Sample layer connections (skip connections)
            if i > 0:
                connections = np.random.binomial(1, 0.5, i)
                architecture['connections'].append(connections.tolist())
            else:
                architecture['connections'].append([])
                
        return architecture

class ArchitectureEvaluator:
    """Evaluates sampled architectures"""
    def __init__(self, 
                 search_space: SearchSpace,
                 device: str = 'cuda'):
        self.search_space = search_space
        self.device = device
        
    def build_model(self, architecture: Dict) -> nn.Module:
        """Convert architecture dict to PyTorch model"""
        class DynamicModel(nn.Module):
            def __init__(self, arch: Dict, input_size: Tuple[int, ...], n_classes: int):
                super().__init__()
                self.layers = nn.ModuleList()
                self.connections = arch['connections']
                
                current_channels = input_size[0]
                for i, op in enumerate(arch['operations']):
                    if op == 'conv3x3':
                        layer = nn.Conv2d(current_channels, current_channels*2, 3, padding=1)
                        current_channels *= 2
                    elif op == 'conv1x1':
                        layer = nn.Conv2d(current_channels, current_channels, 1)
                    elif op == 'maxpool':
                        layer = nn.MaxPool2d(2)
                    elif op == 'avgpool':
                        layer = nn.AvgPool2d(2)
                    
                    self.layers.append(layer)
                
                self.classifier = nn.Linear(current_channels, n_classes)
                
            def forward(self, x):
                features = [x]
                
                for i, layer in enumerate(self.layers):
                    # Process current layer
                    out = layer(features[-1])
                    
                    # Add skip connections
                    if i > 0:
                        for j, connected in enumerate(self.connections[i]):
                            if connected:
                                out += features[j]
                    
                    features.append(out)
                
                # Global average pooling
                out = torch.mean(features[-1], dim=(2, 3))
                return self.classifier(out)
        
        return DynamicModel(architecture, 
                           self.search_space.input_size,
                           self.search_space.n_classes)
    
    def evaluate(self, architecture: Dict, 
                dataloader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                n_epochs: int = 1) -> float:
        """Train and evaluate architecture"""
        model = self.build_model(architecture).to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        
        best_acc = 0
        for epoch in range(n_epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    pred = output.argmax(dim=1, keepdim=True)
                    acc = pred.eq(target.view_as(pred)).float().mean()
                    best_acc = max(best_acc, acc.item())
                    
        return best_acc

class EvolutionarySearch:
    """Evolutionary Neural Architecture Search"""
    def __init__(self,
                 search_space: SearchSpace,
                 evaluator: ArchitectureEvaluator,
                 population_size: int = 100,
                 n_generations: int = 50):
        self.search_space = search_space
        self.evaluator = evaluator
        self.population_size = population_size
        self.n_generations = n_generations
        
    def mutate(self, architecture: Dict) -> Dict:
        """Mutate an architecture"""
        new_arch = architecture.copy()
        
        # Randomly modify operations
        for i in range(len(new_arch['operations'])):
            if np.random.random() < 0.1:
                new_arch['operations'][i] = np.random.choice(self.search_space.operations)
        
        # Randomly modify connections
        for i in range(1, len(new_arch['connections'])):
            for j in range(len(new_arch['connections'][i])):
                if np.random.random() < 0.1:
                    new_arch['connections'][i][j] = 1 - new_arch['connections'][i][j]
                    
        return new_arch
    
    def search(self, 
               dataloader: torch.utils.data.DataLoader,
               criterion: nn.Module) -> Tuple[Dict, float]:
        """Run evolutionary search"""
        # Initialize population
        population = []
        for _ in range(self.population_size):
            arch = self.search_space.sample_architecture()
            acc = self.evaluator.evaluate(arch, dataloader, criterion)
            population.append((arch, acc))
            
        best_arch, best_acc = max(population, key=lambda x: x[1])
        
        # Evolution
        for gen in range(self.n_generations):
            # Sort by accuracy
            population.sort(key=lambda x: x[1], reverse=True)
            
            # Keep top 20%
            population = population[:self.population_size//5]
            
            # Mutate to regenerate population
            while len(population) < self.population_size:
                parent_arch, _ = population[np.random.randint(len(population))]
                child_arch = self.mutate(parent_arch)
                acc = self.evaluator.evaluate(child_arch, dataloader, criterion)
                population.append((child_arch, acc))
                
                if acc > best_acc:
                    best_arch, best_acc = child_arch, acc
                    
            print(f"Generation {gen}: Best accuracy = {best_acc:.4f}")
            
        return best_arch, best_acc 