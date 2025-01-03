import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math
from dataclasses import dataclass

@dataclass
class NeuromorphicConfig:
    """Configuration for neuromorphic computing"""
    membrane_threshold: float = 1.0
    refractory_period: int = 5
    leak_rate: float = 0.1
    reset_potential: float = 0.0
    time_steps: int = 100
    input_channels: int = 64
    hidden_channels: int = 128
    spike_grad: str = "surrogate"  # [surrogate, straight_through]

class SpikingNeuron(nn.Module):
    """Biologically inspired spiking neuron"""
    def __init__(self, config: NeuromorphicConfig):
        super().__init__()
        self.config = config
        self.membrane_potential = None
        self.refractory_count = None
        self.spike_history = []
        
    def reset_state(self, batch_size: int, device: torch.device):
        """Reset neuron state"""
        self.membrane_potential = torch.zeros(batch_size, device=device)
        self.refractory_count = torch.zeros(batch_size, device=device)
        self.spike_history = []
        
    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        if self.membrane_potential is None:
            self.reset_state(input_current.size(0), input_current.device)
            
        # Update refractory period
        self.refractory_count = torch.maximum(
            self.refractory_count - 1,
            torch.zeros_like(self.refractory_count)
        )
        
        # Membrane potential dynamics
        self.membrane_potential = (
            (1 - self.config.leak_rate) * self.membrane_potential +
            input_current * (self.refractory_count == 0).float()
        )
        
        # Spike generation
        spike = self.generate_spike(self.membrane_potential)
        self.spike_history.append(spike)
        
        # Reset membrane potential after spike
        self.membrane_potential = torch.where(
            spike == 1,
            torch.tensor(self.config.reset_potential, device=spike.device),
            self.membrane_potential
        )
        
        # Update refractory period
        self.refractory_count = torch.where(
            spike == 1,
            torch.tensor(self.config.refractory_period, device=spike.device),
            self.refractory_count
        )
        
        return spike
        
    def generate_spike(self, membrane_potential: torch.Tensor) -> torch.Tensor:
        """Generate spike using surrogate gradient"""
        if self.config.spike_grad == "surrogate":
            # Surrogate gradient function
            alpha = 10.0
            return SurrogateSpike.apply(membrane_potential - self.config.membrane_threshold, alpha)
        else:
            # Straight through estimator
            return StraightThroughSpike.apply(membrane_potential - self.config.membrane_threshold)

class SurrogateSpike(torch.autograd.Function):
    """Surrogate gradient for spike function"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).float()
        
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output * alpha * torch.exp(-abs(x) * alpha)
        return grad_input, None

class StraightThroughSpike(torch.autograd.Function):
    """Straight through estimator for spike function"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class NeuromorphicLayer(nn.Module):
    """Layer of spiking neurons with synaptic connections"""
    def __init__(self, in_features: int, out_features: int, config: NeuromorphicConfig):
        super().__init__()
        self.config = config
        
        # Synaptic weights
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / math.sqrt(in_features)
        )
        
        # Spiking neurons
        self.neurons = nn.ModuleList([
            SpikingNeuron(config) for _ in range(out_features)
        ])
        
        # Synaptic plasticity parameters
        self.plasticity = nn.Parameter(torch.ones(out_features, in_features) * 0.1)
        
    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        # Compute synaptic current
        synaptic_current = F.linear(input_spikes, self.weight)
        
        # Generate output spikes
        output_spikes = torch.zeros_like(synaptic_current)
        for i, neuron in enumerate(self.neurons):
            output_spikes[:, i] = neuron(synaptic_current[:, i])
            
        # Apply synaptic plasticity
        if self.training:
            self.update_synapses(input_spikes, output_spikes)
            
        return output_spikes
        
    def update_synapses(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """Update synaptic weights using STDP-inspired rule"""
        with torch.no_grad():
            # Compute correlation between pre and post synaptic spikes
            correlation = torch.einsum('bi,bj->ij', post_spikes, pre_spikes)
            
            # Update weights based on correlation and plasticity
            delta_w = self.plasticity * correlation
            self.weight.data += delta_w

class NeuromorphicNetwork(nn.Module):
    """Complete neuromorphic neural network"""
    def __init__(self, config: NeuromorphicConfig):
        super().__init__()
        self.config = config
        
        # Network layers
        self.input_layer = NeuromorphicLayer(
            config.input_channels,
            config.hidden_channels,
            config
        )
        
        self.hidden_layers = nn.ModuleList([
            NeuromorphicLayer(config.hidden_channels, config.hidden_channels, config)
            for _ in range(3)
        ])
        
        self.output_layer = NeuromorphicLayer(
            config.hidden_channels,
            config.input_channels,
            config
        )
        
        # Temporal integration
        self.temporal_filter = nn.Parameter(
            torch.exp(-torch.arange(20).float() / 5)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert input to spike train
        input_spikes = self.rate_coding(x)
        
        # Process through layers
        spikes = []
        for t in range(self.config.time_steps):
            h = input_spikes[:, t]
            
            # Forward through layers
            h = self.input_layer(h)
            for layer in self.hidden_layers:
                h = layer(h)
            h = self.output_layer(h)
            
            spikes.append(h)
            
        # Temporal integration of spikes
        spike_train = torch.stack(spikes, dim=1)
        return self.temporal_integration(spike_train)
        
    def rate_coding(self, x: torch.Tensor) -> torch.Tensor:
        """Convert continuous input to spike train"""
        rates = torch.sigmoid(x)
        spikes = torch.rand_like(
            rates.unsqueeze(1).expand(-1, self.config.time_steps, -1)
        ) < rates.unsqueeze(1)
        return spikes.float()
        
    def temporal_integration(self, spike_train: torch.Tensor) -> torch.Tensor:
        """Integrate spikes over time"""
        # Apply temporal filter
        filtered = F.conv1d(
            spike_train.transpose(1, 2),
            self.temporal_filter.view(1, 1, -1),
            padding=self.temporal_filter.size(0)-1
        )
        
        # Return final output
        return filtered.transpose(1, 2)[:, -1]

def create_neuromorphic_model(config: NeuromorphicConfig) -> nn.Module:
    """Create a neuromorphic model with advanced features"""
    return nn.Sequential(
        NeuromorphicNetwork(config),
        nn.LayerNorm(config.input_channels),
        nn.Linear(config.input_channels, config.input_channels)
    ) 