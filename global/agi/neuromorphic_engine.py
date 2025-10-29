"""
Neuromorphic Computing Engine for Phase 5
Implements brain-inspired computing for ultra-low latency trading
"""

import numpy as np
import pandas as pd
import asyncio
import logging
import json
import yaml
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict
import threading
import time
import warnings
warnings.filterwarnings('ignore')

class NeuronType(Enum):
    LEAKY_INTEGRATE_FIRE = "lif"
    ADAPTIVE_LIF = "alif"
    IZHIKEVICH = "izhikevich"
    HODGKIN_HUXLEY = "hodgkin_huxley"

class SynapseType(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    PLASTIC = "plastic"
    STDP = "stdp"  # Spike-Timing Dependent Plasticity

class NetworkTopology(Enum):
    FEEDFORWARD = "feedforward"
    RECURRENT = "recurrent"
    RESERVOIR = "reservoir"
    LIQUID_STATE_MACHINE = "lsm"

@dataclass
class SpikeEvent:
    neuron_id: int
    timestamp: float
    amplitude: float
    layer_id: int

@dataclass
class NeuromorphicConfig:
    num_input_neurons: int
    num_hidden_layers: int
    neurons_per_layer: List[int]
    num_output_neurons: int
    neuron_type: NeuronType
    synapse_type: SynapseType
    topology: NetworkTopology
    learning_rate: float = 0.001
    time_step: float = 0.001  # 1ms time steps
    simulation_time: float = 0.1  # 100ms simulation
    threshold: float = 1.0
    decay_rate: float = 0.9
    refractory_period: float = 0.002  # 2ms
    plasticity_enabled: bool = True

class SpikingNeuron:
    """
    Individual spiking neuron with various dynamics
    """
    
    def __init__(self, neuron_id: int, neuron_type: NeuronType, config: Dict[str, Any]):
        self.neuron_id = neuron_id
        self.neuron_type = neuron_type
        self.config = config
        
        # Neuron state variables
        self.membrane_potential = 0.0
        self.threshold = config.get('threshold', 1.0)
        self.decay_rate = config.get('decay_rate', 0.9)
        self.refractory_period = config.get('refractory_period', 0.002)
        self.last_spike_time = -float('inf')
        
        # Adaptive parameters (for ALIF neurons)
        self.adaptation = 0.0
        self.adaptation_decay = config.get('adaptation_decay', 0.95)
        self.adaptation_strength = config.get('adaptation_strength', 0.1)
        
        # Izhikevich parameters
        self.a = config.get('a', 0.02)  # Recovery time constant
        self.b = config.get('b', 0.2)   # Sensitivity of recovery
        self.c = config.get('c', -65.0) # Reset potential
        self.d = config.get('d', 8.0)   # After-spike reset of recovery
        self.recovery = 0.0
        
        # Spike history
        self.spike_times = deque(maxlen=1000)
        self.spike_count = 0
        
    def update(self, input_current: float, dt: float, current_time: float) -> bool:
        """
        Update neuron state and return True if spike occurs
        """
        # Check refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            return False
        
        spike_occurred = False
        
        if self.neuron_type == NeuronType.LEAKY_INTEGRATE_FIRE:
            spike_occurred = self._update_lif(input_current, dt, current_time)
        elif self.neuron_type == NeuronType.ADAPTIVE_LIF:
            spike_occurred = self._update_alif(input_current, dt, current_time)
        elif self.neuron_type == NeuronType.IZHIKEVICH:
            spike_occurred = self._update_izhikevich(input_current, dt, current_time)
        
        if spike_occurred:
            self.last_spike_time = current_time
            self.spike_times.append(current_time)
            self.spike_count += 1
        
        return spike_occurred
    
    def _update_lif(self, input_current: float, dt: float, current_time: float) -> bool:
        """Leaky Integrate-and-Fire neuron dynamics"""
        # Update membrane potential
        self.membrane_potential = (
            self.decay_rate * self.membrane_potential + input_current * dt
        )
        
        # Check for spike
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = 0.0  # Reset
            return True
        
        return False
    
    def _update_alif(self, input_current: float, dt: float, current_time: float) -> bool:
        """Adaptive Leaky Integrate-and-Fire neuron dynamics"""
        # Update adaptation
        self.adaptation *= self.adaptation_decay
        
        # Adaptive threshold
        adaptive_threshold = self.threshold + self.adaptation
        
        # Update membrane potential
        self.membrane_potential = (
            self.decay_rate * self.membrane_potential + input_current * dt
        )
        
        # Check for spike
        if self.membrane_potential >= adaptive_threshold:
            self.membrane_potential = 0.0  # Reset
            self.adaptation += self.adaptation_strength  # Increase adaptation
            return True
        
        return False
    
    def _update_izhikevich(self, input_current: float, dt: float, current_time: float) -> bool:
        """Izhikevich neuron dynamics"""
        v = self.membrane_potential
        u = self.recovery
        
        # Izhikevich equations
        dv = (0.04 * v * v + 5 * v + 140 - u + input_current) * dt
        du = self.a * (self.b * v - u) * dt
        
        self.membrane_potential = v + dv
        self.recovery = u + du
        
        # Check for spike
        if self.membrane_potential >= 30.0:  # Spike threshold for Izhikevich
            self.membrane_potential = self.c  # Reset potential
            self.recovery += self.d  # Reset recovery
            return True
        
        return False
    
    def get_firing_rate(self, time_window: float = 0.1) -> float:
        """Calculate firing rate over recent time window"""
        current_time = time.time()
        recent_spikes = [t for t in self.spike_times if current_time - t <= time_window]
        return len(recent_spikes) / time_window

class PlasticSynapse:
    """
    Plastic synapse with STDP (Spike-Timing Dependent Plasticity)
    """
    
    def __init__(self, pre_neuron_id: int, post_neuron_id: int, 
                 initial_weight: float = 0.5, synapse_type: SynapseType = SynapseType.STDP):
        self.pre_neuron_id = pre_neuron_id
        self.post_neuron_id = post_neuron_id
        self.weight = initial_weight
        self.synapse_type = synapse_type
        
        # STDP parameters
        self.tau_plus = 0.020  # 20ms
        self.tau_minus = 0.020  # 20ms
        self.A_plus = 0.01     # LTP amplitude
        self.A_minus = 0.01    # LTD amplitude
        self.w_max = 1.0       # Maximum weight
        self.w_min = 0.0       # Minimum weight
        
        # Spike timing history
        self.pre_spike_times = deque(maxlen=100)
        self.post_spike_times = deque(maxlen=100)
        
        # Transmission delay
        self.delay = 0.001  # 1ms synaptic delay
        
    def update_weight(self, pre_spike_time: Optional[float], post_spike_time: Optional[float]):
        """Update synaptic weight based on spike timing"""
        if self.synapse_type != SynapseType.STDP:
            return
        
        if pre_spike_time is not None:
            self.pre_spike_times.append(pre_spike_time)
        
        if post_spike_time is not None:
            self.post_spike_times.append(post_spike_time)
        
        # Apply STDP rule
        if pre_spike_time is not None and post_spike_time is not None:
            dt = post_spike_time - pre_spike_time
            
            if dt > 0:  # Pre before post - LTP (potentiation)
                dw = self.A_plus * np.exp(-dt / self.tau_plus)
                self.weight = min(self.w_max, self.weight + dw)
            elif dt < 0:  # Post before pre - LTD (depression)
                dw = -self.A_minus * np.exp(dt / self.tau_minus)
                self.weight = max(self.w_min, self.weight + dw)
    
    def transmit(self, spike_amplitude: float) -> float:
        """Transmit spike through synapse"""
        return self.weight * spike_amplitude

class NeuromorphicNetwork:
    """
    Complete neuromorphic network for trading applications
    """
    
    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Network structure
        self.layers = []
        self.synapses = {}
        self.neurons = {}
        
        # Simulation parameters
        self.current_time = 0.0
        self.dt = config.time_step
        
        # Input/output buffers
        self.input_buffer = deque(maxlen=1000)
        self.output_buffer = deque(maxlen=1000)
        
        # Performance metrics
        self.metrics = {
            'total_spikes': 0,
            'avg_firing_rate': 0.0,
            'network_activity': 0.0,
            'processing_latency': 0.0,
            'energy_consumption': 0.0
        }
        
        # Build network
        self._build_network()
        
    def _build_network(self):
        """Build the neuromorphic network structure"""
        neuron_id = 0
        
        # Input layer
        input_layer = []
        for i in range(self.config.num_input_neurons):
            neuron = SpikingNeuron(
                neuron_id=neuron_id,
                neuron_type=self.config.neuron_type,
                config=asdict(self.config)
            )
            input_layer.append(neuron)
            self.neurons[neuron_id] = neuron
            neuron_id += 1
        
        self.layers.append(input_layer)
        
        # Hidden layers
        for layer_idx in range(self.config.num_hidden_layers):
            hidden_layer = []
            num_neurons = self.config.neurons_per_layer[layer_idx]
            
            for i in range(num_neurons):
                neuron = SpikingNeuron(
                    neuron_id=neuron_id,
                    neuron_type=self.config.neuron_type,
                    config=asdict(self.config)
                )
                hidden_layer.append(neuron)
                self.neurons[neuron_id] = neuron
                neuron_id += 1
            
            self.layers.append(hidden_layer)
        
        # Output layer
        output_layer = []
        for i in range(self.config.num_output_neurons):
            neuron = SpikingNeuron(
                neuron_id=neuron_id,
                neuron_type=self.config.neuron_type,
                config=asdict(self.config)
            )
            output_layer.append(neuron)
            self.neurons[neuron_id] = neuron
            neuron_id += 1
        
        self.layers.append(output_layer)
        
        # Create synapses
        self._create_synapses()
        
    def _create_synapses(self):
        """Create synaptic connections between layers"""
        for layer_idx in range(len(self.layers) - 1):
            pre_layer = self.layers[layer_idx]
            post_layer = self.layers[layer_idx + 1]
            
            for pre_neuron in pre_layer:
                for post_neuron in post_layer:
                    # Random initial weights
                    initial_weight = np.random.uniform(0.1, 0.9)
                    
                    synapse = PlasticSynapse(
                        pre_neuron_id=pre_neuron.neuron_id,
                        post_neuron_id=post_neuron.neuron_id,
                        initial_weight=initial_weight,
                        synapse_type=self.config.synapse_type
                    )
                    
                    synapse_key = (pre_neuron.neuron_id, post_neuron.neuron_id)
                    self.synapses[synapse_key] = synapse
    
    def encode_input(self, data: np.ndarray, encoding_type: str = "rate") -> List[List[float]]:
        """
        Encode input data into spike trains
        """
        if encoding_type == "rate":
            return self._rate_encoding(data)
        elif encoding_type == "temporal":
            return self._temporal_encoding(data)
        elif encoding_type == "population":
            return self._population_encoding(data)
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    def _rate_encoding(self, data: np.ndarray) -> List[List[float]]:
        """Rate-based encoding: higher values = higher firing rates"""
        # Normalize data to [0, 1]
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        
        # Convert to firing rates (spikes per second)
        max_rate = 100.0  # 100 Hz maximum
        firing_rates = normalized_data * max_rate
        
        # Generate spike trains
        spike_trains = []
        simulation_steps = int(self.config.simulation_time / self.dt)
        
        for rate in firing_rates:
            spike_train = []
            for step in range(simulation_steps):
                # Poisson process for spike generation
                if np.random.random() < rate * self.dt:
                    spike_train.append(1.0)
                else:
                    spike_train.append(0.0)
            spike_trains.append(spike_train)
        
        return spike_trains
    
    def _temporal_encoding(self, data: np.ndarray) -> List[List[float]]:
        """Temporal encoding: values encoded as spike timing"""
        spike_trains = []
        simulation_steps = int(self.config.simulation_time / self.dt)
        
        # Normalize data to [0, simulation_time]
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
        spike_times = normalized_data * self.config.simulation_time
        
        for spike_time in spike_times:
            spike_train = [0.0] * simulation_steps
            spike_step = int(spike_time / self.dt)
            if 0 <= spike_step < simulation_steps:
                spike_train[spike_step] = 1.0
            spike_trains.append(spike_train)
        
        return spike_trains
    
    def _population_encoding(self, data: np.ndarray) -> List[List[float]]:
        """Population encoding: multiple neurons encode each value"""
        # For simplicity, use rate encoding with multiple neurons per input
        neurons_per_input = max(1, self.config.num_input_neurons // len(data))
        
        spike_trains = []
        for value in data:
            # Create multiple spike trains for this value with slight variations
            base_rate = (value - np.min(data)) / (np.max(data) - np.min(data) + 1e-8) * 100.0
            
            for _ in range(neurons_per_input):
                # Add noise to create population diversity
                rate = base_rate + np.random.normal(0, 5.0)
                rate = max(0, min(100, rate))  # Clamp to [0, 100]
                
                spike_train = []
                simulation_steps = int(self.config.simulation_time / self.dt)
                for step in range(simulation_steps):
                    if np.random.random() < rate * self.dt:
                        spike_train.append(1.0)
                    else:
                        spike_train.append(0.0)
                spike_trains.append(spike_train)
        
        return spike_trains
    
    async def process_input(self, input_data: np.ndarray, encoding_type: str = "rate") -> Dict[str, Any]:
        """
        Process input through the neuromorphic network
        """
        start_time = time.time()
        
        try:
            # Encode input data
            spike_trains = self.encode_input(input_data, encoding_type)
            
            # Ensure we have the right number of input spike trains
            if len(spike_trains) > self.config.num_input_neurons:
                spike_trains = spike_trains[:self.config.num_input_neurons]
            elif len(spike_trains) < self.config.num_input_neurons:
                # Pad with zero spike trains
                while len(spike_trains) < self.config.num_input_neurons:
                    simulation_steps = int(self.config.simulation_time / self.dt)
                    spike_trains.append([0.0] * simulation_steps)
            
            # Simulate network
            output_spikes = await self._simulate_network(spike_trains)
            
            # Decode output
            output_values = self._decode_output(output_spikes)
            
            processing_time = time.time() - start_time
            
            # Update metrics
            self.metrics['processing_latency'] = processing_time
            self.metrics['total_spikes'] += sum(sum(train) for train in spike_trains)
            
            return {
                'success': True,
                'output_values': output_values,
                'output_spikes': output_spikes,
                'processing_time': processing_time,
                'network_activity': self._calculate_network_activity(),
                'energy_consumption': self._estimate_energy_consumption(),
                'neuromorphic_advantage': True
            }
            
        except Exception as e:
            self.logger.error(f"Neuromorphic processing error: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'neuromorphic_advantage': False
            }
    
    async def _simulate_network(self, input_spike_trains: List[List[float]]) -> List[List[float]]:
        """
        Simulate the neuromorphic network
        """
        simulation_steps = len(input_spike_trains[0])
        output_spikes = [[] for _ in range(self.config.num_output_neurons)]
        
        # Initialize neuron states
        for neuron in self.neurons.values():
            neuron.membrane_potential = 0.0
            neuron.adaptation = 0.0
            neuron.recovery = 0.0
        
        # Simulate each time step
        for step in range(simulation_steps):
            current_time = step * self.dt
            
            # Process each layer
            layer_spikes = {}
            
            for layer_idx, layer in enumerate(self.layers):
                layer_spikes[layer_idx] = []
                
                for neuron_idx, neuron in enumerate(layer):
                    input_current = 0.0
                    
                    if layer_idx == 0:
                        # Input layer - use external input
                        if neuron_idx < len(input_spike_trains):
                            input_current = input_spike_trains[neuron_idx][step]
                    else:
                        # Hidden/output layers - sum synaptic inputs
                        for prev_layer_idx in range(layer_idx):
                            prev_layer = self.layers[prev_layer_idx]
                            for prev_neuron_idx, prev_neuron in enumerate(prev_layer):
                                synapse_key = (prev_neuron.neuron_id, neuron.neuron_id)
                                if synapse_key in self.synapses:
                                    synapse = self.synapses[synapse_key]
                                    # Check if previous neuron spiked
                                    if (prev_layer_idx in layer_spikes and 
                                        prev_neuron_idx < len(layer_spikes[prev_layer_idx]) and
                                        layer_spikes[prev_layer_idx][prev_neuron_idx]):
                                        input_current += synapse.transmit(1.0)
                    
                    # Update neuron
                    spike_occurred = neuron.update(input_current, self.dt, current_time)
                    layer_spikes[layer_idx].append(spike_occurred)
                    
                    # Record output spikes
                    if layer_idx == len(self.layers) - 1:  # Output layer
                        output_spikes[neuron_idx].append(1.0 if spike_occurred else 0.0)
                    
                    # Update synaptic plasticity
                    if spike_occurred and self.config.plasticity_enabled:
                        self._update_plasticity(neuron.neuron_id, current_time)
        
        return output_spikes
    
    def _update_plasticity(self, neuron_id: int, spike_time: float):
        """Update synaptic plasticity based on spike timing"""
        # Update all synapses connected to this neuron
        for synapse_key, synapse in self.synapses.items():
            pre_id, post_id = synapse_key
            
            if pre_id == neuron_id:
                # This neuron is presynaptic
                synapse.update_weight(spike_time, None)
            elif post_id == neuron_id:
                # This neuron is postsynaptic
                synapse.update_weight(None, spike_time)
    
    def _decode_output(self, output_spikes: List[List[float]]) -> List[float]:
        """
        Decode output spike trains to numerical values
        """
        output_values = []
        
        for spike_train in output_spikes:
            # Rate-based decoding: count spikes
            spike_count = sum(spike_train)
            firing_rate = spike_count / self.config.simulation_time
            
            # Normalize to [0, 1]
            normalized_value = min(1.0, firing_rate / 100.0)  # Assuming max 100 Hz
            output_values.append(normalized_value)
        
        return output_values
    
    def _calculate_network_activity(self) -> float:
        """Calculate overall network activity level"""
        total_activity = 0.0
        total_neurons = 0
        
        for neuron in self.neurons.values():
            firing_rate = neuron.get_firing_rate()
            total_activity += firing_rate
            total_neurons += 1
        
        return total_activity / total_neurons if total_neurons > 0 else 0.0
    
    def _estimate_energy_consumption(self) -> float:
        """Estimate energy consumption (simplified model)"""
        # Energy per spike (in arbitrary units)
        energy_per_spike = 1e-12  # Picojoules per spike
        
        total_spikes = sum(neuron.spike_count for neuron in self.neurons.values())
        total_energy = total_spikes * energy_per_spike
        
        return total_energy
    
    def get_network_state(self) -> Dict[str, Any]:
        """Get current network state and statistics"""
        neuron_states = {}
        synapse_weights = {}
        
        for neuron_id, neuron in self.neurons.items():
            neuron_states[neuron_id] = {
                'membrane_potential': neuron.membrane_potential,
                'spike_count': neuron.spike_count,
                'firing_rate': neuron.get_firing_rate(),
                'last_spike_time': neuron.last_spike_time
            }
        
        for synapse_key, synapse in self.synapses.items():
            synapse_weights[f"{synapse_key[0]}->{synapse_key[1]}"] = synapse.weight
        
        return {
            'neuron_states': neuron_states,
            'synapse_weights': synapse_weights,
            'network_metrics': self.metrics,
            'current_time': self.current_time,
            'total_neurons': len(self.neurons),
            'total_synapses': len(self.synapses)
        }

class NeuromorphicTradingEngine:
    """
    High-level neuromorphic trading engine
    """
    
    def __init__(self, config_path: str = "config/neuromorphic_config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Create neuromorphic networks for different trading tasks
        self.networks = {}
        self._initialize_networks()
        
        # Performance tracking
        self.performance_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'avg_latency': 0.0,
            'energy_efficiency': 0.0
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default neuromorphic configuration"""
        return {
            'price_prediction': {
                'num_input_neurons': 20,
                'num_hidden_layers': 2,
                'neurons_per_layer': [50, 30],
                'num_output_neurons': 3,  # up, down, sideways
                'neuron_type': 'alif',
                'synapse_type': 'stdp',
                'topology': 'feedforward'
            },
            'risk_assessment': {
                'num_input_neurons': 15,
                'num_hidden_layers': 2,
                'neurons_per_layer': [40, 25],
                'num_output_neurons': 5,  # risk levels
                'neuron_type': 'lif',
                'synapse_type': 'plastic',
                'topology': 'recurrent'
            },
            'pattern_recognition': {
                'num_input_neurons': 30,
                'num_hidden_layers': 3,
                'neurons_per_layer': [60, 40, 20],
                'num_output_neurons': 10,  # pattern types
                'neuron_type': 'izhikevich',
                'synapse_type': 'stdp',
                'topology': 'reservoir'
            }
        }
    
    def _initialize_networks(self):
        """Initialize neuromorphic networks for different tasks"""
        for task_name, task_config in self.config.items():
            try:
                # Convert string enums to actual enums
                neuron_type = NeuronType(task_config['neuron_type'])
                synapse_type = SynapseType(task_config['synapse_type'])
                topology = NetworkTopology(task_config['topology'])
                
                # Create neuromorphic config
                config = NeuromorphicConfig(
                    num_input_neurons=task_config['num_input_neurons'],
                    num_hidden_layers=task_config['num_hidden_layers'],
                    neurons_per_layer=task_config['neurons_per_layer'],
                    num_output_neurons=task_config['num_output_neurons'],
                    neuron_type=neuron_type,
                    synapse_type=synapse_type,
                    topology=topology
                )
                
                # Create network
                network = NeuromorphicNetwork(config)
                self.networks[task_name] = network
                
                self.logger.info(f"Initialized neuromorphic network for {task_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize network for {task_name}: {e}")
    
    async def predict_price_movement(self, market_data: np.ndarray) -> Dict[str, Any]:
        """Predict price movement using neuromorphic network"""
        if 'price_prediction' not in self.networks:
            return {'success': False, 'error': 'Price prediction network not available'}
        
        network = self.networks['price_prediction']
        result = await network.process_input(market_data, encoding_type="rate")
        
        if result['success']:
            # Interpret output as price movement probabilities
            outputs = result['output_values']
            if len(outputs) >= 3:
                movement_probs = {
                    'up': outputs[0],
                    'down': outputs[1],
                    'sideways': outputs[2]
                }
                
                # Determine prediction
                prediction = max(movement_probs, key=movement_probs.get)
                confidence = movement_probs[prediction]
                
                result.update({
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': movement_probs
                })
        
        return result
    
    async def assess_risk(self, portfolio_data: np.ndarray) -> Dict[str, Any]:
        """Assess portfolio risk using neuromorphic network"""
        if 'risk_assessment' not in self.networks:
            return {'success': False, 'error': 'Risk assessment network not available'}
        
        network = self.networks['risk_assessment']
        result = await network.process_input(portfolio_data, encoding_type="temporal")
        
        if result['success']:
            # Interpret output as risk levels
            outputs = result['output_values']
            risk_score = np.mean(outputs)  # Average risk across all outputs
            
            # Categorize risk
            if risk_score < 0.2:
                risk_level = 'very_low'
            elif risk_score < 0.4:
                risk_level = 'low'
            elif risk_score < 0.6:
                risk_level = 'medium'
            elif risk_score < 0.8:
                risk_level = 'high'
            else:
                risk_level = 'very_high'
            
            result.update({
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_components': outputs
            })
        
        return result
    
    async def recognize_patterns(self, time_series_data: np.ndarray) -> Dict[str, Any]:
        """Recognize patterns in time series data"""
        if 'pattern_recognition' not in self.networks:
            return {'success': False, 'error': 'Pattern recognition network not available'}
        
        network = self.networks['pattern_recognition']
        result = await network.process_input(time_series_data, encoding_type="population")
        
        if result['success']:
            # Interpret output as pattern types
            outputs = result['output_values']
            
            # Define pattern types
            pattern_types = [
                'trend_up', 'trend_down', 'sideways', 'breakout_up', 'breakout_down',
                'reversal_up', 'reversal_down', 'consolidation', 'volatility_spike', 'calm'
            ]
            
            # Find dominant pattern
            if len(outputs) >= len(pattern_types):
                pattern_scores = dict(zip(pattern_types, outputs[:len(pattern_types)]))
                dominant_pattern = max(pattern_scores, key=pattern_scores.get)
                confidence = pattern_scores[dominant_pattern]
                
                result.update({
                    'dominant_pattern': dominant_pattern,
                    'confidence': confidence,
                    'pattern_scores': pattern_scores
                })
        
        return result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all networks"""
        metrics = {
            'overall': self.performance_metrics,
            'networks': {}
        }
        
        for task_name, network in self.networks.items():
            network_state = network.get_network_state()
            metrics['networks'][task_name] = {
                'network_activity': network_state['network_metrics']['network_activity'],
                'total_spikes': network_state['network_metrics']['total_spikes'],
                'processing_latency': network_state['network_metrics']['processing_latency'],
                'energy_consumption': network_state['network_metrics']['energy_consumption'],
                'total_neurons': network_state['total_neurons'],
                'total_synapses': network_state['total_synapses']
            }
        
        return metrics

# Example usage and testing
async def main():
    """
    Example usage of Neuromorphic Trading Engine
    """
    print("🧠 Neuromorphic Computing Engine - Phase 5 Testing")
    print("=" * 60)
    
    # Initialize neuromorphic trading engine
    engine = NeuromorphicTradingEngine()
    
    print(f"\n🔬 Testing Neuromorphic Networks...")
    print(f"Available networks: {list(engine.networks.keys())}")
    
    # Generate sample market data
    np.random.seed(42)
    
    # Test price prediction
    print(f"\n📈 Testing Price Prediction...")
    market_data = np.random.randn(20)  # 20 market indicators
    price_result = await engine.predict_price_movement(market_data)
    
    if price_result['success']:
        print(f"  Prediction: {price_result.get('prediction', 'N/A')}")
        print(f"  Confidence: {price_result.get('confidence', 0):.3f}")
        print(f"  Processing Time: {price_result['processing_time']:.4f}s")
        print(f"  Network Activity: {price_result['network_activity']:.3f}")
    
    # Test risk assessment
    print(f"\n⚠️  Testing Risk Assessment...")
    portfolio_data = np.random.randn(15)  # 15 portfolio metrics
    risk_result = await engine.assess_risk(portfolio_data)
    
    if risk_result['success']:
        print(f"  Risk Level: {risk_result.get('risk_level', 'N/A')}")
        print(f"  Risk Score: {risk_result.get('risk_score', 0):.3f}")
        print(f"  Processing Time: {risk_result['processing_time']:.4f}s")
        print(f"  Energy Consumption: {risk_result['energy_consumption']:.2e}")
    
    # Test pattern recognition
    print(f"\n🔍 Testing Pattern Recognition...")
    time_series = np.random.randn(30)  # 30 time series points
    pattern_result = await engine.recognize_patterns(time_series)
    
    if pattern_result['success']:
        print(f"  Dominant Pattern: {pattern_result.get('dominant_pattern', 'N/A')}")
        print(f"  Confidence: {pattern_result.get('confidence', 0):.3f}")
        print(f"  Processing Time: {pattern_result['processing_time']:.4f}s")
    
    # Show performance metrics
    print(f"\n📊 Performance Metrics:")
    metrics = engine.get_performance_metrics()
    
    for network_name, network_metrics in metrics['networks'].items():
        print(f"  {network_name}:")
        print(f"    Neurons: {network_metrics['total_neurons']}")
        print(f"    Synapses: {network_metrics['total_synapses']}")
        print(f"    Latency: {network_metrics['processing_latency']:.4f}s")
        print(f"    Energy: {network_metrics['energy_consumption']:.2e}")
    
    print(f"\n✅ Neuromorphic engine testing completed!")
    print(f"🚀 Ultra-low latency neuromorphic processing ready for Phase 5!")

if __name__ == "__main__":
    asyncio.run(main())