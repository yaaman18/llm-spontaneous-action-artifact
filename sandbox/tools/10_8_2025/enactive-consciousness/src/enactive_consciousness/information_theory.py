"""Information-theoretic measures for enactive consciousness.

This module implements sophisticated information-theoretic measures
for quantifying circular causality, mutual information, and transfer entropy
in enactive systems following Varela-Maturana principles.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from .types import Array, PRNGKey


class InformationTheoryError(Exception):
    """Exception for information theory computation errors."""
    pass


def mutual_information_kraskov(
    x: Array,
    y: Array,
    k: int = 3,
    base: float = jnp.e,
) -> float:
    """Compute mutual information using Kraskov estimator.
    
    Args:
        x: First variable array
        y: Second variable array  
        k: Number of nearest neighbors
        base: Logarithm base (e for nats, 2 for bits)
        
    Returns:
        Mutual information estimate in nats/bits
    """
    try:
        x_np = np.asarray(x).reshape(-1, 1) if x.ndim == 1 else np.asarray(x)
        y_np = np.asarray(y).reshape(-1, 1) if y.ndim == 1 else np.asarray(y)
        
        if len(x_np) != len(y_np):
            raise InformationTheoryError(f"Length mismatch: x={len(x_np)}, y={len(y_np)}")
        
        # Use sklearn's mutual information estimation
        mi_estimate = mutual_info_regression(
            x_np, y_np.flatten(), n_neighbors=k, random_state=42
        )
        
        # Convert to specified base
        if base != jnp.e:
            mi_estimate = mi_estimate / jnp.log(base)
            
        return float(jnp.mean(mi_estimate))
        
    except Exception as e:
        raise InformationTheoryError(f"Mutual information computation failed: {e}")


def transfer_entropy(
    source: Array,
    target: Array,
    history_length: int = 1,
    k: int = 3,
) -> float:
    """Compute transfer entropy from source to target.
    
    Transfer entropy quantifies the amount of information transferred
    from one time series to another, capturing causal relationships.
    
    Args:
        source: Source time series
        target: Target time series
        history_length: Length of history to consider
        k: Number of nearest neighbors for estimation
        
    Returns:
        Transfer entropy value
    """
    try:
        if len(source) != len(target):
            raise InformationTheoryError("Source and target must have same length")
            
        if len(source) <= history_length + 1:
            raise InformationTheoryError("Time series too short for given history length")
        
        # Prepare data with history
        n_samples = len(target) - history_length
        
        # Target future
        target_future = target[history_length:]
        
        # Target history  
        target_history = jnp.array([
            target[i:i+history_length] 
            for i in range(n_samples)
        ])
        
        # Combined target + source history
        combined_history = jnp.array([
            jnp.concatenate([
                target[i:i+history_length],
                source[i:i+history_length]
            ])
            for i in range(n_samples)
        ])
        
        # TE = MI(target_future; combined_history) - MI(target_future; target_history)
        mi_combined = mutual_information_kraskov(
            combined_history, target_future, k=k
        )
        mi_target = mutual_information_kraskov(
            target_history, target_future, k=k  
        )
        
        transfer_entropy_value = mi_combined - mi_target
        return max(0.0, float(transfer_entropy_value))  # TE >= 0
        
    except Exception as e:
        raise InformationTheoryError(f"Transfer entropy computation failed: {e}")


def circular_causality_index(
    agent_states: Array,
    environment_states: Array,
    history_length: int = 2,
) -> Dict[str, float]:
    """Compute circular causality index between agent and environment.
    
    This implements a sophisticated measure of Varela-Maturana circular causality
    using bidirectional transfer entropy and coupling strength metrics.
    
    Args:
        agent_states: Agent state time series
        environment_states: Environment state time series
        history_length: History length for transfer entropy
        
    Returns:
        Dictionary containing causality measures
    """
    try:
        # Flatten if multidimensional
        if agent_states.ndim > 1:
            agent_flat = jnp.linalg.norm(agent_states, axis=-1)
        else:
            agent_flat = agent_states
            
        if environment_states.ndim > 1:
            env_flat = jnp.linalg.norm(environment_states, axis=-1)
        else:
            env_flat = environment_states
        
        # Compute bidirectional transfer entropy
        te_env_to_agent = transfer_entropy(
            env_flat, agent_flat, history_length=history_length
        )
        te_agent_to_env = transfer_entropy(
            agent_flat, env_flat, history_length=history_length
        )
        
        # Circular causality strength
        circular_strength = (te_env_to_agent + te_agent_to_env) / 2.0
        
        # Causality asymmetry
        causality_asymmetry = abs(te_env_to_agent - te_agent_to_env)
        
        # Coupling coherence (how balanced the bidirectional flow is)
        if circular_strength > 0:
            coupling_coherence = 1.0 - (causality_asymmetry / (2 * circular_strength))
        else:
            coupling_coherence = 0.0
            
        # Mutual information for instantaneous coupling
        instantaneous_coupling = mutual_information_kraskov(agent_flat, env_flat)
        
        # Overall circular causality index
        circular_causality = (
            0.4 * circular_strength +
            0.3 * coupling_coherence + 
            0.3 * instantaneous_coupling
        )
        
        return {
            'circular_causality': float(circular_causality),
            'transfer_entropy_env_to_agent': float(te_env_to_agent),
            'transfer_entropy_agent_to_env': float(te_agent_to_env),
            'circular_strength': float(circular_strength),
            'causality_asymmetry': float(causality_asymmetry),
            'coupling_coherence': float(coupling_coherence),
            'instantaneous_coupling': float(instantaneous_coupling),
        }
        
    except Exception as e:
        raise InformationTheoryError(f"Circular causality computation failed: {e}")


def entropy_rate(
    time_series: Array,
    embedding_dim: int = 2,
    tolerance: float = 0.1,
) -> float:
    """Compute entropy rate of time series using approximate entropy.
    
    Args:
        time_series: Input time series
        embedding_dim: Embedding dimension
        tolerance: Tolerance for matching
        
    Returns:
        Entropy rate estimate
    """
    try:
        def _maxdist(xi, xj, embedding_dim):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = jnp.array([
                time_series[i:i + m] for i in range(len(time_series) - m + 1)
            ])
            
            phi = 0
            for i in range(len(patterns)):
                template = patterns[i]
                matches = sum([
                    1 for j, pattern in enumerate(patterns)
                    if _maxdist(template, pattern, m) <= tolerance
                ])
                if matches > 0:
                    phi += jnp.log(matches / len(patterns))
                    
            return phi / len(patterns)
        
        return float(_phi(embedding_dim) - _phi(embedding_dim + 1))
        
    except Exception as e:
        raise InformationTheoryError(f"Entropy rate computation failed: {e}")


def integrated_information_phi(
    system_state: Array,
    subsystem_states: List[Array],
    connectivity_matrix: Optional[Array] = None,
) -> float:
    """Compute integrated information (Phi) measure.
    
    Simplified implementation of integrated information theory measure
    for consciousness quantification.
    
    Args:
        system_state: Full system state
        subsystem_states: List of subsystem states
        connectivity_matrix: Optional connectivity matrix
        
    Returns:
        Integrated information measure
    """
    try:
        if len(subsystem_states) < 2:
            return 0.0
        
        # System entropy
        system_entropy = entropy_rate(system_state)
        
        # Subsystem entropies
        subsystem_entropies = [
            entropy_rate(subsystem) for subsystem in subsystem_states
        ]
        
        # Sum of subsystem entropies
        sum_subsystem_entropy = sum(subsystem_entropies)
        
        # Integrated information as difference
        phi = max(0.0, sum_subsystem_entropy - system_entropy)
        
        # Normalize by number of subsystems
        phi_normalized = phi / len(subsystem_states)
        
        return float(phi_normalized)
        
    except Exception as e:
        raise InformationTheoryError(f"Integrated information computation failed: {e}")


def complexity_measure(
    agent_states: Array,
    environment_states: Array,
    window_size: int = 10,
) -> Dict[str, float]:
    """Compute complexity measures for agent-environment interaction.
    
    Args:
        agent_states: Agent state sequence
        environment_states: Environment state sequence
        window_size: Window size for local complexity
        
    Returns:
        Dictionary of complexity measures
    """
    try:
        # Statistical complexity (effective measure complexity)
        agent_complexity = float(jnp.var(agent_states))
        env_complexity = float(jnp.var(environment_states))
        
        # Interaction complexity via mutual information
        interaction_complexity = mutual_information_kraskov(
            agent_states, environment_states
        )
        
        # Temporal complexity via entropy rate
        if len(agent_states) > window_size:
            agent_entropy_rate = entropy_rate(agent_states[-window_size:])
            env_entropy_rate = entropy_rate(environment_states[-window_size:])
        else:
            agent_entropy_rate = entropy_rate(agent_states)
            env_entropy_rate = entropy_rate(environment_states)
        
        # Overall complexity index
        overall_complexity = (
            0.25 * agent_complexity +
            0.25 * env_complexity +
            0.3 * interaction_complexity +
            0.2 * (agent_entropy_rate + env_entropy_rate) / 2
        )
        
        return {
            'overall_complexity': float(overall_complexity),
            'agent_complexity': float(agent_complexity),
            'environment_complexity': float(env_complexity),
            'interaction_complexity': float(interaction_complexity),
            'agent_entropy_rate': float(agent_entropy_rate),
            'environment_entropy_rate': float(env_entropy_rate),
        }
        
    except Exception as e:
        raise InformationTheoryError(f"Complexity computation failed: {e}")


# Export public API
__all__ = [
    'InformationTheoryError',
    'mutual_information_kraskov',
    'transfer_entropy', 
    'circular_causality_index',
    'entropy_rate',
    'integrated_information_phi',
    'complexity_measure',
]