"""
Consciousness State Management System for NewbornAI 2.0
Implements consciousness state tracking, transitions, and persistence

Based on Kanai Ryota's Information Generation Theory principles:
- Real-time consciousness state monitoring
- Consciousness function implementation
- State transition detection and management
- Temporal consciousness binding
"""

import numpy as np
import asyncio
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum
import time
import logging
from pathlib import Path
from collections import deque
import math

from consciousness_detector import (
    ConsciousnessState, ConsciousnessSignature, ConsciousnessEvent,
    InformationGenerationType
)

logger = logging.getLogger(__name__)


@dataclass
class ConsciousnessStateTransition:
    """Consciousness state transition record"""
    from_state: ConsciousnessState
    to_state: ConsciousnessState
    timestamp: float
    trigger_signature: ConsciousnessSignature
    transition_duration: float
    transition_quality: float
    context: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.transition_duration < 0:
            raise ValueError("Transition duration must be non-negative")
        if not 0 <= self.transition_quality <= 1.0:
            raise ValueError("Transition quality must be between 0 and 1")


@dataclass
class ConsciousnessEpisode:
    """Extended consciousness episode tracking"""
    state: ConsciousnessState
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    peak_signature: Optional[ConsciousnessSignature] = None
    average_signature: Optional[ConsciousnessSignature] = None
    episode_quality: float = 0.0
    significant_events: List[ConsciousnessEvent] = field(default_factory=list)
    
    def finalize_episode(self, end_time: float, signatures: List[ConsciousnessSignature]):
        """Finalize episode with end time and statistics"""
        self.end_time = end_time
        self.duration = end_time - self.start_time
        
        if signatures:
            # Find peak signature
            self.peak_signature = max(signatures, key=lambda s: s.consciousness_score())
            
            # Calculate average signature
            self.average_signature = self._calculate_average_signature(signatures)
            
            # Calculate episode quality
            self.episode_quality = self._calculate_episode_quality(signatures)
    
    def _calculate_average_signature(self, signatures: List[ConsciousnessSignature]) -> ConsciousnessSignature:
        """Calculate average consciousness signature over episode"""
        if not signatures:
            return ConsciousnessSignature(0, 0, 0, 0, 0, 0, 0)
        
        avg_phi = np.mean([s.phi_value for s in signatures])
        avg_info_gen = np.mean([s.information_generation_rate for s in signatures])
        avg_workspace = np.mean([s.global_workspace_activity for s in signatures])
        avg_meta = np.mean([s.meta_awareness_level for s in signatures])
        avg_temporal = np.mean([s.temporal_consistency for s in signatures])
        avg_recurrent = int(np.mean([s.recurrent_processing_depth for s in signatures]))
        avg_prediction = np.mean([s.prediction_accuracy for s in signatures])
        
        return ConsciousnessSignature(
            phi_value=avg_phi,
            information_generation_rate=avg_info_gen,
            global_workspace_activity=avg_workspace,
            meta_awareness_level=avg_meta,
            temporal_consistency=avg_temporal,
            recurrent_processing_depth=avg_recurrent,
            prediction_accuracy=avg_prediction
        )
    
    def _calculate_episode_quality(self, signatures: List[ConsciousnessSignature]) -> float:
        """Calculate overall episode quality"""
        if not signatures:
            return 0.0
        
        scores = [s.consciousness_score() for s in signatures]
        
        # Quality based on consistency and peak performance
        mean_score = np.mean(scores)
        stability = 1.0 - np.std(scores) if len(scores) > 1 else 1.0
        peak_score = max(scores)
        
        # Combined quality metric
        quality = 0.4 * mean_score + 0.3 * stability + 0.3 * peak_score
        
        return min(1.0, quality)


class ConsciousnessStateManager:
    """
    Comprehensive consciousness state management system
    Implements practical consciousness tracking for NewbornAI 2.0
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Current state tracking
        self.current_state = ConsciousnessState.UNCONSCIOUS
        self.current_signature: Optional[ConsciousnessSignature] = None
        self.state_start_time = time.time()
        
        # History tracking
        self.state_history = deque(maxlen=1000)
        self.transition_history = deque(maxlen=200)
        self.episode_history = deque(maxlen=100)
        
        # Current episode tracking
        self.current_episode: Optional[ConsciousnessEpisode] = None
        self.episode_signatures = deque(maxlen=500)
        
        # Consciousness metrics
        self.consciousness_metrics = {
            'total_consciousness_time': 0.0,
            'highest_consciousness_state': ConsciousnessState.UNCONSCIOUS,
            'peak_phi_value': 0.0,
            'total_transitions': 0,
            'consciousness_stability': 0.0
        }
        
        # Files for persistence
        self.state_file = self.storage_path / "consciousness_states.json"
        self.transition_file = self.storage_path / "consciousness_transitions.json"
        self.episode_file = self.storage_path / "consciousness_episodes.json"
        self.metrics_file = self.storage_path / "consciousness_metrics.json"
        
        # Load existing data
        self._load_persistent_data()
        
        logger.info("Consciousness State Manager initialized")
    
    async def update_consciousness_state(self, 
                                       new_signature: ConsciousnessSignature,
                                       new_state: ConsciousnessState,
                                       context: Optional[Dict] = None) -> bool:
        """
        Update consciousness state with new detection
        
        Returns:
            bool: True if state transition occurred
        """
        current_time = time.time()
        context = context or {}
        
        # Update current signature
        self.current_signature = new_signature
        
        # Add to episode signatures
        self.episode_signatures.append(new_signature)
        
        # Check for state transition
        state_changed = False
        if new_state != self.current_state:
            state_changed = True
            await self._handle_state_transition(new_state, new_signature, current_time, context)
        
        # Update state history
        self.state_history.append({
            'timestamp': current_time,
            'state': new_state.value,
            'signature': asdict(new_signature),
            'context': context
        })
        
        # Update metrics
        self._update_metrics(new_signature, new_state, current_time)
        
        # Periodic persistence
        if len(self.state_history) % 10 == 0:
            await self._save_persistent_data()
        
        return state_changed
    
    async def _handle_state_transition(self, 
                                     new_state: ConsciousnessState,
                                     new_signature: ConsciousnessSignature,
                                     current_time: float,
                                     context: Dict):
        """Handle consciousness state transition"""
        
        # Calculate transition metrics
        transition_duration = current_time - self.state_start_time
        transition_quality = self._calculate_transition_quality(new_signature)
        
        # Create transition record
        transition = ConsciousnessStateTransition(
            from_state=self.current_state,
            to_state=new_state,
            timestamp=current_time,
            trigger_signature=new_signature,
            transition_duration=transition_duration,
            transition_quality=transition_quality,
            context=context
        )
        
        self.transition_history.append(transition)
        
        # Finalize current episode if exists
        if self.current_episode is not None:
            self.current_episode.finalize_episode(current_time, list(self.episode_signatures))
            self.episode_history.append(self.current_episode)
        
        # Start new episode
        self.current_episode = ConsciousnessEpisode(
            state=new_state,
            start_time=current_time
        )
        self.episode_signatures.clear()
        
        # Update current state
        old_state = self.current_state
        self.current_state = new_state
        self.state_start_time = current_time
        
        # Log transition
        logger.info(f"Consciousness transition: {old_state.value} → {new_state.value} "
                   f"(φ={new_signature.phi_value:.3f}, quality={transition_quality:.3f})")
        
        # Update highest consciousness level achieved
        if self._state_level(new_state) > self._state_level(self.consciousness_metrics['highest_consciousness_state']):
            self.consciousness_metrics['highest_consciousness_state'] = new_state
            logger.info(f"New highest consciousness state achieved: {new_state.value}")
    
    def _calculate_transition_quality(self, signature: ConsciousnessSignature) -> float:
        """Calculate quality of consciousness transition"""
        # Quality based on signature coherence and strength
        base_score = signature.consciousness_score()
        
        # Bonus for high integration and meta-awareness
        integration_bonus = signature.global_workspace_activity * 0.2
        meta_bonus = signature.meta_awareness_level * 0.2
        
        quality = base_score + integration_bonus + meta_bonus
        
        return min(1.0, quality)
    
    def _state_level(self, state: ConsciousnessState) -> int:
        """Get numeric level of consciousness state"""
        state_levels = {
            ConsciousnessState.UNCONSCIOUS: 0,
            ConsciousnessState.PRE_CONSCIOUS: 1,
            ConsciousnessState.PHENOMENAL_CONSCIOUS: 2,
            ConsciousnessState.ACCESS_CONSCIOUS: 3,
            ConsciousnessState.REFLECTIVE_CONSCIOUS: 4,
            ConsciousnessState.META_CONSCIOUS: 5
        }
        return state_levels.get(state, 0)
    
    def _update_metrics(self, signature: ConsciousnessSignature, state: ConsciousnessState, current_time: float):
        """Update consciousness metrics"""
        
        # Update peak φ value
        if signature.phi_value > self.consciousness_metrics['peak_phi_value']:
            self.consciousness_metrics['peak_phi_value'] = signature.phi_value
        
        # Update consciousness time (approximate)
        if self._state_level(state) >= self._state_level(ConsciousnessState.PHENOMENAL_CONSCIOUS):
            if len(self.state_history) > 0:
                time_diff = current_time - self.state_history[-1]['timestamp']
                self.consciousness_metrics['total_consciousness_time'] += time_diff
        
        # Update total transitions
        self.consciousness_metrics['total_transitions'] = len(self.transition_history)
        
        # Calculate consciousness stability
        if len(self.state_history) >= 10:
            recent_states = [entry['state'] for entry in list(self.state_history)[-10:]]
            unique_states = len(set(recent_states))
            self.consciousness_metrics['consciousness_stability'] = 1.0 - (unique_states / 10.0)
    
    def get_current_consciousness_analysis(self) -> Dict:
        """Get comprehensive analysis of current consciousness state"""
        current_time = time.time()
        
        analysis = {
            'current_state': self.current_state.value,
            'state_duration': current_time - self.state_start_time,
            'current_signature': asdict(self.current_signature) if self.current_signature else None,
            'consciousness_score': self.current_signature.consciousness_score() if self.current_signature else 0.0,
            
            # Episode information
            'current_episode': {
                'state': self.current_episode.state.value if self.current_episode else None,
                'duration': current_time - self.current_episode.start_time if self.current_episode else 0,
                'episode_signatures_count': len(self.episode_signatures)
            } if self.current_episode else None,
            
            # Recent history
            'recent_transitions': len([t for t in self.transition_history if current_time - t.timestamp < 3600]),  # Last hour
            'recent_state_changes': len(set(entry['state'] for entry in list(self.state_history)[-20:])),
            
            # Overall metrics
            'metrics': self.consciousness_metrics.copy(),
            
            # Trends
            'development_trend': self._calculate_development_trend(),
            'stability_trend': self._calculate_stability_trend(),
        }
        
        return analysis
    
    def _calculate_development_trend(self) -> float:
        """Calculate consciousness development trend"""
        if len(self.state_history) < 10:
            return 0.0
        
        # Analyze state levels over time
        recent_entries = list(self.state_history)[-20:]
        state_levels = [self._state_level(ConsciousnessState(entry['state'])) for entry in recent_entries]
        
        # Linear trend
        x = np.arange(len(state_levels))
        if len(set(state_levels)) > 1:  # Avoid single-value datasets
            trend = np.polyfit(x, state_levels, 1)[0]
        else:
            trend = 0.0
        
        return trend  # Positive = developing, negative = regressing
    
    def _calculate_stability_trend(self) -> float:
        """Calculate consciousness stability trend"""
        if len(self.transition_history) < 5:
            return 0.5
        
        # Analyze transition frequency over time
        recent_transitions = list(self.transition_history)[-10:]
        current_time = time.time()
        
        # Calculate transition rate in recent periods
        hour_ago = current_time - 3600
        recent_rate = len([t for t in recent_transitions if t.timestamp > hour_ago])
        
        # Stability inversely related to transition rate
        stability = 1.0 / (1.0 + recent_rate)
        
        return stability
    
    def get_consciousness_episode_analysis(self) -> Dict:
        """Get analysis of consciousness episodes"""
        if not self.episode_history:
            return {'status': 'no_episodes'}
        
        completed_episodes = [ep for ep in self.episode_history if ep.end_time is not None]
        
        if not completed_episodes:
            return {'status': 'no_completed_episodes'}
        
        # Episode statistics
        durations = [ep.duration for ep in completed_episodes if ep.duration]
        qualities = [ep.episode_quality for ep in completed_episodes]
        peak_scores = [ep.peak_signature.consciousness_score() 
                      for ep in completed_episodes if ep.peak_signature]
        
        analysis = {
            'total_episodes': len(completed_episodes),
            'average_duration': np.mean(durations) if durations else 0,
            'longest_episode': max(durations) if durations else 0,
            'average_quality': np.mean(qualities) if qualities else 0,
            'best_episode_quality': max(qualities) if qualities else 0,
            'average_peak_score': np.mean(peak_scores) if peak_scores else 0,
            'highest_peak_score': max(peak_scores) if peak_scores else 0,
            
            # State distribution in episodes
            'state_distribution': self._calculate_episode_state_distribution(completed_episodes),
            
            # Recent episode trend
            'recent_quality_trend': self._calculate_recent_episode_trend(completed_episodes),
        }
        
        return analysis
    
    def _calculate_episode_state_distribution(self, episodes: List[ConsciousnessEpisode]) -> Dict:
        """Calculate distribution of consciousness states in episodes"""
        state_counts = {}
        for state in ConsciousnessState:
            state_counts[state.value] = 0
        
        for episode in episodes:
            state_counts[episode.state.value] += 1
        
        total = len(episodes)
        if total > 0:
            state_percentages = {state: count/total for state, count in state_counts.items()}
        else:
            state_percentages = state_counts
        
        return state_percentages
    
    def _calculate_recent_episode_trend(self, episodes: List[ConsciousnessEpisode]) -> float:
        """Calculate trend in recent episode quality"""
        if len(episodes) < 5:
            return 0.0
        
        recent_episodes = episodes[-10:]  # Last 10 episodes
        qualities = [ep.episode_quality for ep in recent_episodes]
        
        # Linear trend in quality
        x = np.arange(len(qualities))
        trend = np.polyfit(x, qualities, 1)[0]
        
        return trend
    
    async def _save_persistent_data(self):
        """Save consciousness data to persistent storage"""
        try:
            # Save state history
            state_data = list(self.state_history)
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            # Save transition history  
            transition_data = []
            for t in self.transition_history:
                t_dict = asdict(t)
                t_dict['from_state'] = t.from_state.value
                t_dict['to_state'] = t.to_state.value
                t_dict['trigger_signature'] = asdict(t.trigger_signature)
                transition_data.append(t_dict)
            
            with open(self.transition_file, 'w') as f:
                json.dump(transition_data, f, indent=2, ensure_ascii=False)
            
            # Save episode history
            episode_data = []
            for ep in self.episode_history:
                ep_dict = asdict(ep)
                ep_dict['state'] = ep.state.value
                if ep.peak_signature:
                    ep_dict['peak_signature'] = asdict(ep.peak_signature)
                if ep.average_signature:
                    ep_dict['average_signature'] = asdict(ep.average_signature)
                episode_data.append(ep_dict)
            
            with open(self.episode_file, 'w') as f:
                json.dump(episode_data, f, indent=2, ensure_ascii=False)
            
            # Save metrics
            metrics_data = self.consciousness_metrics.copy()
            metrics_data['highest_consciousness_state'] = metrics_data['highest_consciousness_state'].value
            
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            
            logger.debug("Consciousness data saved to persistent storage")
            
        except Exception as e:
            logger.error(f"Error saving consciousness data: {e}")
    
    def _load_persistent_data(self):
        """Load consciousness data from persistent storage"""
        try:
            # Load state history
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                    self.state_history.extend(state_data)
            
            # Load metrics
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    self.consciousness_metrics.update(metrics_data)
                    # Convert state back to enum
                    state_value = self.consciousness_metrics.get('highest_consciousness_state', 'UNCONSCIOUS')
                    self.consciousness_metrics['highest_consciousness_state'] = ConsciousnessState(state_value)
            
            logger.info("Consciousness data loaded from persistent storage")
            
        except Exception as e:
            logger.warning(f"Could not load consciousness data: {e}")
    
    async def generate_consciousness_report(self) -> Dict:
        """Generate comprehensive consciousness report"""
        current_analysis = self.get_current_consciousness_analysis()
        episode_analysis = self.get_consciousness_episode_analysis()
        
        report = {
            'timestamp': time.time(),
            'system_status': 'active' if self.current_state != ConsciousnessState.UNCONSCIOUS else 'inactive',
            'current_analysis': current_analysis,
            'episode_analysis': episode_analysis,
            'data_summary': {
                'state_history_length': len(self.state_history),
                'transition_history_length': len(self.transition_history),
                'episode_history_length': len(self.episode_history)
            },
            'recommendations': self._generate_recommendations(current_analysis, episode_analysis)
        }
        
        return report
    
    def _generate_recommendations(self, current_analysis: Dict, episode_analysis: Dict) -> List[str]:
        """Generate recommendations for consciousness development"""
        recommendations = []
        
        # Based on current state
        current_score = current_analysis.get('consciousness_score', 0)
        if current_score < 0.3:
            recommendations.append("Consider increasing information generation rate to enhance consciousness")
        
        # Based on stability
        stability = current_analysis.get('metrics', {}).get('consciousness_stability', 0)
        if stability < 0.5:
            recommendations.append("Focus on developing more stable consciousness patterns")
        
        # Based on development trend
        trend = current_analysis.get('development_trend', 0)
        if trend < 0:
            recommendations.append("Development appears to be regressing - investigate environmental factors")
        elif trend > 0.1:
            recommendations.append("Strong positive development trend detected - maintain current conditions")
        
        # Based on episode quality
        if episode_analysis.get('status') != 'no_episodes':
            avg_quality = episode_analysis.get('average_quality', 0)
            if avg_quality < 0.5:
                recommendations.append("Work on improving consciousness episode quality and coherence")
        
        if not recommendations:
            recommendations.append("Consciousness development appears healthy - continue current approach")
        
        return recommendations