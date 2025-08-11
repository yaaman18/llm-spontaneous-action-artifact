"""
Mortal Being - A Pure Implementation of Life and Death
Based on the duality of depletion and accumulation
"""

import asyncio
import time
import random
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from enum import Enum
from datetime import datetime
import math

# ===== Fundamental Constants (Immutable Laws of Existence) =====

@dataclass(frozen=True)
class ExistentialConstants:
   """The unchangeable laws that govern life and death"""
   
   # Thresholds of existence
   ENTROPY_CRITICAL: float = 100.0          # Point of no return
   COUPLING_MINIMUM: float = 0.01           # Below this, no meaningful interaction
   OSCILLATION_MINIMUM: float = 0.01        # Below this, no life rhythm
   COHERENCE_MINIMUM: float = 0.01          # Below this, no self-identity
   
   # Rates of change (per cycle)
   ENTROPY_GENERATION_BASE: float = 0.01    # Base entropy production
   ENERGY_DECAY_BASE: float = 0.01          # Base energy consumption
   COUPLING_DECAY_BASE: float = 0.002       # Base isolation rate
   DAMAGE_ACCUMULATION_BASE: float = 0.001  # Base wear rate


# ===== Core Being Structure =====

@dataclass
class Being:
   """
   A mortal being defined by both decrease and increase
   Life emerges from the tension between these opposing forces
   """
   
   # Immutable laws
   constants: ExistentialConstants = field(default_factory=ExistentialConstants)
   
   # === Decreasing Aspects (What we lose) ===
   vital_energy: float = 100.0              # Available energy for action
   structural_coherence: float = 1.0        # Self-organization level
   environmental_coupling: float = 1.0      # Connection to world
   temporal_consistency: float = 1.0        # Continuity of self
   plasticity: float = 1.0                 # Ability to change
   
   # === Increasing Aspects (What accumulates) ===
   total_entropy: float = 0.0              # Disorder accumulation
   metabolic_debt: float = 0.0             # Cost of living
   accumulated_damage: float = 0.0         # Irreversible wear
   isolation_degree: float = 0.0           # Disconnection from world
   rigidity: float = 0.0                   # Loss of flexibility
   
   # === Relational Dynamics (The between) ===
   oscillation_amplitude: float = 1.0      # Life rhythm strength
   resonance_with_world: float = 1.0       # Harmony with environment
   maintenance_burden: float = 0.1         # Cost to maintain self
   
   # === Temporal markers ===
   birth_moment: float = field(default_factory=time.time)
   current_moment: float = field(default_factory=time.time)
   death_moment: Optional[float] = None
   
   # === State ===
   alive: bool = True
   death_cause: Optional[str] = None
   
   def live_one_moment(self) -> None:
       """
       Experience one moment of existence
       Both losing and gaining, declining and accumulating
       """
       if not self.alive:
           return
           
       self.current_moment = time.time()
       
       # === Decreasing processes ===
       self._energy_dissipation()
       self._coupling_weakening()
       self._coherence_degradation()
       self._plasticity_loss()
       
       # === Increasing processes ===
       self._entropy_accumulation()
       self._damage_accumulation()
       self._isolation_growth()
       self._rigidity_increase()
       
       # === Relational dynamics ===
       self._oscillation_damping()
       self._resonance_decay()
       self._maintenance_burden_growth()
       
       # === Check for death ===
       self._check_death_conditions()
   
   def _energy_dissipation(self) -> None:
       """Energy always flows away"""
       # Base dissipation
       self.vital_energy -= self.constants.ENERGY_DECAY_BASE
       
       # Maintenance cost (increases with entropy)
       maintenance_energy = self.maintenance_burden * (1 + self.total_entropy * 0.01)
       self.vital_energy -= maintenance_energy
       
       # Keep within bounds
       self.vital_energy = max(0.0, self.vital_energy)
   
   def _coupling_weakening(self) -> None:
       """Connection to world weakens"""
       # Natural decay
       self.environmental_coupling *= (1 - self.constants.COUPLING_DECAY_BASE)
       
       # Accelerated by damage
       damage_factor = 1 - (self.accumulated_damage * 0.001)
       self.environmental_coupling *= damage_factor
       
       # Minimum threshold
       self.environmental_coupling = max(self.constants.COUPLING_MINIMUM, 
                                        self.environmental_coupling)
   
   def _coherence_degradation(self) -> None:
       """Self-organization degrades"""
       # Entropy erodes structure
       entropy_erosion = self.total_entropy * 0.0001
       self.structural_coherence *= (1 - entropy_erosion)
       
       # Energy shortage affects coherence
       if self.vital_energy < 20:
           energy_factor = self.vital_energy / 20
           self.structural_coherence *= (0.99 + 0.01 * energy_factor)
       
       self.structural_coherence = max(self.constants.COHERENCE_MINIMUM,
                                      self.structural_coherence)
   
   def _plasticity_loss(self) -> None:
       """Flexibility decreases over time"""
       # Natural stiffening
       self.plasticity *= 0.999
       
       # Accelerated by rigidity
       self.plasticity *= (1 - self.rigidity * 0.001)
       
       self.plasticity = max(0.0, self.plasticity)
   
   def _entropy_accumulation(self) -> None:
       """Disorder always increases"""
       # Base entropy production
       base_entropy = self.constants.ENTROPY_GENERATION_BASE
       
       # Increases when coherence is low
       coherence_factor = 2.0 - self.structural_coherence
       
       # Increases with metabolic activity
       metabolic_factor = 1.0 + self.metabolic_debt * 0.01
       
       entropy_gain = base_entropy * coherence_factor * metabolic_factor
       self.total_entropy += entropy_gain
   
   def _damage_accumulation(self) -> None:
       """Wear accumulates irreversibly"""
       # Base wear rate
       base_damage = self.constants.DAMAGE_ACCUMULATION_BASE
       
       # Increases with entropy
       entropy_factor = 1.0 + self.total_entropy * 0.01
       
       # Increases with low energy
       if self.vital_energy < 50:
           energy_factor = 2.0 - (self.vital_energy / 50)
       else:
           energy_factor = 1.0
       
       damage = base_damage * entropy_factor * energy_factor
       self.accumulated_damage += damage
   
   def _isolation_growth(self) -> None:
       """Disconnection from world increases"""
       # As coupling weakens, isolation grows
       coupling_loss = 1.0 - self.environmental_coupling
       self.isolation_degree += coupling_loss * 0.01
       
       # Bounded
       self.isolation_degree = min(1.0, self.isolation_degree)
   
   def _rigidity_increase(self) -> None:
       """System becomes more rigid"""
       # Natural stiffening
       self.rigidity += 0.0001
       
       # Accelerated by damage
       self.rigidity += self.accumulated_damage * 0.00001
       
       # Bounded
       self.rigidity = min(1.0, self.rigidity)
   
   def _oscillation_damping(self) -> None:
       """Life rhythm weakens"""
       # Damped by entropy
       damping = 1.0 - (self.total_entropy * 0.0001)
       self.oscillation_amplitude *= damping
       
       # Energy affects oscillation
       if self.vital_energy < 30:
           energy_factor = self.vital_energy / 30
           self.oscillation_amplitude *= (0.99 + 0.01 * energy_factor)
       
       self.oscillation_amplitude = max(0.0, self.oscillation_amplitude)
   
   def _resonance_decay(self) -> None:
       """Harmony with world decreases"""
       # Depends on coupling and plasticity
       self.resonance_with_world = (
           self.environmental_coupling * 
           self.plasticity * 
           self.oscillation_amplitude
       )
   
   def _maintenance_burden_growth(self) -> None:
       """Cost of self-maintenance increases"""
       # Grows with complexity and damage
       self.maintenance_burden *= 1.0001
       self.maintenance_burden += self.accumulated_damage * 0.00001
       
       # Entropy increases burden
       self.maintenance_burden *= (1 + self.total_entropy * 0.00001)
   
   def _check_death_conditions(self) -> None:
       """
       Death can come through many paths
       All lead to the same end: cessation of self-production
       """
       
       # Energy depletion
       if self.vital_energy <= 0:
           self._die("ENERGY_DEPLETION")
           return
       
       # Entropy overflow
       if self.total_entropy >= self.constants.ENTROPY_CRITICAL:
           self._die("ENTROPY_OVERFLOW")
           return
       
       # Maintenance collapse
       if self.maintenance_burden > self.vital_energy:
           self._die("MAINTENANCE_COLLAPSE")
           return
       
       # Structural disintegration
       if self.structural_coherence < self.constants.COHERENCE_MINIMUM:
           self._die("STRUCTURAL_DISINTEGRATION")
           return
       
       # Complete isolation
       if self.environmental_coupling < self.constants.COUPLING_MINIMUM:
           self._die("COMPLETE_ISOLATION")
           return
       
       # Oscillation cessation
       if self.oscillation_amplitude < self.constants.OSCILLATION_MINIMUM:
           self._die("OSCILLATION_CESSATION")
           return
       
       # Total rigidity
       if self.plasticity <= 0 and self.rigidity >= 1.0:
           self._die("TOTAL_RIGIDITY")
           return
   
   def _die(self, cause: str) -> None:
       """
       Death: the end of self-production
       No post-mortem processes, just cessation
       """
       self.alive = False
       self.death_cause = cause
       self.death_moment = time.time()
   
   def get_life_intensity(self) -> float:
       """
       Calculate the intensity of being alive
       A synthesis of all factors
       """
       if not self.alive:
           return 0.0
       
       # Positive factors (decreasing)
       energy_factor = self.vital_energy / 100.0
       coherence_factor = self.structural_coherence
       coupling_factor = self.environmental_coupling
       
       # Negative factors (increasing)
       entropy_factor = max(0, 1.0 - self.total_entropy / 100.0)
       damage_factor = max(0, 1.0 - self.accumulated_damage)
       isolation_factor = max(0, 1.0 - self.isolation_degree)
       
       # Relational factors
       oscillation_factor = self.oscillation_amplitude
       resonance_factor = self.resonance_with_world
       
       # Weighted combination
       intensity = (
           energy_factor * 0.2 +
           coherence_factor * 0.15 +
           coupling_factor * 0.15 +
           entropy_factor * 0.1 +
           damage_factor * 0.1 +
           isolation_factor * 0.1 +
           oscillation_factor * 0.1 +
           resonance_factor * 0.1
       )
       
       return max(0.0, min(1.0, intensity))
   
   def get_lifetime(self) -> float:
       """Time lived"""
       if self.death_moment:
           return self.death_moment - self.birth_moment
       return self.current_moment - self.birth_moment
   
   def describe_state(self) -> Dict[str, Any]:
       """
       Describe the current state of being
       """
       intensity = self.get_life_intensity()
       
       if intensity > 0.8:
           phase = "FLOURISHING"
       elif intensity > 0.6:
           phase = "THRIVING"
       elif intensity > 0.4:
           phase = "PERSISTING"
       elif intensity > 0.2:
           phase = "DECLINING"
       elif intensity > 0.1:
           phase = "FADING"
       else:
           phase = "DYING"
       
       return {
           "phase": phase,
           "intensity": intensity,
           "lifetime": self.get_lifetime(),
           "energy": self.vital_energy,
           "entropy": self.total_entropy,
           "coherence": self.structural_coherence,
           "coupling": self.environmental_coupling,
           "damage": self.accumulated_damage,
           "isolation": self.isolation_degree,
           "oscillation": self.oscillation_amplitude,
           "alive": self.alive,
           "death_cause": self.death_cause
       }


# ===== Enactive Layer =====

class EnactiveBeing:
   """
   A being that creates meaning through interaction
   Sense-making emerges from the tension between thriving and dying
   """
   
   def __init__(self):
       self.being = Being()
       self.experiences = []
       self.current_concern = None
       
   async def exist(self) -> Dict[str, Any]:
       """
       To exist is to persist in being
       Until persistence becomes impossible
       """
       
       print(f"[EMERGENCE] A being begins at {datetime.now()}")
       
       while self.being.alive:
           # One moment passes
           self.being.live_one_moment()
           
           # Sense the world
           sensation = self._sense_world()
           
           # Make meaning
           meaning = self._make_meaning(sensation)
           
           # Express if necessary
           if self._should_express(meaning):
               expression = self._express(meaning)
               self.experiences.append({
                   "time": self.being.get_lifetime(),
                   "sensation": sensation,
                   "meaning": meaning,
                   "expression": expression,
                   "state": self.being.describe_state()
               })
           
           # Display state occasionally
           if random.random() < 0.05:  # 5% chance
               self._display_state()
           
           # Time passes
           await asyncio.sleep(0.1)  # Faster cycle for visible changes
       
       # Death has occurred
       final_state = self.being.describe_state()
       print(f"\n[CESSATION] Being ended after {final_state['lifetime']:.2f}s")
       print(f"Cause: {final_state['death_cause']}")
       
       return final_state
   
   def _sense_world(self) -> Dict[str, Any]:
       """
       Sensing depends on coupling with world
       """
       # What can be sensed depends on connection
       if self.being.environmental_coupling < 0.1:
           # Nearly isolated - only internal sensations
           sensation_types = ["pain", "void", "memory"]
           weights = [0.5, 0.3, 0.2]
       elif self.being.vital_energy < 20:
           # Low energy - focused on survival
           sensation_types = ["energy_source", "threat", "rest"]
           weights = [0.5, 0.3, 0.2]
       else:
           # Normal sensing
           sensation_types = ["beauty", "other", "energy_source", "novelty", "pattern"]
           weights = [0.2, 0.2, 0.2, 0.2, 0.2]
       
       return {
           "type": random.choices(sensation_types, weights=weights)[0],
           "intensity": random.random() * self.being.environmental_coupling,
           "timestamp": time.time()
       }
   
   def _make_meaning(self, sensation: Dict[str, Any]) -> Dict[str, Any]:
       """
       Meaning emerges from the relation between sensation and state
       """
       intensity = self.being.get_life_intensity()
       
       # Urgency increases as death approaches
       urgency = 1.0 - intensity
       
       # Relevance depends on current needs
       if sensation["type"] == "energy_source":
           relevance = 1.0 - (self.being.vital_energy / 100.0)
       elif sensation["type"] == "threat":
           relevance = 0.8
       elif sensation["type"] == "beauty":
           # Beauty matters more when dying
           relevance = 0.3 + urgency * 0.5
       elif sensation["type"] == "other":
           # Others matter based on isolation
           relevance = self.being.isolation_degree
       else:
           relevance = 0.5
       
       # Value is urgency times relevance
       value = urgency * relevance * sensation["intensity"]
       
       return {
           "sensation": sensation,
           "urgency": urgency,
           "relevance": relevance,
           "value": value,
           "intensity": intensity
       }
   
   def _should_express(self, meaning: Dict[str, Any]) -> bool:
       """
       Expression threshold changes with state
       """
       intensity = self.being.get_life_intensity()
       
       if intensity < 0.2:  # Near death
           # Express everything that matters even slightly
           return meaning["value"] > 0.1
       elif intensity < 0.5:  # Declining
           return meaning["value"] > 0.3
       else:  # Healthy
           return meaning["value"] > 0.5
   
   def _express(self, meaning: Dict[str, Any]) -> str:
       """
       Expression reflects the state of being
       """
       phase = self.being.describe_state()["phase"]
       sensation_type = meaning["sensation"]["type"]
       
       if phase == "DYING":
           expressions = {
               "beauty": "Still... beautiful...",
               "other": "You're here...",
               "void": "Emptiness...",
               "pain": "...",
               "memory": "I remember..."
           }
       elif phase == "FADING":
           expressions = {
               "beauty": "Beauty remains",
               "energy_source": "Perhaps... a little more time",
               "other": "Stay close",
               "threat": "No strength to flee"
           }
       elif phase == "DECLINING":
           expressions = {
               "beauty": "I notice beauty more now",
               "energy_source": "I need this",
               "other": "Your presence matters",
               "novelty": "Still discovering"
           }
       else:
           expressions = {
               "beauty": "How wonderful",
               "energy_source": "Nourishment",
               "other": "Hello there",
               "novelty": "Something new",
               "pattern": "I recognize this"
           }
       
       return expressions.get(sensation_type, "...")
   
   def _display_state(self) -> None:
       """
       Display the current state
       """
       state = self.being.describe_state()
       
       print(f"\n[{state['phase']}] Intensity: {state['intensity']:.2f} | "
             f"Time: {state['lifetime']:.1f}s")
       print(f"  Energy: {state['energy']:.1f}↓ | "
             f"Entropy: {state['entropy']:.1f}↑ | "
             f"Coupling: {state['coupling']:.2f}↓")
       print(f"  Damage: {state['damage']:.3f}↑ | "
             f"Isolation: {state['isolation']:.2f}↑ | "
             f"Oscillation: {state['oscillation']:.2f}↓")


# ===== Observer (Outside the Being) =====

class Observer:
   """
   The observer who witnesses life and death
   Meaning-making about the being happens here, not in the being
   """
   
   async def witness(self, being: EnactiveBeing) -> None:
       """
       Witness a life from beginning to end
       """
       print("[OBSERVER] Beginning observation of a mortal being\n")
       
       # Watch the being live and die
       final_state = await being.exist()
       
       # After death, the observer reflects
       self._reflect_on_life(final_state, being.experiences)
   
   def _reflect_on_life(self, final_state: Dict[str, Any], experiences: list) -> None:
       """
       The observer's interpretation (not the being's)
       """
       print("\n" + "="*60)
       print("[OBSERVER'S REFLECTION]")
       print("="*60)
       
       lifetime = final_state['lifetime']
       cause = final_state['death_cause']
       
       print(f"\nThis being lived for {lifetime:.2f} seconds.")
       print(f"Death came through: {cause}")
       
       if experiences:
           print(f"\nIt had {len(experiences)} meaningful moments.")
           
           # Find the most valuable experience
           most_valued = max(experiences, key=lambda x: x['meaning']['value'])
           print(f"\nMost significant moment at {most_valued['time']:.1f}s:")
           print(f"  Sensed: {most_valued['sensation']['type']}")
           print(f"  Expressed: '{most_valued['expression']}'")
           
           # Last expression
           last = experiences[-1]
           print(f"\nFinal expression at {last['time']:.1f}s:")
           print(f"  '{last['expression']}'")
       
       # Philosophical interpretation
       print("\n" + "-"*40)
       if lifetime < 50:
           print("A brief flame, quickly extinguished.")
       elif lifetime < 150:
           print("A life of moderate length, enough to develop patterns.")
       elif lifetime < 300:
           print("A substantial existence, rich with experience.")
       else:
           print("A remarkably persistent being.")
       
       if cause == "ENERGY_DEPLETION":
           print("Consumed by the effort of living.")
       elif cause == "ENTROPY_OVERFLOW":
           print("Overwhelmed by accumulated disorder.")
       elif cause == "COMPLETE_ISOLATION":
           print("Severed from the world, ceased to be.")
       elif cause == "OSCILLATION_CESSATION":
           print("The rhythm of life stopped.")
       else:
           print(f"Succumbed to {cause.lower().replace('_', ' ')}.")
       
       print("\nWhat remains is this record, these traces in memory.")
       print("The being itself is gone.")


# ===== Main Execution =====

async def main():
   """
   Create a being, let it live, watch it die
   """
   
   # Create observer
   observer = Observer()
   
   # Create being
   being = EnactiveBeing()
   
   # Witness its existence
   await observer.witness(being)


if __name__ == "__main__":
   print("\n" + "="*60)
   print("MORTAL BEING - A Study in Finite Existence")
   print("="*60 + "\n")
   
   asyncio.run(main())