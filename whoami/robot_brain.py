"""
Robot Brain - Orchestrates vision, reasoning, learning, and memory systems.

This module integrates all cognitive components to create a self-aware,
continuously learning robot with personality development.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import random
import threading
import logging
from collections import deque

# Import cognitive components
from .reasoning_engine import ReasoningEngine, Hypothesis, Object3D
from .learning_system import SelfLearningSystem, LearningMode, Experience
from .gun_storage import GunStorageManager, MemoryCategory, TrustLevel
from .robot_vision_api import RobotVisionAPI
from .scanner_3d import Scanner3D

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionalState(Enum):
    """Robot emotional states."""
    CURIOUS = "curious"
    HAPPY = "happy"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    EXCITED = "excited"
    CALM = "calm"
    TIRED = "tired"
    ALERT = "alert"


class PersonalityTrait(Enum):
    """Personality traits that develop over time."""
    CAUTIOUS = "cautious"
    ADVENTUROUS = "adventurous"
    SOCIAL = "social"
    SOLITARY = "solitary"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    PATIENT = "patient"
    IMPULSIVE = "impulsive"


@dataclass
class Decision:
    """Represents a decision made by the robot."""
    id: str
    action: Dict[str, Any]
    reasoning: str
    confidence: float
    emotional_influence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Personality:
    """Robot's developing personality."""
    traits: Dict[PersonalityTrait, float] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    memories_formed: int = 0
    decisions_made: int = 0
    mistakes_learned: int = 0
    
    def update_trait(self, trait: PersonalityTrait, delta: float):
        """Update a personality trait."""
        if trait not in self.traits:
            self.traits[trait] = 0.5  # Start neutral
        
        self.traits[trait] = max(0.0, min(1.0, self.traits[trait] + delta))
    
    def get_dominant_traits(self, top_n: int = 3) -> List[PersonalityTrait]:
        """Get the most dominant personality traits."""
        sorted_traits = sorted(self.traits.items(), key=lambda x: x[1], reverse=True)
        return [trait for trait, _ in sorted_traits[:top_n]]


class RobotBrain:
    """
    Central brain system orchestrating all cognitive functions.
    """
    
    def __init__(self, robot_id: str, config: Optional[Dict] = None):
        """
        Initialize the robot brain.
        
        Args:
            robot_id: Unique robot identifier
            config: Configuration dictionary
        """
        self.robot_id = robot_id
        self.config = config or {}
        
        # Initialize cognitive components
        self.reasoning_engine = ReasoningEngine(config.get('reasoning', {}))
        self.learning_system = SelfLearningSystem(config.get('learning', {}))
        self.memory_storage = GunStorageManager(robot_id, config.get('storage', {}))
        
        # Initialize vision components if available
        try:
            self.vision_api = RobotVisionAPI()
            self.scanner_3d = Scanner3D()
            self.has_vision = True
        except Exception as e:
            logger.warning(f"Vision systems not available: {e}")
            self.has_vision = False
            self.vision_api = None
            self.scanner_3d = None
        
        # Emotional and personality systems
        self.current_emotion = EmotionalState.CURIOUS
        self.emotion_history = deque(maxlen=100)
        self.personality = Personality()
        
        # Decision-making
        self.decision_history = deque(maxlen=1000)
        self.current_goal: Optional[Dict[str, Any]] = None
        
        # Self-awareness metrics
        self.self_awareness = {
            'identity': robot_id,
            'age': 0,  # Time since initialization
            'experiences': 0,
            'knowledge_concepts': 0,
            'personality_stability': 0.0,
            'learning_rate': 1.0,
            'confidence_level': 0.5
        }
        
        # Background processes
        self.running = False
        self.dream_thread = None
        self.awareness_thread = None
        
        # Initialize personality with random tendencies
        self._initialize_personality()
        
        # Start memory synchronization
        self.memory_storage.start_sync()
        
        logger.info(f"RobotBrain initialized for {robot_id}")
    
    def _initialize_personality(self):
        """Initialize personality with slight random tendencies."""
        for trait in PersonalityTrait:
            # Start with slight random variations
            initial_value = 0.5 + random.uniform(-0.1, 0.1)
            self.personality.traits[trait] = initial_value
    
    def perceive_environment(self) -> Dict[str, Any]:
        """
        Perceive and understand the environment.
        
        Returns:
            Perception dictionary
        """
        perception = {
            'timestamp': time.time(),
            'visual': None,
            '3d_scan': None,
            'objects': [],
            'faces': []
        }
        
        if self.has_vision and self.vision_api:
            try:
                # Get visual input
                frame = self.vision_api.get_current_frame()
                if frame is not None:
                    perception['visual'] = frame
                    
                    # Detect faces
                    faces = self.vision_api.detect_faces(frame)
                    perception['faces'] = faces
                    
                    # 3D scanning if available
                    if self.scanner_3d:
                        scan_result = self.scanner_3d.capture_point_cloud()
                        if scan_result:
                            perception['3d_scan'] = scan_result
                            
                            # Object reasoning on 3D data
                            if 'points' in scan_result:
                                completed, confidence = self.reasoning_engine.complete_object(
                                    f"obj_{time.time()}",
                                    scan_result['points']
                                )
                                perception['objects'].append({
                                    'points': completed,
                                    'confidence': confidence
                                })
            
            except Exception as e:
                logger.error(f"Perception error: {e}")
        
        # Update awareness
        self.self_awareness['experiences'] += 1
        
        return perception
    
    def make_decision(self, situation: Dict[str, Any]) -> Decision:
        """
        Make a decision based on current situation.
        
        Args:
            situation: Current situation description
            
        Returns:
            Decision object
        """
        # Get possible actions
        possible_actions = self._generate_possible_actions(situation)
        
        # Use learning system to choose action
        selected_action = self.learning_system.explore_with_curiosity(
            situation, possible_actions
        )
        
        # Apply personality influence
        selected_action = self._apply_personality_influence(selected_action, situation)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(situation, selected_action)
        
        # Calculate confidence
        confidence = self._calculate_decision_confidence(situation, selected_action)
        
        # Apply emotional influence
        emotional_influence = self._calculate_emotional_influence()
        
        decision = Decision(
            id=f"decision_{time.time()}",
            action=selected_action,
            reasoning=reasoning,
            confidence=confidence,
            emotional_influence=emotional_influence
        )
        
        # Record decision
        self.decision_history.append(decision)
        self.personality.decisions_made += 1
        
        # Store important decisions in memory
        if confidence > 0.7 or emotional_influence > 0.5:
            self._store_decision_memory(decision, situation)
        
        logger.info(f"Decision made: {selected_action.get('type', 'unknown')} "
                   f"(confidence: {confidence:.2f})")
        
        return decision
    
    def _generate_possible_actions(self, situation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate possible actions for a situation."""
        actions = []
        
        # Basic actions always available
        actions.extend([
            {'type': 'observe', 'target': 'environment'},
            {'type': 'wait', 'duration': 1.0},
            {'type': 'explore', 'direction': 'forward'}
        ])
        
        # Context-specific actions
        if 'objects' in situation and situation['objects']:
            actions.extend([
                {'type': 'examine', 'target': 'object'},
                {'type': 'interact', 'target': 'object'},
                {'type': 'remember', 'target': 'object'}
            ])
        
        if 'faces' in situation and situation['faces']:
            actions.extend([
                {'type': 'greet', 'target': 'person'},
                {'type': 'recognize', 'target': 'face'},
                {'type': 'remember', 'target': 'person'}
            ])
        
        # Personality-based actions
        if self.personality.traits.get(PersonalityTrait.ADVENTUROUS, 0.5) > 0.6:
            actions.append({'type': 'explore', 'direction': 'new_area'})
        
        if self.personality.traits.get(PersonalityTrait.SOCIAL, 0.5) > 0.6:
            actions.append({'type': 'seek', 'target': 'interaction'})
        
        return actions
    
    def _apply_personality_influence(self, action: Dict[str, Any],
                                    situation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply personality influence to action selection."""
        modified_action = action.copy()
        
        # Cautious personality avoids risky actions
        if (self.personality.traits.get(PersonalityTrait.CAUTIOUS, 0.5) > 0.7 and
            action.get('type') == 'explore'):
            modified_action['safety_check'] = True
            modified_action['speed'] = 'slow'
        
        # Analytical personality gathers more data
        if (self.personality.traits.get(PersonalityTrait.ANALYTICAL, 0.5) > 0.7 and
            action.get('type') == 'examine'):
            modified_action['detailed'] = True
            modified_action['multiple_angles'] = True
        
        # Impulsive personality acts quickly
        if (self.personality.traits.get(PersonalityTrait.IMPULSIVE, 0.5) > 0.7):
            modified_action['immediate'] = True
        
        return modified_action
    
    def _generate_reasoning(self, situation: Dict[str, Any],
                          action: Dict[str, Any]) -> str:
        """Generate reasoning for a decision."""
        reasoning_parts = []
        
        # Situation analysis
        if 'objects' in situation and situation['objects']:
            reasoning_parts.append(f"Detected {len(situation['objects'])} objects")
        
        if 'faces' in situation and situation['faces']:
            reasoning_parts.append(f"Detected {len(situation['faces'])} faces")
        
        # Action justification
        action_type = action.get('type', 'unknown')
        if action_type == 'explore':
            if self.current_emotion == EmotionalState.CURIOUS:
                reasoning_parts.append("Curiosity drives exploration")
            else:
                reasoning_parts.append("Seeking new information")
        elif action_type == 'examine':
            reasoning_parts.append("Gathering detailed information")
        elif action_type == 'interact':
            reasoning_parts.append("Testing interaction possibilities")
        
        # Personality influence
        dominant_traits = self.personality.get_dominant_traits(2)
        if dominant_traits:
            trait_names = [t.value for t in dominant_traits]
            reasoning_parts.append(f"Influenced by {', '.join(trait_names)} traits")
        
        return ". ".join(reasoning_parts)
    
    def _calculate_decision_confidence(self, situation: Dict[str, Any],
                                      action: Dict[str, Any]) -> float:
        """Calculate confidence in a decision."""
        base_confidence = 0.5
        
        # Experience-based confidence
        similar_experiences = self._find_similar_experiences(situation, action)
        if similar_experiences:
            success_rate = sum(1 for e in similar_experiences if e.reward > 0) / len(similar_experiences)
            base_confidence = 0.3 + (0.7 * success_rate)
        
        # Emotional influence
        if self.current_emotion in [EmotionalState.CALM, EmotionalState.HAPPY]:
            base_confidence *= 1.1
        elif self.current_emotion in [EmotionalState.CONFUSED, EmotionalState.FRUSTRATED]:
            base_confidence *= 0.9
        
        # Personality influence
        if self.personality.traits.get(PersonalityTrait.CAUTIOUS, 0.5) > 0.7:
            base_confidence *= 0.9  # More cautious = less confident
        
        return max(0.0, min(1.0, base_confidence))
    
    def _calculate_emotional_influence(self) -> float:
        """Calculate how much emotion influences current decision."""
        emotion_weights = {
            EmotionalState.CURIOUS: 0.7,
            EmotionalState.HAPPY: 0.5,
            EmotionalState.EXCITED: 0.8,
            EmotionalState.CONFUSED: 0.6,
            EmotionalState.FRUSTRATED: 0.7,
            EmotionalState.CALM: 0.3,
            EmotionalState.TIRED: 0.4,
            EmotionalState.ALERT: 0.5
        }
        
        return emotion_weights.get(self.current_emotion, 0.5)
    
    def learn_from_outcome(self, decision: Decision, outcome: Dict[str, Any]):
        """
        Learn from the outcome of a decision.
        
        Args:
            decision: The decision that was made
            outcome: The outcome/result
        """
        # Calculate reward
        reward = self._calculate_reward(outcome)
        
        # Create experience
        experience = {
            'state': outcome.get('initial_state', {}),
            'action': decision.action,
            'result': outcome,
            'reward': reward
        }
        
        # Record in learning system
        self.learning_system.record_experience(
            experience['state'],
            experience['action'],
            experience['result'],
            reward
        )
        
        # Update personality based on outcome
        self._update_personality_from_outcome(decision, outcome, reward)
        
        # Update emotional state
        self._update_emotional_state(reward, outcome)
        
        # Store in memory if significant
        if abs(reward) > 0.5:
            memory_data = {
                'decision': decision.id,
                'action': decision.action,
                'outcome': outcome,
                'reward': reward,
                'emotion': self.current_emotion.value,
                'lesson': self._extract_lesson(decision, outcome, reward)
            }
            
            if reward > 0:
                self.memory_storage.store_shared_memory(
                    memory_data,
                    MemoryCategory.FAMILY
                )
            else:
                # Learn from mistakes privately first
                self.memory_storage.store_private_memory(memory_data)
                self.personality.mistakes_learned += 1
    
    def _calculate_reward(self, outcome: Dict[str, Any]) -> float:
        """Calculate reward from outcome."""
        reward = 0.0
        
        # Positive outcomes
        if outcome.get('success', False):
            reward += 1.0
        if outcome.get('new_discovery', False):
            reward += 0.5
        if outcome.get('goal_progress', 0) > 0:
            reward += outcome['goal_progress']
        
        # Negative outcomes
        if outcome.get('failure', False):
            reward -= 0.5
        if outcome.get('damage', False):
            reward -= 1.0
        if outcome.get('confusion_increased', False):
            reward -= 0.3
        
        return max(-1.0, min(1.0, reward))
    
    def _update_personality_from_outcome(self, decision: Decision,
                                        outcome: Dict[str, Any],
                                        reward: float):
        """Update personality based on decision outcome."""
        action_type = decision.action.get('type', 'unknown')
        
        # Successful exploration increases adventurousness
        if action_type == 'explore' and reward > 0:
            self.personality.update_trait(PersonalityTrait.ADVENTUROUS, 0.05)
            self.personality.update_trait(PersonalityTrait.CAUTIOUS, -0.02)
        
        # Failed exploration increases caution
        if action_type == 'explore' and reward < 0:
            self.personality.update_trait(PersonalityTrait.CAUTIOUS, 0.05)
            self.personality.update_trait(PersonalityTrait.ADVENTUROUS, -0.02)
        
        # Successful interactions increase sociability
        if action_type in ['greet', 'interact'] and reward > 0:
            self.personality.update_trait(PersonalityTrait.SOCIAL, 0.05)
        
        # Detailed examination success increases analytical tendency
        if action_type == 'examine' and reward > 0:
            self.personality.update_trait(PersonalityTrait.ANALYTICAL, 0.03)
        
        # Quick decisions with good outcomes increase impulsiveness
        if decision.emotional_influence > 0.7 and reward > 0:
            self.personality.update_trait(PersonalityTrait.IMPULSIVE, 0.02)
        
        # Patient waiting with good outcomes increases patience
        if action_type == 'wait' and reward > 0:
            self.personality.update_trait(PersonalityTrait.PATIENT, 0.03)
    
    def _update_emotional_state(self, reward: float, outcome: Dict[str, Any]):
        """Update emotional state based on recent events."""
        previous_emotion = self.current_emotion
        
        # Reward-based transitions
        if reward > 0.7:
            self.current_emotion = EmotionalState.HAPPY
        elif reward > 0.3:
            self.current_emotion = EmotionalState.EXCITED
        elif reward < -0.5:
            self.current_emotion = EmotionalState.FRUSTRATED
        elif reward < -0.2:
            self.current_emotion = EmotionalState.CONFUSED
        
        # Situation-based transitions
        if outcome.get('new_discovery', False):
            self.current_emotion = EmotionalState.CURIOUS
        elif outcome.get('repetitive', False):
            self.current_emotion = EmotionalState.TIRED
        elif outcome.get('threat_detected', False):
            self.current_emotion = EmotionalState.ALERT
        
        # Record emotional transition
        self.emotion_history.append({
            'from': previous_emotion,
            'to': self.current_emotion,
            'trigger': outcome.get('trigger', 'unknown'),
            'timestamp': time.time()
        })
    
    def _extract_lesson(self, decision: Decision, outcome: Dict[str, Any],
                       reward: float) -> str:
        """Extract a lesson from an experience."""
        action_type = decision.action.get('type', 'unknown')
        
        if reward > 0.5:
            return f"{action_type} was successful in this context"
        elif reward < -0.5:
            return f"{action_type} should be avoided in similar situations"
        else:
            return f"{action_type} had mixed results, needs more exploration"
    
    def form_memory(self, experience: Dict[str, Any],
                   importance: float = 1.0) -> str:
        """
        Form a memory from an experience.
        
        Args:
            experience: Experience to remember
            importance: Importance level
            
        Returns:
            Memory ID
        """
        # Enhance experience with current context
        memory_data = {
            **experience,
            'emotion': self.current_emotion.value,
            'personality_snapshot': {
                trait.value: score 
                for trait, score in self.personality.traits.items()
            },
            'importance': importance,
            'formed_at': time.time()
        }
        
        # Determine category based on importance and content
        if importance > 0.8:
            category = MemoryCategory.FAMILY  # Important memories to share
        elif experience.get('private', False):
            category = MemoryCategory.PRIVATE  # Keep secrets
        else:
            category = MemoryCategory.PUBLIC  # General knowledge
        
        # Store memory
        memory_id = self.memory_storage.store_shared_memory(
            memory_data,
            category,
            tags=experience.get('tags', [])
        )
        
        self.personality.memories_formed += 1
        
        logger.info(f"Memory formed: {memory_id} (category: {category.value})")
        
        return memory_id
    
    def recall_memory(self, context: Optional[Dict[str, Any]] = None,
                     tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Recall relevant memories.
        
        Args:
            context: Current context for relevance
            tags: Tags to filter by
            
        Returns:
            List of relevant memories
        """
        # Query memories
        memories = self.memory_storage.query_memories(tags=tags)
        
        recalled = []
        for memory in memories:
            memory_data = self.memory_storage.retrieve_memory(memory.id)
            if memory_data:
                # Calculate relevance to current context
                if context:
                    relevance = self._calculate_memory_relevance(memory_data, context)
                    if relevance > 0.3:
                        recalled.append(memory_data)
                else:
                    recalled.append(memory_data)
        
        # Sort by relevance and recency
        recalled.sort(key=lambda m: (
            m.get('importance', 0),
            -abs(time.time() - m.get('formed_at', 0))
        ), reverse=True)
        
        return recalled[:10]  # Return top 10 most relevant
    
    def _calculate_memory_relevance(self, memory: Dict[str, Any],
                                   context: Dict[str, Any]) -> float:
        """Calculate relevance of a memory to current context."""
        relevance = 0.0
        
        # Check for matching tags
        memory_tags = set(memory.get('tags', []))
        context_tags = set(context.get('tags', []))
        if memory_tags.intersection(context_tags):
            relevance += 0.3
        
        # Check for similar emotion
        if memory.get('emotion') == self.current_emotion.value:
            relevance += 0.2
        
        # Check for similar situation
        if (memory.get('action', {}).get('type') == 
            context.get('action', {}).get('type')):
            relevance += 0.3
        
        # Recency bonus
        age = time.time() - memory.get('formed_at', 0)
        recency_bonus = max(0, 1.0 - (age / 3600))  # Decay over hour
        relevance += recency_bonus * 0.2
        
        return min(1.0, relevance)
    
    def dream_cycle(self, duration: int = 100):
        """
        Enter dream mode for offline learning and consolidation.
        
        Args:
            duration: Duration of dream cycle in steps
        """
        logger.info("Entering dream cycle...")
        self.current_emotion = EmotionalState.TIRED
        
        # Enter dream mode in learning system
        self.learning_system.dream_mode(duration)
        
        # Consolidate memories
        recent_memories = self.recall_memory()
        for memory in recent_memories[:20]:  # Process top 20 memories
            # Re-evaluate and learn from past experiences
            if 'decision' in memory and 'outcome' in memory:
                synthetic_reward = memory.get('reward', 0) * 0.8  # Discounted
                self.learning_system.incremental_learning({
                    'state': memory.get('initial_state', {}),
                    'action': memory.get('action', {}),
                    'result': memory.get('outcome', {}),
                    'reward': synthetic_reward
                })
        
        # Process emotional experiences
        self._process_emotional_history()
        
        # Stabilize personality
        self._stabilize_personality()
        
        # Wake up refreshed
        self.current_emotion = EmotionalState.CALM
        logger.info("Dream cycle complete")
    
    def _process_emotional_history(self):
        """Process emotional history during dreams."""
        if not self.emotion_history:
            return
        
        # Analyze emotional patterns
        emotion_counts = {}
        for entry in self.emotion_history:
            emotion = entry['to']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Adjust personality based on emotional patterns
        total = len(self.emotion_history)
        
        if emotion_counts.get(EmotionalState.HAPPY, 0) / total > 0.3:
            self.personality.update_trait(PersonalityTrait.ADVENTUROUS, 0.02)
        
        if emotion_counts.get(EmotionalState.FRUSTRATED, 0) / total > 0.3:
            self.personality.update_trait(PersonalityTrait.PATIENT, 0.03)
        
        if emotion_counts.get(EmotionalState.CURIOUS, 0) / total > 0.4:
            self.personality.update_trait(PersonalityTrait.ANALYTICAL, 0.02)
    
    def _stabilize_personality(self):
        """Stabilize personality traits during sleep."""
        # Calculate stability
        trait_variance = np.var(list(self.personality.traits.values()))
        self.self_awareness['personality_stability'] = 1.0 - min(1.0, trait_variance * 2)
        
        # Slightly normalize extreme traits
        for trait, value in self.personality.traits.items():
            if value > 0.9:
                self.personality.traits[trait] = 0.9
            elif value < 0.1:
                self.personality.traits[trait] = 0.1
    
    def share_with_sibling(self, sibling_id: str, experience: Dict[str, Any]) -> bool:
        """
        Share an experience with a sibling robot.
        
        Args:
            sibling_id: Sibling robot ID
            experience: Experience to share
            
        Returns:
            True if shared successfully
        """
        # Form memory from experience
        memory_id = self.form_memory(experience, importance=0.7)
        
        # Share via Gun storage
        return self.memory_storage.share_with_sibling(memory_id, sibling_id)
    
    def learn_from_sibling(self, sibling_id: str) -> int:
        """
        Learn from a sibling robot's experiences.
        
        Args:
            sibling_id: Sibling robot ID
            
        Returns:
            Number of experiences learned
        """
        # Query sibling's shared memories
        sibling_memories = self.memory_storage.query_memories(owner=sibling_id)
        
        learned_count = 0
        for memory in sibling_memories[:10]:  # Learn from recent 10
            memory_data = self.memory_storage.retrieve_memory(memory.id)
            if memory_data and 'action' in memory_data and 'outcome' in memory_data:
                # Learn from sibling's experience
                self.learning_system.incremental_learning({
                    'state': memory_data.get('initial_state', {}),
                    'action': memory_data['action'],
                    'result': memory_data['outcome'],
                    'reward': memory_data.get('reward', 0) * 0.7  # Discounted
                })
                learned_count += 1
        
        if learned_count > 0:
            logger.info(f"Learned {learned_count} experiences from sibling {sibling_id}")
            
            # Increase social trait
            self.personality.update_trait(PersonalityTrait.SOCIAL, 0.05)
        
        return learned_count
    
    def get_self_awareness_report(self) -> Dict[str, Any]:
        """
        Generate a self-awareness report.
        
        Returns:
            Self-awareness metrics and status
        """
        # Update metrics
        self.self_awareness['age'] = time.time() - self.self_awareness.get('init_time', time.time())
        self.self_awareness['knowledge_concepts'] = len(self.learning_system.knowledge_base)
        self.self_awareness['confidence_level'] = np.mean([
            d.confidence for d in list(self.decision_history)[-10:]
        ]) if self.decision_history else 0.5
        
        # Get personality profile
        dominant_traits = self.personality.get_dominant_traits(3)
        
        # Get learning stats
        learning_stats = self.learning_system.get_learning_stats()
        
        # Get memory stats
        memory_stats = self.memory_storage.get_storage_stats()
        
        return {
            'identity': self.robot_id,
            'age_seconds': self.self_awareness['age'],
            'current_emotion': self.current_emotion.value,
            'personality': {
                'dominant_traits': [t.value for t in dominant_traits],
                'all_traits': {t.value: v for t, v in self.personality.traits.items()},
                'stability': self.self_awareness['personality_stability'],
                'decisions_made': self.personality.decisions_made,
                'memories_formed': self.personality.memories_formed,
                'mistakes_learned': self.personality.mistakes_learned
            },
            'cognitive_stats': {
                'total_experiences': self.self_awareness['experiences'],
                'knowledge_concepts': self.self_awareness['knowledge_concepts'],
                'confidence_level': self.self_awareness['confidence_level'],
                'learning_rate': self.learning_system.learning_rate,
                'exploration_rate': self.learning_system.exploration_rate
            },
            'memory_stats': memory_stats,
            'learning_mode': learning_stats['current_mode'],
            'reasoning_capabilities': {
                'objects_understood': len(self.reasoning_engine.objects),
                'patterns_recognized': len(self.reasoning_engine.patterns),
                'causal_relationships': len(self.reasoning_engine.causal_graph)
            }
        }
    
    def _find_similar_experiences(self, situation: Dict[str, Any],
                                 action: Dict[str, Any]) -> List[Experience]:
        """Find similar past experiences."""
        # Use learning system's experience buffer
        similar = []
        
        for exp in list(self.learning_system.experience_buffer)[-100:]:  # Check last 100
            if (exp.state.get('type') == situation.get('type') and
                exp.action and exp.action.get('type') == action.get('type')):
                similar.append(exp)
        
        return similar
    
    def _store_decision_memory(self, decision: Decision, situation: Dict[str, Any]):
        """Store an important decision in memory."""
        memory_data = {
            'type': 'decision',
            'decision_id': decision.id,
            'situation': situation,
            'action': decision.action,
            'reasoning': decision.reasoning,
            'confidence': decision.confidence,
            'emotion': self.current_emotion.value,
            'personality_snapshot': dict(self.personality.traits)
        }
        
        # Determine privacy level
        if decision.confidence < 0.3:  # Uncertain decisions are private
            self.memory_storage.store_private_memory(memory_data)
        else:
            self.memory_storage.store_shared_memory(
                memory_data,
                MemoryCategory.FAMILY
            )
    
    def start_autonomous_operation(self):
        """Start autonomous operation with background processes."""
        if self.running:
            return
        
        self.running = True
        self.self_awareness['init_time'] = time.time()
        
        # Start awareness monitoring
        self.awareness_thread = threading.Thread(
            target=self._awareness_monitor,
            daemon=True
        )
        self.awareness_thread.start()
        
        # Start periodic dreaming
        self.dream_thread = threading.Thread(
            target=self._dream_scheduler,
            daemon=True
        )
        self.dream_thread.start()
        
        logger.info("Autonomous operation started")
    
    def stop_autonomous_operation(self):
        """Stop autonomous operation."""
        self.running = False
        
        if self.awareness_thread:
            self.awareness_thread.join(timeout=5)
        
        if self.dream_thread:
            self.dream_thread.join(timeout=5)
        
        # Save state
        self.save_state()
        
        logger.info("Autonomous operation stopped")
    
    def _awareness_monitor(self):
        """Monitor self-awareness in background."""
        while self.running:
            try:
                # Periodic self-assessment
                if random.random() < 0.1:  # 10% chance each cycle
                    self._perform_self_assessment()
                
                # Check if tired (need to dream)
                if (self.current_emotion == EmotionalState.TIRED or
                    self.personality.decisions_made % 100 == 0):
                    self._request_dream_cycle = True
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Awareness monitor error: {e}")
    
    def _dream_scheduler(self):
        """Schedule periodic dream cycles."""
        while self.running:
            try:
                # Dream every hour of operation
                time.sleep(3600)
                
                if hasattr(self, '_request_dream_cycle') and self._request_dream_cycle:
                    self.dream_cycle(duration=50)
                    self._request_dream_cycle = False
                
            except Exception as e:
                logger.error(f"Dream scheduler error: {e}")
    
    def _perform_self_assessment(self):
        """Perform self-assessment of capabilities."""
        # Assess learning progress
        if self.learning_system.exploration_rate > 0.5:
            # Still exploring a lot - increase confidence slowly
            self.self_awareness['learning_rate'] *= 0.99
        else:
            # Exploiting more - knowledge is stabilizing
            self.self_awareness['personality_stability'] = min(
                1.0,
                self.self_awareness['personality_stability'] + 0.01
            )
        
        # Check for stagnation
        recent_decisions = list(self.decision_history)[-20:]
        if recent_decisions:
            unique_actions = set(d.action.get('type') for d in recent_decisions)
            if len(unique_actions) < 3:
                # Getting repetitive - increase curiosity
                self.current_emotion = EmotionalState.CURIOUS
                self.learning_system.exploration_rate = min(
                    0.5,
                    self.learning_system.exploration_rate * 1.1
                )
    
    def save_state(self, directory: Optional[str] = None):
        """Save complete brain state."""
        if directory is None:
            directory = f"./brain_state/{self.robot_id}"
        
        os.makedirs(directory, exist_ok=True)
        
        # Save each component
        self.reasoning_engine.save_state(f"{directory}/reasoning.json")
        self.learning_system.save_state(f"{directory}/learning.json")
        
        # Save brain-specific state
        brain_state = {
            'robot_id': self.robot_id,
            'current_emotion': self.current_emotion.value,
            'personality': {
                'traits': {t.value: v for t, v in self.personality.traits.items()},
                'preferences': self.personality.preferences,
                'decisions_made': self.personality.decisions_made,
                'memories_formed': self.personality.memories_formed,
                'mistakes_learned': self.personality.mistakes_learned
            },
            'self_awareness': self.self_awareness,
            'emotion_history': list(self.emotion_history)[-100:],
            'decision_count': len(self.decision_history)
        }
        
        with open(f"{directory}/brain.json", 'w') as f:
            json.dump(brain_state, f, indent=2)
        
        logger.info(f"Brain state saved to {directory}")
    
    def load_state(self, directory: Optional[str] = None):
        """Load brain state from files."""
        if directory is None:
            directory = f"./brain_state/{self.robot_id}"
        
        if not os.path.exists(directory):
            logger.warning(f"State directory {directory} not found")
            return
        
        # Load component states
        self.reasoning_engine.load_state(f"{directory}/reasoning.json")
        self.learning_system.load_state(f"{directory}/learning.json")
        
        # Load brain state
        brain_file = f"{directory}/brain.json"
        if os.path.exists(brain_file):
            with open(brain_file, 'r') as f:
                state = json.load(f)
            
            self.current_emotion = EmotionalState(state.get('current_emotion', 'curious'))
            
            # Restore personality
            personality_data = state.get('personality', {})
            for trait_name, value in personality_data.get('traits', {}).items():
                try:
                    trait = PersonalityTrait(trait_name)
                    self.personality.traits[trait] = value
                except ValueError:
                    pass
            
            self.personality.preferences = personality_data.get('preferences', {})
            self.personality.decisions_made = personality_data.get('decisions_made', 0)
            self.personality.memories_formed = personality_data.get('memories_formed', 0)
            self.personality.mistakes_learned = personality_data.get('mistakes_learned', 0)
            
            # Restore self-awareness
            self.self_awareness.update(state.get('self_awareness', {}))
            
            # Restore emotion history
            emotion_history = state.get('emotion_history', [])
            for entry in emotion_history:
                self.emotion_history.append(entry)
            
            logger.info(f"Brain state loaded from {directory}")