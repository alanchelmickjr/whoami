"""
Self-Learning System for autonomous robot learning.

This module provides autonomous learning capabilities including curiosity-driven
exploration, experience replay, self-supervised learning, and continuous improvement.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import json
import time
import random
from collections import deque
from enum import Enum
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Different modes of learning."""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    CONSOLIDATION = "consolidation"
    DREAMING = "dreaming"
    TRANSFER = "transfer"


@dataclass
class Experience:
    """Represents a single learning experience."""
    id: str
    state: Dict[str, Any]
    action: Optional[Dict[str, Any]]
    result: Dict[str, Any]
    reward: float
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0
    replay_count: int = 0
    
    def increase_importance(self, factor: float = 1.1):
        """Increase importance of this experience."""
        self.importance = min(10.0, self.importance * factor)
        self.replay_count += 1


@dataclass
class Knowledge:
    """Represents learned knowledge."""
    id: str
    concept: str
    confidence: float
    examples: List[Experience] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    
    def access(self):
        """Record knowledge access."""
        self.last_accessed = time.time()
        self.access_count += 1


class SelfLearningSystem:
    """
    Autonomous learning system for robots with curiosity-driven exploration.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the self-learning system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Experience replay buffer
        buffer_size = self.config.get('replay_buffer_size', 10000)
        self.experience_buffer = deque(maxlen=buffer_size)
        
        # Knowledge base
        self.knowledge_base: Dict[str, Knowledge] = {}
        
        # Learning parameters
        self.curiosity_threshold = self.config.get('curiosity_threshold', 0.5)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.exploration_rate = self.config.get('exploration_rate', 0.3)
        self.consolidation_interval = self.config.get('consolidation_interval', 100)
        
        # State tracking
        self.current_mode = LearningMode.EXPLORATION
        self.total_experiences = 0
        self.last_consolidation = 0
        self.mistakes: List[Dict[str, Any]] = []
        self.curiosity_map: Dict[str, float] = {}
        
        # Transfer learning mappings
        self.transfer_mappings: Dict[str, List[str]] = {}
        
        # Sleep/dream state
        self.is_sleeping = False
        self.dream_experiences: List[Experience] = []
        
        logger.info("SelfLearningSystem initialized")
    
    def explore_with_curiosity(self, current_state: Dict[str, Any],
                              possible_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Choose action based on curiosity-driven exploration.
        
        Args:
            current_state: Current state of the robot
            possible_actions: List of possible actions
            
        Returns:
            Selected action to take
        """
        if not possible_actions:
            return {}
        
        # Calculate curiosity scores for each action
        action_scores = []
        for action in possible_actions:
            curiosity_score = self._calculate_curiosity(current_state, action)
            action_scores.append((action, curiosity_score))
        
        # Sort by curiosity score
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Exploration vs exploitation
        if random.random() < self.exploration_rate:
            # Explore: choose based on curiosity
            weights = [score for _, score in action_scores]
            if sum(weights) > 0:
                weights = [w / sum(weights) for w in weights]
                selected_idx = np.random.choice(len(action_scores), p=weights)
                selected_action = action_scores[selected_idx][0]
            else:
                selected_action = random.choice(possible_actions)
            
            logger.info(f"Exploring with curiosity: {selected_action.get('type', 'unknown')}")
        else:
            # Exploit: choose best known action
            selected_action = self._choose_best_action(current_state, possible_actions)
            logger.info(f"Exploiting knowledge: {selected_action.get('type', 'unknown')}")
        
        return selected_action
    
    def _calculate_curiosity(self, state: Dict[str, Any],
                           action: Dict[str, Any]) -> float:
        """Calculate curiosity score for a state-action pair."""
        state_key = self._state_to_key(state)
        action_key = self._action_to_key(action)
        curiosity_key = f"{state_key}_{action_key}"
        
        # Check if we've explored this before
        if curiosity_key in self.curiosity_map:
            # Decay curiosity over time
            base_curiosity = self.curiosity_map[curiosity_key]
            decay = 0.99  # Slow decay
            self.curiosity_map[curiosity_key] = base_curiosity * decay
            return self.curiosity_map[curiosity_key]
        else:
            # New state-action pair: high curiosity
            initial_curiosity = 1.0
            self.curiosity_map[curiosity_key] = initial_curiosity
            return initial_curiosity
    
    def _choose_best_action(self, state: Dict[str, Any],
                           actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Choose best action based on learned knowledge."""
        best_action = None
        best_reward = float('-inf')
        
        for action in actions:
            expected_reward = self._estimate_reward(state, action)
            if expected_reward > best_reward:
                best_reward = expected_reward
                best_action = action
        
        return best_action or random.choice(actions)
    
    def _estimate_reward(self, state: Dict[str, Any],
                        action: Dict[str, Any]) -> float:
        """Estimate expected reward for state-action pair."""
        # Search experience buffer for similar experiences
        similar_experiences = self._find_similar_experiences(state, action)
        
        if not similar_experiences:
            return 0.0  # Unknown reward
        
        # Weight recent experiences more heavily
        weighted_rewards = []
        for exp in similar_experiences:
            age = time.time() - exp.timestamp
            weight = np.exp(-age / 3600)  # Decay over hours
            weighted_rewards.append(exp.reward * weight)
        
        return np.mean(weighted_rewards) if weighted_rewards else 0.0
    
    def record_experience(self, state: Dict[str, Any],
                         action: Dict[str, Any],
                         result: Dict[str, Any],
                         reward: float):
        """
        Record a learning experience.
        
        Args:
            state: Initial state
            action: Action taken
            result: Resulting state/outcome
            reward: Reward received
        """
        experience = Experience(
            id=f"exp_{self.total_experiences}",
            state=state,
            action=action,
            result=result,
            reward=reward
        )
        
        # Add to replay buffer
        self.experience_buffer.append(experience)
        self.total_experiences += 1
        
        # Update curiosity map
        state_key = self._state_to_key(state)
        action_key = self._action_to_key(action)
        curiosity_key = f"{state_key}_{action_key}"
        
        if curiosity_key in self.curiosity_map:
            # Reduce curiosity after experiencing
            self.curiosity_map[curiosity_key] *= 0.8
        
        # Check for mistakes
        if reward < -0.5:  # Negative reward threshold
            self._record_mistake(experience)
        
        # Trigger consolidation if needed
        if self.total_experiences - self.last_consolidation > self.consolidation_interval:
            self.consolidate_knowledge()
        
        logger.info(f"Recorded experience {experience.id} with reward {reward}")
    
    def replay_experiences(self, batch_size: int = 32) -> List[Experience]:
        """
        Replay important experiences for learning.
        
        Args:
            batch_size: Number of experiences to replay
            
        Returns:
            List of replayed experiences
        """
        if len(self.experience_buffer) < batch_size:
            return list(self.experience_buffer)
        
        # Prioritized experience replay
        experiences = list(self.experience_buffer)
        
        # Sort by importance and recency
        def priority_score(exp):
            age = time.time() - exp.timestamp
            recency = np.exp(-age / 3600)  # Decay over hours
            return exp.importance * recency * abs(exp.reward)
        
        experiences.sort(key=priority_score, reverse=True)
        
        # Select batch with some randomness
        selected = []
        for i in range(min(batch_size, len(experiences))):
            if random.random() < 0.8:  # 80% chance to pick by priority
                selected.append(experiences[i])
            else:  # 20% random selection
                selected.append(random.choice(experiences))
        
        # Increase importance of replayed experiences
        for exp in selected:
            exp.increase_importance()
        
        return selected
    
    def learn_from_experience(self, experience: Experience) -> Dict[str, Any]:
        """
        Learn from a single experience.
        
        Args:
            experience: Experience to learn from
            
        Returns:
            Learning outcome dictionary
        """
        # Extract patterns
        patterns = self._extract_patterns(experience)
        
        # Update knowledge base
        for pattern in patterns:
            concept_name = pattern['concept']
            
            if concept_name not in self.knowledge_base:
                self.knowledge_base[concept_name] = Knowledge(
                    id=f"knowledge_{len(self.knowledge_base)}",
                    concept=concept_name,
                    confidence=0.5
                )
            
            knowledge = self.knowledge_base[concept_name]
            knowledge.examples.append(experience)
            
            # Update confidence based on consistency
            if experience.reward > 0:
                knowledge.confidence = min(1.0, knowledge.confidence + self.learning_rate)
            else:
                knowledge.confidence = max(0.0, knowledge.confidence - self.learning_rate * 0.5)
        
        # Check for transfer learning opportunities
        transfer_concepts = self._identify_transfer_opportunities(experience)
        
        return {
            'patterns_learned': len(patterns),
            'concepts_updated': [p['concept'] for p in patterns],
            'transfer_opportunities': transfer_concepts
        }
    
    def _extract_patterns(self, experience: Experience) -> List[Dict[str, Any]]:
        """Extract learnable patterns from experience."""
        patterns = []
        
        # Object interaction patterns
        if 'object' in experience.state:
            patterns.append({
                'concept': f"object_{experience.state['object'].get('type', 'unknown')}",
                'pattern': 'object_interaction',
                'features': experience.state['object']
            })
        
        # Action-outcome patterns
        if experience.action:
            action_type = experience.action.get('type', 'unknown')
            outcome_type = experience.result.get('outcome', 'unknown')
            patterns.append({
                'concept': f"action_{action_type}_causes_{outcome_type}",
                'pattern': 'causal',
                'features': {
                    'action': action_type,
                    'outcome': outcome_type,
                    'reward': experience.reward
                }
            })
        
        # Spatial patterns
        if 'position' in experience.state:
            patterns.append({
                'concept': 'spatial_awareness',
                'pattern': 'spatial',
                'features': experience.state['position']
            })
        
        return patterns
    
    def detect_and_correct_mistakes(self):
        """
        Detect mistakes and self-correct.
        """
        if not self.mistakes:
            return
        
        # Analyze recent mistakes
        recent_mistakes = [m for m in self.mistakes 
                          if time.time() - m['timestamp'] < 3600]  # Last hour
        
        if not recent_mistakes:
            return
        
        # Find common patterns in mistakes
        mistake_patterns = {}
        for mistake in recent_mistakes:
            pattern_key = f"{mistake['state_type']}_{mistake['action_type']}"
            if pattern_key not in mistake_patterns:
                mistake_patterns[pattern_key] = []
            mistake_patterns[pattern_key].append(mistake)
        
        # Generate corrections
        for pattern_key, mistakes in mistake_patterns.items():
            if len(mistakes) >= 2:  # Repeated mistake
                logger.info(f"Detected repeated mistake pattern: {pattern_key}")
                
                # Create negative knowledge
                concept_name = f"avoid_{pattern_key}"
                if concept_name not in self.knowledge_base:
                    self.knowledge_base[concept_name] = Knowledge(
                        id=f"knowledge_{len(self.knowledge_base)}",
                        concept=concept_name,
                        confidence=0.8
                    )
                
                # Reduce exploration rate temporarily
                self.exploration_rate = max(0.1, self.exploration_rate * 0.9)
        
        # Clear old mistakes
        self.mistakes = recent_mistakes
    
    def _record_mistake(self, experience: Experience):
        """Record a mistake for later analysis."""
        mistake = {
            'experience_id': experience.id,
            'state_type': experience.state.get('type', 'unknown'),
            'action_type': experience.action.get('type', 'unknown') if experience.action else 'none',
            'reward': experience.reward,
            'timestamp': experience.timestamp
        }
        self.mistakes.append(mistake)
        logger.info(f"Mistake recorded: {mistake['action_type']} in {mistake['state_type']}")
    
    def consolidate_knowledge(self):
        """
        Consolidate knowledge during 'sleep' cycles.
        """
        logger.info("Starting knowledge consolidation...")
        self.current_mode = LearningMode.CONSOLIDATION
        
        # Replay important experiences
        important_experiences = self.replay_experiences(batch_size=50)
        
        # Re-learn from important experiences
        for exp in important_experiences:
            self.learn_from_experience(exp)
        
        # Prune low-confidence knowledge
        to_remove = []
        for concept_name, knowledge in self.knowledge_base.items():
            # Remove rarely accessed, low-confidence knowledge
            age = time.time() - knowledge.created_at
            if (knowledge.confidence < 0.3 and 
                knowledge.access_count < 2 and 
                age > 3600):  # Older than 1 hour
                to_remove.append(concept_name)
        
        for concept_name in to_remove:
            del self.knowledge_base[concept_name]
            logger.info(f"Pruned low-confidence knowledge: {concept_name}")
        
        # Detect and correct mistakes
        self.detect_and_correct_mistakes()
        
        # Update transfer mappings
        self._update_transfer_mappings()
        
        self.last_consolidation = self.total_experiences
        self.current_mode = LearningMode.EXPLORATION
        logger.info("Knowledge consolidation complete")
    
    def dream_mode(self, duration_steps: int = 100):
        """
        Enter dream mode for offline learning.
        
        Args:
            duration_steps: Number of dream steps
        """
        logger.info("Entering dream mode...")
        self.is_sleeping = True
        self.current_mode = LearningMode.DREAMING
        self.dream_experiences = []
        
        for step in range(duration_steps):
            # Generate synthetic experiences from existing knowledge
            synthetic_exp = self._generate_synthetic_experience()
            
            if synthetic_exp:
                # Learn from synthetic experience
                self.learn_from_experience(synthetic_exp)
                self.dream_experiences.append(synthetic_exp)
            
            # Consolidate every 20 steps
            if step % 20 == 0:
                self.consolidate_knowledge()
        
        self.is_sleeping = False
        self.current_mode = LearningMode.EXPLORATION
        logger.info(f"Dream mode complete. Generated {len(self.dream_experiences)} experiences")
    
    def _generate_synthetic_experience(self) -> Optional[Experience]:
        """Generate synthetic experience from existing knowledge."""
        if len(self.experience_buffer) < 10:
            return None
        
        # Randomly combine elements from different experiences
        exp1 = random.choice(list(self.experience_buffer))
        exp2 = random.choice(list(self.experience_buffer))
        
        # Create hybrid experience
        synthetic_state = {**exp1.state}
        synthetic_action = exp2.action if exp2.action else exp1.action
        
        # Estimate result based on similar experiences
        similar_exps = self._find_similar_experiences(synthetic_state, synthetic_action)
        
        if similar_exps:
            # Average the results
            synthetic_result = similar_exps[0].result
            synthetic_reward = np.mean([e.reward for e in similar_exps])
        else:
            synthetic_result = exp1.result
            synthetic_reward = (exp1.reward + exp2.reward) / 2
        
        return Experience(
            id=f"dream_{time.time()}",
            state=synthetic_state,
            action=synthetic_action,
            result=synthetic_result,
            reward=synthetic_reward,
            importance=0.5  # Lower importance for synthetic experiences
        )
    
    def transfer_learning(self, source_concept: str,
                         target_concept: str,
                         similarity: float = 0.7):
        """
        Transfer learning between similar concepts.
        
        Args:
            source_concept: Source concept to transfer from
            target_concept: Target concept to transfer to
            similarity: Similarity threshold
        """
        if source_concept not in self.knowledge_base:
            logger.warning(f"Source concept {source_concept} not found")
            return
        
        source_knowledge = self.knowledge_base[source_concept]
        
        # Create or update target knowledge
        if target_concept not in self.knowledge_base:
            self.knowledge_base[target_concept] = Knowledge(
                id=f"knowledge_{len(self.knowledge_base)}",
                concept=target_concept,
                confidence=source_knowledge.confidence * similarity
            )
        
        target_knowledge = self.knowledge_base[target_concept]
        
        # Transfer relevant experiences
        for exp in source_knowledge.examples[-10:]:  # Transfer last 10 examples
            # Adjust experience for target concept
            transferred_exp = Experience(
                id=f"transfer_{exp.id}",
                state={**exp.state, 'transferred_from': source_concept},
                action=exp.action,
                result=exp.result,
                reward=exp.reward * similarity,  # Adjust reward by similarity
                importance=exp.importance * similarity
            )
            target_knowledge.examples.append(transferred_exp)
        
        # Update transfer mappings
        if source_concept not in self.transfer_mappings:
            self.transfer_mappings[source_concept] = []
        self.transfer_mappings[source_concept].append(target_concept)
        
        logger.info(f"Transferred learning from {source_concept} to {target_concept}")
    
    def _identify_transfer_opportunities(self, experience: Experience) -> List[str]:
        """Identify opportunities for transfer learning."""
        opportunities = []
        
        # Look for similar concepts in knowledge base
        exp_patterns = self._extract_patterns(experience)
        
        for pattern in exp_patterns:
            concept = pattern['concept']
            
            # Find similar concepts
            for known_concept in self.knowledge_base:
                if known_concept != concept:
                    similarity = self._calculate_concept_similarity(concept, known_concept)
                    if similarity > 0.7:
                        opportunities.append(known_concept)
        
        return opportunities
    
    def _calculate_concept_similarity(self, concept1: str, concept2: str) -> float:
        """Calculate similarity between two concepts."""
        # Simple string similarity
        words1 = set(concept1.lower().split('_'))
        words2 = set(concept2.lower().split('_'))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _update_transfer_mappings(self):
        """Update transfer learning mappings based on knowledge base."""
        for concept1 in self.knowledge_base:
            for concept2 in self.knowledge_base:
                if concept1 != concept2:
                    similarity = self._calculate_concept_similarity(concept1, concept2)
                    if similarity > 0.8:
                        if concept1 not in self.transfer_mappings:
                            self.transfer_mappings[concept1] = []
                        if concept2 not in self.transfer_mappings[concept1]:
                            self.transfer_mappings[concept1].append(concept2)
    
    def incremental_learning(self, new_data: Dict[str, Any]) -> bool:
        """
        Continuously improve knowledge with new data.
        
        Args:
            new_data: New data to learn from
            
        Returns:
            True if learning was successful
        """
        try:
            # Create experience from new data
            experience = Experience(
                id=f"incremental_{time.time()}",
                state=new_data.get('state', {}),
                action=new_data.get('action', {}),
                result=new_data.get('result', {}),
                reward=new_data.get('reward', 0.0)
            )
            
            # Add to buffer
            self.experience_buffer.append(experience)
            
            # Learn immediately if important
            if abs(experience.reward) > 0.5:
                self.learn_from_experience(experience)
            
            # Adjust learning parameters
            self.learning_rate *= 0.999  # Slow decay
            
            return True
            
        except Exception as e:
            logger.error(f"Incremental learning failed: {e}")
            return False
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get current learning statistics.
        
        Returns:
            Dictionary of learning statistics
        """
        return {
            'total_experiences': self.total_experiences,
            'buffer_size': len(self.experience_buffer),
            'knowledge_concepts': len(self.knowledge_base),
            'mistakes_recorded': len(self.mistakes),
            'current_mode': self.current_mode.value,
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate,
            'curiosity_locations': len(self.curiosity_map),
            'transfer_mappings': len(self.transfer_mappings),
            'is_sleeping': self.is_sleeping,
            'dream_experiences': len(self.dream_experiences)
        }
    
    def _find_similar_experiences(self, state: Dict[str, Any],
                                 action: Optional[Dict[str, Any]]) -> List[Experience]:
        """Find similar experiences in buffer."""
        similar = []
        
        state_key = self._state_to_key(state)
        action_key = self._action_to_key(action) if action else None
        
        for exp in self.experience_buffer:
            exp_state_key = self._state_to_key(exp.state)
            exp_action_key = self._action_to_key(exp.action) if exp.action else None
            
            # Check similarity
            if exp_state_key == state_key:
                if action_key is None or exp_action_key == action_key:
                    similar.append(exp)
        
        return similar
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state to string key."""
        key_parts = []
        for k in sorted(['type', 'object', 'position']):
            if k in state:
                if isinstance(state[k], dict):
                    key_parts.append(f"{k}_{state[k].get('type', 'unknown')}")
                else:
                    key_parts.append(f"{k}_{state[k]}")
        return "_".join(key_parts) if key_parts else "unknown_state"
    
    def _action_to_key(self, action: Optional[Dict[str, Any]]) -> str:
        """Convert action to string key."""
        if not action:
            return "no_action"
        
        return f"action_{action.get('type', 'unknown')}_{action.get('target', 'unknown')}"
    
    def save_state(self, filepath: str):
        """Save learning system state to file."""
        state = {
            'config': self.config,
            'total_experiences': self.total_experiences,
            'knowledge_base': {
                k: {
                    'id': v.id,
                    'concept': v.concept,
                    'confidence': v.confidence,
                    'created_at': v.created_at,
                    'access_count': v.access_count
                } for k, v in self.knowledge_base.items()
            },
            'curiosity_map': self.curiosity_map,
            'transfer_mappings': self.transfer_mappings,
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save experience buffer separately
        buffer_file = filepath.replace('.json', '_buffer.pkl')
        with open(buffer_file, 'wb') as f:
            pickle.dump(list(self.experience_buffer), f)
        
        logger.info(f"Learning system state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load learning system state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.config = state.get('config', {})
        self.total_experiences = state.get('total_experiences', 0)
        self.curiosity_map = state.get('curiosity_map', {})
        self.transfer_mappings = state.get('transfer_mappings', {})
        self.exploration_rate = state.get('exploration_rate', 0.3)
        self.learning_rate = state.get('learning_rate', 0.01)
        
        # Reconstruct knowledge base
        self.knowledge_base = {}
        for k, v in state.get('knowledge_base', {}).items():
            self.knowledge_base[k] = Knowledge(
                id=v['id'],
                concept=v['concept'],
                confidence=v['confidence'],
                created_at=v.get('created_at', time.time()),
                access_count=v.get('access_count', 0)
            )
        
        # Load experience buffer
        buffer_file = filepath.replace('.json', '_buffer.pkl')
        try:
            with open(buffer_file, 'rb') as f:
                experiences = pickle.load(f)
                self.experience_buffer = deque(experiences, maxlen=self.config.get('replay_buffer_size', 10000))
        except FileNotFoundError:
            logger.warning(f"Experience buffer file {buffer_file} not found")
        
        logger.info(f"Learning system state loaded from {filepath}")