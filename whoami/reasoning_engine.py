"""
Reasoning Engine for spatial and object reasoning with hypothesis generation.

This module provides advanced reasoning capabilities for robots to understand
incomplete information, predict unseen parts, and reason about the world.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import time
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """Represents a hypothesis about an object or situation."""
    id: str
    description: str
    confidence: float
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def update_confidence(self, new_evidence: Dict[str, Any], adjustment: float):
        """Update confidence based on new evidence."""
        self.evidence.append(new_evidence)
        self.confidence = max(0.0, min(1.0, self.confidence + adjustment))


@dataclass
class Object3D:
    """Represents a 3D object with partial or complete information."""
    id: str
    name: Optional[str] = None
    points: Optional[np.ndarray] = None  # 3D point cloud
    completed_points: Optional[np.ndarray] = None  # Filled-in missing parts
    confidence_map: Optional[np.ndarray] = None  # Confidence for each point
    last_seen: float = field(default_factory=time.time)
    permanence_score: float = 1.0  # Object permanence understanding
    views: List[np.ndarray] = field(default_factory=list)  # Multiple perspectives


class ReasoningEngine:
    """
    Advanced reasoning engine for spatial understanding and object reasoning.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the reasoning engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.objects: Dict[str, Object3D] = {}
        self.hypotheses: Dict[str, List[Hypothesis]] = {}
        self.patterns: Dict[str, Dict[str, Any]] = {}
        self.causal_graph: Dict[str, List[str]] = {}  # Cause -> [Effects]
        self.observation_history = deque(maxlen=1000)
        
        # Reasoning parameters
        self.min_confidence = self.config.get('min_confidence', 0.3)
        self.hypothesis_threshold = self.config.get('hypothesis_threshold', 0.5)
        self.pattern_threshold = self.config.get('pattern_threshold', 3)
        
        logger.info("ReasoningEngine initialized")
    
    def complete_object(self, object_id: str, partial_points: np.ndarray,
                       context: Optional[Dict] = None) -> Tuple[np.ndarray, float]:
        """
        Complete missing parts of a 3D object from partial scan.
        
        Args:
            object_id: Unique identifier for the object
            partial_points: Partial 3D point cloud
            context: Additional context information
            
        Returns:
            Completed point cloud and confidence score
        """
        logger.info(f"Completing object {object_id}")
        
        # Store or update object
        if object_id not in self.objects:
            self.objects[object_id] = Object3D(id=object_id, points=partial_points)
        else:
            self.objects[object_id].points = partial_points
            self.objects[object_id].last_seen = time.time()
        
        # Generate hypotheses about missing parts
        hypotheses = self._generate_object_hypotheses(partial_points, context)
        
        # Use pattern matching to predict missing parts
        completed_points = self._predict_missing_parts(partial_points, hypotheses)
        
        # Calculate confidence based on hypothesis strength and pattern matches
        confidence = self._calculate_completion_confidence(
            partial_points, completed_points, hypotheses
        )
        
        # Store completed object
        self.objects[object_id].completed_points = completed_points
        self.objects[object_id].confidence_map = self._create_confidence_map(
            partial_points, completed_points, confidence
        )
        
        return completed_points, confidence
    
    def _generate_object_hypotheses(self, points: np.ndarray,
                                   context: Optional[Dict]) -> List[Hypothesis]:
        """Generate hypotheses about what the complete object might be."""
        hypotheses = []
        
        # Analyze point cloud characteristics
        bbox = self._calculate_bounding_box(points)
        density = len(points) / np.prod(bbox[1] - bbox[0])
        
        # Generate shape hypotheses
        if self._is_likely_spherical(points):
            hyp = Hypothesis(
                id=f"sphere_{time.time()}",
                description="Object appears spherical",
                confidence=0.7,
                evidence=[{"type": "shape_analysis", "data": "spherical_pattern"}]
            )
            hypotheses.append(hyp)
        
        if self._is_likely_cubic(points):
            hyp = Hypothesis(
                id=f"cube_{time.time()}",
                description="Object appears cubic",
                confidence=0.6,
                evidence=[{"type": "shape_analysis", "data": "cubic_pattern"}]
            )
            hypotheses.append(hyp)
        
        # Check against known patterns
        for pattern_name, pattern_data in self.patterns.items():
            similarity = self._calculate_pattern_similarity(points, pattern_data)
            if similarity > self.hypothesis_threshold:
                hyp = Hypothesis(
                    id=f"pattern_{pattern_name}_{time.time()}",
                    description=f"Matches pattern: {pattern_name}",
                    confidence=similarity,
                    evidence=[{"type": "pattern_match", "pattern": pattern_name}]
                )
                hypotheses.append(hyp)
        
        return hypotheses
    
    def _predict_missing_parts(self, partial_points: np.ndarray,
                              hypotheses: List[Hypothesis]) -> np.ndarray:
        """Predict missing parts based on hypotheses."""
        if not hypotheses:
            # Simple symmetry-based completion
            return self._complete_by_symmetry(partial_points)
        
        # Weight predictions by hypothesis confidence
        completed = partial_points.copy()
        
        for hyp in hypotheses:
            if "sphere" in hyp.description.lower():
                sphere_completion = self._complete_as_sphere(partial_points)
                completed = self._merge_completions(
                    completed, sphere_completion, hyp.confidence
                )
            elif "cube" in hyp.description.lower():
                cube_completion = self._complete_as_cube(partial_points)
                completed = self._merge_completions(
                    completed, cube_completion, hyp.confidence
                )
        
        return completed
    
    def recognize_pattern(self, observations: List[Dict[str, Any]]) -> Optional[str]:
        """
        Recognize patterns from repeated observations.
        
        Args:
            observations: List of observation dictionaries
            
        Returns:
            Pattern name if recognized, None otherwise
        """
        # Add to observation history
        self.observation_history.extend(observations)
        
        # Look for repeating sequences
        pattern_candidates = self._find_repeating_sequences(observations)
        
        for candidate in pattern_candidates:
            if candidate['count'] >= self.pattern_threshold:
                pattern_name = f"pattern_{len(self.patterns)}"
                self.patterns[pattern_name] = {
                    'sequence': candidate['sequence'],
                    'count': candidate['count'],
                    'confidence': candidate['confidence']
                }
                logger.info(f"New pattern recognized: {pattern_name}")
                return pattern_name
        
        return None
    
    def reason_multi_view(self, object_id: str, views: List[np.ndarray]) -> Dict[str, Any]:
        """
        Combine multiple perspectives to understand an object better.
        
        Args:
            object_id: Object identifier
            views: List of point clouds from different angles
            
        Returns:
            Combined understanding dictionary
        """
        if object_id not in self.objects:
            self.objects[object_id] = Object3D(id=object_id)
        
        obj = self.objects[object_id]
        obj.views.extend(views)
        
        # Merge all views into unified point cloud
        merged_points = self._merge_point_clouds(views)
        
        # Calculate view consistency
        consistency = self._calculate_view_consistency(views)
        
        # Update object representation
        obj.points = merged_points
        obj.completed_points, confidence = self.complete_object(
            object_id, merged_points
        )
        
        return {
            'object_id': object_id,
            'merged_points': merged_points,
            'num_views': len(views),
            'consistency': consistency,
            'confidence': confidence,
            'completeness': self._estimate_completeness(merged_points)
        }
    
    def understand_object_permanence(self, object_id: str,
                                    currently_visible: bool) -> float:
        """
        Understand that objects exist even when not visible.
        
        Args:
            object_id: Object identifier
            currently_visible: Whether object is currently visible
            
        Returns:
            Permanence score (0-1)
        """
        if object_id not in self.objects:
            return 0.0
        
        obj = self.objects[object_id]
        time_since_seen = time.time() - obj.last_seen
        
        if currently_visible:
            obj.last_seen = time.time()
            obj.permanence_score = min(1.0, obj.permanence_score + 0.1)
        else:
            # Decay permanence score over time if not seen
            decay_rate = 0.001  # Slow decay
            obj.permanence_score = max(
                0.0,
                obj.permanence_score - (decay_rate * time_since_seen)
            )
        
        return obj.permanence_score
    
    def reason_causality(self, event_a: Dict[str, Any],
                        event_b: Dict[str, Any]) -> float:
        """
        Understand cause and effect relationships.
        
        Args:
            event_a: Potential cause event
            event_b: Potential effect event
            
        Returns:
            Causal confidence score (0-1)
        """
        # Check temporal ordering
        if event_a.get('timestamp', 0) >= event_b.get('timestamp', 0):
            return 0.0  # Effect cannot precede cause
        
        # Look for patterns in observation history
        causal_score = 0.0
        pattern_count = 0
        
        for i in range(len(self.observation_history) - 1):
            if (self._events_match(self.observation_history[i], event_a) and
                self._events_match(self.observation_history[i + 1], event_b)):
                pattern_count += 1
        
        if pattern_count >= 2:
            causal_score = min(1.0, pattern_count / 10.0)
            
            # Update causal graph
            cause_key = self._event_to_key(event_a)
            effect_key = self._event_to_key(event_b)
            
            if cause_key not in self.causal_graph:
                self.causal_graph[cause_key] = []
            
            if effect_key not in self.causal_graph[cause_key]:
                self.causal_graph[cause_key].append(effect_key)
                logger.info(f"Causal relationship learned: {cause_key} -> {effect_key}")
        
        return causal_score
    
    def generate_hypothesis(self, observation: Dict[str, Any]) -> List[Hypothesis]:
        """
        Generate hypotheses about what might happen or what something might be.
        
        Args:
            observation: Current observation
            
        Returns:
            List of generated hypotheses
        """
        hypotheses = []
        
        # Check causal graph for predictions
        obs_key = self._event_to_key(observation)
        if obs_key in self.causal_graph:
            for effect in self.causal_graph[obs_key]:
                hyp = Hypothesis(
                    id=f"causal_{time.time()}",
                    description=f"Predicting effect: {effect}",
                    confidence=0.6,
                    evidence=[{"type": "causal", "cause": obs_key}]
                )
                hypotheses.append(hyp)
        
        # Generate object-based hypotheses
        if 'object_id' in observation:
            obj_hyps = self._generate_object_hypotheses(
                observation.get('points', np.array([])),
                observation
            )
            hypotheses.extend(obj_hyps)
        
        return hypotheses
    
    def calculate_confidence(self, prediction: Dict[str, Any],
                           actual: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate confidence score for a prediction.
        
        Args:
            prediction: The prediction made
            actual: Actual outcome (if available)
            
        Returns:
            Confidence score (0-1)
        """
        base_confidence = prediction.get('confidence', 0.5)
        
        if actual is not None:
            # Adjust confidence based on accuracy
            accuracy = self._calculate_prediction_accuracy(prediction, actual)
            return base_confidence * accuracy
        
        # Use historical accuracy for similar predictions
        similar_predictions = self._find_similar_predictions(prediction)
        if similar_predictions:
            historical_accuracy = np.mean([
                p.get('accuracy', 0.5) for p in similar_predictions
            ])
            return base_confidence * historical_accuracy
        
        return base_confidence
    
    # Helper methods
    def _calculate_bounding_box(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate bounding box of point cloud."""
        if len(points) == 0:
            return np.zeros(3), np.zeros(3)
        return np.min(points, axis=0), np.max(points, axis=0)
    
    def _is_likely_spherical(self, points: np.ndarray) -> bool:
        """Check if points likely form a sphere."""
        if len(points) < 10:
            return False
        
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        std_dev = np.std(distances)
        
        return std_dev < 0.2 * np.mean(distances)
    
    def _is_likely_cubic(self, points: np.ndarray) -> bool:
        """Check if points likely form a cube."""
        if len(points) < 8:
            return False
        
        bbox_min, bbox_max = self._calculate_bounding_box(points)
        dimensions = bbox_max - bbox_min
        
        # Check if dimensions are roughly equal (cube-like)
        return np.std(dimensions) < 0.2 * np.mean(dimensions)
    
    def _calculate_pattern_similarity(self, points: np.ndarray,
                                     pattern: Dict[str, Any]) -> float:
        """Calculate similarity between points and a known pattern."""
        if 'points' not in pattern:
            return 0.0
        
        pattern_points = pattern['points']
        if isinstance(pattern_points, list):
            pattern_points = np.array(pattern_points)
        
        # Simple distance-based similarity
        if len(points) == 0 or len(pattern_points) == 0:
            return 0.0
        
        # Normalize both point clouds
        points_norm = points - np.mean(points, axis=0)
        pattern_norm = pattern_points - np.mean(pattern_points, axis=0)
        
        # Calculate similarity (simplified)
        similarity = 1.0 / (1.0 + np.mean(np.abs(points_norm - pattern_norm[:len(points)])))
        
        return min(1.0, similarity)
    
    def _complete_by_symmetry(self, points: np.ndarray) -> np.ndarray:
        """Complete object using symmetry assumptions."""
        if len(points) == 0:
            return points
        
        center = np.mean(points, axis=0)
        
        # Mirror points across center
        mirrored = 2 * center - points
        
        return np.vstack([points, mirrored])
    
    def _complete_as_sphere(self, points: np.ndarray) -> np.ndarray:
        """Complete points as a sphere."""
        if len(points) == 0:
            return points
        
        center = np.mean(points, axis=0)
        radius = np.mean(np.linalg.norm(points - center, axis=1))
        
        # Generate sphere points
        theta = np.linspace(0, 2 * np.pi, 20)
        phi = np.linspace(0, np.pi, 10)
        
        sphere_points = []
        for t in theta:
            for p in phi:
                x = center[0] + radius * np.sin(p) * np.cos(t)
                y = center[1] + radius * np.sin(p) * np.sin(t)
                z = center[2] + radius * np.cos(p)
                sphere_points.append([x, y, z])
        
        return np.vstack([points, np.array(sphere_points)])
    
    def _complete_as_cube(self, points: np.ndarray) -> np.ndarray:
        """Complete points as a cube."""
        if len(points) == 0:
            return points
        
        bbox_min, bbox_max = self._calculate_bounding_box(points)
        
        # Generate cube vertices
        vertices = []
        for x in [bbox_min[0], bbox_max[0]]:
            for y in [bbox_min[1], bbox_max[1]]:
                for z in [bbox_min[2], bbox_max[2]]:
                    vertices.append([x, y, z])
        
        return np.vstack([points, np.array(vertices)])
    
    def _merge_completions(self, base: np.ndarray, new: np.ndarray,
                          weight: float) -> np.ndarray:
        """Merge two point cloud completions with weighting."""
        # Simple weighted average
        if len(base) == 0:
            return new
        if len(new) == 0:
            return base
        
        # Sample points from new completion based on weight
        num_samples = int(len(new) * weight)
        if num_samples > 0:
            indices = np.random.choice(len(new), num_samples, replace=False)
            sampled = new[indices]
            return np.vstack([base, sampled])
        
        return base
    
    def _calculate_completion_confidence(self, partial: np.ndarray,
                                        completed: np.ndarray,
                                        hypotheses: List[Hypothesis]) -> float:
        """Calculate confidence in object completion."""
        if len(hypotheses) == 0:
            return 0.3  # Low confidence without hypotheses
        
        # Average hypothesis confidence
        avg_confidence = np.mean([h.confidence for h in hypotheses])
        
        # Adjust based on completion ratio
        completion_ratio = len(completed) / max(1, len(partial))
        if completion_ratio > 10:  # Too much completion
            avg_confidence *= 0.5
        
        return min(1.0, avg_confidence)
    
    def _create_confidence_map(self, partial: np.ndarray,
                              completed: np.ndarray,
                              confidence: float) -> np.ndarray:
        """Create confidence map for each point."""
        confidence_map = np.zeros(len(completed))
        
        # Original points have high confidence
        confidence_map[:len(partial)] = 1.0
        
        # Completed points have lower confidence
        if len(completed) > len(partial):
            confidence_map[len(partial):] = confidence
        
        return confidence_map
    
    def _find_repeating_sequences(self, observations: List[Dict[str, Any]]) -> List[Dict]:
        """Find repeating sequences in observations."""
        sequences = []
        
        for length in range(2, min(10, len(observations) // 2)):
            for start in range(len(observations) - length * 2):
                seq = observations[start:start + length]
                
                # Count occurrences
                count = 0
                for i in range(start + length, len(observations) - length + 1):
                    if self._sequences_match(seq, observations[i:i + length]):
                        count += 1
                
                if count >= 2:
                    sequences.append({
                        'sequence': seq,
                        'count': count,
                        'confidence': count / (len(observations) / length)
                    })
        
        return sequences
    
    def _sequences_match(self, seq1: List[Dict], seq2: List[Dict]) -> bool:
        """Check if two sequences match."""
        if len(seq1) != len(seq2):
            return False
        
        for e1, e2 in zip(seq1, seq2):
            if not self._events_match(e1, e2):
                return False
        
        return True
    
    def _events_match(self, event1: Dict, event2: Dict) -> bool:
        """Check if two events match."""
        # Compare key attributes
        for key in ['type', 'object_id', 'action']:
            if key in event1 and key in event2:
                if event1[key] != event2[key]:
                    return False
        
        return True
    
    def _event_to_key(self, event: Dict) -> str:
        """Convert event to string key."""
        return f"{event.get('type', 'unknown')}_{event.get('action', 'unknown')}"
    
    def _merge_point_clouds(self, views: List[np.ndarray]) -> np.ndarray:
        """Merge multiple point cloud views."""
        if not views:
            return np.array([])
        
        # Simple concatenation with duplicate removal
        all_points = np.vstack(views)
        
        # Remove duplicates (simplified)
        unique_points = np.unique(all_points, axis=0)
        
        return unique_points
    
    def _calculate_view_consistency(self, views: List[np.ndarray]) -> float:
        """Calculate consistency between multiple views."""
        if len(views) < 2:
            return 1.0
        
        # Calculate overlap between views
        overlaps = []
        for i in range(len(views) - 1):
            for j in range(i + 1, len(views)):
                overlap = self._calculate_overlap(views[i], views[j])
                overlaps.append(overlap)
        
        return np.mean(overlaps) if overlaps else 0.0
    
    def _calculate_overlap(self, points1: np.ndarray, points2: np.ndarray) -> float:
        """Calculate overlap between two point clouds."""
        if len(points1) == 0 or len(points2) == 0:
            return 0.0
        
        # Simplified overlap calculation
        bbox1_min, bbox1_max = self._calculate_bounding_box(points1)
        bbox2_min, bbox2_max = self._calculate_bounding_box(points2)
        
        # Calculate intersection
        inter_min = np.maximum(bbox1_min, bbox2_min)
        inter_max = np.minimum(bbox1_max, bbox2_max)
        
        if np.any(inter_max < inter_min):
            return 0.0
        
        inter_volume = np.prod(inter_max - inter_min)
        union_volume = (np.prod(bbox1_max - bbox1_min) + 
                       np.prod(bbox2_max - bbox2_min) - inter_volume)
        
        return inter_volume / max(union_volume, 1e-6)
    
    def _estimate_completeness(self, points: np.ndarray) -> float:
        """Estimate how complete an object scan is."""
        if len(points) < 10:
            return 0.1
        
        # Check point distribution
        bbox_min, bbox_max = self._calculate_bounding_box(points)
        volume = np.prod(bbox_max - bbox_min)
        
        if volume == 0:
            return 0.1
        
        density = len(points) / volume
        
        # Normalize to 0-1 (assuming reasonable density)
        completeness = min(1.0, density / 1000.0)
        
        return completeness
    
    def _find_similar_predictions(self, prediction: Dict) -> List[Dict]:
        """Find similar predictions from history."""
        similar = []
        
        # This would search through stored predictions
        # For now, return empty list
        return similar
    
    def _calculate_prediction_accuracy(self, prediction: Dict,
                                      actual: Dict) -> float:
        """Calculate accuracy of a prediction."""
        # Compare prediction to actual outcome
        matching_keys = 0
        total_keys = 0
        
        for key in prediction:
            if key in actual:
                total_keys += 1
                if prediction[key] == actual[key]:
                    matching_keys += 1
        
        if total_keys == 0:
            return 0.5
        
        return matching_keys / total_keys
    
    def save_state(self, filepath: str):
        """Save reasoning engine state to file."""
        state = {
            'objects': {k: {
                'id': v.id,
                'name': v.name,
                'last_seen': v.last_seen,
                'permanence_score': v.permanence_score
            } for k, v in self.objects.items()},
            'patterns': self.patterns,
            'causal_graph': self.causal_graph,
            'config': self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Reasoning engine state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load reasoning engine state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.config = state.get('config', {})
        self.patterns = state.get('patterns', {})
        self.causal_graph = state.get('causal_graph', {})
        
        # Reconstruct objects
        for obj_id, obj_data in state.get('objects', {}).items():
            self.objects[obj_id] = Object3D(
                id=obj_data['id'],
                name=obj_data.get('name'),
                last_seen=obj_data.get('last_seen', time.time()),
                permanence_score=obj_data.get('permanence_score', 1.0)
            )
        
        logger.info(f"Reasoning engine state loaded from {filepath}")