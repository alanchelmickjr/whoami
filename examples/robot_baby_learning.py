#!/usr/bin/env python3
"""
Robot Baby Learning Example

Demonstrates autonomous learning, object reasoning, memory formation,
and knowledge sharing between robot siblings.
"""

import json
import time
import random
import numpy as np
from typing import Dict, Any, List
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whoami.robot_brain import RobotBrain, EmotionalState, PersonalityTrait
from whoami.gun_storage import TrustLevel, MemoryCategory


def simulate_environment() -> Dict[str, Any]:
    """Simulate an environment with objects and stimuli."""
    environments = [
        {
            'type': 'room',
            'objects': [
                {'type': 'ball', 'color': 'red', 'position': [1, 0, 0]},
                {'type': 'cube', 'color': 'blue', 'position': [0, 1, 0]}
            ],
            'brightness': 0.7,
            'temperature': 22
        },
        {
            'type': 'playground',
            'objects': [
                {'type': 'toy', 'color': 'yellow', 'position': [2, 1, 0]},
                {'type': 'robot', 'color': 'silver', 'position': [3, 2, 0]}
            ],
            'brightness': 0.9,
            'has_other_robots': True
        },
        {
            'type': 'obstacle_course',
            'objects': [
                {'type': 'wall', 'height': 2, 'position': [0, 5, 0]},
                {'type': 'ramp', 'angle': 30, 'position': [2, 3, 0]}
            ],
            'difficulty': 'medium'
        }
    ]
    
    return random.choice(environments)


def simulate_3d_scan(object_type: str) -> np.ndarray:
    """Simulate a partial 3D point cloud scan of an object."""
    if object_type == 'ball':
        # Generate partial sphere points
        theta = np.random.uniform(0, np.pi, 50)
        phi = np.random.uniform(0, 2*np.pi, 50)
        r = 1.0 + np.random.normal(0, 0.1, 50)
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        points = np.column_stack((x, y, z))
        # Only return half (partial scan)
        return points[:25]
    
    elif object_type == 'cube':
        # Generate partial cube points
        points = []
        for _ in range(30):
            face = random.choice(['x', 'y', 'z'])
            if face == 'x':
                x = random.choice([-1, 1])
                y = random.uniform(-1, 1)
                z = random.uniform(-1, 1)
            elif face == 'y':
                x = random.uniform(-1, 1)
                y = random.choice([-1, 1])
                z = random.uniform(-1, 1)
            else:
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                z = random.choice([-1, 1])
            
            points.append([x, y, z])
        
        return np.array(points)
    
    else:
        # Random point cloud for unknown objects
        return np.random.randn(20, 3)


def calculate_outcome(action: Dict[str, Any], environment: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate the outcome of an action in the environment."""
    outcome = {
        'success': False,
        'new_discovery': False,
        'reward': 0,
        'confusion_increased': False
    }
    
    action_type = action.get('type', 'unknown')
    
    if action_type == 'explore':
        # Exploration usually leads to discoveries
        if random.random() < 0.7:
            outcome['success'] = True
            outcome['new_discovery'] = random.random() < 0.4
            outcome['reward'] = 0.5 if outcome['new_discovery'] else 0.2
        else:
            outcome['confusion_increased'] = True
            outcome['reward'] = -0.1
    
    elif action_type == 'examine':
        # Examining objects is usually successful
        if 'objects' in environment and environment['objects']:
            outcome['success'] = True
            outcome['learned_property'] = random.choice(['shape', 'color', 'texture'])
            outcome['reward'] = 0.3
        else:
            outcome['reward'] = 0
    
    elif action_type == 'interact':
        # Interaction success depends on object type
        if environment.get('has_other_robots'):
            outcome['success'] = random.random() < 0.8
            outcome['social_success'] = outcome['success']
            outcome['reward'] = 0.6 if outcome['success'] else -0.2
        else:
            outcome['success'] = random.random() < 0.5
            outcome['reward'] = 0.2 if outcome['success'] else 0
    
    elif action_type == 'wait':
        # Waiting is always successful but low reward
        outcome['success'] = True
        outcome['reward'] = 0.1
    
    elif action_type == 'remember':
        # Remembering strengthens memory
        outcome['success'] = True
        outcome['memory_strengthened'] = True
        outcome['reward'] = 0.2
    
    return outcome


def demonstrate_autonomous_learning():
    """Demonstrate the robot baby learning autonomously."""
    print("=" * 80)
    print("ROBOT BABY AUTONOMOUS LEARNING DEMONSTRATION")
    print("=" * 80)
    
    # Load configuration
    config_path = "config/brain_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Create robot baby
    baby_robot = RobotBrain("baby_alpha", config)
    
    # Start autonomous operation
    baby_robot.start_autonomous_operation()
    
    print(f"\nRobot Baby '{baby_robot.robot_id}' is born!")
    print(f"Initial personality traits:")
    for trait, value in baby_robot.personality.traits.items():
        print(f"  {trait.value}: {value:.2f}")
    
    # Simulate learning cycles
    num_cycles = 50
    
    for cycle in range(num_cycles):
        print(f"\n{'='*60}")
        print(f"Learning Cycle {cycle + 1}/{num_cycles}")
        print(f"{'='*60}")
        
        # Perceive environment
        environment = simulate_environment()
        print(f"\nEnvironment: {environment['type']}")
        
        # Add 3D scan data for objects
        if 'objects' in environment and environment['objects']:
            for obj in environment['objects']:
                obj['points'] = simulate_3d_scan(obj['type'])
                
                # Object reasoning
                completed, confidence = baby_robot.reasoning_engine.complete_object(
                    f"obj_{cycle}_{obj['type']}",
                    obj['points']
                )
                print(f"  Scanned {obj['type']}: {len(obj['points'])} points -> "
                      f"{len(completed)} completed (confidence: {confidence:.2f})")
        
        # Make decision
        situation = {
            **environment,
            'cycle': cycle,
            'timestamp': time.time()
        }
        
        decision = baby_robot.make_decision(situation)
        print(f"\nDecision: {decision.action['type']}")
        print(f"  Reasoning: {decision.reasoning}")
        print(f"  Confidence: {decision.confidence:.2f}")
        print(f"  Emotional influence: {decision.emotional_influence:.2f}")
        print(f"  Current emotion: {baby_robot.current_emotion.value}")
        
        # Calculate outcome
        outcome = calculate_outcome(decision.action, environment)
        outcome['initial_state'] = situation
        
        # Learn from outcome
        baby_robot.learn_from_outcome(decision, outcome)
        
        print(f"\nOutcome:")
        print(f"  Success: {outcome.get('success', False)}")
        print(f"  Reward: {outcome.get('reward', 0):.2f}")
        if outcome.get('new_discovery'):
            print(f"  🌟 New discovery made!")
        
        # Form memory periodically
        if cycle % 5 == 0 and outcome.get('reward', 0) > 0:
            memory_id = baby_robot.form_memory(
                {
                    'cycle': cycle,
                    'environment': environment['type'],
                    'action': decision.action['type'],
                    'outcome': outcome,
                    'emotion': baby_robot.current_emotion.value
                },
                importance=abs(outcome.get('reward', 0))
            )
            print(f"  💭 Memory formed: {memory_id}")
        
        # Show learning progress every 10 cycles
        if (cycle + 1) % 10 == 0:
            stats = baby_robot.learning_system.get_learning_stats()
            print(f"\n📊 Learning Progress:")
            print(f"  Total experiences: {stats['total_experiences']}")
            print(f"  Knowledge concepts: {stats['knowledge_concepts']}")
            print(f"  Exploration rate: {stats['exploration_rate']:.2f}")
            print(f"  Learning rate: {stats['learning_rate']:.4f}")
            print(f"  Mistakes recorded: {stats['mistakes_recorded']}")
        
        # Dream cycle every 20 cycles
        if (cycle + 1) % 20 == 0:
            print(f"\n😴 Entering dream cycle for consolidation...")
            baby_robot.dream_cycle(duration=20)
            print(f"  Dreams complete - knowledge consolidated")
        
        # Detect personality development
        if (cycle + 1) % 15 == 0:
            dominant_traits = baby_robot.personality.get_dominant_traits(3)
            print(f"\n🧠 Personality Development:")
            print(f"  Dominant traits: {[t.value for t in dominant_traits]}")
            print(f"  Decisions made: {baby_robot.personality.decisions_made}")
            print(f"  Memories formed: {baby_robot.personality.memories_formed}")
            print(f"  Mistakes learned from: {baby_robot.personality.mistakes_learned}")
    
    # Final self-awareness report
    print(f"\n{'='*80}")
    print("FINAL SELF-AWARENESS REPORT")
    print(f"{'='*80}")
    
    report = baby_robot.get_self_awareness_report()
    
    print(f"\nIdentity: {report['identity']}")
    print(f"Age: {report['age_seconds']:.1f} seconds")
    print(f"Current emotion: {report['current_emotion']}")
    
    print(f"\nPersonality Profile:")
    print(f"  Dominant traits: {', '.join(report['personality']['dominant_traits'])}")
    print(f"  Stability: {report['personality']['stability']:.2f}")
    print(f"  All traits:")
    for trait, value in report['personality']['all_traits'].items():
        print(f"    {trait}: {value:.2f}")
    
    print(f"\nCognitive Statistics:")
    for key, value in report['cognitive_stats'].items():
        print(f"  {key}: {value}")
    
    print(f"\nReasoning Capabilities:")
    for key, value in report['reasoning_capabilities'].items():
        print(f"  {key}: {value}")
    
    # Save state
    baby_robot.save_state()
    print(f"\n💾 Robot brain state saved")
    
    # Stop autonomous operation
    baby_robot.stop_autonomous_operation()
    
    return baby_robot


def demonstrate_sibling_interaction():
    """Demonstrate knowledge sharing between sibling robots."""
    print("\n" + "=" * 80)
    print("SIBLING ROBOT INTERACTION DEMONSTRATION")
    print("=" * 80)
    
    # Create two sibling robots
    config = {}
    if os.path.exists("config/brain_config.json"):
        with open("config/brain_config.json", 'r') as f:
            config = json.load(f)
    
    # Modify ports for different robots
    config1 = config.copy()
    config1['storage'] = config.get('storage', {}).copy()
    config1['storage']['listen_port'] = 8765
    
    config2 = config.copy()
    config2['storage'] = config.get('storage', {}).copy()
    config2['storage']['listen_port'] = 8766
    
    robot1 = RobotBrain("sibling_alpha", config1)
    robot2 = RobotBrain("sibling_beta", config2)
    
    print(f"\nCreated sibling robots:")
    print(f"  - {robot1.robot_id}")
    print(f"  - {robot2.robot_id}")
    
    # Add each other as trusted siblings
    robot1.memory_storage.add_peer(
        robot2.robot_id,
        "localhost",
        8766,
        TrustLevel.SIBLING
    )
    
    robot2.memory_storage.add_peer(
        robot1.robot_id,
        "localhost",
        8765,
        TrustLevel.SIBLING
    )
    
    print(f"\n🤝 Robots added as trusted siblings")
    
    # Robot 1 learns something
    print(f"\n{robot1.robot_id} is learning...")
    
    for i in range(5):
        environment = simulate_environment()
        situation = {'environment': environment, 'step': i}
        decision = robot1.make_decision(situation)
        outcome = calculate_outcome(decision.action, environment)
        robot1.learn_from_outcome(decision, outcome)
        
        # Form memory of successful experience
        if outcome.get('success'):
            memory_id = robot1.form_memory(
                {
                    'experience': f"learned_{i}",
                    'action': decision.action,
                    'outcome': outcome,
                    'lesson': f"Action {decision.action['type']} works well"
                },
                importance=0.8
            )
            print(f"  Learned: {decision.action['type']} -> "
                  f"reward: {outcome['reward']:.2f}")
    
    # Share knowledge with sibling
    print(f"\n📤 {robot1.robot_id} sharing knowledge with {robot2.robot_id}...")
    
    # Get recent memories to share
    memories_to_share = robot1.memory_storage.query_memories(
        category=MemoryCategory.FAMILY
    )[:3]
    
    shared_count = 0
    for memory in memories_to_share:
        if robot1.memory_storage.share_with_sibling(memory.id, robot2.robot_id):
            shared_count += 1
    
    print(f"  Shared {shared_count} memories")
    
    # Robot 2 learns from sibling
    print(f"\n📥 {robot2.robot_id} learning from sibling...")
    learned = robot2.learn_from_sibling(robot1.robot_id)
    print(f"  Learned from {learned} experiences")
    
    # Check personality influence
    print(f"\n🧠 Personality influence from social learning:")
    print(f"  {robot2.robot_id} social trait: "
          f"{robot2.personality.traits.get(PersonalityTrait.SOCIAL, 0.5):.2f}")
    
    # Show knowledge transfer
    print(f"\n📚 Knowledge Transfer Results:")
    print(f"  {robot1.robot_id} knowledge concepts: "
          f"{len(robot1.learning_system.knowledge_base)}")
    print(f"  {robot2.robot_id} knowledge concepts: "
          f"{len(robot2.learning_system.knowledge_base)}")
    
    return robot1, robot2


def demonstrate_mistake_learning():
    """Demonstrate how the robot learns from mistakes."""
    print("\n" + "=" * 80)
    print("LEARNING FROM MISTAKES DEMONSTRATION")
    print("=" * 80)
    
    config = {}
    if os.path.exists("config/brain_config.json"):
        with open("config/brain_config.json", 'r') as f:
            config = json.load(f)
    
    robot = RobotBrain("learner_gamma", config)
    
    print(f"\nRobot '{robot.robot_id}' will make and learn from mistakes")
    
    # Simulate repeated mistakes
    mistake_action = {'type': 'interact', 'target': 'wall'}
    
    for attempt in range(5):
        print(f"\n🔄 Attempt {attempt + 1}")
        
        environment = {
            'type': 'room',
            'objects': [{'type': 'wall', 'solid': True}]
        }
        
        # Force the mistake action for demonstration
        if attempt < 3:
            # Make the same mistake repeatedly
            decision_action = mistake_action
            print(f"  Action: {decision_action['type']} with {decision_action['target']}")
            
            outcome = {
                'success': False,
                'failure': True,
                'damage': True,
                'reward': -0.8,
                'initial_state': environment
            }
            print(f"  Result: ❌ Failed (damage taken)")
        else:
            # After mistakes, robot should avoid this action
            situation = environment
            decision = robot.make_decision(situation)
            decision_action = decision.action
            print(f"  Action: {decision_action['type']}")
            
            if decision_action['type'] != 'interact' or decision_action.get('target') != 'wall':
                outcome = {
                    'success': True,
                    'reward': 0.5,
                    'initial_state': environment
                }
                print(f"  Result: ✅ Success (learned to avoid wall)")
            else:
                outcome = {
                    'success': False,
                    'failure': True,
                    'reward': -0.8,
                    'initial_state': environment
                }
                print(f"  Result: ❌ Failed (still learning)")
        
        # Create a decision object for learning
        from whoami.robot_brain import Decision
        decision = Decision(
            id=f"decision_{attempt}",
            action=decision_action,
            reasoning="Testing mistake learning",
            confidence=0.5
        )
        
        # Learn from outcome
        robot.learn_from_outcome(decision, outcome)
        
        # Check mistake detection
        robot.learning_system.detect_and_correct_mistakes()
    
    # Show learning results
    print(f"\n📊 Learning from Mistakes Summary:")
    print(f"  Mistakes learned: {robot.personality.mistakes_learned}")
    print(f"  Exploration rate: {robot.learning_system.exploration_rate:.2f}")
    print(f"  Cautious trait: "
          f"{robot.personality.traits.get(PersonalityTrait.CAUTIOUS, 0.5):.2f}")
    
    # Check if robot now avoids the mistake
    print(f"\n🧪 Testing if robot learned...")
    test_environment = {
        'type': 'room',
        'objects': [
            {'type': 'wall', 'solid': True},
            {'type': 'ball', 'solid': False}
        ]
    }
    
    final_decision = robot.make_decision(test_environment)
    print(f"  Robot chose: {final_decision.action['type']}")
    if final_decision.action.get('target') != 'wall':
        print(f"  ✅ Robot successfully learned to avoid walls!")
    else:
        print(f"  ⚠️ Robot still learning...")
    
    return robot


def main():
    """Run all demonstrations."""
    try:
        # Demonstration 1: Autonomous Learning
        baby_robot = demonstrate_autonomous_learning()
        
        # Wait a bit between demonstrations
        time.sleep(2)
        
        # Demonstration 2: Sibling Interaction
        # Note: This requires network setup, so we'll skip if it fails
        try:
            robot1, robot2 = demonstrate_sibling_interaction()
        except Exception as e:
            print(f"\n⚠️ Sibling interaction demo skipped (network error): {e}")
        
        # Demonstration 3: Learning from Mistakes
        learner_robot = demonstrate_mistake_learning()
        
        print("\n" + "=" * 80)
        print("🎉 DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print("\nThe robot baby has demonstrated:")
        print("  ✅ Autonomous learning and exploration")
        print("  ✅ Object reasoning and completion")
        print("  ✅ Memory formation and recall")
        print("  ✅ Personality development")
        print("  ✅ Emotional state management")
        print("  ✅ Learning from mistakes")
        print("  ✅ Knowledge consolidation through dreaming")
        print("  ✅ Sibling knowledge sharing (when network available)")
        print("\nThe robot is now capable of continuous self-improvement!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()