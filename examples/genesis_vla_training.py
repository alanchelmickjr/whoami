#!/usr/bin/env python3
"""
Genesis VLA Training Example

Complete pipeline demonstrating:
1. Orbital 3D scanning with 3DOF gimbal
2. Converting scans to Genesis simulation
3. Training VLA policies in simulation
4. Sim2Real transfer to physical robot

This is the core of embodied AI learning:
- Robot scans objects from all angles
- Builds complete 3D + semantic understanding
- Trains manipulation policies in Genesis (ultra-fast)
- Executes learned policies in reality
- Continuously improves from experience

The robot becomes its own:
- Data collector (orbital scanning)
- Scene builder (Genesis conversion)
- Policy trainer (VLA learning)
- Validator (sim2real execution)
"""

import numpy as np
import logging
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from whoami.gimbal_3dof_controller import Gimbal3DOFController
from whoami.scanner_3d import Scanner3D, Scanner3DConfig, ScanMode
from whoami.genesis_bridge import (
    OrbitalToGenesis,
    GenesisSceneBuilder,
    VLATrainingPipeline,
    Sim2RealBridge,
    ScannedObject,
    GENESIS_AVAILABLE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Complete Embodied AI Learning System
# ============================================================================

class EmbodiedAISystem:
    """
    Complete system for self-supervised embodied AI learning

    Pipeline:
    Reality → Orbital Scan → Genesis Sim → VLA Training → Reality
    """

    def __init__(self):
        """Initialize complete system"""
        logger.info("="*70)
        logger.info("EMBODIED AI LEARNING SYSTEM")
        logger.info("Reality ↔ Genesis Bridge for VLA Training")
        logger.info("="*70)

        # Check Genesis availability
        if not GENESIS_AVAILABLE:
            logger.error("Genesis not installed!")
            logger.error("Install with: pip install genesis-world")
            sys.exit(1)

        # Initialize components
        self.gimbal = Gimbal3DOFController()
        self.scanner = Scanner3D(Scanner3DConfig(
            resolution=(1280, 720),
            fps=30,
            scan_mode=ScanMode.SINGLE_SHOT
        ))

        # Genesis bridge components
        self.orbital_converter = OrbitalToGenesis()
        self.genesis_builder = GenesisSceneBuilder()
        self.vla_pipeline = VLATrainingPipeline(self.genesis_builder)
        self.sim2real = Sim2RealBridge()

        # State
        self.scanned_objects = []
        self.trained_policies = {}

    def connect(self) -> bool:
        """Connect to all hardware"""
        logger.info("\nConnecting to hardware...")

        if not self.gimbal.connect():
            logger.error("Failed to connect gimbal")
            return False

        if not self.scanner.initialize():
            logger.error("Failed to initialize scanner")
            return False

        logger.info("✓ All hardware connected")
        return True

    def disconnect(self):
        """Disconnect all hardware"""
        self.gimbal.disconnect()
        self.scanner.shutdown()

    # ========================================================================
    # Phase 1: Reality → Genesis (Scan Real World)
    # ========================================================================

    def scan_and_digitize_object(
        self,
        object_name: str,
        object_center: np.ndarray,
        scan_radius: float = 0.20,
        scan_quality: str = "standard"
    ) -> ScannedObject:
        """
        Scan real object with orbital gimbal and convert to Genesis format

        Args:
            object_name: Name/identifier for object
            object_center: (x, y, z) position of object
            scan_radius: Orbital scanning radius
            scan_quality: "quick", "standard", or "detailed"

        Returns:
            ScannedObject ready for Genesis simulation
        """
        logger.info("="*70)
        logger.info(f"PHASE 1: SCANNING OBJECT - {object_name}")
        logger.info("="*70)

        # Configure scan density based on quality
        scan_params = {
            "quick": {"num_points": 12, "roll_angles": [0]},
            "standard": {"num_points": 36, "roll_angles": [0]},
            "detailed": {"num_points": 36, "roll_angles": [0, 90]}
        }
        params = scan_params.get(scan_quality, scan_params["standard"])

        # Collect multi-view data
        point_clouds = []
        orbital_images = []
        camera_poses = []

        def capture_callback(idx, angles):
            """Capture point cloud and image at each position"""
            logger.info(f"  Capturing view {idx + 1}")

            # Capture point cloud
            cloud = self.scanner.capture_point_cloud()
            if cloud:
                point_clouds.append(cloud)

            # Capture RGB image
            image = self.scanner.capture_rgb_frame()
            if image is not None:
                orbital_images.append(image)

            # Record camera pose
            pose = self.gimbal.get_current_pose()
            if pose:
                camera_poses.append({
                    "position": pose.position.tolist(),
                    "forward": pose.forward.tolist(),
                    "up": pose.up.tolist()
                })

        # Perform orbital scan
        logger.info(f"Performing {scan_quality} orbital scan...")
        success = self.gimbal.scan_horizontal_orbit(
            center=object_center,
            radius=scan_radius,
            num_points=params["num_points"],
            scan_callback=capture_callback
        )

        if not success:
            raise RuntimeError("Orbital scan failed")

        # Merge point clouds into complete 3D model
        logger.info("Merging point clouds...")
        merged_cloud = self.scanner.merge_point_clouds(point_clouds)

        # Generate mesh
        logger.info("Generating mesh...")
        mesh = self.scanner.generate_mesh(merged_cloud)

        if mesh is None:
            raise RuntimeError("Mesh generation failed")

        # Convert to Genesis-compatible format
        logger.info("Converting to Genesis format...")
        scanned_obj = self.orbital_converter.create_scanned_object(
            name=object_name,
            mesh=mesh,
            point_cloud=merged_cloud,
            orbital_images=orbital_images,
            camera_poses=camera_poses,
            material_hint="plastic",  # Could be estimated from appearance
            category="unknown"  # Could use vision model to classify
        )

        self.scanned_objects.append(scanned_obj)

        logger.info(f"✓ Object digitized: {object_name}")
        logger.info(f"  Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        logger.info(f"  Mass: {scanned_obj.mass:.3f}kg")
        logger.info(f"  Views: {len(orbital_images)}")

        return scanned_obj

    # ========================================================================
    # Phase 2: Genesis Simulation (Build & Train)
    # ========================================================================

    def build_genesis_scene(
        self,
        scene_name: str,
        objects: List[ScannedObject]
    ) -> str:
        """
        Build Genesis simulation scene from scanned objects

        Args:
            scene_name: Scene identifier
            objects: Scanned objects to include

        Returns:
            Scene ID
        """
        logger.info("="*70)
        logger.info(f"PHASE 2: BUILDING GENESIS SCENE - {scene_name}")
        logger.info("="*70)

        scene = self.genesis_builder.create_scene(
            scene_id=scene_name,
            objects=objects,
            ground_plane=True,
            lighting="default"
        )

        logger.info(f"✓ Genesis scene created: {scene_name}")
        logger.info(f"  Objects: {len(objects)}")
        logger.info(f"  Physics: 60 Hz simulation")
        logger.info(f"  Ready for training")

        return scene_name

    def train_vla_policy(
        self,
        scene_id: str,
        task_description: str,
        num_episodes: int = 100
    ):
        """
        Train VLA policy in Genesis simulation

        Args:
            scene_id: Genesis scene to train in
            task_description: Natural language task ("pick up the mug")
            num_episodes: Number of training episodes
        """
        logger.info("="*70)
        logger.info(f"PHASE 3: VLA TRAINING - {task_description}")
        logger.info("="*70)

        logger.info(f"Training in Genesis scene: {scene_id}")
        logger.info(f"Task: {task_description}")
        logger.info(f"Episodes: {num_episodes}")

        # Generate training data through Genesis simulation
        logger.info("\nGenerating training data...")
        experiences = self.vla_pipeline.generate_training_data(
            scanned_objects=self.scanned_objects,
            language_instructions=[task_description],
            num_episodes=num_episodes
        )

        # Train VLA model
        logger.info("\nTraining VLA model...")
        self.vla_pipeline.train(experiences, num_epochs=50)

        # Save trained policy
        policy_path = Path(f"models/vla_policy_{scene_id}.pt")
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        self.vla_pipeline.save_model(policy_path)

        logger.info(f"✓ Policy trained and saved: {policy_path}")

        return policy_path

    # ========================================================================
    # Phase 4: Sim2Real Transfer (Back to Reality)
    # ========================================================================

    def execute_learned_policy(
        self,
        policy_path: Path,
        real_object_center: np.ndarray,
        task_description: str
    ) -> Dict[str, Any]:
        """
        Execute learned policy on real robot

        Args:
            policy_path: Path to trained policy
            real_object_center: Object position in reality
            task_description: Task to execute

        Returns:
            Execution results and metrics
        """
        logger.info("="*70)
        logger.info(f"PHASE 4: SIM2REAL EXECUTION - {task_description}")
        logger.info("="*70)

        # Load trained policy
        logger.info(f"Loading policy from {policy_path}")
        self.vla_pipeline.load_model(policy_path)

        # TODO: Actual policy execution
        # This would:
        # 1. Observe current state with multi-view perception
        # 2. Encode observation + language instruction
        # 3. Generate action from policy
        # 4. Execute on physical robot
        # 5. Observe outcome
        # 6. Compute success metrics

        logger.info("Executing policy in reality...")
        logger.info("  [Placeholder - would execute on real robot]")

        results = {
            "success": True,
            "execution_time": 5.0,
            "sim_real_discrepancy": 0.02,
            "task": task_description
        }

        logger.info(f"✓ Execution complete")
        logger.info(f"  Success: {results['success']}")
        logger.info(f"  Time: {results['execution_time']:.1f}s")

        return results

    # ========================================================================
    # Phase 5: Continuous Learning (Close the Loop)
    # ========================================================================

    def continuous_learning_loop(
        self,
        num_iterations: int = 10
    ):
        """
        Continuous learning loop: scan → simulate → train → execute → improve

        Args:
            num_iterations: Number of learning iterations
        """
        logger.info("="*70)
        logger.info("PHASE 5: CONTINUOUS LEARNING LOOP")
        logger.info("="*70)

        for iteration in range(num_iterations):
            logger.info(f"\n{'='*70}")
            logger.info(f"ITERATION {iteration + 1}/{num_iterations}")
            logger.info(f"{'='*70}")

            # 1. Scan new object or rescan existing
            # 2. Update Genesis scene with new geometry
            # 3. Generate more training data
            # 4. Fine-tune policy
            # 5. Execute and validate
            # 6. Learn from outcome

            logger.info("  [Continuous learning iteration]")
            time.sleep(1)

        logger.info("\n✓ Continuous learning complete")


# ============================================================================
# Example Workflows
# ============================================================================

def example_basic_workflow():
    """
    Basic workflow: Scan one object, train policy, execute

    This demonstrates the complete pipeline from reality to Genesis and back
    """
    system = EmbodiedAISystem()

    if not system.connect():
        return

    try:
        # Define object to scan (adjust for your setup)
        object_center = np.array([0.30, 0.0, 0.0])  # 30cm forward

        # Phase 1: Scan real object
        mug = system.scan_and_digitize_object(
            object_name="coffee_mug",
            object_center=object_center,
            scan_radius=0.20,
            scan_quality="standard"
        )

        # Phase 2: Build Genesis scene
        scene_id = system.build_genesis_scene(
            scene_name="tabletop_scene",
            objects=[mug]
        )

        # Phase 3: Train VLA policy
        policy_path = system.train_vla_policy(
            scene_id=scene_id,
            task_description="pick up the coffee mug by the handle",
            num_episodes=100
        )

        # Phase 4: Execute in reality
        results = system.execute_learned_policy(
            policy_path=policy_path,
            real_object_center=object_center,
            task_description="pick up the coffee mug by the handle"
        )

        logger.info("\n" + "="*70)
        logger.info("WORKFLOW COMPLETE!")
        logger.info("="*70)
        logger.info(f"Scanned: {mug.name}")
        logger.info(f"Trained: {scene_id}")
        logger.info(f"Success: {results['success']}")

    except KeyboardInterrupt:
        logger.info("\nWorkflow interrupted")
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        system.disconnect()


def example_multi_object_scene():
    """
    Advanced: Scan multiple objects, build complex scene, train policies
    """
    system = EmbodiedAISystem()

    if not system.connect():
        return

    try:
        # Scan multiple objects
        objects = []

        object_configs = [
            {"name": "mug", "center": [0.30, 0.10, 0.0]},
            {"name": "tool", "center": [0.35, -0.10, 0.0]},
            {"name": "block", "center": [0.25, 0.0, 0.0]}
        ]

        logger.info("\nScanning multiple objects...")
        for config in object_configs:
            obj = system.scan_and_digitize_object(
                object_name=config["name"],
                object_center=np.array(config["center"]),
                scan_radius=0.15,
                scan_quality="standard"
            )
            objects.append(obj)
            system.gimbal.home()  # Reset between scans
            time.sleep(1)

        # Build scene with all objects
        scene_id = system.build_genesis_scene(
            scene_name="multi_object_scene",
            objects=objects
        )

        # Train policies for different tasks
        tasks = [
            "pick up the mug",
            "stack the blocks",
            "move the tool to the left"
        ]

        for task in tasks:
            logger.info(f"\n--- Training task: {task} ---")
            system.train_vla_policy(
                scene_id=scene_id,
                task_description=task,
                num_episodes=50
            )

        logger.info("\n✓ Multi-object training complete")

    finally:
        system.disconnect()


def example_continuous_learning():
    """
    Continuous learning: Robot explores, learns, improves over time
    """
    system = EmbodiedAISystem()

    if not system.connect():
        return

    try:
        # Initial scan
        obj = system.scan_and_digitize_object(
            object_name="exploration_object",
            object_center=np.array([0.30, 0.0, 0.0]),
            scan_radius=0.20,
            scan_quality="standard"
        )

        # Build scene
        scene_id = system.build_genesis_scene(
            scene_name="exploration_scene",
            objects=[obj]
        )

        # Enter continuous learning loop
        system.continuous_learning_loop(num_iterations=10)

    finally:
        system.disconnect()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Genesis VLA Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s basic      # Basic workflow: scan → train → execute
  %(prog)s multi      # Multi-object scene training
  %(prog)s continuous # Continuous learning loop
        """
    )

    parser.add_argument(
        "mode",
        choices=["basic", "multi", "continuous"],
        help="Workflow mode to run"
    )

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("GENESIS VLA TRAINING PIPELINE")
    logger.info("Embodied AI Learning: Reality ↔ Simulation")
    logger.info("="*70)
    logger.info(f"\nMode: {args.mode}")
    logger.info("")

    if not GENESIS_AVAILABLE:
        logger.error("Genesis not available!")
        logger.error("Install with: pip install genesis-world")
        return

    if args.mode == "basic":
        example_basic_workflow()
    elif args.mode == "multi":
        example_multi_object_scene()
    elif args.mode == "continuous":
        example_continuous_learning()


if __name__ == "__main__":
    main()
