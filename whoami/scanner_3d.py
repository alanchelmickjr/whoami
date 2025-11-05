"""
3D Scanner Module for OAK-D Camera
Provides 3D scanning, point cloud processing, mesh generation, and STL export
for robot tool creation and object manipulation
"""

import numpy as np
import cv2
import depthai as dai
import open3d as o3d
import trimesh
import logging
import os
import time
import threading
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes and Enums
# ============================================================================

class ScanMode(Enum):
    """3D scanning modes"""
    SINGLE_SHOT = "single_shot"      # Single point cloud capture
    TURNTABLE = "turntable"          # Rotating object scan
    MULTI_ANGLE = "multi_angle"      # Multiple manual angles
    CONTINUOUS = "continuous"        # Continuous streaming


class MeshQuality(Enum):
    """Mesh generation quality levels"""
    DRAFT = "draft"          # Fast, low quality
    STANDARD = "standard"    # Balanced quality/speed
    HIGH = "high"           # High quality, slower


class ToolType(Enum):
    """Types of tools for inverse generation"""
    GRIPPER = "gripper"          # Gripping tool
    SOCKET = "socket"            # Socket wrench type
    MOLD = "mold"               # Casting mold
    FIXTURE = "fixture"         # Holding fixture
    CUSTOM = "custom"           # Custom tool type


@dataclass
class PointCloudData:
    """Container for point cloud data"""
    points: np.ndarray              # Nx3 array of 3D points
    colors: Optional[np.ndarray]    # Nx3 array of RGB colors
    normals: Optional[np.ndarray]   # Nx3 array of normals
    timestamp: float                # Capture timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResult:
    """Result of a 3D scan"""
    point_cloud: o3d.geometry.PointCloud
    mesh: Optional[trimesh.Trimesh]
    scan_mode: ScanMode
    num_captures: int
    total_points: int
    bounding_box: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scanner3DConfig:
    """Configuration for 3D scanner"""
    # Camera settings
    resolution: Tuple[int, int] = (1280, 720)
    fps: int = 30
    
    # Depth settings
    min_depth: int = 200          # Minimum depth in mm
    max_depth: int = 10000        # Maximum depth in mm
    depth_confidence: int = 6     # Confidence threshold (0-255)
    
    # Point cloud settings
    voxel_size: float = 0.005     # Voxel size for downsampling (meters)
    statistical_outlier_neighbors: int = 20
    statistical_outlier_std_ratio: float = 2.0
    
    # Mesh generation
    mesh_quality: MeshQuality = MeshQuality.STANDARD
    poisson_depth: int = 9        # Poisson reconstruction depth
    simplification_target: Optional[int] = None  # Target face count
    
    # Scanning behavior
    scan_mode: ScanMode = ScanMode.SINGLE_SHOT
    auto_align: bool = True       # Auto-align multiple scans
    icp_threshold: float = 0.02   # ICP alignment threshold
    
    # Output settings
    output_directory: str = "scans"
    auto_save: bool = True
    
    # Processing
    enable_threading: bool = True
    log_level: str = "INFO"


# ============================================================================
# OAK-D 3D Camera Interface
# ============================================================================

class OakD3DCamera:
    """OAK-D camera interface for 3D scanning with depth and point cloud"""
    
    def __init__(self, config: Scanner3DConfig):
        self.config = config
        self.pipeline = None
        self.device = None
        self.depth_queue = None
        self.rgb_queue = None
        self.pointcloud_queue = None
        self._lock = threading.Lock()
        self._running = False
        
        # Calibration data
        self.calibration_data = None
        self.intrinsics = None
        
        logger.debug("Initialized OakD3DCamera for 3D scanning")
    
    def start(self) -> bool:
        """Start the OAK-D camera for 3D capture"""
        try:
            with self._lock:
                self.pipeline = self._create_3d_pipeline()
                self.device = dai.Device(self.pipeline)
                
                # Get calibration data
                self.calibration_data = self.device.readCalibration()
                self.intrinsics = self._get_intrinsics()
                
                # Create output queues
                self.depth_queue = self.device.getOutputQueue("depth", maxSize=4, blocking=False)
                self.rgb_queue = self.device.getOutputQueue("rgb", maxSize=4, blocking=False)
                
                # Note: Point cloud generation will be done in software
                # as OAK-D doesn't have direct point cloud output
                
                self._running = True
                logger.info("OAK-D 3D camera started successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start OAK-D 3D camera: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the OAK-D camera"""
        with self._lock:
            self._running = False
            if self.device:
                self.device.close()
                self.device = None
            self.pipeline = None
            logger.info("OAK-D 3D camera stopped")
    
    def capture_point_cloud(self) -> Optional[PointCloudData]:
        """Capture a single point cloud from the camera"""
        with self._lock:
            if not self._running:
                logger.error("Camera not running")
                return None
            
            try:
                # Get depth frame
                depth_msg = self.depth_queue.get()
                depth_frame = depth_msg.getFrame()
                
                # Get RGB frame
                rgb_msg = self.rgb_queue.get()
                rgb_frame = rgb_msg.getCvFrame()
                
                # Generate point cloud from depth
                point_cloud = self._depth_to_pointcloud(depth_frame, rgb_frame)
                
                return point_cloud
                
            except Exception as e:
                logger.error(f"Failed to capture point cloud: {e}")
                return None
    
    def _create_3d_pipeline(self) -> dai.Pipeline:
        """Create DepthAI pipeline for 3D scanning"""
        pipeline = dai.Pipeline()
        
        # Create mono cameras
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        
        # Create depth node
        depth = pipeline.create(dai.node.StereoDepth)
        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        depth.setLeftRightCheck(True)
        depth.setSubpixel(True)
        depth.setDepthAlign(dai.CameraBoardSocket.RGB)
        
        # Set depth range
        depth.setOutputSize(self.config.resolution[0], self.config.resolution[1])
        config = depth.initialConfig.get()
        config.postProcessing.speckleFilter.enable = True
        config.postProcessing.speckleFilter.speckleRange = 50
        config.postProcessing.temporalFilter.enable = True
        config.postProcessing.spatialFilter.enable = True
        depth.initialConfig.set(config)
        
        # Connect mono cameras to depth
        mono_left.out.link(depth.left)
        mono_right.out.link(depth.right)
        
        # Create RGB camera
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setIspScale(2, 3)  # Scale to 720p
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam_rgb.initialControl.setManualFocus(130)
        
        # Create outputs
        depth_out = pipeline.create(dai.node.XLinkOut)
        depth_out.setStreamName("depth")
        depth.depth.link(depth_out.input)
        
        rgb_out = pipeline.create(dai.node.XLinkOut)
        rgb_out.setStreamName("rgb")
        cam_rgb.isp.link(rgb_out.input)
        
        return pipeline
    
    def _depth_to_pointcloud(self, depth_frame: np.ndarray, 
                           rgb_frame: np.ndarray) -> PointCloudData:
        """Convert depth image to point cloud"""
        height, width = depth_frame.shape
        
        # Get camera intrinsics
        fx, fy, cx, cy = self.intrinsics
        
        # Create mesh grid
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        
        # Calculate 3D points
        z = depth_frame.astype(np.float32) / 1000.0  # Convert mm to meters
        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy
        
        # Stack coordinates
        points = np.stack([x, y, z], axis=-1)
        
        # Filter out invalid points
        mask = (z > self.config.min_depth / 1000.0) & (z < self.config.max_depth / 1000.0)
        points = points[mask]
        
        # Get corresponding colors
        colors = rgb_frame[mask] / 255.0
        
        return PointCloudData(
            points=points.reshape(-1, 3),
            colors=colors.reshape(-1, 3),
            normals=None,
            timestamp=time.time()
        )
    
    def _get_intrinsics(self) -> Tuple[float, float, float, float]:
        """Get camera intrinsics for point cloud generation"""
        # Get calibration data for RGB camera
        w, h = self.config.resolution
        calib_data = self.calibration_data.getCameraIntrinsics(
            dai.CameraBoardSocket.RGB,
            w, h
        )
        
        # Extract focal lengths and principal point
        fx = calib_data[0][0]
        fy = calib_data[1][1]
        cx = calib_data[0][2]
        cy = calib_data[1][2]
        
        return fx, fy, cx, cy
    
    @property
    def is_running(self) -> bool:
        """Check if camera is running"""
        return self._running


# ============================================================================
# Point Cloud Processor
# ============================================================================

class PointCloudProcessor:
    """Processes and filters point clouds"""
    
    def __init__(self, config: Scanner3DConfig):
        self.config = config
        logger.debug("Initialized PointCloudProcessor")
    
    def process(self, point_cloud_data: PointCloudData) -> o3d.geometry.PointCloud:
        """Process raw point cloud data into Open3D point cloud"""
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_data.points)
        
        if point_cloud_data.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(point_cloud_data.colors)
        
        # Apply filtering
        pcd = self.downsample(pcd)
        pcd = self.remove_outliers(pcd)
        pcd = self.estimate_normals(pcd)
        
        return pcd
    
    def downsample(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Downsample point cloud using voxel grid"""
        if self.config.voxel_size > 0:
            pcd = pcd.voxel_down_sample(self.config.voxel_size)
            logger.debug(f"Downsampled to {len(pcd.points)} points")
        return pcd
    
    def remove_outliers(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Remove statistical outliers from point cloud"""
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=self.config.statistical_outlier_neighbors,
            std_ratio=self.config.statistical_outlier_std_ratio
        )
        logger.debug(f"Removed outliers, {len(pcd.points)} points remaining")
        return pcd
    
    def estimate_normals(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Estimate point cloud normals"""
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30
            )
        )
        pcd.orient_normals_consistent_tangent_plane(30)
        return pcd
    
    def merge_point_clouds(self, point_clouds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
        """Merge multiple point clouds into one"""
        if not point_clouds:
            return o3d.geometry.PointCloud()
        
        if len(point_clouds) == 1:
            return point_clouds[0]
        
        # Merge all point clouds
        merged = point_clouds[0]
        for pcd in point_clouds[1:]:
            if self.config.auto_align:
                # Align using ICP
                transformation = self.align_point_clouds(merged, pcd)
                pcd.transform(transformation)
            merged += pcd
        
        # Post-process merged cloud
        merged = self.downsample(merged)
        merged = self.remove_outliers(merged)
        merged = self.estimate_normals(merged)
        
        return merged
    
    def align_point_clouds(self, source: o3d.geometry.PointCloud, 
                          target: o3d.geometry.PointCloud) -> np.ndarray:
        """Align two point clouds using ICP"""
        # Initial alignment using RANSAC
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target,
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.075),
            max_correspondence_distance=0.075,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ransac_n=3,
            checkers=[],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        # Fine alignment using ICP
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target,
            self.config.icp_threshold,
            result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        return result_icp.transformation


# ============================================================================
# Mesh Generator
# ============================================================================

class MeshGenerator:
    """Generates meshes from point clouds"""
    
    def __init__(self, config: Scanner3DConfig):
        self.config = config
        logger.debug("Initialized MeshGenerator")
    
    def generate_mesh(self, pcd: o3d.geometry.PointCloud) -> trimesh.Trimesh:
        """Generate mesh from point cloud using Poisson reconstruction"""
        # Ensure normals are computed
        if not pcd.has_normals():
            pcd.estimate_normals()
        
        # Poisson surface reconstruction
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=self._get_poisson_depth(),
            width=0,
            scale=1.1,
            linear_fit=False
        )
        
        # Remove spurious vertices
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # Convert to trimesh for more operations
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Simplify if needed
        if self.config.simplification_target:
            trimesh_mesh = trimesh_mesh.simplify_quadric_decimation(
                self.config.simplification_target
            )
        
        # Clean up
        trimesh_mesh.remove_duplicate_faces()
        trimesh_mesh.remove_degenerate_faces()
        trimesh_mesh.fill_holes()
        
        logger.info(f"Generated mesh with {len(trimesh_mesh.vertices)} vertices and {len(trimesh_mesh.faces)} faces")
        
        return trimesh_mesh
    
    def _get_poisson_depth(self) -> int:
        """Get Poisson reconstruction depth based on quality setting"""
        depth_map = {
            MeshQuality.DRAFT: 7,
            MeshQuality.STANDARD: 9,
            MeshQuality.HIGH: 11
        }
        return depth_map.get(self.config.mesh_quality, 9)
    
    def generate_tool_inverse(self, mesh: trimesh.Trimesh, 
                            tool_type: ToolType = ToolType.GRIPPER,
                            offset: float = 2.0) -> trimesh.Trimesh:
        """Generate inverse tool mesh (negative space) for gripping/molding"""
        
        # Get bounding box
        bounds = mesh.bounds
        center = mesh.center_mass
        
        # Create enclosing shape based on tool type
        if tool_type == ToolType.GRIPPER:
            # Create gripper fingers around object
            tool_mesh = self._create_gripper_inverse(mesh, offset)
        elif tool_type == ToolType.SOCKET:
            # Create socket wrench inverse
            tool_mesh = self._create_socket_inverse(mesh, offset)
        elif tool_type == ToolType.MOLD:
            # Create casting mold
            tool_mesh = self._create_mold_inverse(mesh, offset)
        elif tool_type == ToolType.FIXTURE:
            # Create holding fixture
            tool_mesh = self._create_fixture_inverse(mesh, offset)
        else:
            # Default: create simple enclosing box
            tool_mesh = self._create_box_inverse(mesh, offset)
        
        logger.info(f"Generated {tool_type.value} inverse tool")
        return tool_mesh
    
    def _create_gripper_inverse(self, mesh: trimesh.Trimesh, offset: float) -> trimesh.Trimesh:
        """Create gripper fingers that conform to object shape"""
        # Create two gripper fingers
        bounds = mesh.bounds
        width = bounds[1][0] - bounds[0][0]
        height = bounds[1][1] - bounds[0][1]
        depth = bounds[1][2] - bounds[0][2]
        
        # Create finger boxes
        finger_width = width / 3
        finger_height = height + offset * 2
        finger_depth = depth + offset * 2
        
        # Left finger
        left_finger = trimesh.creation.box(
            extents=[finger_width, finger_height, finger_depth]
        )
        left_finger.apply_translation([
            bounds[0][0] - finger_width/2 - offset,
            mesh.center_mass[1],
            mesh.center_mass[2]
        ])
        
        # Right finger
        right_finger = trimesh.creation.box(
            extents=[finger_width, finger_height, finger_depth]
        )
        right_finger.apply_translation([
            bounds[1][0] + finger_width/2 + offset,
            mesh.center_mass[1],
            mesh.center_mass[2]
        ])
        
        # Combine fingers
        gripper = trimesh.util.concatenate([left_finger, right_finger])
        
        # Subtract object shape to create conforming grip
        gripper = gripper.difference(mesh)
        
        return gripper
    
    def _create_socket_inverse(self, mesh: trimesh.Trimesh, offset: float) -> trimesh.Trimesh:
        """Create socket wrench inverse"""
        # Find the convex hull for simpler socket shape
        hull = mesh.convex_hull
        
        # Create cylinder around the hull
        bounds = hull.bounds
        radius = max(bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1]) / 2 + offset
        height = (bounds[1][2] - bounds[0][2]) + offset * 2
        
        socket = trimesh.creation.cylinder(
            radius=radius,
            height=height,
            sections=32
        )
        socket.apply_translation(hull.center_mass)
        
        # Subtract the object to create socket cavity
        socket = socket.difference(hull)
        
        return socket
    
    def _create_mold_inverse(self, mesh: trimesh.Trimesh, offset: float) -> trimesh.Trimesh:
        """Create casting mold (two-part)"""
        bounds = mesh.bounds
        
        # Create enclosing box
        box_extents = [
            bounds[1][0] - bounds[0][0] + offset * 2,
            bounds[1][1] - bounds[0][1] + offset * 2,
            bounds[1][2] - bounds[0][2] + offset * 2
        ]
        
        mold_box = trimesh.creation.box(extents=box_extents)
        mold_box.apply_translation(mesh.center_mass)
        
        # Subtract the object
        mold = mold_box.difference(mesh)
        
        # Split into two parts at the center
        plane_origin = mesh.center_mass
        plane_normal = [0, 0, 1]  # Split horizontally
        
        top_mold = mold.slice_plane(plane_origin, plane_normal)
        bottom_mold = mold.slice_plane(plane_origin, -np.array(plane_normal))
        
        # Add registration features (alignment pins)
        # This is simplified - real molds need more complex features
        
        return trimesh.util.concatenate([top_mold, bottom_mold])
    
    def _create_fixture_inverse(self, mesh: trimesh.Trimesh, offset: float) -> trimesh.Trimesh:
        """Create holding fixture"""
        # Create a cradle-like fixture
        bounds = mesh.bounds
        
        # Create base plate
        base_thickness = 10  # mm
        base = trimesh.creation.box(
            extents=[
                bounds[1][0] - bounds[0][0] + offset * 4,
                bounds[1][1] - bounds[0][1] + offset * 4,
                base_thickness
            ]
        )
        base.apply_translation([
            mesh.center_mass[0],
            mesh.center_mass[1],
            bounds[0][2] - base_thickness/2 - offset
        ])
        
        # Create support walls
        wall_height = (bounds[1][2] - bounds[0][2]) / 2
        wall_thickness = 5  # mm
        
        # Create V-shaped cradle
        support = trimesh.creation.box(
            extents=[
                bounds[1][0] - bounds[0][0] + offset * 2,
                bounds[1][1] - bounds[0][1] + offset * 2,
                wall_height
            ]
        )
        support.apply_translation([
            mesh.center_mass[0],
            mesh.center_mass[1],
            bounds[0][2] + wall_height/2
        ])
        
        # Combine base and support
        fixture = trimesh.util.concatenate([base, support])
        
        # Subtract object shape to create conforming cradle
        fixture = fixture.difference(mesh)
        
        return fixture
    
    def _create_box_inverse(self, mesh: trimesh.Trimesh, offset: float) -> trimesh.Trimesh:
        """Create simple box inverse (default)"""
        bounds = mesh.bounds
        
        # Create enclosing box
        box_extents = [
            bounds[1][0] - bounds[0][0] + offset * 2,
            bounds[1][1] - bounds[0][1] + offset * 2,
            bounds[1][2] - bounds[0][2] + offset * 2
        ]
        
        enclosing_box = trimesh.creation.box(extents=box_extents)
        enclosing_box.apply_translation(mesh.center_mass)
        
        # Subtract the object
        inverse = enclosing_box.difference(mesh)
        
        return inverse


# ============================================================================
# Main Scanner3D Class
# ============================================================================

class Scanner3D:
    """
    Main 3D Scanner class for OAK-D camera
    Provides complete 3D scanning pipeline from capture to STL export
    """
    
    def __init__(self, config: Optional[Scanner3DConfig] = None):
        """
        Initialize 3D Scanner
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or Scanner3DConfig()
        
        # Configure logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        # Initialize components
        self.camera = OakD3DCamera(self.config)
        self.processor = PointCloudProcessor(self.config)
        self.mesh_generator = MeshGenerator(self.config)
        
        # Scanning state
        self._scanning = False
        self._point_clouds: List[o3d.geometry.PointCloud] = []
        self._current_scan: Optional[ScanResult] = None
        self._scan_metadata: Dict[str, Any] = {}
        
        # Thread safety
        self._lock = threading.RLock() if self.config.enable_threading else None
        
        # Create output directory
        Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("3D Scanner initialized")
    
    # ========================================================================
    # Camera Control
    # ========================================================================
    
    def start_camera(self) -> bool:
        """Start the OAK-D camera for 3D scanning"""
        with self._get_lock():
            return self.camera.start()
    
    def stop_camera(self) -> None:
        """Stop the OAK-D camera"""
        with self._get_lock():
            self.camera.stop()
    
    def is_camera_running(self) -> bool:
        """Check if camera is running"""
        with self._get_lock():
            return self.camera.is_running
    
    # ========================================================================
    # Scanning Operations
    # ========================================================================
    
    def start_scan(self, scan_mode: Optional[ScanMode] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Start a new 3D scanning session
        
        Args:
            scan_mode: Scanning mode to use
            metadata: Optional metadata for the scan
        
        Returns:
            True if scan started successfully
        """
        with self._get_lock():
            if self._scanning:
                logger.warning("Scan already in progress")
                return False
            
            if not self.camera.is_running:
                if not self.start_camera():
                    return False
            
            self._scanning = True
            self._point_clouds.clear()
            self._current_scan = None
            self._scan_metadata = metadata or {}
            self._scan_metadata['start_time'] = time.time()
            self._scan_metadata['scan_mode'] = (scan_mode or self.config.scan_mode).value
            
            logger.info(f"Started {self._scan_metadata['scan_mode']} scan")
            return True
    
    def capture_point_cloud(self) -> Optional[o3d.geometry.PointCloud]:
        """
        Capture a single point cloud
        
        Returns:
            Processed point cloud or None if failed
        """
        with self._get_lock():
            if not self._scanning:
                logger.error("No scan in progress. Call start_scan() first")
                return None
            
            # Capture raw point cloud
            raw_data = self.camera.capture_point_cloud()
            if raw_data is None:
                return None
            
            # Process point cloud
            pcd = self.processor.process(raw_data)
            
            # Store for later merging
            self._point_clouds.append(pcd)
            
            logger.info(f"Captured point cloud {len(self._point_clouds)} with {len(pcd.points)} points")
            
            return pcd
    
    def capture_multiple(self, num_captures: int, 
                        delay_seconds: float = 1.0) -> List[o3d.geometry.PointCloud]:
        """
        Capture multiple point clouds with delay
        
        Args:
            num_captures: Number of captures to perform
            delay_seconds: Delay between captures
        
        Returns:
            List of captured point clouds
        """
        captures = []
        
        for i in range(num_captures):
            logger.info(f"Capturing {i+1}/{num_captures}")
            pcd = self.capture_point_cloud()
            if pcd:
                captures.append(pcd)
            
            if i < num_captures - 1:
                time.sleep(delay_seconds)
        
        return captures
    
    def complete_scan(self, generate_mesh: bool = True) -> Optional[ScanResult]:
        """
        Complete the current scan and generate final result
        
        Args:
            generate_mesh: Whether to generate mesh from point cloud
        
        Returns:
            ScanResult object or None if failed
        """
        with self._get_lock():
            if not self._scanning:
                logger.error("No scan in progress")
                return None
            
            if not self._point_clouds:
                logger.error("No point clouds captured")
                return None
            
            # Merge all point clouds
            logger.info(f"Merging {len(self._point_clouds)} point clouds")
            merged_pcd = self.processor.merge_point_clouds(self._point_clouds)
            
            # Generate mesh if requested
            mesh = None
            if generate_mesh:
                logger.info("Generating mesh from point cloud")
                mesh = self.mesh_generator.generate_mesh(merged_pcd)
            
            # Calculate bounding box
            bbox = merged_pcd.get_axis_aligned_bounding_box()
            bbox_dict = {
                'min': bbox.min_bound.tolist(),
                'max': bbox.max_bound.tolist(),
                'center': bbox.get_center().tolist(),
                'extents': (bbox.max_bound - bbox.min_bound).tolist()
            }
            
            # Update metadata
            self._scan_metadata['end_time'] = time.time()
            self._scan_metadata['duration'] = (
                self._scan_metadata['end_time'] - self._scan_metadata['start_time']
            )
            
            # Create result
            self._current_scan = ScanResult(
                point_cloud=merged_pcd,
                mesh=mesh,
                scan_mode=ScanMode(self._scan_metadata['scan_mode']),
                num_captures=len(self._point_clouds),
                total_points=len(merged_pcd.points),
                bounding_box=bbox_dict,
                metadata=self._scan_metadata.copy()
            )
            
            # Reset scanning state
            self._scanning = False
            
            logger.info(f"Scan completed: {self._current_scan.total_points} points, "
                       f"{self._current_scan.num_captures} captures")
            
            # Auto-save if configured
            if self.config.auto_save:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                self.save_scan(f"scan_{timestamp}")
            
            return self._current_scan
    
    def cancel_scan(self) -> None:
        """Cancel the current scan"""
        with self._get_lock():
            if self._scanning:
                self._scanning = False
                self._point_clouds.clear()
                self._scan_metadata.clear()
                logger.info("Scan cancelled")
    
    # ========================================================================
    # Tool Generation
    # ========================================================================
    
    def generate_tool_inverse(self, tool_type: ToolType = ToolType.GRIPPER,
                            offset: float = 2.0,
                            scan_result: Optional[ScanResult] = None) -> Optional[trimesh.Trimesh]:
        """
        Generate inverse tool mesh for gripping/molding
        
        Args:
            tool_type: Type of tool to generate
            offset: Offset from object surface in mm
            scan_result: Scan result to use (uses current if None)
        
        Returns:
            Tool mesh or None if failed
        """
        with self._get_lock():
            scan = scan_result or self._current_scan
            
            if not scan:
                logger.error("No scan available")
                return None
            
            if not scan.mesh:
                logger.info("Generating mesh for tool creation")
                scan.mesh = self.mesh_generator.generate_mesh(scan.point_cloud)
            
            # Generate inverse tool
            tool_mesh = self.mesh_generator.generate_tool_inverse(
                scan.mesh, tool_type, offset
            )
            
            return tool_mesh
    
    # ========================================================================
    # Export Functions
    # ========================================================================
    
    def export_stl(self, filename: str, mesh: Optional[trimesh.Trimesh] = None) -> bool:
        """
        Export mesh as STL file
        
        Args:
            filename: Output filename (without extension)
            mesh: Mesh to export (uses current scan mesh if None)
        
        Returns:
            True if successful
        """
        with self._get_lock():
            if mesh is None:
                if not self._current_scan or not self._current_scan.mesh:
                    logger.error("No mesh available to export")
                    return False
                mesh = self._current_scan.mesh
            
            # Ensure .stl extension
            if not filename.endswith('.stl'):
                filename += '.stl'
            
            # Full path
            filepath = Path(self.config.output_directory) / filename
            
            try:
                mesh.export(filepath)
                logger.info(f"Exported STL to {filepath}")
                return True
            except Exception as e:
                logger.error(f"Failed to export STL: {e}")
                return False
    
    def export_pointcloud(self, filename: str, 
                         pcd: Optional[o3d.geometry.PointCloud] = None) -> bool:
        """
        Export point cloud as PLY file
        
        Args:
            filename: Output filename (without extension)
            pcd: Point cloud to export (uses current scan if None)
        
        Returns:
            True if successful
        """
        with self._get_lock():
            if pcd is None:
                if not self._current_scan:
                    logger.error("No point cloud available to export")
                    return False
                pcd = self._current_scan.point_cloud
            
            # Ensure .ply extension
            if not filename.endswith('.ply'):
                filename += '.ply'
            
            # Full path
            filepath = Path(self.config.output_directory) / filename
            
            try:
                o3d.io.write_point_cloud(str(filepath), pcd)
                logger.info(f"Exported point cloud to {filepath}")
                return True
            except Exception as e:
                logger.error(f"Failed to export point cloud: {e}")
                return False
    
    def save_scan(self, name: str) -> bool:
        """
        Save complete scan with all data
        
        Args:
            name: Base name for saved files
        
        Returns:
            True if successful
        """
        with self._get_lock():
            if not self._current_scan:
                logger.error("No scan to save")
                return False
            
            success = True
            
            # Export point cloud
            if not self.export_pointcloud(f"{name}_pointcloud"):
                success = False
            
            # Export mesh
            if self._current_scan.mesh:
                if not self.export_stl(f"{name}_mesh"):
                    success = False
            
            # Save metadata
            metadata_file = Path(self.config.output_directory) / f"{name}_metadata.json"
            try:
                with open(metadata_file, 'w') as f:
                    json.dump(self._current_scan.metadata, f, indent=2)
                logger.info(f"Saved metadata to {metadata_file}")
            except Exception as e:
                logger.error(f"Failed to save metadata: {e}")
                success = False
            
            return success
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_current_scan(self) -> Optional[ScanResult]:
        """Get the current scan result"""
        with self._get_lock():
            return self._current_scan
    
    def clear_scan(self) -> None:
        """Clear the current scan data"""
        with self._get_lock():
            self._current_scan = None
            self._point_clouds.clear()
            self._scan_metadata.clear()
    
    def _get_lock(self):
        """Get thread lock if threading is enabled"""
        if self._lock:
            return self._lock
        else:
            # Return a dummy context manager if threading is disabled
            from contextlib import contextmanager
            return contextmanager(lambda: iter([None]))()
    
    # ========================================================================
    # Context Manager Support
    # ========================================================================
    
    def __enter__(self):
        """Context manager entry"""
        self.start_camera()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._scanning:
            self.cancel_scan()
        self.stop_camera()


# ============================================================================
# Factory Functions
# ============================================================================

def create_3d_scanner(
    output_directory: str = "scans",
    mesh_quality: MeshQuality = MeshQuality.STANDARD,
    scan_mode: ScanMode = ScanMode.SINGLE_SHOT,
    **kwargs
) -> Scanner3D:
    """
    Factory function to create 3D Scanner with common settings
    
    Args:
        output_directory: Directory for saving scans
        mesh_quality: Quality level for mesh generation
        scan_mode: Default scanning mode
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured Scanner3D instance
    """
    config = Scanner3DConfig(
        output_directory=output_directory,
        mesh_quality=mesh_quality,
        scan_mode=scan_mode,
        **kwargs
    )
    
    return Scanner3D(config)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Scan an object and create a gripper tool
    scanner = create_3d_scanner(
        output_directory="robot_tools",
        mesh_quality=MeshQuality.HIGH
    )
    
    with scanner:
        # Start scanning
        scanner.start_scan(ScanMode.MULTI_ANGLE)
        
        # Capture from multiple angles
        print("Position object and press Enter for each capture...")
        for i in range(3):
            input(f"Capture {i+1}/3 - Press Enter...")
            scanner.capture_point_cloud()
        
        # Complete scan and generate mesh
        result = scanner.complete_scan()
        
        if result:
            print(f"Scan complete: {result.total_points} points")
            
            # Generate gripper tool
            tool = scanner.generate_tool_inverse(
                tool_type=ToolType.GRIPPER,
                offset=3.0
            )
            
            if tool:
                # Export for 3D printing
                scanner.export_stl("gripper_tool", tool)
                print("Gripper tool STL exported!")