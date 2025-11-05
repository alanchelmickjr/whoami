"""
Robot Vision API - Master orchestrator for robot vision capabilities
Integrates face recognition, 3D scanning, gimbal control, and future modules (SLAM, personas, etc.)
"""

import logging
import threading
from typing import Optional, Dict, Any, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import numpy as np
import time

# Import vision modules
from .face_recognition_api import (
    FaceRecognitionAPI,
    RecognitionConfig,
    CameraType,
    RecognitionResult
)
from .scanner_3d import (
    Scanner3D,
    Scanner3DConfig,
    ScanMode,
    MeshQuality,
    ToolType,
    ScanResult
)

# Import gimbal control modules
try:
    from .vision_behaviors import (
        VisionTrackingSystem,
        BehaviorType,
        BehaviorConfig,
        ScanPattern,
        create_vision_tracking_system
    )
    from .gimbal_control import MovementProfile
    GIMBAL_AVAILABLE = True
except ImportError:
    logger.warning("Gimbal control modules not available - servo control disabled")
    GIMBAL_AVAILABLE = False
    VisionTrackingSystem = None
    BehaviorType = None
    BehaviorConfig = None
    ScanPattern = None
    MovementProfile = None

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================

class VisionModule(Enum):
    """Available vision modules"""
    FACE_RECOGNITION = "face_recognition"
    SCANNER_3D = "3d_scanner"
    GIMBAL_CONTROL = "gimbal_control"    # Gimbal/servo control for camera movement
    SLAM = "slam"                    # Future: Simultaneous Localization and Mapping
    OBJECT_DETECTION = "object_detection"  # Future: General object detection
    PERSONA = "persona"               # Future: Personality/behavior system
    GESTURE = "gesture"               # Future: Gesture recognition
    DEPTH = "depth"                   # Future: Depth perception


class RobotTask(Enum):
    """Common robot vision tasks"""
    IDENTIFY_PERSON = "identify_person"
    SCAN_OBJECT = "scan_object"
    CREATE_TOOL = "create_tool"
    MAP_ENVIRONMENT = "map_environment"
    TRACK_PERSON = "track_person"
    RECOGNIZE_GESTURE = "recognize_gesture"
    LEARN_OBJECT = "learn_object"


@dataclass
class VisionCapability:
    """Description of a vision capability"""
    module: VisionModule
    name: str
    description: str
    enabled: bool = True
    instance: Optional[Any] = None
    config: Optional[Any] = None


@dataclass
class RobotVisionConfig:
    """Configuration for Robot Vision API"""
    # Camera settings
    camera_type: CameraType = CameraType.OAK_D
    camera_resolution: tuple = (1280, 720)
    camera_fps: int = 30
    
    # Module enablement
    enable_face_recognition: bool = True
    enable_3d_scanning: bool = True
    enable_gimbal_control: bool = False  # Gimbal/servo control
    enable_slam: bool = False  # Future feature
    enable_object_detection: bool = False  # Future feature
    
    # Module configurations
    face_recognition_config: Optional[RecognitionConfig] = None
    scanner_3d_config: Optional[Scanner3DConfig] = None
    gimbal_config: Optional[Dict[str, Any]] = None
    behavior_config: Optional[Any] = None  # BehaviorConfig when available
    
    # Gimbal settings
    gimbal_serial_port: str = "/dev/ttyUSB0"
    gimbal_baudrate: int = 1000000
    enable_vision_tracking: bool = True  # Auto-track faces
    enable_curiosity_mode: bool = True  # Random exploration when idle
    
    # Behavior settings
    auto_start_camera: bool = True
    share_camera: bool = True  # Share camera between modules
    
    # Threading
    enable_threading: bool = True
    
    # Logging
    log_level: str = "INFO"
    
    # Data paths
    data_directory: str = "robot_vision_data"
    face_database_path: str = "robot_faces.pkl"
    scan_output_directory: str = "robot_scans"


@dataclass
class VisionEvent:
    """Event from vision system"""
    module: VisionModule
    event_type: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Module Adapters
# ============================================================================

class FaceRecognitionAdapter:
    """Adapter for Face Recognition module"""
    
    def __init__(self, config: Optional[RecognitionConfig] = None):
        self.api = FaceRecognitionAPI(config or RecognitionConfig())
        self.active = False
    
    def start(self) -> bool:
        """Start face recognition"""
        if self.api.start_camera():
            self.active = True
            return True
        return False
    
    def stop(self) -> None:
        """Stop face recognition"""
        self.api.stop_camera()
        self.active = False
    
    def process_frame(self, frame: Optional[np.ndarray] = None) -> List[RecognitionResult]:
        """Process a frame for face recognition"""
        return self.api.process_frame(frame)
    
    def add_face(self, name: str, frame: Optional[np.ndarray] = None) -> bool:
        """Add a face to the database"""
        return self.api.add_face(name, frame)
    
    def get_known_people(self) -> List[str]:
        """Get list of known people"""
        return self.api.get_all_names()


class Scanner3DAdapter:
    """Adapter for 3D Scanner module"""
    
    def __init__(self, config: Optional[Scanner3DConfig] = None):
        self.scanner = Scanner3D(config or Scanner3DConfig())
        self.active = False
    
    def start(self) -> bool:
        """Start 3D scanner"""
        if self.scanner.start_camera():
            self.active = True
            return True
        return False
    
    def stop(self) -> None:
        """Stop 3D scanner"""
        self.scanner.stop_camera()
        self.active = False
    
    def start_scan(self, mode: ScanMode = ScanMode.SINGLE_SHOT) -> bool:
        """Start a scanning session"""
        return self.scanner.start_scan(mode)
    
    def capture_point_cloud(self):
        """Capture a point cloud"""
        return self.scanner.capture_point_cloud()
    
    def complete_scan(self) -> Optional[ScanResult]:
        """Complete the scan and generate mesh"""
        return self.scanner.complete_scan()
    
    def create_tool(self, tool_type: ToolType, offset: float = 2.0):
        """Create a tool from the last scan"""
        return self.scanner.generate_tool_inverse(tool_type, offset)
    
    def export_stl(self, filename: str, mesh=None) -> bool:
        """Export mesh as STL"""
        return self.scanner.export_stl(filename, mesh)
    
    
    class GimbalControlAdapter:
        """Adapter for Gimbal Control module"""
        
        def __init__(self, gimbal_config: Optional[Dict] = None,
                     behavior_config: Optional[Any] = None):
            if not GIMBAL_AVAILABLE:
                raise ImportError("Gimbal control modules not available")
            
            self.system = create_vision_tracking_system(
                gimbal_config=gimbal_config or {},
                behavior_config=behavior_config
            )
            self.active = False
            self.tracking_enabled = False
        
        def start(self) -> bool:
            """Start gimbal control system"""
            if self.system.connect():
                self.active = True
                return True
            return False
        
        def stop(self) -> None:
            """Stop gimbal control system"""
            self.system.disconnect()
            self.active = False
            self.tracking_enabled = False
        
        def enable_tracking(self, mode: Optional[Any] = None) -> None:
            """Enable automatic vision tracking"""
            if mode is None and BehaviorType:
                mode = BehaviorType.TRACK_PERSON
            if mode:
                self.system.enable_tracking(mode)
                self.tracking_enabled = True
        
        def disable_tracking(self) -> None:
            """Disable automatic vision tracking"""
            self.system.disable_tracking()
            self.tracking_enabled = False
        
        def process_frame(self, recognition_results: List[RecognitionResult],
                         target_name: Optional[str] = None) -> None:
            """Process vision frame for tracking"""
            if self.tracking_enabled:
                self.system.process_vision_frame(recognition_results, target_name)
        
        def center_face(self, face_bbox: Tuple[int, int, int, int]) -> bool:
            """Center on detected face"""
            return self.system.center_on_face(face_bbox)
        
        def scan_for_3d(self, num_positions: int = 8) -> None:
            """Perform scanning pattern for 3D capture"""
            self.system.scan_for_3d(num_positions)
        
        def explore_environment(self) -> None:
            """Start environmental exploration"""
            self.system.explore_environment()
        
        def enable_curiosity(self) -> None:
            """Enable curiosity mode"""
            self.system.enable_curiosity()
        
        def search_for_target(self) -> None:
            """Search for lost target"""
            self.system.search_for_target()
        
        def look_at(self, x: float, y: float, z: float) -> None:
            """Look at 3D point"""
            self.system.look_at(x, y, z)
        
        def home(self) -> None:
            """Return to home position"""
            self.system.home()
        
        def get_position(self) -> Tuple[float, float]:
            """Get current gimbal position"""
            return self.system.get_position()


# ============================================================================
# Main Robot Vision API
# ============================================================================

class RobotVisionAPI:
    """
    Master Robot Vision API - Orchestrates multiple vision capabilities
    
    This is the main entry point for robot vision systems, providing
    unified access to face recognition, 3D scanning, SLAM, and more.
    """
    
    def __init__(self, config: Optional[RobotVisionConfig] = None):
        """
        Initialize Robot Vision API
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or RobotVisionConfig()
        
        # Configure logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        # Initialize capabilities registry
        self.capabilities: Dict[VisionModule, VisionCapability] = {}
        
        # Initialize enabled modules
        self._initialize_modules()
        
        # Camera sharing state
        self._camera_owner: Optional[VisionModule] = None
        
        # Thread safety
        self._lock = threading.RLock() if self.config.enable_threading else None
        
        # Event callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'on_person_identified': [],
            'on_object_scanned': [],
            'on_tool_created': [],
            'on_module_started': [],
            'on_module_stopped': [],
            'on_error': [],
            'on_gimbal_movement': [],
            'on_tracking_target': [],
            'on_behavior_change': []
        }
        
        # Task execution history
        self._task_history: List[Dict[str, Any]] = []
        
        logger.info("Robot Vision API initialized")
    
    def _initialize_modules(self) -> None:
        """Initialize enabled vision modules"""
        
        # Face Recognition
        if self.config.enable_face_recognition:
            face_config = self.config.face_recognition_config or RecognitionConfig(
                camera_type=self.config.camera_type,
                camera_resolution=self.config.camera_resolution,
                database_path=self.config.face_database_path
            )
            
            self.capabilities[VisionModule.FACE_RECOGNITION] = VisionCapability(
                module=VisionModule.FACE_RECOGNITION,
                name="Face Recognition",
                description="Identify and remember people",
                enabled=True,
                instance=FaceRecognitionAdapter(face_config),
                config=face_config
            )
            logger.info("Face Recognition module initialized")
        
        # 3D Scanner
        if self.config.enable_3d_scanning:
            scanner_config = self.config.scanner_3d_config or Scanner3DConfig(
                resolution=self.config.camera_resolution,
                output_directory=self.config.scan_output_directory
            )
            
            self.capabilities[VisionModule.SCANNER_3D] = VisionCapability(
                module=VisionModule.SCANNER_3D,
                name="3D Scanner",
                description="Scan objects and create 3D models",
                enabled=True,
                instance=Scanner3DAdapter(scanner_config),
                config=scanner_config
            )
            logger.info("3D Scanner module initialized")
        
        # Gimbal Control
        if self.config.enable_gimbal_control and GIMBAL_AVAILABLE:
            try:
                gimbal_config = self.config.gimbal_config or {
                    'serial_port': self.config.gimbal_serial_port,
                    'baudrate': self.config.gimbal_baudrate
                }
                
                self.capabilities[VisionModule.GIMBAL_CONTROL] = VisionCapability(
                    module=VisionModule.GIMBAL_CONTROL,
                    name="Gimbal Control",
                    description="Camera movement and tracking control",
                    enabled=True,
                    instance=GimbalControlAdapter(gimbal_config, self.config.behavior_config),
                    config=gimbal_config
                )
                logger.info("Gimbal Control module initialized")
            except Exception as e:
                logger.error(f"Failed to initialize gimbal control: {e}")
                self.config.enable_gimbal_control = False
        
        # Future modules can be added here
        # if self.config.enable_slam:
        #     self.capabilities[VisionModule.SLAM] = ...
    
    # ========================================================================
    # Module Management
    # ========================================================================
    
    def get_available_modules(self) -> List[VisionModule]:
        """Get list of available vision modules"""
        return list(self.capabilities.keys())
    
    def is_module_available(self, module: VisionModule) -> bool:
        """Check if a module is available"""
        return module in self.capabilities and self.capabilities[module].enabled
    
    def get_module(self, module: VisionModule) -> Optional[Any]:
        """Get a specific module instance"""
        with self._get_lock():
            if module in self.capabilities:
                return self.capabilities[module].instance
            return None
    
    def start_module(self, module: VisionModule) -> bool:
        """
        Start a specific vision module
        
        Args:
            module: Module to start
        
        Returns:
            True if successful
        """
        with self._get_lock():
            if not self.is_module_available(module):
                logger.error(f"Module {module.value} not available")
                return False
            
            capability = self.capabilities[module]
            
            # Handle camera sharing
            if self.config.share_camera and self._camera_owner and self._camera_owner != module:
                logger.warning(f"Camera in use by {self._camera_owner.value}")
                return False
            
            try:
                if capability.instance.start():
                    self._camera_owner = module
                    self._trigger_callback('on_module_started', module)
                    logger.info(f"Started {module.value} module")
                    return True
                return False
                
            except Exception as e:
                logger.error(f"Failed to start {module.value}: {e}")
                self._trigger_callback('on_error', e, module)
                return False
    
    def stop_module(self, module: VisionModule) -> None:
        """Stop a specific vision module"""
        with self._get_lock():
            if not self.is_module_available(module):
                return
            
            capability = self.capabilities[module]
            
            try:
                capability.instance.stop()
                if self._camera_owner == module:
                    self._camera_owner = None
                self._trigger_callback('on_module_stopped', module)
                logger.info(f"Stopped {module.value} module")
                
            except Exception as e:
                logger.error(f"Failed to stop {module.value}: {e}")
                self._trigger_callback('on_error', e, module)
    
    def switch_module(self, from_module: VisionModule, to_module: VisionModule) -> bool:
        """
        Switch from one module to another
        
        Args:
            from_module: Currently active module
            to_module: Module to switch to
        
        Returns:
            True if successful
        """
        with self._get_lock():
            self.stop_module(from_module)
            time.sleep(0.5)  # Brief delay for camera release
            return self.start_module(to_module)
    
    # ========================================================================
    # Face Recognition Tasks
    # ========================================================================
    
    def identify_person(self, frame: Optional[np.ndarray] = None) -> List[RecognitionResult]:
        """
        Identify people in the current view
        
        Args:
            frame: Optional frame to process
        
        Returns:
            List of recognition results
        """
        with self._get_lock():
            face_module = self.get_module(VisionModule.FACE_RECOGNITION)
            if not face_module:
                logger.error("Face recognition module not available")
                return []
            
            # Ensure module is started
            if not face_module.active:
                if not self.start_module(VisionModule.FACE_RECOGNITION):
                    return []
            
            results = face_module.process_frame(frame)
            
            # Send to gimbal for tracking if enabled
            if self.config.enable_gimbal_control and self.config.enable_vision_tracking:
                gimbal_module = self.get_module(VisionModule.GIMBAL_CONTROL)
                if gimbal_module and gimbal_module.active:
                    gimbal_module.process_frame(results)
            
            # Trigger callbacks for identified people
            for result in results:
                if result.name != "Unknown":
                    self._trigger_callback('on_person_identified', result)
            
            # Log task
            self._log_task(RobotTask.IDENTIFY_PERSON, {'results': len(results)})
            
            return results
    
    def learn_person(self, name: str, frame: Optional[np.ndarray] = None) -> bool:
        """
        Learn a new person's face
        
        Args:
            name: Person's name
            frame: Optional frame containing the face
        
        Returns:
            True if successful
        """
        with self._get_lock():
            face_module = self.get_module(VisionModule.FACE_RECOGNITION)
            if not face_module:
                logger.error("Face recognition module not available")
                return False
            
            # Ensure module is started
            if not face_module.active:
                if not self.start_module(VisionModule.FACE_RECOGNITION):
                    return False
            
            success = face_module.add_face(name, frame)
            
            # Log task
            self._log_task(RobotTask.IDENTIFY_PERSON, {'action': 'learn', 'name': name, 'success': success})
            
            return success
    
    def get_known_people(self) -> List[str]:
        """Get list of people the robot knows"""
        with self._get_lock():
            face_module = self.get_module(VisionModule.FACE_RECOGNITION)
            if face_module:
                return face_module.get_known_people()
            return []
    
    # ========================================================================
    # 3D Scanning Tasks
    # ========================================================================
    
    def scan_object(self, scan_mode: ScanMode = ScanMode.SINGLE_SHOT,
                   num_captures: int = 1) -> Optional[ScanResult]:
        """
        Scan a 3D object
        
        Args:
            scan_mode: Scanning mode to use
            num_captures: Number of captures for multi-angle mode
        
        Returns:
            Scan result or None if failed
        """
        with self._get_lock():
            scanner = self.get_module(VisionModule.SCANNER_3D)
            if not scanner:
                logger.error("3D scanner module not available")
                return None
            
            # Ensure module is started
            if not scanner.active:
                if not self.start_module(VisionModule.SCANNER_3D):
                    return None
            
            # Start scan
            if not scanner.start_scan(scan_mode):
                return None
            
            # Capture point clouds
            if scan_mode == ScanMode.SINGLE_SHOT:
                scanner.capture_point_cloud()
            else:
                for i in range(num_captures):
                    logger.info(f"Capturing {i+1}/{num_captures}")
                    scanner.capture_point_cloud()
                    if i < num_captures - 1:
                        time.sleep(1.0)  # Delay between captures
            
            # Complete scan
            result = scanner.complete_scan()
            
            if result:
                self._trigger_callback('on_object_scanned', result)
                self._log_task(RobotTask.SCAN_OBJECT, {
                    'mode': scan_mode.value,
                    'captures': num_captures,
                    'points': result.total_points
                })
            
            return result
    
    def create_tool(self, tool_type: ToolType = ToolType.GRIPPER,
                   offset: float = 2.0,
                   export_stl: bool = True,
                   filename: Optional[str] = None) -> bool:
        """
        Create a tool from the last scanned object
        
        Args:
            tool_type: Type of tool to create
            offset: Offset from object surface
            export_stl: Whether to export as STL file
            filename: Optional filename for STL export
        
        Returns:
            True if successful
        """
        with self._get_lock():
            scanner = self.get_module(VisionModule.SCANNER_3D)
            if not scanner:
                logger.error("3D scanner module not available")
                return False
            
            # Generate tool
            tool_mesh = scanner.create_tool(tool_type, offset)
            if not tool_mesh:
                logger.error("Failed to create tool")
                return False
            
            # Export if requested
            if export_stl:
                if not filename:
                    filename = f"{tool_type.value}_tool_{int(time.time())}"
                
                if not scanner.export_stl(filename, tool_mesh):
                    logger.error("Failed to export STL")
                    return False
            
            self._trigger_callback('on_tool_created', {
                'tool_type': tool_type,
                'filename': filename
            })
            
            self._log_task(RobotTask.CREATE_TOOL, {
                'tool_type': tool_type.value,
                'offset': offset,
                'exported': export_stl
            })
            
            return True
    
    # ========================================================================
    # Gimbal Control Tasks
    # ========================================================================
    
    def start_gimbal_control(self) -> bool:
        """
        Start gimbal control system
        
        Returns:
            True if successful
        """
        if not self.config.enable_gimbal_control:
            logger.warning("Gimbal control not enabled in configuration")
            return False
        
        with self._get_lock():
            return self.start_module(VisionModule.GIMBAL_CONTROL)
    
    def stop_gimbal_control(self) -> None:
        """Stop gimbal control system"""
        with self._get_lock():
            self.stop_module(VisionModule.GIMBAL_CONTROL)
    
    def enable_face_tracking(self, target_name: Optional[str] = None) -> bool:
        """
        Enable automatic face tracking
        
        Args:
            target_name: Optional specific person to track
        
        Returns:
            True if tracking enabled
        """
        with self._get_lock():
            gimbal_module = self.get_module(VisionModule.GIMBAL_CONTROL)
            if not gimbal_module:
                logger.error("Gimbal control module not available")
                return False
            
            if not gimbal_module.active:
                if not self.start_gimbal_control():
                    return False
            
            gimbal_module.enable_tracking()
            self._trigger_callback('on_behavior_change', 'face_tracking')
            return True
    
    def disable_face_tracking(self) -> None:
        """Disable automatic face tracking"""
        with self._get_lock():
            gimbal_module = self.get_module(VisionModule.GIMBAL_CONTROL)
            if gimbal_module:
                gimbal_module.disable_tracking()
    
    def center_on_face(self, face_bbox: Tuple[int, int, int, int]) -> bool:
        """
        Center gimbal on a detected face
        
        Args:
            face_bbox: Face bounding box (x, y, width, height)
        
        Returns:
            True if movement initiated
        """
        with self._get_lock():
            gimbal_module = self.get_module(VisionModule.GIMBAL_CONTROL)
            if not gimbal_module or not gimbal_module.active:
                return False
            
            return gimbal_module.center_face(face_bbox)
    
    def look_at_3d_point(self, x: float, y: float, z: float) -> bool:
        """
        Point camera at a 3D coordinate
        
        Args:
            x, y, z: Target point in camera space (meters)
        
        Returns:
            True if successful
        """
        with self._get_lock():
            gimbal_module = self.get_module(VisionModule.GIMBAL_CONTROL)
            if not gimbal_module or not gimbal_module.active:
                return False
            
            gimbal_module.look_at(x, y, z)
            self._trigger_callback('on_gimbal_movement', (x, y, z))
            return True
    
    def scan_with_gimbal(self, num_positions: int = 8) -> bool:
        """
        Perform scanning pattern with gimbal for 3D capture
        
        Args:
            num_positions: Number of scan positions
        
        Returns:
            True if scan started
        """
        with self._get_lock():
            gimbal_module = self.get_module(VisionModule.GIMBAL_CONTROL)
            if not gimbal_module or not gimbal_module.active:
                return False
            
            gimbal_module.scan_for_3d(num_positions)
            self._trigger_callback('on_behavior_change', 'scanning')
            return True
    
    def explore_with_gimbal(self) -> bool:
        """
        Start environmental exploration with gimbal
        
        Returns:
            True if exploration started
        """
        with self._get_lock():
            gimbal_module = self.get_module(VisionModule.GIMBAL_CONTROL)
            if not gimbal_module or not gimbal_module.active:
                return False
            
            gimbal_module.explore_environment()
            self._trigger_callback('on_behavior_change', 'exploring')
            return True
    
    def enable_curiosity_mode(self) -> bool:
        """
        Enable curiosity mode for idle exploration
        
        Returns:
            True if enabled
        """
        with self._get_lock():
            gimbal_module = self.get_module(VisionModule.GIMBAL_CONTROL)
            if not gimbal_module or not gimbal_module.active:
                return False
            
            gimbal_module.enable_curiosity()
            self._trigger_callback('on_behavior_change', 'curiosity')
            return True
    
    def search_for_person(self) -> bool:
        """
        Search for a lost tracked person
        
        Returns:
            True if search started
        """
        with self._get_lock():
            gimbal_module = self.get_module(VisionModule.GIMBAL_CONTROL)
            if not gimbal_module or not gimbal_module.active:
                return False
            
            gimbal_module.search_for_target()
            self._trigger_callback('on_behavior_change', 'searching')
            return True
    
    def gimbal_home(self) -> bool:
        """
        Return gimbal to home position
        
        Returns:
            True if successful
        """
        with self._get_lock():
            gimbal_module = self.get_module(VisionModule.GIMBAL_CONTROL)
            if not gimbal_module or not gimbal_module.active:
                return False
            
            gimbal_module.home()
            self._trigger_callback('on_gimbal_movement', 'home')
            return True
    
    def get_gimbal_position(self) -> Optional[Tuple[float, float]]:
        """
        Get current gimbal position
        
        Returns:
            (pan, tilt) angles in degrees or None if not available
        """
        with self._get_lock():
            gimbal_module = self.get_module(VisionModule.GIMBAL_CONTROL)
            if gimbal_module and gimbal_module.active:
                return gimbal_module.get_position()
            return None
    
    # ========================================================================
    # Combined Operations
    # ========================================================================
    
    def scan_and_create_tool(self, tool_type: ToolType = ToolType.GRIPPER,
                            scan_mode: ScanMode = ScanMode.MULTI_ANGLE,
                            num_captures: int = 3,
                            offset: float = 2.0) -> Optional[str]:
        """
        Complete workflow: scan object and create tool
        
        Args:
            tool_type: Type of tool to create
            scan_mode: Scanning mode
            num_captures: Number of captures
            offset: Tool offset
        
        Returns:
            Filename of exported STL or None if failed
        """
        with self._get_lock():
            # Scan the object
            logger.info("Starting object scan...")
            scan_result = self.scan_object(scan_mode, num_captures)
            
            if not scan_result:
                logger.error("Scan failed")
                return None
            
            logger.info(f"Scan complete: {scan_result.total_points} points")
            
            # Create tool
            logger.info(f"Creating {tool_type.value} tool...")
            filename = f"{tool_type.value}_{int(time.time())}"
            
            if self.create_tool(tool_type, offset, True, filename):
                logger.info(f"Tool created and exported: {filename}.stl")
                return f"{filename}.stl"
            
            return None
    
    def identify_and_scan_for_person(self, target_name: str) -> Optional[ScanResult]:
        """
        Identify a specific person, then scan an object for them
        
        Args:
            target_name: Name of person to identify
        
        Returns:
            Scan result if person identified and scan completed
        """
        with self._get_lock():
            # First, identify the person
            logger.info(f"Looking for {target_name}...")
            
            found = False
            for _ in range(30):  # Try for 30 frames
                results = self.identify_person()
                for result in results:
                    if result.name == target_name:
                        logger.info(f"Found {target_name}!")
                        found = True
                        break
                if found:
                    break
                time.sleep(0.1)
            
            if not found:
                logger.warning(f"Could not find {target_name}")
                return None
            
            # Person identified, now scan object
            logger.info(f"Scanning object for {target_name}...")
            
            # Switch to 3D scanner
            if not self.switch_module(VisionModule.FACE_RECOGNITION, VisionModule.SCANNER_3D):
                logger.error("Failed to switch to 3D scanner")
                return None
            
            # Perform scan (can use gimbal if available)
            if self.config.enable_gimbal_control:
                # Use gimbal to scan around object
                logger.info("Using gimbal for enhanced 3D scanning...")
                self.scan_with_gimbal(num_positions=6)
            
            return self.scan_object(ScanMode.SINGLE_SHOT)
    
    # ========================================================================
    # Event Callbacks
    # ========================================================================
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for an event"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
            logger.debug(f"Registered callback for {event}")
        else:
            logger.warning(f"Unknown event: {event}")
    
    def unregister_callback(self, event: str, callback: Callable) -> None:
        """Unregister a callback"""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
            logger.debug(f"Unregistered callback for {event}")
    
    def _trigger_callback(self, event: str, *args, **kwargs) -> None:
        """Trigger callbacks for an event"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    # ========================================================================
    # Task History and Statistics
    # ========================================================================
    
    def _log_task(self, task: RobotTask, data: Dict[str, Any]) -> None:
        """Log a task execution"""
        self._task_history.append({
            'task': task.value,
            'timestamp': time.time(),
            'data': data
        })
        
        # Keep only last 100 tasks
        if len(self._task_history) > 100:
            self._task_history = self._task_history[-100:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        with self._get_lock():
            stats = {
                'modules': {
                    module.value: {
                        'available': self.is_module_available(module),
                        'active': (self.capabilities[module].instance.active 
                                 if module in self.capabilities else False)
                    }
                    for module in VisionModule
                },
                'camera_owner': self._camera_owner.value if self._camera_owner else None,
                'task_history_count': len(self._task_history),
                'known_people': len(self.get_known_people())
            }
            
            # Add module-specific stats
            face_module = self.get_module(VisionModule.FACE_RECOGNITION)
            if face_module and hasattr(face_module.api, 'get_statistics'):
                stats['face_recognition'] = face_module.api.get_statistics()
            
            # Add gimbal stats if available
            gimbal_module = self.get_module(VisionModule.GIMBAL_CONTROL)
            if gimbal_module and gimbal_module.active:
                stats['gimbal'] = {
                    'position': gimbal_module.get_position(),
                    'tracking_enabled': gimbal_module.tracking_enabled
                }
            
            return stats
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def _get_lock(self):
        """Get thread lock if threading is enabled"""
        if self._lock:
            return self._lock
        else:
            # Return a dummy context manager if threading is disabled
            return contextmanager(lambda: iter([None]))()
    
    def stop_all_modules(self) -> None:
        """Stop all active modules"""
        with self._get_lock():
            for module in self.capabilities:
                self.stop_module(module)
    
    # ========================================================================
    # Context Manager Support
    # ========================================================================
    
    def __enter__(self):
        """Context manager entry"""
        if self.config.auto_start_camera:
            # Start default module
            if self.config.enable_face_recognition:
                self.start_module(VisionModule.FACE_RECOGNITION)
            elif self.config.enable_3d_scanning:
                self.start_module(VisionModule.SCANNER_3D)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_all_modules()


# ============================================================================
# Factory Functions
# ============================================================================

def create_robot_vision(
    enable_face_recognition: bool = True,
    enable_3d_scanning: bool = True,
    enable_gimbal_control: bool = False,
    camera_type: CameraType = CameraType.OAK_D,
    **kwargs
) -> RobotVisionAPI:
    """
    Factory function to create Robot Vision API
    
    Args:
        enable_face_recognition: Enable face recognition module
        enable_3d_scanning: Enable 3D scanning module
        enable_gimbal_control: Enable gimbal/servo control
        camera_type: Type of camera to use
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured RobotVisionAPI instance
    """
    config = RobotVisionConfig(
        enable_face_recognition=enable_face_recognition,
        enable_3d_scanning=enable_3d_scanning,
        enable_gimbal_control=enable_gimbal_control,
        camera_type=camera_type,
        **kwargs
    )
    
    return RobotVisionAPI(config)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Robot that identifies a person and creates a custom tool
    robot = create_robot_vision()
    
    with robot:
        # Learn a person
        print("Teaching robot to recognize John...")
        robot.learn_person("John")
        
        # Later, identify John and create a tool for them
        print("Looking for John...")
        results = robot.identify_person()
        
        for result in results:
            if result.name == "John":
                print(f"Found John! Creating custom tool...")
                
                # Scan an object John is holding
                scan = robot.scan_object(ScanMode.MULTI_ANGLE, num_captures=3)
                
                if scan:
                    # Create a gripper tool for the object
                    robot.create_tool(
                        tool_type=ToolType.GRIPPER,
                        offset=3.0,
                        filename=f"john_gripper_{int(time.time())}"
                    )
                    print("Custom gripper tool created for John!")