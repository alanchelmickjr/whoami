"""
Configuration Module
Handles settings for the facial recognition system
"""

import json
import os
from typing import Dict, Any


class Config:
    """Configuration manager for the application"""
    
    DEFAULT_CONFIG = {
        "camera": {
            "preview_width": 640,
            "preview_height": 480,
            "fps": 30
        },
        "recognition": {
            "tolerance": 0.6,
            "database_path": "face_database.pkl"
        },
        "gui": {
            "window_width": 1000,
            "window_height": 700,
            "video_width": 640,
            "video_height": 480
        }
    }
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        self.load()
    
    def load(self):
        """Load configuration from file"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
            except Exception as e:
                print(f"Error loading config: {e}")
    
    def save(self):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (can use dot notation like 'camera.fps')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value
        
        Args:
            key: Configuration key (can use dot notation like 'camera.fps')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save()
