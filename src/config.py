"""Configuration management for the chess vision system."""
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Configuration manager that loads from YAML file."""
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config:
            self.load()
    
    def load(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            # Look for config in project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config.yaml"
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            self._config = self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'camera': {
                'device_id': 0,
                'width': 1280,
                'height': 720,
                'fps': 30
            },
            'vision': {
                'board_detection': {
                    'method': 'contour',
                    'min_board_area': 50000
                },
                'piece_detection': {
                    'method': 'color',
                    'empty_threshold': 0.3,
                    'white_threshold': 0.6
                }
            },
            'engine': {
                'path': 'stockfish/stockfish.exe',
                'skill_level': 10,
                'move_time': 1.0
            },
            'theatre': {
                'window_size': 800,
                'board_size': 640
            },
            'game': {
                'player_color': 'white'
            },
            'logging': {
                'level': 'INFO'
            }
        }
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a configuration value by nested keys."""
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, *keys_and_value: Any) -> None:
        """Set a configuration value by nested keys."""
        if len(keys_and_value) < 2:
            raise ValueError("Need at least one key and a value")
        
        keys = keys_and_value[:-1]
        value = keys_and_value[-1]
        
        current = self._config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    @property
    def camera(self) -> Dict[str, Any]:
        return self._config.get('camera', {})
    
    @property
    def vision(self) -> Dict[str, Any]:
        return self._config.get('vision', {})
    
    @property
    def engine(self) -> Dict[str, Any]:
        return self._config.get('engine', {})
    
    @property
    def theatre(self) -> Dict[str, Any]:
        return self._config.get('theatre', {})
    
    @property
    def game(self) -> Dict[str, Any]:
        return self._config.get('game', {})
    
    @property
    def logging(self) -> Dict[str, Any]:
        return self._config.get('logging', {})


# Global config instance
config = Config()
