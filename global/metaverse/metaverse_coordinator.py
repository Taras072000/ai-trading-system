"""
Metaverse Coordinator for Phase 5
Implements VR/AR trading interfaces and virtual trading halls
"""

import asyncio
import logging
import json
import yaml
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timezone, timedelta
import hashlib
import hmac
import base64
from decimal import Decimal
import threading
from collections import defaultdict, deque
import numpy as np
import uuid
import warnings
warnings.filterwarnings('ignore')

class VRPlatform(Enum):
    OCULUS_QUEST = "oculus_quest"
    HTCVIVE = "htc_vive"
    VALVE_INDEX = "valve_index"
    PICO = "pico"
    WEBXR = "webxr"
    MOBILE_VR = "mobile_vr"

class ARPlatform(Enum):
    HOLOLENS = "hololens"
    MAGIC_LEAP = "magic_leap"
    ARKIT = "arkit"
    ARCORE = "arcore"
    WEBXR_AR = "webxr_ar"

class VirtualEnvironment(Enum):
    TRADING_FLOOR = "trading_floor"
    CONFERENCE_ROOM = "conference_room"
    EDUCATION_CENTER = "education_center"
    SOCIAL_LOUNGE = "social_lounge"
    ANALYTICS_LAB = "analytics_lab"
    NFT_GALLERY = "nft_gallery"
    STRATEGY_WORKSHOP = "strategy_workshop"
    PRIVATE_OFFICE = "private_office"

class InteractionType(Enum):
    GESTURE = "gesture"
    VOICE = "voice"
    GAZE = "gaze"
    CONTROLLER = "controller"
    HAND_TRACKING = "hand_tracking"
    BRAIN_INTERFACE = "brain_interface"

class VisualizationType(Enum):
    CANDLESTICK_3D = "candlestick_3d"
    VOLUME_SPHERES = "volume_spheres"
    PRICE_WAVES = "price_waves"
    CORRELATION_NETWORK = "correlation_network"
    RISK_HEATMAP = "risk_heatmap"
    PORTFOLIO_GALAXY = "portfolio_galaxy"
    ORDER_FLOW = "order_flow"
    SENTIMENT_CLOUDS = "sentiment_clouds"

@dataclass
class VRUser:
    """VR/AR user representation"""
    user_id: str
    username: str
    avatar_url: str
    
    # VR/AR capabilities
    vr_platforms: List[VRPlatform] = field(default_factory=list)
    ar_platforms: List[ARPlatform] = field(default_factory=list)
    preferred_interactions: List[InteractionType] = field(default_factory=list)
    
    # Virtual presence
    current_environment: Optional[VirtualEnvironment] = None
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_online: bool = False
    
    # Preferences
    ui_scale: float = 1.0
    comfort_settings: Dict[str, Any] = field(default_factory=dict)
    accessibility_options: Dict[str, Any] = field(default_factory=dict)
    
    # Session data
    session_start: Optional[datetime] = None
    total_vr_time: int = 0  # seconds
    environments_visited: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.user_id:
            self.user_id = str(uuid.uuid4())

@dataclass
class VirtualTradingHall:
    """Virtual trading hall configuration"""
    hall_id: str
    name: str
    environment_type: VirtualEnvironment
    max_capacity: int
    
    # 3D Environment settings
    scene_url: str
    lighting_config: Dict[str, Any] = field(default_factory=dict)
    audio_config: Dict[str, Any] = field(default_factory=dict)
    physics_enabled: bool = True
    
    # Trading features
    available_markets: List[str] = field(default_factory=list)
    visualization_types: List[VisualizationType] = field(default_factory=list)
    collaboration_tools: List[str] = field(default_factory=list)
    
    # Current state
    active_users: List[str] = field(default_factory=list)
    shared_screens: List[Dict[str, Any]] = field(default_factory=list)
    active_presentations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Access control
    is_private: bool = False
    access_level: str = "public"  # public, premium, vip, private
    required_tokens: Decimal = Decimal('0')
    
    def __post_init__(self):
        if not self.hall_id:
            self.hall_id = str(uuid.uuid4())

@dataclass
class VR3DVisualization:
    """3D visualization configuration"""
    viz_id: str
    name: str
    viz_type: VisualizationType
    data_source: str
    
    # 3D properties
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Visual properties
    color_scheme: str = "default"
    opacity: float = 1.0
    animation_speed: float = 1.0
    interactive: bool = True
    
    # Data configuration
    update_frequency: int = 1000  # milliseconds
    data_range: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # User interaction
    hover_info: bool = True
    click_actions: List[str] = field(default_factory=list)
    gesture_controls: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.viz_id:
            self.viz_id = str(uuid.uuid4())

@dataclass
class VRSession:
    """VR/AR session data"""
    session_id: str
    user_id: str
    platform: Union[VRPlatform, ARPlatform]
    environment: VirtualEnvironment
    
    # Session timing
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration: int = 0  # seconds
    
    # Performance metrics
    fps_average: float = 0.0
    latency_average: float = 0.0
    comfort_score: float = 10.0  # 1-10 scale
    motion_sickness_events: int = 0
    
    # Interaction data
    gestures_performed: int = 0
    voice_commands: int = 0
    controller_actions: int = 0
    trades_executed: int = 0
    
    # Technical data
    headset_model: str = ""
    tracking_quality: float = 1.0
    battery_level: float = 1.0
    
    def __post_init__(self):
        if not self.session_id:
            self.session_id = str(uuid.uuid4())

class VRTradingInterface:
    """VR trading interface manager"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Interface components
        self.virtual_keyboards: Dict[str, Dict[str, Any]] = {}
        self.gesture_recognizer = VRGestureRecognizer()
        self.voice_processor = VRVoiceProcessor()
        self.haptic_feedback = VRHapticFeedback()
        
        # Trading widgets
        self.trading_widgets = {
            'price_chart_3d': self._create_3d_price_chart,
            'order_book_cylinder': self._create_order_book_cylinder,
            'portfolio_sphere': self._create_portfolio_sphere,
            'risk_heatmap_plane': self._create_risk_heatmap,
            'news_feed_scroll': self._create_news_feed_scroll,
            'strategy_dashboard': self._create_strategy_dashboard
        }
    
    async def initialize_vr_interface(self, user: VRUser, platform: VRPlatform) -> Dict[str, Any]:
        """Initialize VR trading interface for user"""
        try:
            # Create user-specific interface
            interface_config = {
                'user_id': user.user_id,
                'platform': platform.value,
                'ui_scale': user.ui_scale,
                'comfort_settings': user.comfort_settings,
                'widgets': [],
                'layouts': self._get_default_layouts(),
                'interaction_methods': user.preferred_interactions
            }
            
            # Initialize default widgets
            for widget_name, widget_creator in self.trading_widgets.items():
                widget_config = await widget_creator(user)
                interface_config['widgets'].append(widget_config)
            
            # Setup gesture recognition
            await self.gesture_recognizer.initialize(user.user_id)
            
            # Setup voice commands
            await self.voice_processor.initialize(user.user_id)
            
            # Configure haptic feedback
            await self.haptic_feedback.initialize(platform)
            
            self.logger.info(f"VR interface initialized for user {user.user_id}")
            return interface_config
            
        except Exception as e:
            self.logger.error(f"VR interface initialization failed: {e}")
            return {}
    
    async def _create_3d_price_chart(self, user: VRUser) -> Dict[str, Any]:
        """Create 3D price chart widget"""
        return {
            'widget_id': str(uuid.uuid4()),
            'type': 'price_chart_3d',
            'name': '3D Price Chart',
            'position': (0.0, 1.5, -2.0),
            'scale': (2.0, 1.5, 0.5),
            'data_source': 'market_data',
            'visualization': {
                'type': 'candlestick_3d',
                'time_depth': 100,  # Number of candles in depth
                'color_scheme': 'bull_bear',
                'animation': True,
                'interactive': True
            },
            'interactions': {
                'zoom': True,
                'rotate': True,
                'time_travel': True,
                'hover_info': True
            }
        }
    
    async def _create_order_book_cylinder(self, user: VRUser) -> Dict[str, Any]:
        """Create cylindrical order book widget"""
        return {
            'widget_id': str(uuid.uuid4()),
            'type': 'order_book_cylinder',
            'name': 'Order Book Cylinder',
            'position': (2.0, 1.0, -1.5),
            'scale': (0.8, 2.0, 0.8),
            'data_source': 'order_book',
            'visualization': {
                'type': 'cylindrical_book',
                'bid_color': '#00ff00',
                'ask_color': '#ff0000',
                'depth_levels': 20,
                'rotation_speed': 0.1
            },
            'interactions': {
                'place_order': True,
                'cancel_order': True,
                'order_size_gesture': True
            }
        }
    
    async def _create_portfolio_sphere(self, user: VRUser) -> Dict[str, Any]:
        """Create portfolio sphere widget"""
        return {
            'widget_id': str(uuid.uuid4()),
            'type': 'portfolio_sphere',
            'name': 'Portfolio Sphere',
            'position': (-2.0, 1.0, -1.5),
            'scale': (1.0, 1.0, 1.0),
            'data_source': 'portfolio_data',
            'visualization': {
                'type': 'asset_sphere',
                'size_by': 'allocation',
                'color_by': 'performance',
                'orbit_animation': True,
                'particle_effects': True
            },
            'interactions': {
                'asset_details': True,
                'rebalance': True,
                'add_asset': True,
                'remove_asset': True
            }
        }
    
    async def _create_risk_heatmap(self, user: VRUser) -> Dict[str, Any]:
        """Create risk heatmap widget"""
        return {
            'widget_id': str(uuid.uuid4()),
            'type': 'risk_heatmap_plane',
            'name': 'Risk Heatmap',
            'position': (0.0, 0.5, -3.0),
            'scale': (3.0, 2.0, 0.1),
            'data_source': 'risk_metrics',
            'visualization': {
                'type': 'heatmap_3d',
                'risk_levels': ['low', 'medium', 'high', 'extreme'],
                'color_gradient': ['green', 'yellow', 'orange', 'red'],
                'height_mapping': True,
                'real_time': True
            },
            'interactions': {
                'drill_down': True,
                'risk_adjustment': True,
                'alert_setup': True
            }
        }
    
    async def _create_news_feed_scroll(self, user: VRUser) -> Dict[str, Any]:
        """Create scrolling news feed widget"""
        return {
            'widget_id': str(uuid.uuid4()),
            'type': 'news_feed_scroll',
            'name': 'News Feed',
            'position': (3.0, 2.0, -1.0),
            'scale': (1.5, 3.0, 0.1),
            'data_source': 'news_feed',
            'visualization': {
                'type': 'scrolling_text',
                'scroll_speed': 1.0,
                'sentiment_colors': True,
                'importance_sizing': True,
                'auto_translate': True
            },
            'interactions': {
                'read_full': True,
                'sentiment_analysis': True,
                'related_trades': True,
                'bookmark': True
            }
        }
    
    async def _create_strategy_dashboard(self, user: VRUser) -> Dict[str, Any]:
        """Create strategy dashboard widget"""
        return {
            'widget_id': str(uuid.uuid4()),
            'type': 'strategy_dashboard',
            'name': 'Strategy Dashboard',
            'position': (0.0, 0.0, -1.0),
            'scale': (4.0, 1.0, 0.1),
            'data_source': 'strategy_metrics',
            'visualization': {
                'type': 'holographic_dashboard',
                'performance_graphs': True,
                'real_time_metrics': True,
                'comparison_mode': True,
                'ai_insights': True
            },
            'interactions': {
                'strategy_selection': True,
                'parameter_tuning': True,
                'backtest_launch': True,
                'live_deployment': True
            }
        }
    
    def _get_default_layouts(self) -> Dict[str, Any]:
        """Get default VR interface layouts"""
        return {
            'beginner': {
                'name': 'Beginner Layout',
                'description': 'Simple layout for new VR traders',
                'widgets': ['price_chart_3d', 'portfolio_sphere', 'news_feed_scroll'],
                'complexity': 'low'
            },
            'professional': {
                'name': 'Professional Layout',
                'description': 'Full-featured layout for experienced traders',
                'widgets': ['price_chart_3d', 'order_book_cylinder', 'portfolio_sphere', 
                           'risk_heatmap_plane', 'strategy_dashboard'],
                'complexity': 'high'
            },
            'analyst': {
                'name': 'Analyst Layout',
                'description': 'Analysis-focused layout',
                'widgets': ['price_chart_3d', 'risk_heatmap_plane', 'strategy_dashboard', 
                           'news_feed_scroll'],
                'complexity': 'medium'
            }
        }

class VRGestureRecognizer:
    """VR gesture recognition system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gesture_models = {}
        self.active_sessions = {}
        
        # Define trading gestures
        self.trading_gestures = {
            'buy_gesture': {
                'pattern': 'thumbs_up',
                'confidence_threshold': 0.8,
                'action': 'place_buy_order'
            },
            'sell_gesture': {
                'pattern': 'thumbs_down',
                'confidence_threshold': 0.8,
                'action': 'place_sell_order'
            },
            'zoom_in': {
                'pattern': 'pinch_expand',
                'confidence_threshold': 0.7,
                'action': 'zoom_chart_in'
            },
            'zoom_out': {
                'pattern': 'pinch_contract',
                'confidence_threshold': 0.7,
                'action': 'zoom_chart_out'
            },
            'rotate_chart': {
                'pattern': 'circular_motion',
                'confidence_threshold': 0.6,
                'action': 'rotate_3d_chart'
            },
            'dismiss': {
                'pattern': 'wave_away',
                'confidence_threshold': 0.7,
                'action': 'close_widget'
            },
            'grab_widget': {
                'pattern': 'grab_motion',
                'confidence_threshold': 0.8,
                'action': 'move_widget'
            },
            'portfolio_expand': {
                'pattern': 'spread_hands',
                'confidence_threshold': 0.7,
                'action': 'expand_portfolio_view'
            }
        }
    
    async def initialize(self, user_id: str):
        """Initialize gesture recognition for user"""
        try:
            # Load user-specific gesture preferences
            self.active_sessions[user_id] = {
                'calibrated_gestures': {},
                'gesture_history': [],
                'accuracy_stats': {},
                'custom_gestures': {}
            }
            
            self.logger.info(f"Gesture recognition initialized for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Gesture recognition initialization failed: {e}")
    
    async def recognize_gesture(self, user_id: str, hand_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Recognize gesture from hand tracking data"""
        try:
            if user_id not in self.active_sessions:
                return None
            
            # Process hand tracking data
            gesture_result = await self._process_hand_data(hand_data)
            
            if gesture_result:
                # Match against known gestures
                for gesture_name, gesture_config in self.trading_gestures.items():
                    if (gesture_result['pattern'] == gesture_config['pattern'] and
                        gesture_result['confidence'] >= gesture_config['confidence_threshold']):
                        
                        # Record gesture
                        self.active_sessions[user_id]['gesture_history'].append({
                            'gesture': gesture_name,
                            'confidence': gesture_result['confidence'],
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        })
                        
                        return {
                            'gesture': gesture_name,
                            'action': gesture_config['action'],
                            'confidence': gesture_result['confidence'],
                            'parameters': gesture_result.get('parameters', {})
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Gesture recognition failed: {e}")
            return None
    
    async def _process_hand_data(self, hand_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process raw hand tracking data"""
        # Simplified gesture processing - in real implementation would use ML models
        try:
            left_hand = hand_data.get('left_hand', {})
            right_hand = hand_data.get('right_hand', {})
            
            # Example: Detect thumbs up
            if self._is_thumbs_up(right_hand):
                return {
                    'pattern': 'thumbs_up',
                    'confidence': 0.9,
                    'hand': 'right'
                }
            
            # Example: Detect pinch gesture
            if self._is_pinch_gesture(right_hand):
                return {
                    'pattern': 'pinch_expand',
                    'confidence': 0.8,
                    'hand': 'right',
                    'parameters': {
                        'distance': self._calculate_pinch_distance(right_hand)
                    }
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Hand data processing failed: {e}")
            return None
    
    def _is_thumbs_up(self, hand_data: Dict[str, Any]) -> bool:
        """Detect thumbs up gesture"""
        # Simplified detection logic
        thumb_up = hand_data.get('thumb', {}).get('extended', False)
        fingers_down = all(
            not hand_data.get(finger, {}).get('extended', True)
            for finger in ['index', 'middle', 'ring', 'pinky']
        )
        return thumb_up and fingers_down
    
    def _is_pinch_gesture(self, hand_data: Dict[str, Any]) -> bool:
        """Detect pinch gesture"""
        # Simplified detection logic
        thumb_pos = hand_data.get('thumb', {}).get('position', [0, 0, 0])
        index_pos = hand_data.get('index', {}).get('position', [0, 0, 0])
        
        if thumb_pos and index_pos:
            distance = np.linalg.norm(np.array(thumb_pos) - np.array(index_pos))
            return distance < 0.05  # Close proximity threshold
        
        return False
    
    def _calculate_pinch_distance(self, hand_data: Dict[str, Any]) -> float:
        """Calculate distance between thumb and index finger"""
        thumb_pos = hand_data.get('thumb', {}).get('position', [0, 0, 0])
        index_pos = hand_data.get('index', {}).get('position', [0, 0, 0])
        
        if thumb_pos and index_pos:
            return float(np.linalg.norm(np.array(thumb_pos) - np.array(index_pos)))
        
        return 0.0

class VRVoiceProcessor:
    """VR voice command processor"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.voice_commands = {
            'buy': ['buy', 'purchase', 'long', 'go long'],
            'sell': ['sell', 'short', 'go short', 'close position'],
            'show_chart': ['show chart', 'display chart', 'chart view'],
            'hide_chart': ['hide chart', 'close chart', 'remove chart'],
            'zoom_in': ['zoom in', 'magnify', 'closer'],
            'zoom_out': ['zoom out', 'zoom back', 'further'],
            'next_timeframe': ['next timeframe', 'higher timeframe', 'zoom out time'],
            'previous_timeframe': ['previous timeframe', 'lower timeframe', 'zoom in time'],
            'show_portfolio': ['show portfolio', 'my portfolio', 'holdings'],
            'show_orders': ['show orders', 'open orders', 'my orders'],
            'cancel_all': ['cancel all orders', 'close all', 'stop all'],
            'help': ['help', 'commands', 'what can I do'],
            'switch_layout': ['switch layout', 'change layout', 'new layout']
        }
        
        self.active_sessions = {}
    
    async def initialize(self, user_id: str):
        """Initialize voice processing for user"""
        try:
            self.active_sessions[user_id] = {
                'language': 'en-US',
                'voice_profile': None,
                'command_history': [],
                'accuracy_stats': {},
                'custom_commands': {}
            }
            
            self.logger.info(f"Voice processing initialized for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Voice processing initialization failed: {e}")
    
    async def process_voice_command(self, user_id: str, audio_data: bytes) -> Optional[Dict[str, Any]]:
        """Process voice command from audio data"""
        try:
            if user_id not in self.active_sessions:
                return None
            
            # Convert speech to text (simplified)
            text = await self._speech_to_text(audio_data)
            
            if text:
                # Match command
                command = self._match_command(text.lower())
                
                if command:
                    # Record command
                    self.active_sessions[user_id]['command_history'].append({
                        'text': text,
                        'command': command,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                    
                    return {
                        'command': command,
                        'text': text,
                        'confidence': 0.9,  # Simplified
                        'parameters': self._extract_parameters(text, command)
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Voice command processing failed: {e}")
            return None
    
    async def _speech_to_text(self, audio_data: bytes) -> Optional[str]:
        """Convert speech to text"""
        # Simplified implementation - would use actual STT service
        # For demo purposes, return mock text
        mock_commands = [
            "buy bitcoin",
            "sell ethereum", 
            "show chart",
            "zoom in",
            "show portfolio",
            "help"
        ]
        
        import random
        return random.choice(mock_commands)
    
    def _match_command(self, text: str) -> Optional[str]:
        """Match text to known commands"""
        for command, variations in self.voice_commands.items():
            for variation in variations:
                if variation in text:
                    return command
        return None
    
    def _extract_parameters(self, text: str, command: str) -> Dict[str, Any]:
        """Extract parameters from voice command"""
        parameters = {}
        
        # Extract asset names
        assets = ['bitcoin', 'ethereum', 'btc', 'eth', 'ada', 'sol', 'matic']
        for asset in assets:
            if asset in text.lower():
                parameters['asset'] = asset.upper()
                break
        
        # Extract quantities
        import re
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            parameters['quantity'] = float(numbers[0])
        
        # Extract timeframes
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d', '1w']
        for tf in timeframes:
            if tf in text.lower():
                parameters['timeframe'] = tf
                break
        
        return parameters

class VRHapticFeedback:
    """VR haptic feedback system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feedback_patterns = {
            'order_filled': {
                'pattern': 'double_pulse',
                'intensity': 0.8,
                'duration': 200
            },
            'price_alert': {
                'pattern': 'continuous_buzz',
                'intensity': 0.6,
                'duration': 500
            },
            'profit_milestone': {
                'pattern': 'celebration',
                'intensity': 1.0,
                'duration': 1000
            },
            'loss_warning': {
                'pattern': 'warning_pulse',
                'intensity': 0.9,
                'duration': 300
            },
            'button_press': {
                'pattern': 'single_click',
                'intensity': 0.4,
                'duration': 50
            },
            'gesture_confirm': {
                'pattern': 'gentle_pulse',
                'intensity': 0.5,
                'duration': 100
            }
        }
        
        self.active_controllers = {}
    
    async def initialize(self, platform: VRPlatform):
        """Initialize haptic feedback for platform"""
        try:
            self.platform = platform
            self.active_controllers = {
                'left': {'connected': True, 'battery': 1.0},
                'right': {'connected': True, 'battery': 1.0}
            }
            
            self.logger.info(f"Haptic feedback initialized for {platform.value}")
            
        except Exception as e:
            self.logger.error(f"Haptic feedback initialization failed: {e}")
    
    async def trigger_feedback(self, event_type: str, controller: str = 'both', 
                             custom_intensity: float = None) -> bool:
        """Trigger haptic feedback"""
        try:
            if event_type not in self.feedback_patterns:
                return False
            
            pattern = self.feedback_patterns[event_type]
            intensity = custom_intensity or pattern['intensity']
            
            # Send haptic command to controller(s)
            if controller == 'both':
                await self._send_haptic_command('left', pattern, intensity)
                await self._send_haptic_command('right', pattern, intensity)
            else:
                await self._send_haptic_command(controller, pattern, intensity)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Haptic feedback failed: {e}")
            return False
    
    async def _send_haptic_command(self, controller: str, pattern: Dict[str, Any], 
                                 intensity: float):
        """Send haptic command to specific controller"""
        # Simplified implementation - would interface with actual VR SDK
        self.logger.debug(f"Haptic feedback: {controller} controller, "
                         f"pattern: {pattern['pattern']}, intensity: {intensity}")

class MetaverseCoordinator:
    """Main metaverse coordinator"""
    
    def __init__(self, config_path: str = "config/metaverse_config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Core components
        self.vr_interface = VRTradingInterface()
        self.users: Dict[str, VRUser] = {}
        self.trading_halls: Dict[str, VirtualTradingHall] = {}
        self.active_sessions: Dict[str, VRSession] = {}
        self.visualizations: Dict[str, VR3DVisualization] = {}
        
        # Metaverse metrics
        self.metrics = {
            'total_users': 0,
            'active_sessions': 0,
            'total_trading_halls': 0,
            'total_vr_time': 0,
            'average_session_duration': 0,
            'popular_environments': {},
            'gesture_accuracy': 0.0,
            'voice_accuracy': 0.0
        }
        
        # Background tasks
        self.background_tasks = []
        self.is_running = False
    
    def _load_config(self, config_path: str) -> Dict:
        """Load metaverse configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default metaverse configuration"""
        return {
            'vr_settings': {
                'default_fov': 110,
                'max_fps': 90,
                'comfort_settings': {
                    'motion_sickness_reduction': True,
                    'teleport_movement': True,
                    'snap_turning': True
                }
            },
            'environments': {
                'max_users_per_hall': 50,
                'voice_chat_enabled': True,
                'spatial_audio': True,
                'physics_simulation': True
            },
            'performance': {
                'adaptive_quality': True,
                'dynamic_lod': True,
                'occlusion_culling': True,
                'texture_streaming': True
            }
        }
    
    async def initialize(self):
        """Initialize metaverse coordinator"""
        self.logger.info("Initializing metaverse coordinator...")
        
        # Create default trading halls
        await self._create_default_trading_halls()
        
        # Start background tasks
        self.is_running = True
        self.background_tasks = [
            asyncio.create_task(self._session_monitor()),
            asyncio.create_task(self._performance_optimizer()),
            asyncio.create_task(self._metrics_collector())
        ]
        
        self.logger.info("Metaverse coordinator initialized")
    
    async def register_vr_user(self, username: str, avatar_url: str,
                             vr_platforms: List[VRPlatform] = None,
                             ar_platforms: List[ARPlatform] = None) -> str:
        """Register new VR user"""
        try:
            user = VRUser(
                user_id="",  # Will be generated
                username=username,
                avatar_url=avatar_url,
                vr_platforms=vr_platforms or [VRPlatform.WEBXR],
                ar_platforms=ar_platforms or [ARPlatform.WEBXR_AR],
                preferred_interactions=[InteractionType.CONTROLLER, InteractionType.VOICE]
            )
            
            self.users[user.user_id] = user
            self.metrics['total_users'] += 1
            
            self.logger.info(f"VR user registered: {user.user_id}")
            return user.user_id
            
        except Exception as e:
            self.logger.error(f"VR user registration failed: {e}")
            return ""
    
    async def start_vr_session(self, user_id: str, platform: VRPlatform,
                             environment: VirtualEnvironment) -> Optional[str]:
        """Start VR trading session"""
        try:
            if user_id not in self.users:
                return None
            
            user = self.users[user_id]
            
            # Create VR session
            session = VRSession(
                session_id="",  # Will be generated
                user_id=user_id,
                platform=platform,
                environment=environment
            )
            
            # Initialize VR interface
            interface_config = await self.vr_interface.initialize_vr_interface(user, platform)
            
            # Update user state
            user.current_environment = environment
            user.is_online = True
            user.session_start = session.start_time
            
            # Store session
            self.active_sessions[session.session_id] = session
            
            # Add user to appropriate trading hall
            hall = await self._find_or_create_hall(environment)
            if hall and len(hall.active_users) < hall.max_capacity:
                hall.active_users.append(user_id)
            
            # Update metrics
            self.metrics['active_sessions'] += 1
            
            self.logger.info(f"VR session started: {session.session_id}")
            return session.session_id
            
        except Exception as e:
            self.logger.error(f"VR session start failed: {e}")
            return None
    
    async def end_vr_session(self, session_id: str) -> bool:
        """End VR trading session"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            user = self.users.get(session.user_id)
            
            # Update session data
            session.end_time = datetime.now(timezone.utc)
            session.duration = int((session.end_time - session.start_time).total_seconds())
            
            # Update user state
            if user:
                user.is_online = False
                user.current_environment = None
                user.total_vr_time += session.duration
                
                # Add environment to visited list
                if session.environment.value not in user.environments_visited:
                    user.environments_visited.append(session.environment.value)
            
            # Remove from trading hall
            for hall in self.trading_halls.values():
                if session.user_id in hall.active_users:
                    hall.active_users.remove(session.user_id)
                    break
            
            # Update metrics
            self.metrics['active_sessions'] -= 1
            self.metrics['total_vr_time'] += session.duration
            
            # Calculate average session duration
            total_sessions = len([s for s in self.active_sessions.values() if s.end_time])
            if total_sessions > 0:
                avg_duration = sum(s.duration for s in self.active_sessions.values() if s.end_time) / total_sessions
                self.metrics['average_session_duration'] = avg_duration
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            self.logger.info(f"VR session ended: {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"VR session end failed: {e}")
            return False
    
    async def create_3d_visualization(self, viz_type: VisualizationType,
                                    data_source: str, position: Tuple[float, float, float],
                                    user_id: str = None) -> Optional[str]:
        """Create 3D visualization"""
        try:
            visualization = VR3DVisualization(
                viz_id="",  # Will be generated
                name=f"{viz_type.value.replace('_', ' ').title()}",
                viz_type=viz_type,
                data_source=data_source,
                position=position
            )
            
            self.visualizations[visualization.viz_id] = visualization
            
            self.logger.info(f"3D visualization created: {visualization.viz_id}")
            return visualization.viz_id
            
        except Exception as e:
            self.logger.error(f"3D visualization creation failed: {e}")
            return None
    
    async def get_trading_halls(self) -> List[Dict[str, Any]]:
        """Get available trading halls"""
        halls = []
        
        for hall in self.trading_halls.values():
            halls.append({
                'hall_id': hall.hall_id,
                'name': hall.name,
                'environment_type': hall.environment_type.value,
                'current_users': len(hall.active_users),
                'max_capacity': hall.max_capacity,
                'is_private': hall.is_private,
                'access_level': hall.access_level,
                'required_tokens': hall.required_tokens,
                'available_markets': hall.available_markets,
                'visualization_types': [vt.value for vt in hall.visualization_types]
            })
        
        return halls
    
    async def get_user_session_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user's current session information"""
        if user_id not in self.users:
            return None
        
        user = self.users[user_id]
        
        # Find active session
        active_session = None
        for session in self.active_sessions.values():
            if session.user_id == user_id:
                active_session = session
                break
        
        return {
            'user_id': user.user_id,
            'username': user.username,
            'is_online': user.is_online,
            'current_environment': user.current_environment.value if user.current_environment else None,
            'total_vr_time': user.total_vr_time,
            'environments_visited': user.environments_visited,
            'active_session': {
                'session_id': active_session.session_id,
                'platform': active_session.platform.value,
                'start_time': active_session.start_time.isoformat(),
                'duration': int((datetime.now(timezone.utc) - active_session.start_time).total_seconds()),
                'performance': {
                    'fps_average': active_session.fps_average,
                    'latency_average': active_session.latency_average,
                    'comfort_score': active_session.comfort_score
                }
            } if active_session else None
        }
    
    async def get_metaverse_metrics(self) -> Dict[str, Any]:
        """Get metaverse metrics"""
        # Update popular environments
        env_counts = defaultdict(int)
        for session in self.active_sessions.values():
            env_counts[session.environment.value] += 1
        
        self.metrics['popular_environments'] = dict(env_counts)
        
        return {
            'metaverse_metrics': self.metrics,
            'active_trading_halls': [
                {
                    'name': hall.name,
                    'environment': hall.environment_type.value,
                    'active_users': len(hall.active_users),
                    'capacity': hall.max_capacity
                }
                for hall in self.trading_halls.values()
                if hall.active_users
            ],
            'platform_distribution': self._get_platform_distribution(),
            'performance_stats': self._get_performance_stats()
        }
    
    async def _create_default_trading_halls(self):
        """Create default trading halls"""
        default_halls = [
            {
                'name': 'Main Trading Floor',
                'environment_type': VirtualEnvironment.TRADING_FLOOR,
                'max_capacity': 100,
                'available_markets': ['crypto', 'forex', 'stocks'],
                'visualization_types': [VisualizationType.CANDLESTICK_3D, VisualizationType.VOLUME_SPHERES]
            },
            {
                'name': 'Analytics Laboratory',
                'environment_type': VirtualEnvironment.ANALYTICS_LAB,
                'max_capacity': 30,
                'available_markets': ['crypto'],
                'visualization_types': [VisualizationType.CORRELATION_NETWORK, VisualizationType.RISK_HEATMAP]
            },
            {
                'name': 'Education Center',
                'environment_type': VirtualEnvironment.EDUCATION_CENTER,
                'max_capacity': 50,
                'available_markets': ['demo'],
                'visualization_types': [VisualizationType.CANDLESTICK_3D]
            },
            {
                'name': 'NFT Gallery',
                'environment_type': VirtualEnvironment.NFT_GALLERY,
                'max_capacity': 25,
                'available_markets': ['nft'],
                'visualization_types': [VisualizationType.PORTFOLIO_GALAXY]
            }
        ]
        
        for hall_config in default_halls:
            hall = VirtualTradingHall(
                hall_id="",  # Will be generated
                name=hall_config['name'],
                environment_type=hall_config['environment_type'],
                max_capacity=hall_config['max_capacity'],
                scene_url=f"scenes/{hall_config['environment_type'].value}.glb",
                available_markets=hall_config['available_markets'],
                visualization_types=hall_config['visualization_types']
            )
            
            self.trading_halls[hall.hall_id] = hall
            self.metrics['total_trading_halls'] += 1
    
    async def _find_or_create_hall(self, environment: VirtualEnvironment) -> Optional[VirtualTradingHall]:
        """Find existing hall or create new one for environment"""
        # Find existing hall with capacity
        for hall in self.trading_halls.values():
            if (hall.environment_type == environment and 
                len(hall.active_users) < hall.max_capacity):
                return hall
        
        # Create new hall if needed
        hall = VirtualTradingHall(
            hall_id="",  # Will be generated
            name=f"{environment.value.replace('_', ' ').title()} Hall",
            environment_type=environment,
            max_capacity=50,
            scene_url=f"scenes/{environment.value}.glb"
        )
        
        self.trading_halls[hall.hall_id] = hall
        self.metrics['total_trading_halls'] += 1
        
        return hall
    
    def _get_platform_distribution(self) -> Dict[str, int]:
        """Get distribution of VR platforms"""
        platform_counts = defaultdict(int)
        
        for session in self.active_sessions.values():
            platform_counts[session.platform.value] += 1
        
        return dict(platform_counts)
    
    def _get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.active_sessions:
            return {}
        
        fps_values = [s.fps_average for s in self.active_sessions.values() if s.fps_average > 0]
        latency_values = [s.latency_average for s in self.active_sessions.values() if s.latency_average > 0]
        comfort_values = [s.comfort_score for s in self.active_sessions.values() if s.comfort_score > 0]
        
        return {
            'average_fps': np.mean(fps_values) if fps_values else 0.0,
            'average_latency': np.mean(latency_values) if latency_values else 0.0,
            'average_comfort_score': np.mean(comfort_values) if comfort_values else 0.0,
            'total_motion_sickness_events': sum(s.motion_sickness_events for s in self.active_sessions.values())
        }
    
    async def _session_monitor(self):
        """Background task to monitor VR sessions"""
        while self.is_running:
            try:
                # Update session metrics
                for session in self.active_sessions.values():
                    if not session.end_time:
                        # Update session duration
                        session.duration = int((datetime.now(timezone.utc) - session.start_time).total_seconds())
                        
                        # Simulate performance metrics (in real implementation, get from VR runtime)
                        session.fps_average = 85.0 + np.random.normal(0, 5)
                        session.latency_average = 15.0 + np.random.normal(0, 3)
                        session.comfort_score = max(1.0, min(10.0, session.comfort_score + np.random.normal(0, 0.1)))
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Session monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _performance_optimizer(self):
        """Background task to optimize VR performance"""
        while self.is_running:
            try:
                # Optimize performance based on current load
                active_count = len(self.active_sessions)
                
                if active_count > 50:
                    # High load - reduce quality
                    self.logger.info("High VR load detected, optimizing performance")
                elif active_count < 10:
                    # Low load - increase quality
                    self.logger.info("Low VR load detected, increasing quality")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Performance optimizer error: {e}")
                await asyncio.sleep(30)
    
    async def _metrics_collector(self):
        """Background task to collect metaverse metrics"""
        while self.is_running:
            try:
                # Collect and update metrics
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(60)
    
    async def cleanup(self):
        """Cleanup metaverse coordinator"""
        self.is_running = False
        
        # End all active sessions
        for session_id in list(self.active_sessions.keys()):
            await self.end_vr_session(session_id)
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.logger.info("Metaverse coordinator cleaned up")

# Example usage and testing
async def main():
    """
    Example usage of Metaverse Coordinator
    """
    print("🌐 Metaverse Coordinator - Phase 5 Testing")
    print("=" * 50)
    
    # Initialize metaverse
    metaverse = MetaverseCoordinator()
    await metaverse.initialize()
    
    # Register VR users
    print("\n👥 Registering VR Users:")
    test_users = []
    
    for i in range(5):
        user_id = await metaverse.register_vr_user(
            username=f"VRTrader{i+1}",
            avatar_url=f"https://example.com/avatar_{i+1}.glb",
            vr_platforms=[VRPlatform.WEBXR, VRPlatform.OCULUS_QUEST],
            ar_platforms=[ARPlatform.WEBXR_AR]
        )
        
        if user_id:
            test_users.append(user_id)
            print(f"  VRTrader{i+1}: {user_id}")
    
    # Start VR sessions
    print(f"\n🥽 Starting VR Sessions:")
    active_sessions = []
    
    environments = [
        VirtualEnvironment.TRADING_FLOOR,
        VirtualEnvironment.ANALYTICS_LAB,
        VirtualEnvironment.EDUCATION_CENTER,
        VirtualEnvironment.NFT_GALLERY,
        VirtualEnvironment.SOCIAL_LOUNGE
    ]
    
    for i, user_id in enumerate(test_users):
        session_id = await metaverse.start_vr_session(
            user_id=user_id,
            platform=VRPlatform.WEBXR,
            environment=environments[i % len(environments)]
        )
        
        if session_id:
            active_sessions.append(session_id)
            print(f"  Session {i+1}: {session_id} in {environments[i % len(environments)].value}")
    
    # Create 3D visualizations
    print(f"\n📊 Creating 3D Visualizations:")
    visualizations = []
    
    viz_configs = [
        (VisualizationType.CANDLESTICK_3D, "btc_price_data", (0.0, 1.5, -2.0)),
        (VisualizationType.VOLUME_SPHERES, "volume_data", (2.0, 1.0, -1.5)),
        (VisualizationType.PORTFOLIO_GALAXY, "portfolio_data", (-2.0, 1.0, -1.5)),
        (VisualizationType.RISK_HEATMAP, "risk_data", (0.0, 0.5, -3.0)),
        (VisualizationType.CORRELATION_NETWORK, "correlation_data", (0.0, 2.0, -1.0))
    ]
    
    for viz_type, data_source, position in viz_configs:
        viz_id = await metaverse.create_3d_visualization(
            viz_type=viz_type,
            data_source=data_source,
            position=position
        )
        
        if viz_id:
            visualizations.append(viz_id)
            print(f"  {viz_type.value}: {viz_id}")
    
    # Get trading halls
    print(f"\n🏛️ Available Trading Halls:")
    halls = await metaverse.get_trading_halls()
    
    for hall in halls:
        print(f"  {hall['name']} ({hall['environment_type']}):")
        print(f"    Users: {hall['current_users']}/{hall['max_capacity']}")
        print(f"    Markets: {', '.join(hall['available_markets'])}")
        print(f"    Access: {hall['access_level']}")
    
    # Test gesture recognition
    print(f"\n👋 Testing Gesture Recognition:")
    if test_users:
        gesture_recognizer = VRGestureRecognizer()
        await gesture_recognizer.initialize(test_users[0])
        
        # Simulate hand tracking data
        mock_hand_data = {
            'right_hand': {
                'thumb': {'extended': True, 'position': [0.1, 0.1, 0.1]},
                'index': {'extended': False, 'position': [0.0, 0.0, 0.0]},
                'middle': {'extended': False, 'position': [0.0, 0.0, 0.0]},
                'ring': {'extended': False, 'position': [0.0, 0.0, 0.0]},
                'pinky': {'extended': False, 'position': [0.0, 0.0, 0.0]}
            }
        }
        
        gesture_result = await gesture_recognizer.recognize_gesture(test_users[0], mock_hand_data)
        if gesture_result:
            print(f"  Gesture recognized: {gesture_result['gesture']} -> {gesture_result['action']}")
            print(f"  Confidence: {gesture_result['confidence']:.2f}")
    
    # Test voice commands
    print(f"\n🎤 Testing Voice Commands:")
    if test_users:
        voice_processor = VRVoiceProcessor()
        await voice_processor.initialize(test_users[0])
        
        # Simulate voice command
        mock_audio = b"mock_audio_data"  # In real implementation, this would be actual audio
        voice_result = await voice_processor.process_voice_command(test_users[0], mock_audio)
        
        if voice_result:
            print(f"  Voice command: '{voice_result['text']}'")
            print(f"  Command: {voice_result['command']}")
            print(f"  Parameters: {voice_result['parameters']}")
    
    # Get user session info
    print(f"\n👤 User Session Information:")
    if test_users:
        session_info = await metaverse.get_user_session_info(test_users[0])
        if session_info:
            print(f"  User: {session_info['username']}")
            print(f"  Online: {session_info['is_online']}")
            print(f"  Environment: {session_info['current_environment']}")
            print(f"  Total VR Time: {session_info['total_vr_time']} seconds")
            
            if session_info['active_session']:
                session = session_info['active_session']
                print(f"  Current Session:")
                print(f"    Platform: {session['platform']}")
                print(f"    Duration: {session['duration']} seconds")
                print(f"    FPS: {session['performance']['fps_average']:.1f}")
                print(f"    Latency: {session['performance']['latency_average']:.1f}ms")
                print(f"    Comfort: {session['performance']['comfort_score']:.1f}/10")
    
    # Get metaverse metrics
    print(f"\n📈 Metaverse Metrics:")
    metrics = await metaverse.get_metaverse_metrics()
    metaverse_metrics = metrics['metaverse_metrics']
    
    print(f"  Total Users: {metaverse_metrics['total_users']}")
    print(f"  Active Sessions: {metaverse_metrics['active_sessions']}")
    print(f"  Trading Halls: {metaverse_metrics['total_trading_halls']}")
    print(f"  Total VR Time: {metaverse_metrics['total_vr_time']} seconds")
    print(f"  Avg Session Duration: {metaverse_metrics['average_session_duration']:.1f} seconds")
    
    print(f"\n  Popular Environments:")
    for env, count in metaverse_metrics['popular_environments'].items():
        print(f"    {env}: {count} users")
    
    print(f"\n  Platform Distribution:")
    for platform, count in metrics['platform_distribution'].items():
        print(f"    {platform}: {count} users")
    
    print(f"\n  Performance Stats:")
    perf_stats = metrics['performance_stats']
    if perf_stats:
        print(f"    Average FPS: {perf_stats.get('average_fps', 0):.1f}")
        print(f"    Average Latency: {perf_stats.get('average_latency', 0):.1f}ms")
        print(f"    Average Comfort: {perf_stats.get('average_comfort_score', 0):.1f}/10")
    
    # Wait a bit to simulate activity
    print(f"\n⏳ Simulating VR activity...")
    await asyncio.sleep(5)
    
    # End sessions
    print(f"\n🔚 Ending VR Sessions:")
    for i, session_id in enumerate(active_sessions):
        success = await metaverse.end_vr_session(session_id)
        print(f"  Session {i+1}: {'✅' if success else '❌'}")
    
    # Cleanup
    await metaverse.cleanup()
    
    print(f"\n✅ Metaverse testing completed!")
    print(f"🌐 VR/AR trading metaverse ready for Phase 5!")

if __name__ == "__main__":
    asyncio.run(main())