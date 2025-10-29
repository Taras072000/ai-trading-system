"""
WebXR Interface for Phase 5
Implements browser-based VR/AR trading with Three.js and WebXR
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timezone
import uuid
import numpy as np
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

class WebXRMode(Enum):
    VR = "immersive-vr"
    AR = "immersive-ar"
    INLINE = "inline"

class WebXRDevice(Enum):
    OCULUS_BROWSER = "oculus_browser"
    CHROME_VR = "chrome_vr"
    FIREFOX_VR = "firefox_vr"
    SAFARI_AR = "safari_ar"
    EDGE_VR = "edge_vr"
    MOBILE_VR = "mobile_vr"
    DESKTOP_VR = "desktop_vr"

class WebXRFeature(Enum):
    HAND_TRACKING = "hand-tracking"
    EYE_TRACKING = "eye-tracking"
    FACE_TRACKING = "face-tracking"
    PLANE_DETECTION = "plane-detection"
    MESH_DETECTION = "mesh-detection"
    LIGHT_ESTIMATION = "light-estimation"
    ANCHORS = "anchors"
    HIT_TEST = "hit-test"
    DOM_OVERLAY = "dom-overlay"
    LAYERS = "layers"

class TradingWidget3D(Enum):
    PRICE_CHART = "price_chart_3d"
    ORDER_BOOK = "order_book_3d"
    PORTFOLIO = "portfolio_3d"
    NEWS_FEED = "news_feed_3d"
    TRADING_PANEL = "trading_panel_3d"
    RISK_MONITOR = "risk_monitor_3d"
    MARKET_DEPTH = "market_depth_3d"
    STRATEGY_VIEWER = "strategy_viewer_3d"

@dataclass
class WebXRSession:
    """WebXR session configuration"""
    session_id: str
    user_id: str
    mode: WebXRMode
    device: WebXRDevice
    
    # Session state
    is_active: bool = False
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Capabilities
    supported_features: List[WebXRFeature] = field(default_factory=list)
    input_sources: List[str] = field(default_factory=list)
    
    # Performance
    frame_rate: float = 60.0
    resolution: Tuple[int, int] = (1920, 1080)
    fov: float = 110.0
    
    # Tracking
    head_pose: Dict[str, float] = field(default_factory=dict)
    controller_poses: Dict[str, Dict[str, float]] = field(default_factory=dict)
    hand_poses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.session_id:
            self.session_id = str(uuid.uuid4())

@dataclass
class WebXRWidget:
    """3D trading widget for WebXR"""
    widget_id: str
    widget_type: TradingWidget3D
    name: str
    
    # 3D Transform
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    # Visual properties
    material: str = "standard"
    color: str = "#ffffff"
    opacity: float = 1.0
    visible: bool = True
    
    # Interaction
    interactive: bool = True
    hover_enabled: bool = True
    click_enabled: bool = True
    drag_enabled: bool = False
    
    # Data
    data_source: str = ""
    update_frequency: int = 1000  # milliseconds
    last_update: Optional[datetime] = None
    
    # Animation
    animated: bool = False
    animation_duration: float = 1.0
    animation_easing: str = "ease-in-out"
    
    def __post_init__(self):
        if not self.widget_id:
            self.widget_id = str(uuid.uuid4())

@dataclass
class WebXRScene:
    """WebXR 3D scene configuration"""
    scene_id: str
    name: str
    environment_type: str
    
    # Scene properties
    background_color: str = "#000000"
    background_texture: Optional[str] = None
    fog_enabled: bool = False
    fog_color: str = "#ffffff"
    fog_density: float = 0.01
    
    # Lighting
    ambient_light: Dict[str, Any] = field(default_factory=lambda: {
        "color": "#404040",
        "intensity": 0.4
    })
    directional_light: Dict[str, Any] = field(default_factory=lambda: {
        "color": "#ffffff",
        "intensity": 1.0,
        "position": [10, 10, 5],
        "cast_shadow": True
    })
    
    # Physics
    physics_enabled: bool = False
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)
    
    # Audio
    spatial_audio: bool = True
    reverb_enabled: bool = True
    
    # Widgets
    widgets: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.scene_id:
            self.scene_id = str(uuid.uuid4())

class WebXRTradingInterface:
    """WebXR trading interface manager"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # WebXR sessions
        self.active_sessions: Dict[str, WebXRSession] = {}
        self.scenes: Dict[str, WebXRScene] = {}
        self.widgets: Dict[str, WebXRWidget] = {}
        
        # Device capabilities
        self.device_capabilities = {
            WebXRDevice.OCULUS_BROWSER: {
                'max_resolution': (2880, 1700),
                'max_refresh_rate': 90,
                'hand_tracking': True,
                'eye_tracking': False,
                'controllers': 2
            },
            WebXRDevice.CHROME_VR: {
                'max_resolution': (2160, 1200),
                'max_refresh_rate': 90,
                'hand_tracking': True,
                'eye_tracking': False,
                'controllers': 2
            },
            WebXRDevice.SAFARI_AR: {
                'max_resolution': (1920, 1080),
                'max_refresh_rate': 60,
                'hand_tracking': False,
                'eye_tracking': False,
                'controllers': 0
            }
        }
        
        # Widget templates
        self.widget_templates = {
            TradingWidget3D.PRICE_CHART: self._create_price_chart_template,
            TradingWidget3D.ORDER_BOOK: self._create_order_book_template,
            TradingWidget3D.PORTFOLIO: self._create_portfolio_template,
            TradingWidget3D.NEWS_FEED: self._create_news_feed_template,
            TradingWidget3D.TRADING_PANEL: self._create_trading_panel_template,
            TradingWidget3D.RISK_MONITOR: self._create_risk_monitor_template,
            TradingWidget3D.MARKET_DEPTH: self._create_market_depth_template,
            TradingWidget3D.STRATEGY_VIEWER: self._create_strategy_viewer_template
        }
    
    async def initialize_webxr_session(self, user_id: str, mode: WebXRMode, 
                                     device: WebXRDevice) -> Optional[str]:
        """Initialize WebXR session"""
        try:
            # Check device capabilities
            if device not in self.device_capabilities:
                self.logger.error(f"Unsupported device: {device}")
                return None
            
            capabilities = self.device_capabilities[device]
            
            # Create session
            session = WebXRSession(
                session_id="",  # Will be generated
                user_id=user_id,
                mode=mode,
                device=device,
                is_active=True,
                start_time=datetime.now(timezone.utc),
                resolution=capabilities['max_resolution'],
                frame_rate=capabilities['max_refresh_rate']
            )
            
            # Set supported features based on device
            session.supported_features = self._get_supported_features(device, mode)
            
            # Store session
            self.active_sessions[session.session_id] = session
            
            # Create default scene
            scene_id = await self._create_default_scene(session.session_id)
            
            self.logger.info(f"WebXR session initialized: {session.session_id}")
            return session.session_id
            
        except Exception as e:
            self.logger.error(f"WebXR session initialization failed: {e}")
            return None
    
    async def create_trading_scene(self, session_id: str, environment_type: str = "trading_floor") -> Optional[str]:
        """Create 3D trading scene"""
        try:
            if session_id not in self.active_sessions:
                return None
            
            session = self.active_sessions[session_id]
            
            # Create scene based on environment type
            scene = WebXRScene(
                scene_id="",  # Will be generated
                name=f"{environment_type.replace('_', ' ').title()}",
                environment_type=environment_type
            )
            
            # Configure scene based on environment
            if environment_type == "trading_floor":
                scene.background_color = "#1a1a2e"
                scene.ambient_light = {
                    "color": "#404080",
                    "intensity": 0.6
                }
                scene.directional_light = {
                    "color": "#ffffff",
                    "intensity": 1.2,
                    "position": [10, 15, 5],
                    "cast_shadow": True
                }
            elif environment_type == "space_station":
                scene.background_color = "#000000"
                scene.background_texture = "textures/space_background.jpg"
                scene.ambient_light = {
                    "color": "#202040",
                    "intensity": 0.3
                }
            elif environment_type == "cyberpunk_city":
                scene.background_color = "#0a0a0a"
                scene.fog_enabled = True
                scene.fog_color = "#ff00ff"
                scene.fog_density = 0.02
                scene.ambient_light = {
                    "color": "#ff0080",
                    "intensity": 0.5
                }
            
            # Add default trading widgets
            widget_configs = [
                (TradingWidget3D.PRICE_CHART, (0.0, 1.5, -2.0)),
                (TradingWidget3D.ORDER_BOOK, (2.0, 1.0, -1.5)),
                (TradingWidget3D.PORTFOLIO, (-2.0, 1.0, -1.5)),
                (TradingWidget3D.TRADING_PANEL, (0.0, 0.5, -1.0)),
                (TradingWidget3D.NEWS_FEED, (3.0, 2.0, -1.0))
            ]
            
            for widget_type, position in widget_configs:
                widget_id = await self.create_3d_widget(
                    session_id=session_id,
                    widget_type=widget_type,
                    position=position
                )
                if widget_id:
                    scene.widgets.append(widget_id)
            
            # Store scene
            self.scenes[scene.scene_id] = scene
            
            self.logger.info(f"Trading scene created: {scene.scene_id}")
            return scene.scene_id
            
        except Exception as e:
            self.logger.error(f"Trading scene creation failed: {e}")
            return None
    
    async def create_3d_widget(self, session_id: str, widget_type: TradingWidget3D,
                             position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                             custom_config: Dict[str, Any] = None) -> Optional[str]:
        """Create 3D trading widget"""
        try:
            if session_id not in self.active_sessions:
                return None
            
            # Get widget template
            if widget_type not in self.widget_templates:
                return None
            
            template_func = self.widget_templates[widget_type]
            widget_config = template_func()
            
            # Apply custom configuration
            if custom_config:
                widget_config.update(custom_config)
            
            # Create widget
            widget = WebXRWidget(
                widget_id="",  # Will be generated
                widget_type=widget_type,
                name=widget_config['name'],
                position=position,
                material=widget_config.get('material', 'standard'),
                color=widget_config.get('color', '#ffffff'),
                opacity=widget_config.get('opacity', 1.0),
                interactive=widget_config.get('interactive', True),
                data_source=widget_config.get('data_source', ''),
                update_frequency=widget_config.get('update_frequency', 1000),
                animated=widget_config.get('animated', False)
            )
            
            # Store widget
            self.widgets[widget.widget_id] = widget
            
            self.logger.info(f"3D widget created: {widget.widget_id} ({widget_type.value})")
            return widget.widget_id
            
        except Exception as e:
            self.logger.error(f"3D widget creation failed: {e}")
            return None
    
    async def update_widget_data(self, widget_id: str, data: Dict[str, Any]) -> bool:
        """Update 3D widget data"""
        try:
            if widget_id not in self.widgets:
                return False
            
            widget = self.widgets[widget_id]
            
            # Update widget based on type
            if widget.widget_type == TradingWidget3D.PRICE_CHART:
                await self._update_price_chart_data(widget, data)
            elif widget.widget_type == TradingWidget3D.ORDER_BOOK:
                await self._update_order_book_data(widget, data)
            elif widget.widget_type == TradingWidget3D.PORTFOLIO:
                await self._update_portfolio_data(widget, data)
            elif widget.widget_type == TradingWidget3D.NEWS_FEED:
                await self._update_news_feed_data(widget, data)
            
            widget.last_update = datetime.now(timezone.utc)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Widget data update failed: {e}")
            return False
    
    async def handle_webxr_interaction(self, session_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle WebXR user interaction"""
        try:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            interaction_type = interaction_data.get('type')
            target_id = interaction_data.get('target_id')
            
            if interaction_type == 'widget_click':
                return await self._handle_widget_click(target_id, interaction_data)
            elif interaction_type == 'widget_hover':
                return await self._handle_widget_hover(target_id, interaction_data)
            elif interaction_type == 'widget_drag':
                return await self._handle_widget_drag(target_id, interaction_data)
            elif interaction_type == 'gesture':
                return await self._handle_gesture_interaction(session_id, interaction_data)
            elif interaction_type == 'voice':
                return await self._handle_voice_interaction(session_id, interaction_data)
            
            return {"status": "interaction_processed"}
            
        except Exception as e:
            self.logger.error(f"WebXR interaction handling failed: {e}")
            return {"error": str(e)}
    
    async def get_webxr_scene_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get WebXR scene data for rendering"""
        try:
            if session_id not in self.active_sessions:
                return None
            
            session = self.active_sessions[session_id]
            
            # Find scene for session
            scene = None
            for s in self.scenes.values():
                if session_id in s.widgets or len(s.widgets) > 0:  # Simplified scene finding
                    scene = s
                    break
            
            if not scene:
                return None
            
            # Get widgets data
            widgets_data = []
            for widget_id in scene.widgets:
                if widget_id in self.widgets:
                    widget = self.widgets[widget_id]
                    widgets_data.append({
                        'id': widget.widget_id,
                        'type': widget.widget_type.value,
                        'name': widget.name,
                        'transform': {
                            'position': widget.position,
                            'rotation': widget.rotation,
                            'scale': widget.scale
                        },
                        'material': {
                            'type': widget.material,
                            'color': widget.color,
                            'opacity': widget.opacity
                        },
                        'properties': {
                            'visible': widget.visible,
                            'interactive': widget.interactive,
                            'animated': widget.animated
                        },
                        'data_source': widget.data_source,
                        'last_update': widget.last_update.isoformat() if widget.last_update else None
                    })
            
            return {
                'session': {
                    'id': session.session_id,
                    'mode': session.mode.value,
                    'device': session.device.value,
                    'features': [f.value for f in session.supported_features],
                    'performance': {
                        'frame_rate': session.frame_rate,
                        'resolution': session.resolution,
                        'fov': session.fov
                    }
                },
                'scene': {
                    'id': scene.scene_id,
                    'name': scene.name,
                    'environment': scene.environment_type,
                    'background': {
                        'color': scene.background_color,
                        'texture': scene.background_texture
                    },
                    'lighting': {
                        'ambient': scene.ambient_light,
                        'directional': scene.directional_light
                    },
                    'fog': {
                        'enabled': scene.fog_enabled,
                        'color': scene.fog_color,
                        'density': scene.fog_density
                    },
                    'physics': {
                        'enabled': scene.physics_enabled,
                        'gravity': scene.gravity
                    },
                    'audio': {
                        'spatial': scene.spatial_audio,
                        'reverb': scene.reverb_enabled
                    }
                },
                'widgets': widgets_data
            }
            
        except Exception as e:
            self.logger.error(f"WebXR scene data retrieval failed: {e}")
            return None
    
    async def generate_webxr_html(self, session_id: str) -> Optional[str]:
        """Generate WebXR HTML page"""
        try:
            scene_data = await self.get_webxr_scene_data(session_id)
            if not scene_data:
                return None
            
            html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Peper Binance v4 - WebXR Trading</title>
    <script src="https://aframe.io/releases/1.4.0/aframe.min.js"></script>
    <script src="https://cdn.jsdelivr.net/gh/donmccurdy/aframe-extras@v6.1.1/dist/aframe-extras.min.js"></script>
    <script src="https://unpkg.com/aframe-environment-component@1.3.1/dist/aframe-environment-component.min.js"></script>
    <script src="https://unpkg.com/aframe-event-set-component@5.0.0/dist/aframe-event-set-component.min.js"></script>
    <style>
        body {
            margin: 0;
            font-family: 'Arial', sans-serif;
            background: #000;
        }
        
        #ui-overlay {
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 1000;
            color: white;
            background: rgba(0, 0, 0, 0.7);
            padding: 15px;
            border-radius: 10px;
            font-size: 14px;
        }
        
        .trading-hud {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            display: flex;
            gap: 10px;
        }
        
        .hud-button {
            background: rgba(0, 150, 255, 0.8);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 12px;
            transition: background 0.3s;
        }
        
        .hud-button:hover {
            background: rgba(0, 150, 255, 1);
        }
        
        .price-display {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            color: #00ff00;
            background: rgba(0, 0, 0, 0.8);
            padding: 10px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <!-- UI Overlay -->
    <div id="ui-overlay">
        <div><strong>Peper Binance v4 - WebXR Trading</strong></div>
        <div>Session: {session_id}</div>
        <div>Mode: {mode}</div>
        <div>Device: {device}</div>
        <div>FPS: <span id="fps-counter">60</span></div>
    </div>
    
    <!-- Price Display -->
    <div class="price-display">
        <div>BTC/USDT: <span id="btc-price">$45,000.00</span></div>
        <div>ETH/USDT: <span id="eth-price">$3,200.00</span></div>
        <div>Portfolio: <span id="portfolio-value">$125,000.00</span></div>
    </div>
    
    <!-- Trading HUD -->
    <div class="trading-hud">
        <button class="hud-button" onclick="enterVR()">Enter VR</button>
        <button class="hud-button" onclick="enterAR()">Enter AR</button>
        <button class="hud-button" onclick="toggleLayout()">Layout</button>
        <button class="hud-button" onclick="showHelp()">Help</button>
    </div>
    
    <!-- A-Frame Scene -->
    <a-scene 
        vr-mode-ui="enabled: true"
        embedded
        style="height: 100vh; width: 100vw;"
        background="color: {background_color}"
        fog="type: exponential; color: {fog_color}; density: {fog_density}"
        physics="driver: ammo; debug: false"
        webxr="requiredFeatures: hand-tracking,local-floor; optionalFeatures: eye-tracking,face-tracking">
        
        <!-- Assets -->
        <a-assets>
            <!-- Textures -->
            <img id="space-texture" src="https://cdn.aframe.io/360-image-gallery-boilerplate/img/sechelt.jpg">
            <img id="grid-texture" src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZGVmcz4KICAgIDxwYXR0ZXJuIGlkPSJncmlkIiB3aWR0aD0iMTAiIGhlaWdodD0iMTAiIHBhdHRlcm5Vbml0cz0idXNlclNwYWNlT25Vc2UiPgogICAgICA8cGF0aCBkPSJNIDEwIDAgTCAwIDAgMCAxMCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMzMzIiBzdHJva2Utd2lkdGg9IjEiLz4KICAgIDwvcGF0dGVybj4KICA8L2RlZnM+CiAgPHJlY3Qgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSIgZmlsbD0idXJsKCNncmlkKSIvPgo8L3N2Zz4=">
            
            <!-- 3D Models -->
            <a-asset-item id="trading-desk" src="models/trading_desk.glb"></a-asset-item>
            <a-asset-item id="hologram-frame" src="models/hologram_frame.glb"></a-asset-item>
            
            <!-- Sounds -->
            <audio id="order-fill-sound" src="sounds/order_fill.mp3" preload="auto"></audio>
            <audio id="alert-sound" src="sounds/alert.mp3" preload="auto"></audio>
        </a-assets>
        
        <!-- Environment -->
        <a-entity environment="preset: {environment_preset}; groundColor: #1a1a2e; grid: 1x1; gridColor: #333; playArea: 1"></a-entity>
        
        <!-- Lighting -->
        <a-light type="ambient" color="{ambient_color}" intensity="{ambient_intensity}"></a-light>
        <a-light type="directional" 
                 position="{light_position}" 
                 color="{light_color}" 
                 intensity="{light_intensity}"
                 light="castShadow: true; shadowMapHeight: 2048; shadowMapWidth: 2048"></a-light>
        
        <!-- Trading Floor -->
        <a-plane position="0 0 0" 
                 rotation="-90 0 0" 
                 width="20" 
                 height="20" 
                 color="#1a1a2e" 
                 material="src: #grid-texture; repeat: 20 20; transparent: true; opacity: 0.8"
                 shadow="receive: true"></a-plane>
        
        <!-- Price Chart 3D -->
        <a-entity id="price-chart" 
                  position="{price_chart_position}"
                  animation="property: rotation; to: 0 360 0; loop: true; dur: 60000; easing: linear">
            <a-box position="0 0 0" 
                   width="3" 
                   height="2" 
                   depth="0.1" 
                   color="#003366" 
                   material="transparent: true; opacity: 0.8"
                   shadow="cast: true">
                <a-text value="BTC/USDT Chart" 
                        position="0 1.2 0.1" 
                        align="center" 
                        color="#00ff00" 
                        font="kelsonsans"
                        width="8"></a-text>
            </a-box>
            
            <!-- Candlesticks -->
            <a-entity id="candlesticks">
                <!-- Dynamic candlesticks will be added here -->
            </a-entity>
        </a-entity>
        
        <!-- Order Book Cylinder -->
        <a-entity id="order-book" position="{order_book_position}">
            <a-cylinder position="0 1 0" 
                        radius="0.8" 
                        height="2" 
                        color="#004400" 
                        material="transparent: true; opacity: 0.7"
                        shadow="cast: true">
                <a-text value="Order Book" 
                        position="0 1.2 0" 
                        align="center" 
                        color="#ffffff" 
                        font="kelsonsans"
                        width="6"></a-text>
            </a-cylinder>
            
            <!-- Order levels -->
            <a-entity id="order-levels">
                <!-- Dynamic order levels will be added here -->
            </a-entity>
        </a-entity>
        
        <!-- Portfolio Sphere -->
        <a-entity id="portfolio" position="{portfolio_position}">
            <a-sphere position="0 1 0" 
                      radius="1" 
                      color="#440044" 
                      material="transparent: true; opacity: 0.6"
                      shadow="cast: true"
                      animation="property: rotation; to: 0 360 0; loop: true; dur: 30000; easing: linear">
                <a-text value="Portfolio" 
                        position="0 1.5 0" 
                        align="center" 
                        color="#ff00ff" 
                        font="kelsonsans"
                        width="6"></a-text>
            </a-sphere>
            
            <!-- Asset orbits -->
            <a-entity id="asset-orbits">
                <!-- Dynamic asset representations will be added here -->
            </a-entity>
        </a-entity>
        
        <!-- Trading Panel -->
        <a-entity id="trading-panel" position="{trading_panel_position}">
            <a-box position="0 0.5 0" 
                   width="2" 
                   height="1" 
                   depth="0.1" 
                   color="#333333" 
                   material="transparent: true; opacity: 0.9"
                   shadow="cast: true">
                <a-text value="Trading Panel" 
                        position="0 0.6 0.1" 
                        align="center" 
                        color="#ffffff" 
                        font="kelsonsans"
                        width="6"></a-text>
            </a-box>
            
            <!-- Trading buttons -->
            <a-entity id="trading-buttons">
                <a-box position="-0.5 0.2 0.1" 
                       width="0.3" 
                       height="0.2" 
                       depth="0.05" 
                       color="#00aa00" 
                       class="clickable"
                       event-set__enter="_event: mouseenter; color: #00ff00"
                       event-set__leave="_event: mouseleave; color: #00aa00"
                       onclick="placeBuyOrder()">
                    <a-text value="BUY" 
                            position="0 0 0.03" 
                            align="center" 
                            color="#ffffff" 
                            font="kelsonsans"
                            width="8"></a-text>
                </a-box>
                
                <a-box position="0.5 0.2 0.1" 
                       width="0.3" 
                       height="0.2" 
                       depth="0.05" 
                       color="#aa0000" 
                       class="clickable"
                       event-set__enter="_event: mouseenter; color: #ff0000"
                       event-set__leave="_event: mouseleave; color: #aa0000"
                       onclick="placeSellOrder()">
                    <a-text value="SELL" 
                            position="0 0 0.03" 
                            align="center" 
                            color="#ffffff" 
                            font="kelsonsans"
                            width="8"></a-text>
                </a-box>
            </a-entity>
        </a-entity>
        
        <!-- News Feed -->
        <a-entity id="news-feed" position="{news_feed_position}">
            <a-box position="0 2 0" 
                   width="1.5" 
                   height="3" 
                   depth="0.1" 
                   color="#001122" 
                   material="transparent: true; opacity: 0.8"
                   shadow="cast: true">
                <a-text value="Market News" 
                        position="0 1.6 0.1" 
                        align="center" 
                        color="#00aaff" 
                        font="kelsonsans"
                        width="6"></a-text>
            </a-box>
            
            <!-- News items -->
            <a-entity id="news-items">
                <!-- Dynamic news items will be added here -->
            </a-entity>
        </a-entity>
        
        <!-- VR Controllers -->
        <a-entity id="leftHand" 
                  hand-controls="hand: left; handModelStyle: lowPoly; color: #ffcccc"
                  laser-controls="hand: left"
                  raycaster="objects: .clickable"
                  line="color: #00ff00; opacity: 0.75"></a-entity>
        
        <a-entity id="rightHand" 
                  hand-controls="hand: right; handModelStyle: lowPoly; color: #ffcccc"
                  laser-controls="hand: right"
                  raycaster="objects: .clickable"
                  line="color: #00ff00; opacity: 0.75"></a-entity>
        
        <!-- Camera Rig -->
        <a-entity id="cameraRig" 
                  movement-controls="fly: false; constrainToNavMesh: false"
                  position="0 1.6 3">
            <a-camera id="camera" 
                      look-controls="pointerLockEnabled: true"
                      wasd-controls="acceleration: 20"
                      cursor="rayOrigin: mouse; fuse: false"
                      raycaster="objects: .clickable">
                
                <!-- VR UI -->
                <a-entity id="vr-ui" 
                          position="0 -0.5 -1"
                          visible="false">
                    <a-text value="Peper Binance v4 VR" 
                            position="0 0.3 0" 
                            align="center" 
                            color="#ffffff" 
                            font="kelsonsans"
                            width="4"></a-text>
                    
                    <a-text id="vr-status" 
                            value="Ready for Trading" 
                            position="0 0.1 0" 
                            align="center" 
                            color="#00ff00" 
                            font="kelsonsans"
                            width="3"></a-text>
                </a-entity>
            </a-camera>
        </a-entity>
        
        <!-- Particle Systems for Effects -->
        <a-entity id="profit-particles" 
                  position="0 3 0" 
                  visible="false"
                  particle-system="preset: snow; particleCount: 100; color: #00ff00,#ffff00"></a-entity>
        
        <a-entity id="loss-particles" 
                  position="0 3 0" 
                  visible="false"
                  particle-system="preset: dust; particleCount: 50; color: #ff0000,#ff4444"></a-entity>
    </a-scene>
    
    <script>
        // WebXR Trading Interface JavaScript
        let session = null;
        let isVRActive = false;
        let isARActive = false;
        let tradingData = {session_data};
        
        // Initialize WebXR
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Peper Binance v4 WebXR Trading Interface Loaded');
            initializeTradingInterface();
            startDataUpdates();
        });
        
        function initializeTradingInterface() {
            // Setup event listeners
            document.querySelector('#price-chart').addEventListener('click', function() {
                console.log('Price chart clicked');
                showChartDetails();
            });
            
            document.querySelector('#order-book').addEventListener('click', function() {
                console.log('Order book clicked');
                showOrderBookDetails();
            });
            
            document.querySelector('#portfolio').addEventListener('click', function() {
                console.log('Portfolio clicked');
                showPortfolioDetails();
            });
            
            // Setup hand tracking
            if (navigator.xr) {
                navigator.xr.isSessionSupported('immersive-vr').then(function(supported) {
                    if (supported) {
                        console.log('VR supported');
                        document.querySelector('.hud-button').style.display = 'block';
                    }
                });
                
                navigator.xr.isSessionSupported('immersive-ar').then(function(supported) {
                    if (supported) {
                        console.log('AR supported');
                    }
                });
            }
        }
        
        function enterVR() {
            const sceneEl = document.querySelector('a-scene');
            if (sceneEl.is('vr-mode')) {
                sceneEl.exitVR();
            } else {
                sceneEl.enterVR();
                isVRActive = true;
                document.querySelector('#vr-ui').setAttribute('visible', true);
                updateVRStatus('VR Mode Active');
            }
        }
        
        function enterAR() {
            if (navigator.xr) {
                navigator.xr.requestSession('immersive-ar', {
                    requiredFeatures: ['local-floor'],
                    optionalFeatures: ['hand-tracking', 'plane-detection']
                }).then(function(session) {
                    console.log('AR session started');
                    isARActive = true;
                    updateVRStatus('AR Mode Active');
                }).catch(function(error) {
                    console.error('AR session failed:', error);
                });
            }
        }
        
        function toggleLayout() {
            // Cycle through different widget layouts
            const layouts = ['default', 'compact', 'expanded', 'minimal'];
            const currentLayout = localStorage.getItem('webxr-layout') || 'default';
            const currentIndex = layouts.indexOf(currentLayout);
            const nextLayout = layouts[(currentIndex + 1) % layouts.length];
            
            applyLayout(nextLayout);
            localStorage.setItem('webxr-layout', nextLayout);
            updateVRStatus(`Layout: ${nextLayout}`);
        }
        
        function applyLayout(layout) {
            const widgets = {
                'price-chart': document.querySelector('#price-chart'),
                'order-book': document.querySelector('#order-book'),
                'portfolio': document.querySelector('#portfolio'),
                'trading-panel': document.querySelector('#trading-panel'),
                'news-feed': document.querySelector('#news-feed')
            };
            
            const layouts = {
                'default': {
                    'price-chart': '0 1.5 -2',
                    'order-book': '2 1 -1.5',
                    'portfolio': '-2 1 -1.5',
                    'trading-panel': '0 0.5 -1',
                    'news-feed': '3 2 -1'
                },
                'compact': {
                    'price-chart': '0 1 -1.5',
                    'order-book': '1.5 1 -1',
                    'portfolio': '-1.5 1 -1',
                    'trading-panel': '0 0.3 -0.8',
                    'news-feed': '2 1.5 -0.8'
                },
                'expanded': {
                    'price-chart': '0 2 -3',
                    'order-book': '3 1.5 -2',
                    'portfolio': '-3 1.5 -2',
                    'trading-panel': '0 0.5 -1.5',
                    'news-feed': '4 2.5 -1.5'
                },
                'minimal': {
                    'price-chart': '0 1.2 -1.8',
                    'order-book': '0 0 0',  // Hidden
                    'portfolio': '0 0 0',   // Hidden
                    'trading-panel': '0 0.4 -1.2',
                    'news-feed': '0 0 0'    // Hidden
                }
            };
            
            const positions = layouts[layout];
            for (const [widgetId, position] of Object.entries(positions)) {
                if (widgets[widgetId]) {
                    widgets[widgetId].setAttribute('position', position);
                    widgets[widgetId].setAttribute('visible', position !== '0 0 0');
                }
            }
        }
        
        function showHelp() {
            const helpText = `
WebXR Trading Controls:
• Look around: Move your head
• Select: Point and click with controllers
• Move widgets: Grab and drag
• Voice commands: "Buy Bitcoin", "Show portfolio"
• Gestures: Thumbs up = Buy, Thumbs down = Sell
• Layout: Click Layout button to change view
            `;
            
            updateVRStatus(helpText);
            setTimeout(() => updateVRStatus('Ready for Trading'), 5000);
        }
        
        function updateVRStatus(message) {
            const statusEl = document.querySelector('#vr-status');
            if (statusEl) {
                statusEl.setAttribute('value', message);
            }
        }
        
        function placeBuyOrder() {
            console.log('Buy order placed');
            playSound('order-fill-sound');
            showParticleEffect('profit-particles');
            updateVRStatus('Buy Order Placed!');
            
            // Simulate order execution
            setTimeout(() => {
                updateVRStatus('Order Filled!');
                updatePriceDisplay();
            }, 2000);
        }
        
        function placeSellOrder() {
            console.log('Sell order placed');
            playSound('order-fill-sound');
            showParticleEffect('loss-particles');
            updateVRStatus('Sell Order Placed!');
            
            // Simulate order execution
            setTimeout(() => {
                updateVRStatus('Order Filled!');
                updatePriceDisplay();
            }, 2000);
        }
        
        function playSound(soundId) {
            const sound = document.querySelector(`#${soundId}`);
            if (sound) {
                sound.currentTime = 0;
                sound.play();
            }
        }
        
        function showParticleEffect(particleId) {
            const particles = document.querySelector(`#${particleId}`);
            if (particles) {
                particles.setAttribute('visible', true);
                setTimeout(() => {
                    particles.setAttribute('visible', false);
                }, 3000);
            }
        }
        
        function updatePriceDisplay() {
            // Simulate price updates
            const btcPrice = 45000 + (Math.random() - 0.5) * 1000;
            const ethPrice = 3200 + (Math.random() - 0.5) * 200;
            const portfolioValue = 125000 + (Math.random() - 0.5) * 5000;
            
            document.querySelector('#btc-price').textContent = `$${btcPrice.toFixed(2)}`;
            document.querySelector('#eth-price').textContent = `$${ethPrice.toFixed(2)}`;
            document.querySelector('#portfolio-value').textContent = `$${portfolioValue.toFixed(2)}`;
        }
        
        function startDataUpdates() {
            // Update prices every 2 seconds
            setInterval(updatePriceDisplay, 2000);
            
            // Update FPS counter
            let frameCount = 0;
            let lastTime = performance.now();
            
            function updateFPS() {
                frameCount++;
                const currentTime = performance.now();
                
                if (currentTime - lastTime >= 1000) {
                    const fps = Math.round(frameCount * 1000 / (currentTime - lastTime));
                    document.querySelector('#fps-counter').textContent = fps;
                    frameCount = 0;
                    lastTime = currentTime;
                }
                
                requestAnimationFrame(updateFPS);
            }
            
            updateFPS();
        }
        
        function showChartDetails() {
            updateVRStatus('Price Chart: BTC/USDT 24h +2.5%');
        }
        
        function showOrderBookDetails() {
            updateVRStatus('Order Book: Bid $44,950 | Ask $45,050');
        }
        
        function showPortfolioDetails() {
            updateVRStatus('Portfolio: 2.5 BTC, 15 ETH, +$2,500 today');
        }
        
        // WebXR Session Management
        function onSessionStarted(session) {
            session.addEventListener('end', onSessionEnded);
            console.log('WebXR session started');
        }
        
        function onSessionEnded() {
            session = null;
            isVRActive = false;
            isARActive = false;
            console.log('WebXR session ended');
        }
        
        // Export functions for external use
        window.WebXRTrading = {
            enterVR,
            enterAR,
            toggleLayout,
            placeBuyOrder,
            placeSellOrder,
            updatePriceDisplay
        };
    </script>
</body>
</html>
            """.format(
                session_id=scene_data['session']['id'],
                mode=scene_data['session']['mode'],
                device=scene_data['session']['device'],
                background_color=scene_data['scene']['background']['color'],
                fog_color=scene_data['scene']['fog']['color'],
                fog_density=scene_data['scene']['fog']['density'],
                environment_preset="forest",  # Default environment
                ambient_color=scene_data['scene']['lighting']['ambient']['color'],
                ambient_intensity=scene_data['scene']['lighting']['ambient']['intensity'],
                light_position=" ".join(map(str, scene_data['scene']['lighting']['directional']['position'])),
                light_color=scene_data['scene']['lighting']['directional']['color'],
                light_intensity=scene_data['scene']['lighting']['directional']['intensity'],
                price_chart_position=" ".join(map(str, scene_data['widgets'][0]['transform']['position'])) if scene_data['widgets'] else "0 1.5 -2",
                order_book_position=" ".join(map(str, scene_data['widgets'][1]['transform']['position'])) if len(scene_data['widgets']) > 1 else "2 1 -1.5",
                portfolio_position=" ".join(map(str, scene_data['widgets'][2]['transform']['position'])) if len(scene_data['widgets']) > 2 else "-2 1 -1.5",
                trading_panel_position=" ".join(map(str, scene_data['widgets'][3]['transform']['position'])) if len(scene_data['widgets']) > 3 else "0 0.5 -1",
                news_feed_position=" ".join(map(str, scene_data['widgets'][4]['transform']['position'])) if len(scene_data['widgets']) > 4 else "3 2 -1",
                session_data=json.dumps(scene_data)
            )
            
            return html_template
            
        except Exception as e:
            self.logger.error(f"WebXR HTML generation failed: {e}")
            return None
    
    def _get_supported_features(self, device: WebXRDevice, mode: WebXRMode) -> List[WebXRFeature]:
        """Get supported WebXR features for device and mode"""
        features = []
        
        if device == WebXRDevice.OCULUS_BROWSER:
            features.extend([
                WebXRFeature.HAND_TRACKING,
                WebXRFeature.ANCHORS,
                WebXRFeature.LAYERS
            ])
        elif device == WebXRDevice.SAFARI_AR:
            features.extend([
                WebXRFeature.PLANE_DETECTION,
                WebXRFeature.HIT_TEST,
                WebXRFeature.LIGHT_ESTIMATION
            ])
        elif device == WebXRDevice.CHROME_VR:
            features.extend([
                WebXRFeature.HAND_TRACKING,
                WebXRFeature.DOM_OVERLAY
            ])
        
        if mode == WebXRMode.AR:
            features.extend([
                WebXRFeature.PLANE_DETECTION,
                WebXRFeature.HIT_TEST,
                WebXRFeature.ANCHORS
            ])
        
        return features
    
    async def _create_default_scene(self, session_id: str) -> str:
        """Create default trading scene"""
        return await self.create_trading_scene(session_id, "trading_floor")
    
    def _create_price_chart_template(self) -> Dict[str, Any]:
        """Create price chart widget template"""
        return {
            'name': '3D Price Chart',
            'material': 'standard',
            'color': '#003366',
            'opacity': 0.8,
            'interactive': True,
            'data_source': 'price_data',
            'update_frequency': 1000,
            'animated': True
        }
    
    def _create_order_book_template(self) -> Dict[str, Any]:
        """Create order book widget template"""
        return {
            'name': 'Order Book',
            'material': 'standard',
            'color': '#004400',
            'opacity': 0.7,
            'interactive': True,
            'data_source': 'order_book_data',
            'update_frequency': 500,
            'animated': False
        }
    
    def _create_portfolio_template(self) -> Dict[str, Any]:
        """Create portfolio widget template"""
        return {
            'name': 'Portfolio Sphere',
            'material': 'standard',
            'color': '#440044',
            'opacity': 0.6,
            'interactive': True,
            'data_source': 'portfolio_data',
            'update_frequency': 2000,
            'animated': True
        }
    
    def _create_news_feed_template(self) -> Dict[str, Any]:
        """Create news feed widget template"""
        return {
            'name': 'News Feed',
            'material': 'standard',
            'color': '#001122',
            'opacity': 0.8,
            'interactive': True,
            'data_source': 'news_data',
            'update_frequency': 5000,
            'animated': False
        }
    
    def _create_trading_panel_template(self) -> Dict[str, Any]:
        """Create trading panel widget template"""
        return {
            'name': 'Trading Panel',
            'material': 'standard',
            'color': '#333333',
            'opacity': 0.9,
            'interactive': True,
            'data_source': 'trading_data',
            'update_frequency': 1000,
            'animated': False
        }
    
    def _create_risk_monitor_template(self) -> Dict[str, Any]:
        """Create risk monitor widget template"""
        return {
            'name': 'Risk Monitor',
            'material': 'standard',
            'color': '#660000',
            'opacity': 0.8,
            'interactive': True,
            'data_source': 'risk_data',
            'update_frequency': 1000,
            'animated': True
        }
    
    def _create_market_depth_template(self) -> Dict[str, Any]:
        """Create market depth widget template"""
        return {
            'name': 'Market Depth',
            'material': 'standard',
            'color': '#006600',
            'opacity': 0.7,
            'interactive': True,
            'data_source': 'depth_data',
            'update_frequency': 500,
            'animated': False
        }
    
    def _create_strategy_viewer_template(self) -> Dict[str, Any]:
        """Create strategy viewer widget template"""
        return {
            'name': 'Strategy Viewer',
            'material': 'standard',
            'color': '#000066',
            'opacity': 0.8,
            'interactive': True,
            'data_source': 'strategy_data',
            'update_frequency': 2000,
            'animated': True
        }
    
    async def _update_price_chart_data(self, widget: WebXRWidget, data: Dict[str, Any]):
        """Update price chart widget data"""
        # Implementation for updating 3D price chart
        pass
    
    async def _update_order_book_data(self, widget: WebXRWidget, data: Dict[str, Any]):
        """Update order book widget data"""
        # Implementation for updating 3D order book
        pass
    
    async def _update_portfolio_data(self, widget: WebXRWidget, data: Dict[str, Any]):
        """Update portfolio widget data"""
        # Implementation for updating 3D portfolio
        pass
    
    async def _update_news_feed_data(self, widget: WebXRWidget, data: Dict[str, Any]):
        """Update news feed widget data"""
        # Implementation for updating 3D news feed
        pass
    
    async def _handle_widget_click(self, widget_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle widget click interaction"""
        if widget_id not in self.widgets:
            return {"error": "Widget not found"}
        
        widget = self.widgets[widget_id]
        
        return {
            "action": "widget_clicked",
            "widget_id": widget_id,
            "widget_type": widget.widget_type.value,
            "interaction_point": interaction_data.get('point', [0, 0, 0])
        }
    
    async def _handle_widget_hover(self, widget_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle widget hover interaction"""
        if widget_id not in self.widgets:
            return {"error": "Widget not found"}
        
        return {
            "action": "widget_hovered",
            "widget_id": widget_id,
            "hover_info": interaction_data.get('info', {})
        }
    
    async def _handle_widget_drag(self, widget_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle widget drag interaction"""
        if widget_id not in self.widgets:
            return {"error": "Widget not found"}
        
        widget = self.widgets[widget_id]
        
        if widget.drag_enabled:
            new_position = interaction_data.get('new_position', widget.position)
            widget.position = tuple(new_position)
            
            return {
                "action": "widget_moved",
                "widget_id": widget_id,
                "new_position": widget.position
            }
        
        return {"error": "Widget drag not enabled"}
    
    async def _handle_gesture_interaction(self, session_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle gesture interaction"""
        gesture_type = interaction_data.get('gesture_type')
        
        if gesture_type == 'thumbs_up':
            return {"action": "buy_signal", "confidence": interaction_data.get('confidence', 0.8)}
        elif gesture_type == 'thumbs_down':
            return {"action": "sell_signal", "confidence": interaction_data.get('confidence', 0.8)}
        elif gesture_type == 'pinch':
            return {"action": "zoom", "direction": "in", "amount": interaction_data.get('amount', 1.0)}
        elif gesture_type == 'spread':
            return {"action": "zoom", "direction": "out", "amount": interaction_data.get('amount', 1.0)}
        
        return {"action": "gesture_recognized", "gesture": gesture_type}
    
    async def _handle_voice_interaction(self, session_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle voice interaction"""
        command = interaction_data.get('command', '').lower()
        
        if 'buy' in command:
            return {"action": "voice_buy_command", "text": command}
        elif 'sell' in command:
            return {"action": "voice_sell_command", "text": command}
        elif 'show' in command and 'chart' in command:
            return {"action": "show_chart", "text": command}
        elif 'hide' in command:
            return {"action": "hide_widget", "text": command}
        
        return {"action": "voice_command", "text": command}

# Example usage
async def main():
    """
    Example usage of WebXR Trading Interface
    """
    print("🌐 WebXR Trading Interface - Phase 5 Testing")
    print("=" * 50)
    
    # Initialize WebXR interface
    webxr = WebXRTradingInterface()
    
    # Test WebXR session creation
    print("\n🥽 Creating WebXR Sessions:")
    
    test_sessions = []
    
    # VR session
    vr_session_id = await webxr.initialize_webxr_session(
        user_id="user_001",
        mode=WebXRMode.VR,
        device=WebXRDevice.OCULUS_BROWSER
    )
    
    if vr_session_id:
        test_sessions.append(vr_session_id)
        print(f"  VR Session: {vr_session_id}")
    
    # AR session
    ar_session_id = await webxr.initialize_webxr_session(
        user_id="user_002",
        mode=WebXRMode.AR,
        device=WebXRDevice.SAFARI_AR
    )
    
    if ar_session_id:
        test_sessions.append(ar_session_id)
        print(f"  AR Session: {ar_session_id}")
    
    # Create trading scenes
    print(f"\n🏛️ Creating Trading Scenes:")
    
    for session_id in test_sessions:
        scene_id = await webxr.create_trading_scene(session_id, "trading_floor")
        if scene_id:
            print(f"  Scene for {session_id}: {scene_id}")
    
    # Create additional widgets
    print(f"\n📊 Creating Additional Widgets:")
    
    if test_sessions:
        session_id = test_sessions[0]
        
        widget_configs = [
            (TradingWidget3D.RISK_MONITOR, (1.0, 0.5, -2.5)),
            (TradingWidget3D.MARKET_DEPTH, (-1.0, 0.5, -2.5)),
            (TradingWidget3D.STRATEGY_VIEWER, (0.0, 2.5, -1.0))
        ]
        
        for widget_type, position in widget_configs:
            widget_id = await webxr.create_3d_widget(
                session_id=session_id,
                widget_type=widget_type,
                position=position
            )
            if widget_id:
                print(f"  {widget_type.value}: {widget_id}")
    
    # Test widget data updates
    print(f"\n📈 Testing Widget Data Updates:")
    
    if webxr.widgets:
        widget_id = list(webxr.widgets.keys())[0]
        test_data = {
            'price': 45000.0,
            'volume': 1250000,
            'change_24h': 2.5,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        success = await webxr.update_widget_data(widget_id, test_data)
        print(f"  Widget data update: {'✅' if success else '❌'}")
    
    # Test interactions
    print(f"\n👆 Testing WebXR Interactions:")
    
    if test_sessions:
        session_id = test_sessions[0]
        
        # Test widget click
        if webxr.widgets:
            widget_id = list(webxr.widgets.keys())[0]
            click_result = await webxr.handle_webxr_interaction(
                session_id=session_id,
                interaction_data={
                    'type': 'widget_click',
                    'target_id': widget_id,
                    'point': [0.5, 0.5, 0.0]
                }
            )
            print(f"  Widget click: {click_result.get('action', 'unknown')}")
        
        # Test gesture interaction
        gesture_result = await webxr.handle_webxr_interaction(
            session_id=session_id,
            interaction_data={
                'type': 'gesture',
                'gesture_type': 'thumbs_up',
                'confidence': 0.95
            }
        )
        print(f"  Gesture: {gesture_result.get('action', 'unknown')}")
        
        # Test voice interaction
        voice_result = await webxr.handle_webxr_interaction(
            session_id=session_id,
            interaction_data={
                'type': 'voice',
                'command': 'buy bitcoin'
            }
        )
        print(f"  Voice: {voice_result.get('action', 'unknown')}")
    
    # Generate WebXR HTML
    print(f"\n🌐 Generating WebXR HTML:")
    
    if test_sessions:
        session_id = test_sessions[0]
        html_content = await webxr.generate_webxr_html(session_id)
        
        if html_content:
            print(f"  HTML generated: {len(html_content)} characters")
            
            # Save HTML file for testing
            html_file = "/tmp/webxr_trading.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"  HTML saved to: {html_file}")
        else:
            print("  ❌ HTML generation failed")
    
    # Get scene data
    print(f"\n📋 Scene Data Summary:")
    
    if test_sessions:
        session_id = test_sessions[0]
        scene_data = await webxr.get_webxr_scene_data(session_id)
        
        if scene_data:
            print(f"  Session ID: {scene_data['session']['id']}")
            print(f"  Mode: {scene_data['session']['mode']}")
            print(f"  Device: {scene_data['session']['device']}")
            print(f"  Features: {len(scene_data['session']['features'])}")
            print(f"  Scene: {scene_data['scene']['name']}")
            print(f"  Widgets: {len(scene_data['widgets'])}")
            print(f"  Environment: {scene_data['scene']['environment']}")
    
    print(f"\n✅ WebXR Trading Interface testing completed!")
    print(f"📊 Active sessions: {len(webxr.active_sessions)}")
    print(f"🏛️ Scenes created: {len(webxr.scenes)}")
    print(f"📱 Widgets created: {len(webxr.widgets)}")

if __name__ == "__main__":
    asyncio.run(main())