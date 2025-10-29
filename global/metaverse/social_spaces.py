"""
Social Spaces for Phase 5 Metaverse
Implements virtual social trading environments and community features
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timezone, timedelta
import uuid
import numpy as np
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

class SocialSpaceType(Enum):
    TRADING_HALL = "trading_hall"
    CONFERENCE_ROOM = "conference_room"
    EDUCATION_CENTER = "education_center"
    NETWORKING_LOUNGE = "networking_lounge"
    STRATEGY_LAB = "strategy_lab"
    NEWS_PLAZA = "news_plaza"
    MENTORSHIP_ZONE = "mentorship_zone"
    COMPETITION_ARENA = "competition_arena"

class UserRole(Enum):
    TRADER = "trader"
    MENTOR = "mentor"
    EDUCATOR = "educator"
    ANALYST = "analyst"
    MODERATOR = "moderator"
    VIP = "vip"
    ADMIN = "admin"
    GUEST = "guest"

class InteractionType(Enum):
    VOICE_CHAT = "voice_chat"
    TEXT_CHAT = "text_chat"
    SCREEN_SHARE = "screen_share"
    WHITEBOARD = "whiteboard"
    STRATEGY_SHARE = "strategy_share"
    TRADE_COPY = "trade_copy"
    GESTURE = "gesture"
    EMOTE = "emote"

class EventType(Enum):
    WEBINAR = "webinar"
    WORKSHOP = "workshop"
    TRADING_COMPETITION = "trading_competition"
    MARKET_ANALYSIS = "market_analysis"
    STRATEGY_PRESENTATION = "strategy_presentation"
    NETWORKING_SESSION = "networking_session"
    Q_AND_A = "q_and_a"
    LIVE_TRADING = "live_trading"

@dataclass
class SocialUser:
    """Social metaverse user"""
    user_id: str
    username: str
    display_name: str
    
    # Profile
    role: UserRole = UserRole.TRADER
    level: int = 1
    experience_points: int = 0
    reputation_score: float = 0.0
    
    # Avatar
    avatar_url: str = ""
    avatar_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    avatar_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    avatar_animation: str = "idle"
    
    # Status
    is_online: bool = False
    current_space: Optional[str] = None
    last_activity: Optional[datetime] = None
    
    # Trading stats
    win_rate: float = 0.0
    total_trades: int = 0
    profit_loss: float = 0.0
    
    # Social features
    followers: Set[str] = field(default_factory=set)
    following: Set[str] = field(default_factory=set)
    friends: Set[str] = field(default_factory=set)
    
    # Preferences
    voice_enabled: bool = True
    notifications_enabled: bool = True
    privacy_mode: bool = False
    
    def __post_init__(self):
        if not self.user_id:
            self.user_id = str(uuid.uuid4())

@dataclass
class SocialSpace:
    """Virtual social space"""
    space_id: str
    name: str
    space_type: SocialSpaceType
    
    # Configuration
    max_capacity: int = 50
    is_public: bool = True
    requires_invitation: bool = False
    
    # Environment
    environment_theme: str = "modern_office"
    background_music: bool = True
    ambient_sounds: bool = True
    
    # Features
    voice_chat_enabled: bool = True
    screen_sharing_enabled: bool = True
    whiteboard_enabled: bool = True
    recording_enabled: bool = False
    
    # Current state
    current_users: Set[str] = field(default_factory=set)
    active_interactions: List[str] = field(default_factory=list)
    
    # Moderation
    moderators: Set[str] = field(default_factory=set)
    banned_users: Set[str] = field(default_factory=set)
    
    # Analytics
    total_visits: int = 0
    peak_concurrent_users: int = 0
    average_session_duration: float = 0.0
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.space_id:
            self.space_id = str(uuid.uuid4())

@dataclass
class SocialEvent:
    """Social metaverse event"""
    event_id: str
    title: str
    description: str
    event_type: EventType
    
    # Scheduling
    start_time: datetime
    end_time: datetime
    timezone_str: str = "UTC"
    
    # Location
    space_id: str
    
    # Participants
    host_id: str
    speakers: List[str] = field(default_factory=list)
    attendees: Set[str] = field(default_factory=set)
    max_attendees: int = 100
    
    # Content
    agenda: List[str] = field(default_factory=list)
    materials: List[str] = field(default_factory=list)
    recording_url: Optional[str] = None
    
    # Status
    is_active: bool = False
    is_recorded: bool = False
    
    # Registration
    requires_registration: bool = True
    registration_fee: float = 0.0
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())

@dataclass
class SocialInteraction:
    """Social interaction record"""
    interaction_id: str
    interaction_type: InteractionType
    
    # Participants
    initiator_id: str
    participants: List[str] = field(default_factory=list)
    
    # Context
    space_id: str
    content: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration: float = 0.0
    
    # Status
    is_active: bool = True
    
    def __post_init__(self):
        if not self.interaction_id:
            self.interaction_id = str(uuid.uuid4())

class SocialSpaceManager:
    """Social spaces manager for metaverse"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.users: Dict[str, SocialUser] = {}
        self.spaces: Dict[str, SocialSpace] = {}
        self.events: Dict[str, SocialEvent] = {}
        self.interactions: Dict[str, SocialInteraction] = {}
        
        # Real-time connections
        self.user_connections: Dict[str, Dict[str, Any]] = {}
        self.space_channels: Dict[str, Set[str]] = {}
        
        # Analytics
        self.analytics_data: Dict[str, Any] = {
            'total_users': 0,
            'active_users': 0,
            'total_spaces': 0,
            'total_events': 0,
            'total_interactions': 0
        }
        
        # Initialize default spaces
        asyncio.create_task(self._initialize_default_spaces())
    
    async def _initialize_default_spaces(self):
        """Initialize default social spaces"""
        try:
            default_spaces = [
                {
                    'name': 'Main Trading Hall',
                    'space_type': SocialSpaceType.TRADING_HALL,
                    'max_capacity': 100,
                    'environment_theme': 'trading_floor'
                },
                {
                    'name': 'Education Center',
                    'space_type': SocialSpaceType.EDUCATION_CENTER,
                    'max_capacity': 50,
                    'environment_theme': 'classroom'
                },
                {
                    'name': 'Strategy Lab',
                    'space_type': SocialSpaceType.STRATEGY_LAB,
                    'max_capacity': 30,
                    'environment_theme': 'laboratory'
                },
                {
                    'name': 'Networking Lounge',
                    'space_type': SocialSpaceType.NETWORKING_LOUNGE,
                    'max_capacity': 75,
                    'environment_theme': 'lounge'
                },
                {
                    'name': 'Competition Arena',
                    'space_type': SocialSpaceType.COMPETITION_ARENA,
                    'max_capacity': 200,
                    'environment_theme': 'arena'
                }
            ]
            
            for space_config in default_spaces:
                space = SocialSpace(
                    space_id="",  # Will be generated
                    name=space_config['name'],
                    space_type=space_config['space_type'],
                    max_capacity=space_config['max_capacity'],
                    environment_theme=space_config['environment_theme']
                )
                
                self.spaces[space.space_id] = space
                self.space_channels[space.space_id] = set()
                
                self.logger.info(f"Default space created: {space.name}")
            
        except Exception as e:
            self.logger.error(f"Default spaces initialization failed: {e}")
    
    async def register_user(self, username: str, display_name: str, 
                          role: UserRole = UserRole.TRADER) -> Optional[str]:
        """Register new social user"""
        try:
            # Check if username exists
            for user in self.users.values():
                if user.username == username:
                    return None
            
            user = SocialUser(
                user_id="",  # Will be generated
                username=username,
                display_name=display_name,
                role=role,
                is_online=True,
                last_activity=datetime.now(timezone.utc)
            )
            
            self.users[user.user_id] = user
            self.analytics_data['total_users'] += 1
            self.analytics_data['active_users'] += 1
            
            self.logger.info(f"User registered: {username} ({user.user_id})")
            return user.user_id
            
        except Exception as e:
            self.logger.error(f"User registration failed: {e}")
            return None
    
    async def join_space(self, user_id: str, space_id: str) -> bool:
        """User joins social space"""
        try:
            if user_id not in self.users or space_id not in self.spaces:
                return False
            
            user = self.users[user_id]
            space = self.spaces[space_id]
            
            # Check capacity
            if len(space.current_users) >= space.max_capacity:
                return False
            
            # Check if banned
            if user_id in space.banned_users:
                return False
            
            # Leave current space if any
            if user.current_space:
                await self.leave_space(user_id, user.current_space)
            
            # Join new space
            space.current_users.add(user_id)
            user.current_space = space_id
            user.last_activity = datetime.now(timezone.utc)
            
            # Add to space channel
            if space_id not in self.space_channels:
                self.space_channels[space_id] = set()
            self.space_channels[space_id].add(user_id)
            
            # Update analytics
            space.total_visits += 1
            if len(space.current_users) > space.peak_concurrent_users:
                space.peak_concurrent_users = len(space.current_users)
            
            # Notify other users
            await self._notify_space_users(space_id, {
                'type': 'user_joined',
                'user_id': user_id,
                'username': user.username,
                'display_name': user.display_name
            })
            
            self.logger.info(f"User {user.username} joined space {space.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Join space failed: {e}")
            return False
    
    async def leave_space(self, user_id: str, space_id: str) -> bool:
        """User leaves social space"""
        try:
            if user_id not in self.users or space_id not in self.spaces:
                return False
            
            user = self.users[user_id]
            space = self.spaces[space_id]
            
            # Remove from space
            space.current_users.discard(user_id)
            user.current_space = None
            
            # Remove from space channel
            if space_id in self.space_channels:
                self.space_channels[space_id].discard(user_id)
            
            # Notify other users
            await self._notify_space_users(space_id, {
                'type': 'user_left',
                'user_id': user_id,
                'username': user.username
            })
            
            self.logger.info(f"User {user.username} left space {space.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Leave space failed: {e}")
            return False
    
    async def create_event(self, host_id: str, title: str, description: str,
                         event_type: EventType, start_time: datetime,
                         end_time: datetime, space_id: str) -> Optional[str]:
        """Create social event"""
        try:
            if host_id not in self.users or space_id not in self.spaces:
                return None
            
            event = SocialEvent(
                event_id="",  # Will be generated
                title=title,
                description=description,
                event_type=event_type,
                start_time=start_time,
                end_time=end_time,
                space_id=space_id,
                host_id=host_id
            )
            
            self.events[event.event_id] = event
            self.analytics_data['total_events'] += 1
            
            # Notify space users
            await self._notify_space_users(space_id, {
                'type': 'event_created',
                'event_id': event.event_id,
                'title': title,
                'start_time': start_time.isoformat(),
                'event_type': event_type.value
            })
            
            self.logger.info(f"Event created: {title} ({event.event_id})")
            return event.event_id
            
        except Exception as e:
            self.logger.error(f"Event creation failed: {e}")
            return None
    
    async def start_interaction(self, initiator_id: str, interaction_type: InteractionType,
                              participants: List[str], space_id: str,
                              content: Dict[str, Any] = None) -> Optional[str]:
        """Start social interaction"""
        try:
            if initiator_id not in self.users or space_id not in self.spaces:
                return None
            
            # Validate participants
            valid_participants = []
            for participant_id in participants:
                if participant_id in self.users and participant_id != initiator_id:
                    valid_participants.append(participant_id)
            
            interaction = SocialInteraction(
                interaction_id="",  # Will be generated
                interaction_type=interaction_type,
                initiator_id=initiator_id,
                participants=valid_participants,
                space_id=space_id,
                content=content or {}
            )
            
            self.interactions[interaction.interaction_id] = interaction
            self.analytics_data['total_interactions'] += 1
            
            # Add to space active interactions
            space = self.spaces[space_id]
            space.active_interactions.append(interaction.interaction_id)
            
            # Notify participants
            all_participants = [initiator_id] + valid_participants
            for participant_id in all_participants:
                await self._notify_user(participant_id, {
                    'type': 'interaction_started',
                    'interaction_id': interaction.interaction_id,
                    'interaction_type': interaction_type.value,
                    'initiator': self.users[initiator_id].display_name
                })
            
            self.logger.info(f"Interaction started: {interaction_type.value} ({interaction.interaction_id})")
            return interaction.interaction_id
            
        except Exception as e:
            self.logger.error(f"Start interaction failed: {e}")
            return None
    
    async def end_interaction(self, interaction_id: str) -> bool:
        """End social interaction"""
        try:
            if interaction_id not in self.interactions:
                return False
            
            interaction = self.interactions[interaction_id]
            interaction.is_active = False
            interaction.end_time = datetime.now(timezone.utc)
            interaction.duration = (interaction.end_time - interaction.start_time).total_seconds()
            
            # Remove from space active interactions
            space = self.spaces[interaction.space_id]
            if interaction_id in space.active_interactions:
                space.active_interactions.remove(interaction_id)
            
            # Notify participants
            all_participants = [interaction.initiator_id] + interaction.participants
            for participant_id in all_participants:
                await self._notify_user(participant_id, {
                    'type': 'interaction_ended',
                    'interaction_id': interaction_id,
                    'duration': interaction.duration
                })
            
            self.logger.info(f"Interaction ended: {interaction_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"End interaction failed: {e}")
            return False
    
    async def send_message(self, sender_id: str, space_id: str, message: str,
                         message_type: str = "text") -> bool:
        """Send message to space"""
        try:
            if sender_id not in self.users or space_id not in self.spaces:
                return False
            
            sender = self.users[sender_id]
            space = self.spaces[space_id]
            
            # Check if user is in space
            if sender_id not in space.current_users:
                return False
            
            message_data = {
                'type': 'message',
                'message_type': message_type,
                'sender_id': sender_id,
                'sender_name': sender.display_name,
                'message': message,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'space_id': space_id
            }
            
            # Notify all users in space
            await self._notify_space_users(space_id, message_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Send message failed: {e}")
            return False
    
    async def follow_user(self, follower_id: str, target_id: str) -> bool:
        """Follow another user"""
        try:
            if follower_id not in self.users or target_id not in self.users:
                return False
            
            if follower_id == target_id:
                return False
            
            follower = self.users[follower_id]
            target = self.users[target_id]
            
            # Add to following/followers
            follower.following.add(target_id)
            target.followers.add(follower_id)
            
            # Notify target user
            await self._notify_user(target_id, {
                'type': 'new_follower',
                'follower_id': follower_id,
                'follower_name': follower.display_name
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Follow user failed: {e}")
            return False
    
    async def get_space_info(self, space_id: str) -> Optional[Dict[str, Any]]:
        """Get space information"""
        try:
            if space_id not in self.spaces:
                return None
            
            space = self.spaces[space_id]
            
            # Get user info for current users
            users_info = []
            for user_id in space.current_users:
                if user_id in self.users:
                    user = self.users[user_id]
                    users_info.append({
                        'user_id': user_id,
                        'username': user.username,
                        'display_name': user.display_name,
                        'role': user.role.value,
                        'level': user.level,
                        'avatar_position': user.avatar_position,
                        'avatar_animation': user.avatar_animation
                    })
            
            # Get active events
            active_events = []
            current_time = datetime.now(timezone.utc)
            for event in self.events.values():
                if (event.space_id == space_id and 
                    event.start_time <= current_time <= event.end_time):
                    active_events.append({
                        'event_id': event.event_id,
                        'title': event.title,
                        'event_type': event.event_type.value,
                        'host_id': event.host_id,
                        'attendees_count': len(event.attendees)
                    })
            
            return {
                'space_id': space_id,
                'name': space.name,
                'space_type': space.space_type.value,
                'environment_theme': space.environment_theme,
                'current_users': users_info,
                'user_count': len(space.current_users),
                'max_capacity': space.max_capacity,
                'active_events': active_events,
                'active_interactions': len(space.active_interactions),
                'features': {
                    'voice_chat': space.voice_chat_enabled,
                    'screen_sharing': space.screen_sharing_enabled,
                    'whiteboard': space.whiteboard_enabled,
                    'recording': space.recording_enabled
                },
                'analytics': {
                    'total_visits': space.total_visits,
                    'peak_concurrent_users': space.peak_concurrent_users
                }
            }
            
        except Exception as e:
            self.logger.error(f"Get space info failed: {e}")
            return None
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile"""
        try:
            if user_id not in self.users:
                return None
            
            user = self.users[user_id]
            
            return {
                'user_id': user_id,
                'username': user.username,
                'display_name': user.display_name,
                'role': user.role.value,
                'level': user.level,
                'experience_points': user.experience_points,
                'reputation_score': user.reputation_score,
                'is_online': user.is_online,
                'current_space': user.current_space,
                'trading_stats': {
                    'win_rate': user.win_rate,
                    'total_trades': user.total_trades,
                    'profit_loss': user.profit_loss
                },
                'social_stats': {
                    'followers_count': len(user.followers),
                    'following_count': len(user.following),
                    'friends_count': len(user.friends)
                },
                'avatar': {
                    'url': user.avatar_url,
                    'position': user.avatar_position,
                    'rotation': user.avatar_rotation,
                    'animation': user.avatar_animation
                }
            }
            
        except Exception as e:
            self.logger.error(f"Get user profile failed: {e}")
            return None
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get social spaces analytics"""
        try:
            # Update active users count
            active_count = sum(1 for user in self.users.values() if user.is_online)
            self.analytics_data['active_users'] = active_count
            
            # Space analytics
            space_analytics = {}
            for space_id, space in self.spaces.items():
                space_analytics[space_id] = {
                    'name': space.name,
                    'type': space.space_type.value,
                    'current_users': len(space.current_users),
                    'total_visits': space.total_visits,
                    'peak_concurrent': space.peak_concurrent_users
                }
            
            # Event analytics
            current_time = datetime.now(timezone.utc)
            active_events = sum(1 for event in self.events.values() 
                              if event.start_time <= current_time <= event.end_time)
            
            return {
                'overview': self.analytics_data,
                'spaces': space_analytics,
                'events': {
                    'total_events': len(self.events),
                    'active_events': active_events
                },
                'interactions': {
                    'total_interactions': len(self.interactions),
                    'active_interactions': sum(1 for i in self.interactions.values() if i.is_active)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Get analytics failed: {e}")
            return {}
    
    async def _notify_space_users(self, space_id: str, message: Dict[str, Any]):
        """Notify all users in a space"""
        try:
            if space_id in self.space_channels:
                for user_id in self.space_channels[space_id]:
                    await self._notify_user(user_id, message)
        except Exception as e:
            self.logger.error(f"Space notification failed: {e}")
    
    async def _notify_user(self, user_id: str, message: Dict[str, Any]):
        """Notify specific user"""
        try:
            # In a real implementation, this would send the message
            # through WebSocket or other real-time communication
            self.logger.debug(f"Notify user {user_id}: {message['type']}")
        except Exception as e:
            self.logger.error(f"User notification failed: {e}")

# Example usage
async def main():
    """
    Example usage of Social Spaces Manager
    """
    print("🌐 Social Spaces Manager - Phase 5 Testing")
    print("=" * 50)
    
    # Initialize social manager
    social_manager = SocialSpaceManager()
    
    # Wait for default spaces to be created
    await asyncio.sleep(1)
    
    # Test user registration
    print("\n👥 Registering Users:")
    
    test_users = [
        ("alice_trader", "Alice Johnson", UserRole.TRADER),
        ("bob_mentor", "Bob Smith", UserRole.MENTOR),
        ("carol_analyst", "Carol Davis", UserRole.ANALYST),
        ("dave_educator", "Dave Wilson", UserRole.EDUCATOR)
    ]
    
    user_ids = []
    for username, display_name, role in test_users:
        user_id = await social_manager.register_user(username, display_name, role)
        if user_id:
            user_ids.append(user_id)
            print(f"  {display_name}: {user_id}")
    
    # Test joining spaces
    print(f"\n🏛️ Users Joining Spaces:")
    
    spaces = list(social_manager.spaces.keys())
    for i, user_id in enumerate(user_ids[:3]):  # First 3 users join spaces
        space_id = spaces[i % len(spaces)]
        success = await social_manager.join_space(user_id, space_id)
        if success:
            user = social_manager.users[user_id]
            space = social_manager.spaces[space_id]
            print(f"  {user.display_name} joined {space.name}")
    
    # Test creating events
    print(f"\n📅 Creating Events:")
    
    if user_ids and spaces:
        host_id = user_ids[1]  # Bob as host
        space_id = spaces[0]   # Main Trading Hall
        
        start_time = datetime.now(timezone.utc) + timedelta(hours=1)
        end_time = start_time + timedelta(hours=2)
        
        event_id = await social_manager.create_event(
            host_id=host_id,
            title="Advanced Trading Strategies Workshop",
            description="Learn advanced trading techniques and risk management",
            event_type=EventType.WORKSHOP,
            start_time=start_time,
            end_time=end_time,
            space_id=space_id
        )
        
        if event_id:
            print(f"  Workshop created: {event_id}")
    
    # Test interactions
    print(f"\n💬 Testing Interactions:")
    
    if len(user_ids) >= 2:
        # Voice chat interaction
        interaction_id = await social_manager.start_interaction(
            initiator_id=user_ids[0],
            interaction_type=InteractionType.VOICE_CHAT,
            participants=[user_ids[1]],
            space_id=spaces[0],
            content={'quality': 'high', 'codec': 'opus'}
        )
        
        if interaction_id:
            print(f"  Voice chat started: {interaction_id}")
            
            # End interaction after a moment
            await asyncio.sleep(0.1)
            await social_manager.end_interaction(interaction_id)
            print(f"  Voice chat ended: {interaction_id}")
    
    # Test messaging
    print(f"\n💬 Testing Messaging:")
    
    if user_ids and spaces:
        success = await social_manager.send_message(
            sender_id=user_ids[0],
            space_id=spaces[0],
            message="Hello everyone! Ready for some trading?",
            message_type="text"
        )
        print(f"  Message sent: {'✅' if success else '❌'}")
    
    # Test following
    print(f"\n👥 Testing Social Features:")
    
    if len(user_ids) >= 2:
        success = await social_manager.follow_user(user_ids[0], user_ids[1])
        print(f"  Follow user: {'✅' if success else '❌'}")
    
    # Get space information
    print(f"\n🏛️ Space Information:")
    
    if spaces:
        space_info = await social_manager.get_space_info(spaces[0])
        if space_info:
            print(f"  Space: {space_info['name']}")
            print(f"  Users: {space_info['user_count']}/{space_info['max_capacity']}")
            print(f"  Type: {space_info['space_type']}")
            print(f"  Active Events: {len(space_info['active_events'])}")
    
    # Get user profiles
    print(f"\n👤 User Profiles:")
    
    for user_id in user_ids[:2]:
        profile = await social_manager.get_user_profile(user_id)
        if profile:
            print(f"  {profile['display_name']} ({profile['role']})")
            print(f"    Level: {profile['level']}")
            print(f"    Followers: {profile['social_stats']['followers_count']}")
    
    # Get analytics
    print(f"\n📊 Analytics:")
    
    analytics = await social_manager.get_analytics()
    if analytics:
        print(f"  Total Users: {analytics['overview']['total_users']}")
        print(f"  Active Users: {analytics['overview']['active_users']}")
        print(f"  Total Spaces: {len(analytics['spaces'])}")
        print(f"  Total Events: {analytics['events']['total_events']}")
        print(f"  Total Interactions: {analytics['interactions']['total_interactions']}")
    
    print(f"\n✅ Social Spaces testing completed!")

if __name__ == "__main__":
    asyncio.run(main())