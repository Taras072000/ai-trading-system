"""
NFT Trading System for Phase 5
Implements NFT trading, collection, and marketplace functionality
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

class NFTStandard(Enum):
    ERC721 = "erc721"
    ERC1155 = "erc1155"
    SPL_TOKEN = "spl_token"  # Solana
    METAPLEX = "metaplex"    # Solana NFT standard

class NFTCategory(Enum):
    ART = "art"
    COLLECTIBLES = "collectibles"
    GAMING = "gaming"
    MUSIC = "music"
    SPORTS = "sports"
    VIRTUAL_WORLDS = "virtual_worlds"
    TRADING_CARDS = "trading_cards"
    UTILITY = "utility"
    DOMAIN_NAMES = "domain_names"
    PHOTOGRAPHY = "photography"

class NFTRarity(Enum):
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    EPIC = "epic"
    LEGENDARY = "legendary"
    MYTHIC = "mythic"

class ListingType(Enum):
    FIXED_PRICE = "fixed_price"
    AUCTION = "auction"
    DUTCH_AUCTION = "dutch_auction"
    BUNDLE = "bundle"
    OFFER = "offer"

class AuctionStatus(Enum):
    ACTIVE = "active"
    ENDED = "ended"
    CANCELLED = "cancelled"
    SETTLED = "settled"

@dataclass
class NFTMetadata:
    """NFT metadata structure"""
    name: str
    description: str
    image: str
    external_url: Optional[str] = None
    animation_url: Optional[str] = None
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def add_attribute(self, trait_type: str, value: Any, display_type: str = None):
        """Add attribute to NFT metadata"""
        attribute = {
            "trait_type": trait_type,
            "value": value
        }
        if display_type:
            attribute["display_type"] = display_type
        
        self.attributes.append(attribute)
    
    def get_rarity_score(self) -> float:
        """Calculate rarity score based on attributes"""
        # Simple rarity calculation - can be enhanced
        base_score = 1.0
        
        for attr in self.attributes:
            trait_type = attr.get("trait_type", "")
            value = attr.get("value", "")
            
            # Assign rarity multipliers based on trait types
            if "rare" in str(value).lower():
                base_score *= 2.0
            elif "epic" in str(value).lower():
                base_score *= 3.0
            elif "legendary" in str(value).lower():
                base_score *= 5.0
        
        return base_score

@dataclass
class NFTToken:
    """NFT token representation"""
    token_id: str
    contract_address: str
    blockchain: str
    standard: NFTStandard
    owner_address: str
    creator_address: str
    
    # Metadata
    metadata: NFTMetadata
    metadata_uri: str
    
    # Trading info
    current_price: Optional[Decimal] = None
    last_sale_price: Optional[Decimal] = None
    currency: str = "ETH"
    
    # Collection info
    collection_name: str = ""
    collection_slug: str = ""
    
    # Rarity and stats
    rarity: NFTRarity = NFTRarity.COMMON
    rarity_rank: Optional[int] = None
    total_supply: int = 1
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_transfer_at: Optional[datetime] = None
    
    # Trading history
    transfer_count: int = 0
    view_count: int = 0
    favorite_count: int = 0
    
    def __post_init__(self):
        if not self.token_id:
            self.token_id = self._generate_token_id()
        
        # Calculate rarity if not set
        if self.rarity == NFTRarity.COMMON and self.metadata:
            rarity_score = self.metadata.get_rarity_score()
            self.rarity = self._calculate_rarity_from_score(rarity_score)
    
    def _generate_token_id(self) -> str:
        """Generate unique token ID"""
        data = f"{self.contract_address}{self.owner_address}{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _calculate_rarity_from_score(self, score: float) -> NFTRarity:
        """Calculate rarity tier from score"""
        if score >= 10.0:
            return NFTRarity.MYTHIC
        elif score >= 5.0:
            return NFTRarity.LEGENDARY
        elif score >= 3.0:
            return NFTRarity.EPIC
        elif score >= 2.0:
            return NFTRarity.RARE
        elif score >= 1.5:
            return NFTRarity.UNCOMMON
        else:
            return NFTRarity.COMMON

@dataclass
class NFTListing:
    """NFT marketplace listing"""
    listing_id: str
    nft_token: NFTToken
    seller_address: str
    listing_type: ListingType
    
    # Pricing
    price: Decimal
    currency: str
    
    # Auction specific
    auction_end_time: Optional[datetime] = None
    reserve_price: Optional[Decimal] = None
    current_bid: Optional[Decimal] = None
    highest_bidder: Optional[str] = None
    bid_count: int = 0
    
    # Bundle specific
    bundle_tokens: List[NFTToken] = field(default_factory=list)
    
    # Status and timestamps
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.listing_id:
            self.listing_id = self._generate_listing_id()
    
    def _generate_listing_id(self) -> str:
        """Generate unique listing ID"""
        data = f"{self.nft_token.token_id}{self.seller_address}{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def is_auction_ended(self) -> bool:
        """Check if auction has ended"""
        if self.listing_type not in [ListingType.AUCTION, ListingType.DUTCH_AUCTION]:
            return False
        
        if not self.auction_end_time:
            return False
        
        return datetime.now(timezone.utc) >= self.auction_end_time
    
    def get_current_price(self) -> Decimal:
        """Get current price (for Dutch auctions, calculate current price)"""
        if self.listing_type == ListingType.DUTCH_AUCTION:
            return self._calculate_dutch_auction_price()
        elif self.listing_type == ListingType.AUCTION and self.current_bid:
            return self.current_bid
        else:
            return self.price
    
    def _calculate_dutch_auction_price(self) -> Decimal:
        """Calculate current price for Dutch auction"""
        if not self.auction_end_time or not self.reserve_price:
            return self.price
        
        now = datetime.now(timezone.utc)
        total_duration = (self.auction_end_time - self.created_at).total_seconds()
        elapsed = (now - self.created_at).total_seconds()
        
        if elapsed >= total_duration:
            return self.reserve_price
        
        # Linear price decrease
        price_range = self.price - self.reserve_price
        price_decrease = price_range * (elapsed / total_duration)
        
        return self.price - price_decrease

@dataclass
class NFTBid:
    """NFT auction bid"""
    bid_id: str
    listing_id: str
    bidder_address: str
    amount: Decimal
    currency: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    
    def __post_init__(self):
        if not self.bid_id:
            self.bid_id = self._generate_bid_id()
    
    def _generate_bid_id(self) -> str:
        """Generate unique bid ID"""
        data = f"{self.listing_id}{self.bidder_address}{self.amount}{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

@dataclass
class NFTCollection:
    """NFT collection information"""
    collection_id: str
    name: str
    slug: str
    description: str
    creator_address: str
    
    # Collection metadata
    image_url: str
    banner_url: str
    website_url: Optional[str] = None
    discord_url: Optional[str] = None
    twitter_url: Optional[str] = None
    
    # Collection stats
    total_supply: int = 0
    owners_count: int = 0
    floor_price: Optional[Decimal] = None
    volume_traded: Decimal = Decimal('0')
    
    # Categories and tags
    category: NFTCategory = NFTCategory.ART
    tags: List[str] = field(default_factory=list)
    
    # Verification and status
    is_verified: bool = False
    is_featured: bool = False
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.collection_id:
            self.collection_id = self._generate_collection_id()
    
    def _generate_collection_id(self) -> str:
        """Generate unique collection ID"""
        data = f"{self.name}{self.creator_address}{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

class NFTMarketplace:
    """NFT marketplace for trading and discovery"""
    
    def __init__(self, config_path: str = "config/nft_config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Core data structures
        self.nft_tokens: Dict[str, NFTToken] = {}
        self.collections: Dict[str, NFTCollection] = {}
        self.listings: Dict[str, NFTListing] = {}
        self.bids: Dict[str, List[NFTBid]] = defaultdict(list)
        
        # User data
        self.user_portfolios: Dict[str, List[str]] = defaultdict(list)  # user -> token_ids
        self.user_favorites: Dict[str, List[str]] = defaultdict(list)   # user -> token_ids
        self.user_watchlists: Dict[str, List[str]] = defaultdict(list) # user -> collection_ids
        
        # Trading history
        self.trade_history: List[Dict[str, Any]] = []
        self.price_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Marketplace metrics
        self.metrics = {
            'total_nfts': 0,
            'total_collections': 0,
            'total_listings': 0,
            'total_volume': Decimal('0'),
            'total_trades': 0,
            'active_users': 0,
            'floor_prices': {},
            'trending_collections': []
        }
        
        # Background tasks
        self.background_tasks = []
        self.is_running = False
    
    def _load_config(self, config_path: str) -> Dict:
        """Load NFT marketplace configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default NFT marketplace configuration"""
        return {
            'marketplace': {
                'fee_percentage': 2.5,
                'royalty_percentage': 5.0,
                'min_bid_increment': 0.01,
                'auction_extension_time': 600,  # 10 minutes
                'max_auction_duration': 30 * 24 * 3600  # 30 days
            },
            'supported_currencies': ['ETH', 'WETH', 'USDC', 'PEPER'],
            'supported_blockchains': ['ethereum', 'polygon', 'solana', 'binance_smart_chain'],
            'ipfs': {
                'gateway': 'https://ipfs.io/ipfs/',
                'pinning_service': 'pinata'
            }
        }
    
    async def initialize(self):
        """Initialize NFT marketplace"""
        self.logger.info("Initializing NFT marketplace...")
        
        # Start background tasks
        self.is_running = True
        self.background_tasks = [
            asyncio.create_task(self._auction_monitor()),
            asyncio.create_task(self._metrics_updater()),
            asyncio.create_task(self._trending_calculator())
        ]
        
        self.logger.info("NFT marketplace initialized")
    
    async def mint_nft(self, creator_address: str, metadata: NFTMetadata,
                      contract_address: str, blockchain: str,
                      collection_id: str = None) -> Optional[str]:
        """Mint new NFT"""
        try:
            # Create NFT token
            nft_token = NFTToken(
                token_id="",  # Will be generated
                contract_address=contract_address,
                blockchain=blockchain,
                standard=NFTStandard.ERC721,  # Default
                owner_address=creator_address,
                creator_address=creator_address,
                metadata=metadata,
                metadata_uri=f"ipfs://{uuid.uuid4().hex}",  # Placeholder
                collection_name=self.collections.get(collection_id, {}).name if collection_id else "",
                collection_slug=self.collections.get(collection_id, {}).slug if collection_id else ""
            )
            
            # Store NFT
            self.nft_tokens[nft_token.token_id] = nft_token
            
            # Add to user portfolio
            self.user_portfolios[creator_address].append(nft_token.token_id)
            
            # Update collection stats
            if collection_id and collection_id in self.collections:
                collection = self.collections[collection_id]
                collection.total_supply += 1
                if creator_address not in [owner for owners in self.user_portfolios.values() for owner in owners]:
                    collection.owners_count += 1
            
            # Update metrics
            self.metrics['total_nfts'] += 1
            
            self.logger.info(f"NFT minted: {nft_token.token_id} by {creator_address}")
            return nft_token.token_id
            
        except Exception as e:
            self.logger.error(f"NFT minting failed: {e}")
            return None
    
    async def create_collection(self, creator_address: str, name: str,
                              description: str, image_url: str,
                              category: NFTCategory = NFTCategory.ART) -> Optional[str]:
        """Create new NFT collection"""
        try:
            collection = NFTCollection(
                collection_id="",  # Will be generated
                name=name,
                slug=name.lower().replace(" ", "-"),
                description=description,
                creator_address=creator_address,
                image_url=image_url,
                banner_url=image_url,  # Use same image as banner for now
                category=category
            )
            
            # Store collection
            self.collections[collection.collection_id] = collection
            
            # Update metrics
            self.metrics['total_collections'] += 1
            
            self.logger.info(f"Collection created: {collection.collection_id} by {creator_address}")
            return collection.collection_id
            
        except Exception as e:
            self.logger.error(f"Collection creation failed: {e}")
            return None
    
    async def create_listing(self, seller_address: str, token_id: str,
                           listing_type: ListingType, price: Decimal,
                           currency: str = "ETH",
                           auction_duration: int = None,
                           reserve_price: Decimal = None) -> Optional[str]:
        """Create NFT listing"""
        try:
            if token_id not in self.nft_tokens:
                return None
            
            nft_token = self.nft_tokens[token_id]
            
            # Check ownership
            if nft_token.owner_address != seller_address:
                return None
            
            # Set auction end time if applicable
            auction_end_time = None
            if listing_type in [ListingType.AUCTION, ListingType.DUTCH_AUCTION]:
                duration = auction_duration or 7 * 24 * 3600  # Default 7 days
                auction_end_time = datetime.now(timezone.utc) + timedelta(seconds=duration)
            
            # Create listing
            listing = NFTListing(
                listing_id="",  # Will be generated
                nft_token=nft_token,
                seller_address=seller_address,
                listing_type=listing_type,
                price=price,
                currency=currency,
                auction_end_time=auction_end_time,
                reserve_price=reserve_price
            )
            
            # Store listing
            self.listings[listing.listing_id] = listing
            
            # Update NFT current price
            nft_token.current_price = price
            nft_token.currency = currency
            
            # Update metrics
            self.metrics['total_listings'] += 1
            
            self.logger.info(f"Listing created: {listing.listing_id} for {token_id}")
            return listing.listing_id
            
        except Exception as e:
            self.logger.error(f"Listing creation failed: {e}")
            return None
    
    async def place_bid(self, bidder_address: str, listing_id: str,
                       amount: Decimal, currency: str = "ETH") -> bool:
        """Place bid on auction"""
        try:
            if listing_id not in self.listings:
                return False
            
            listing = self.listings[listing_id]
            
            # Check if auction is active
            if listing.listing_type not in [ListingType.AUCTION, ListingType.DUTCH_AUCTION]:
                return False
            
            if listing.is_auction_ended():
                return False
            
            # Check bid amount
            min_bid = listing.current_bid or listing.reserve_price or listing.price
            min_increment = Decimal(str(self.config['marketplace']['min_bid_increment']))
            
            if amount < min_bid + min_increment:
                return False
            
            # Create bid
            bid = NFTBid(
                bid_id="",  # Will be generated
                listing_id=listing_id,
                bidder_address=bidder_address,
                amount=amount,
                currency=currency
            )
            
            # Store bid
            self.bids[listing_id].append(bid)
            
            # Update listing
            listing.current_bid = amount
            listing.highest_bidder = bidder_address
            listing.bid_count += 1
            listing.updated_at = datetime.now(timezone.utc)
            
            # Extend auction if near end (anti-sniping)
            if listing.auction_end_time:
                time_left = (listing.auction_end_time - datetime.now(timezone.utc)).total_seconds()
                extension_time = self.config['marketplace']['auction_extension_time']
                
                if time_left < extension_time:
                    listing.auction_end_time += timedelta(seconds=extension_time)
            
            self.logger.info(f"Bid placed: {amount} {currency} on {listing_id} by {bidder_address}")
            return True
            
        except Exception as e:
            self.logger.error(f"Bid placement failed: {e}")
            return False
    
    async def buy_now(self, buyer_address: str, listing_id: str) -> bool:
        """Buy NFT at fixed price"""
        try:
            if listing_id not in self.listings:
                return False
            
            listing = self.listings[listing_id]
            
            # Check if it's a fixed price listing
            if listing.listing_type != ListingType.FIXED_PRICE:
                return False
            
            if not listing.is_active:
                return False
            
            # Execute trade
            success = await self._execute_trade(
                listing=listing,
                buyer_address=buyer_address,
                price=listing.price,
                currency=listing.currency
            )
            
            if success:
                # Deactivate listing
                listing.is_active = False
                listing.updated_at = datetime.now(timezone.utc)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Buy now failed: {e}")
            return False
    
    async def settle_auction(self, listing_id: str) -> bool:
        """Settle ended auction"""
        try:
            if listing_id not in self.listings:
                return False
            
            listing = self.listings[listing_id]
            
            # Check if auction has ended
            if not listing.is_auction_ended():
                return False
            
            # Check if there are bids
            if not listing.highest_bidder or not listing.current_bid:
                # No bids, cancel auction
                listing.is_active = False
                return True
            
            # Check reserve price
            if listing.reserve_price and listing.current_bid < listing.reserve_price:
                # Reserve not met, cancel auction
                listing.is_active = False
                return True
            
            # Execute trade with highest bidder
            success = await self._execute_trade(
                listing=listing,
                buyer_address=listing.highest_bidder,
                price=listing.current_bid,
                currency=listing.currency
            )
            
            if success:
                listing.is_active = False
                listing.updated_at = datetime.now(timezone.utc)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Auction settlement failed: {e}")
            return False
    
    async def _execute_trade(self, listing: NFTListing, buyer_address: str,
                           price: Decimal, currency: str) -> bool:
        """Execute NFT trade"""
        try:
            nft_token = listing.nft_token
            seller_address = listing.seller_address
            
            # Calculate fees
            marketplace_fee = price * Decimal(str(self.config['marketplace']['fee_percentage'] / 100))
            royalty_fee = price * Decimal(str(self.config['marketplace']['royalty_percentage'] / 100))
            seller_proceeds = price - marketplace_fee - royalty_fee
            
            # Transfer NFT ownership
            nft_token.owner_address = buyer_address
            nft_token.last_sale_price = price
            nft_token.last_transfer_at = datetime.now(timezone.utc)
            nft_token.transfer_count += 1
            
            # Update user portfolios
            if nft_token.token_id in self.user_portfolios[seller_address]:
                self.user_portfolios[seller_address].remove(nft_token.token_id)
            self.user_portfolios[buyer_address].append(nft_token.token_id)
            
            # Record trade
            trade_record = {
                'trade_id': uuid.uuid4().hex,
                'token_id': nft_token.token_id,
                'seller_address': seller_address,
                'buyer_address': buyer_address,
                'price': price,
                'currency': currency,
                'marketplace_fee': marketplace_fee,
                'royalty_fee': royalty_fee,
                'seller_proceeds': seller_proceeds,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'listing_type': listing.listing_type.value
            }
            
            self.trade_history.append(trade_record)
            
            # Update price history
            self.price_history[nft_token.token_id].append({
                'price': price,
                'currency': currency,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            # Update collection stats
            if nft_token.collection_slug:
                for collection in self.collections.values():
                    if collection.slug == nft_token.collection_slug:
                        collection.volume_traded += price
                        
                        # Update floor price
                        if not collection.floor_price or price < collection.floor_price:
                            collection.floor_price = price
                        
                        break
            
            # Update metrics
            self.metrics['total_volume'] += price
            self.metrics['total_trades'] += 1
            
            self.logger.info(f"Trade executed: {nft_token.token_id} sold for {price} {currency}")
            return True
            
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}")
            return False
    
    async def get_nft_details(self, token_id: str) -> Optional[Dict[str, Any]]:
        """Get NFT details"""
        if token_id not in self.nft_tokens:
            return None
        
        nft = self.nft_tokens[token_id]
        
        # Get current listing
        current_listing = None
        for listing in self.listings.values():
            if listing.nft_token.token_id == token_id and listing.is_active:
                current_listing = {
                    'listing_id': listing.listing_id,
                    'type': listing.listing_type.value,
                    'price': listing.price,
                    'currency': listing.currency,
                    'seller': listing.seller_address,
                    'created_at': listing.created_at.isoformat()
                }
                
                if listing.listing_type in [ListingType.AUCTION, ListingType.DUTCH_AUCTION]:
                    current_listing.update({
                        'auction_end_time': listing.auction_end_time.isoformat() if listing.auction_end_time else None,
                        'current_bid': listing.current_bid,
                        'highest_bidder': listing.highest_bidder,
                        'bid_count': listing.bid_count
                    })
                break
        
        # Get price history
        price_history = self.price_history.get(token_id, [])
        
        return {
            'token_id': nft.token_id,
            'contract_address': nft.contract_address,
            'blockchain': nft.blockchain,
            'standard': nft.standard.value,
            'owner_address': nft.owner_address,
            'creator_address': nft.creator_address,
            'metadata': {
                'name': nft.metadata.name,
                'description': nft.metadata.description,
                'image': nft.metadata.image,
                'attributes': nft.metadata.attributes,
                'properties': nft.metadata.properties
            },
            'collection': {
                'name': nft.collection_name,
                'slug': nft.collection_slug
            },
            'pricing': {
                'current_price': nft.current_price,
                'last_sale_price': nft.last_sale_price,
                'currency': nft.currency
            },
            'rarity': {
                'tier': nft.rarity.value,
                'rank': nft.rarity_rank,
                'score': nft.metadata.get_rarity_score()
            },
            'stats': {
                'transfer_count': nft.transfer_count,
                'view_count': nft.view_count,
                'favorite_count': nft.favorite_count
            },
            'timestamps': {
                'created_at': nft.created_at.isoformat(),
                'last_transfer_at': nft.last_transfer_at.isoformat() if nft.last_transfer_at else None
            },
            'current_listing': current_listing,
            'price_history': price_history[-10:]  # Last 10 sales
        }
    
    async def get_collection_details(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """Get collection details"""
        if collection_id not in self.collections:
            return None
        
        collection = self.collections[collection_id]
        
        # Get collection NFTs
        collection_nfts = [
            nft for nft in self.nft_tokens.values()
            if nft.collection_slug == collection.slug
        ]
        
        # Calculate stats
        total_volume = sum(
            trade['price'] for trade in self.trade_history
            if any(nft.token_id == trade['token_id'] for nft in collection_nfts)
        )
        
        return {
            'collection_id': collection.collection_id,
            'name': collection.name,
            'slug': collection.slug,
            'description': collection.description,
            'creator_address': collection.creator_address,
            'images': {
                'image_url': collection.image_url,
                'banner_url': collection.banner_url
            },
            'links': {
                'website_url': collection.website_url,
                'discord_url': collection.discord_url,
                'twitter_url': collection.twitter_url
            },
            'stats': {
                'total_supply': collection.total_supply,
                'owners_count': collection.owners_count,
                'floor_price': collection.floor_price,
                'volume_traded': total_volume
            },
            'metadata': {
                'category': collection.category.value,
                'tags': collection.tags,
                'is_verified': collection.is_verified,
                'is_featured': collection.is_featured
            },
            'created_at': collection.created_at.isoformat()
        }
    
    async def search_nfts(self, query: str = "", category: NFTCategory = None,
                         min_price: Decimal = None, max_price: Decimal = None,
                         rarity: NFTRarity = None, blockchain: str = None,
                         sort_by: str = "created_at", limit: int = 50) -> List[Dict[str, Any]]:
        """Search NFTs with filters"""
        results = []
        
        for nft in self.nft_tokens.values():
            # Apply filters
            if query and query.lower() not in nft.metadata.name.lower():
                continue
            
            if category and nft.collection_name:
                collection = next((c for c in self.collections.values() 
                                 if c.name == nft.collection_name), None)
                if not collection or collection.category != category:
                    continue
            
            if min_price and (not nft.current_price or nft.current_price < min_price):
                continue
            
            if max_price and (not nft.current_price or nft.current_price > max_price):
                continue
            
            if rarity and nft.rarity != rarity:
                continue
            
            if blockchain and nft.blockchain != blockchain:
                continue
            
            # Get NFT details
            nft_details = await self.get_nft_details(nft.token_id)
            if nft_details:
                results.append(nft_details)
        
        # Sort results
        if sort_by == "price_low":
            results.sort(key=lambda x: x['pricing']['current_price'] or Decimal('0'))
        elif sort_by == "price_high":
            results.sort(key=lambda x: x['pricing']['current_price'] or Decimal('0'), reverse=True)
        elif sort_by == "rarity":
            results.sort(key=lambda x: x['rarity']['score'], reverse=True)
        elif sort_by == "recent":
            results.sort(key=lambda x: x['timestamps']['created_at'], reverse=True)
        
        return results[:limit]
    
    async def get_user_portfolio(self, user_address: str) -> Dict[str, Any]:
        """Get user's NFT portfolio"""
        user_nfts = []
        total_value = Decimal('0')
        
        for token_id in self.user_portfolios.get(user_address, []):
            nft_details = await self.get_nft_details(token_id)
            if nft_details:
                user_nfts.append(nft_details)
                if nft_details['pricing']['current_price']:
                    total_value += nft_details['pricing']['current_price']
        
        # Get user's active listings
        active_listings = []
        for listing in self.listings.values():
            if listing.seller_address == user_address and listing.is_active:
                active_listings.append({
                    'listing_id': listing.listing_id,
                    'token_id': listing.nft_token.token_id,
                    'type': listing.listing_type.value,
                    'price': listing.price,
                    'currency': listing.currency,
                    'created_at': listing.created_at.isoformat()
                })
        
        # Get user's bids
        active_bids = []
        for listing_id, bids in self.bids.items():
            user_bids = [bid for bid in bids if bid.bidder_address == user_address and bid.is_active]
            for bid in user_bids:
                active_bids.append({
                    'bid_id': bid.bid_id,
                    'listing_id': bid.listing_id,
                    'amount': bid.amount,
                    'currency': bid.currency,
                    'timestamp': bid.timestamp.isoformat()
                })
        
        return {
            'user_address': user_address,
            'portfolio': {
                'nfts': user_nfts,
                'total_count': len(user_nfts),
                'total_value': total_value
            },
            'activity': {
                'active_listings': active_listings,
                'active_bids': active_bids
            },
            'favorites': self.user_favorites.get(user_address, []),
            'watchlists': self.user_watchlists.get(user_address, [])
        }
    
    async def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics"""
        # Calculate trending collections
        recent_trades = [
            trade for trade in self.trade_history
            if datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00')) > 
               datetime.now(timezone.utc) - timedelta(days=7)
        ]
        
        collection_volumes = defaultdict(Decimal)
        for trade in recent_trades:
            token_id = trade['token_id']
            if token_id in self.nft_tokens:
                nft = self.nft_tokens[token_id]
                if nft.collection_slug:
                    collection_volumes[nft.collection_slug] += trade['price']
        
        trending_collections = sorted(
            collection_volumes.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'marketplace_stats': self.metrics,
            'trending_collections': [
                {
                    'collection_slug': slug,
                    'volume_7d': volume
                }
                for slug, volume in trending_collections
            ],
            'recent_sales': self.trade_history[-10:],  # Last 10 sales
            'top_collections_by_volume': [
                {
                    'name': collection.name,
                    'slug': collection.slug,
                    'volume': collection.volume_traded,
                    'floor_price': collection.floor_price
                }
                for collection in sorted(
                    self.collections.values(),
                    key=lambda x: x.volume_traded,
                    reverse=True
                )[:10]
            ]
        }
    
    async def _auction_monitor(self):
        """Background task to monitor auctions"""
        while self.is_running:
            try:
                now = datetime.now(timezone.utc)
                
                for listing in self.listings.values():
                    if (listing.is_active and 
                        listing.listing_type in [ListingType.AUCTION, ListingType.DUTCH_AUCTION] and
                        listing.auction_end_time and
                        now >= listing.auction_end_time):
                        
                        # Settle ended auction
                        await self.settle_auction(listing.listing_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Auction monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_updater(self):
        """Background task to update marketplace metrics"""
        while self.is_running:
            try:
                # Update metrics every 5 minutes
                await asyncio.sleep(300)
                
                # Update active users (users with activity in last 30 days)
                thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
                active_users = set()
                
                for trade in self.trade_history:
                    trade_time = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
                    if trade_time > thirty_days_ago:
                        active_users.add(trade['seller_address'])
                        active_users.add(trade['buyer_address'])
                
                self.metrics['active_users'] = len(active_users)
                
                # Update floor prices
                for collection in self.collections.values():
                    collection_nfts = [
                        nft for nft in self.nft_tokens.values()
                        if nft.collection_slug == collection.slug
                    ]
                    
                    active_listings = [
                        listing for listing in self.listings.values()
                        if (listing.is_active and 
                            listing.nft_token.collection_slug == collection.slug and
                            listing.listing_type == ListingType.FIXED_PRICE)
                    ]
                    
                    if active_listings:
                        floor_price = min(listing.price for listing in active_listings)
                        collection.floor_price = floor_price
                        self.metrics['floor_prices'][collection.slug] = floor_price
                
            except Exception as e:
                self.logger.error(f"Metrics updater error: {e}")
    
    async def _trending_calculator(self):
        """Background task to calculate trending collections"""
        while self.is_running:
            try:
                # Update trending every hour
                await asyncio.sleep(3600)
                
                # Calculate trending based on recent activity
                recent_trades = [
                    trade for trade in self.trade_history
                    if datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00')) > 
                       datetime.now(timezone.utc) - timedelta(hours=24)
                ]
                
                collection_activity = defaultdict(lambda: {'volume': Decimal('0'), 'trades': 0})
                
                for trade in recent_trades:
                    token_id = trade['token_id']
                    if token_id in self.nft_tokens:
                        nft = self.nft_tokens[token_id]
                        if nft.collection_slug:
                            collection_activity[nft.collection_slug]['volume'] += trade['price']
                            collection_activity[nft.collection_slug]['trades'] += 1
                
                # Sort by combined score (volume + trade count)
                trending = sorted(
                    collection_activity.items(),
                    key=lambda x: float(x[1]['volume']) + x[1]['trades'] * 100,
                    reverse=True
                )
                
                self.metrics['trending_collections'] = [
                    {
                        'collection_slug': slug,
                        'volume_24h': activity['volume'],
                        'trades_24h': activity['trades']
                    }
                    for slug, activity in trending[:20]
                ]
                
            except Exception as e:
                self.logger.error(f"Trending calculator error: {e}")
    
    async def cleanup(self):
        """Cleanup NFT marketplace"""
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.logger.info("NFT marketplace cleaned up")

# Example usage and testing
async def main():
    """
    Example usage of NFT Trading System
    """
    print("🎨 NFT Trading System - Phase 5 Testing")
    print("=" * 50)
    
    # Initialize marketplace
    marketplace = NFTMarketplace()
    await marketplace.initialize()
    
    # Create test collection
    print("\n🏛️ Creating Test Collection:")
    creator_address = "0x1234567890123456789012345678901234567890"
    collection_id = await marketplace.create_collection(
        creator_address=creator_address,
        name="Peper AI Art Collection",
        description="AI-generated art collection for trading enthusiasts",
        image_url="https://example.com/collection.jpg",
        category=NFTCategory.ART
    )
    
    if collection_id:
        print(f"  Collection created: {collection_id}")
        collection_details = await marketplace.get_collection_details(collection_id)
        print(f"  Name: {collection_details['name']}")
        print(f"  Category: {collection_details['metadata']['category']}")
    
    # Mint test NFTs
    print(f"\n🎨 Minting Test NFTs:")
    test_nfts = []
    
    for i in range(5):
        metadata = NFTMetadata(
            name=f"Peper AI Art #{i+1}",
            description=f"Unique AI-generated artwork #{i+1} for trading platform",
            image=f"https://example.com/nft_{i+1}.jpg"
        )
        
        # Add attributes
        metadata.add_attribute("Style", ["Abstract", "Realistic", "Surreal", "Minimalist"][i % 4])
        metadata.add_attribute("Rarity", ["Common", "Uncommon", "Rare", "Epic", "Legendary"][i])
        metadata.add_attribute("AI Model", "GPT-5 Vision")
        metadata.add_attribute("Generation", i + 1, "number")
        
        token_id = await marketplace.mint_nft(
            creator_address=creator_address,
            metadata=metadata,
            contract_address="0xabcdef1234567890abcdef1234567890abcdef12",
            blockchain="ethereum",
            collection_id=collection_id
        )
        
        if token_id:
            test_nfts.append(token_id)
            print(f"  NFT #{i+1} minted: {token_id}")
    
    # Create listings
    print(f"\n🏪 Creating Test Listings:")
    listings = []
    
    # Fixed price listing
    listing_id = await marketplace.create_listing(
        seller_address=creator_address,
        token_id=test_nfts[0],
        listing_type=ListingType.FIXED_PRICE,
        price=Decimal('1.5'),
        currency="ETH"
    )
    if listing_id:
        listings.append(listing_id)
        print(f"  Fixed price listing: {listing_id} (1.5 ETH)")
    
    # Auction listing
    listing_id = await marketplace.create_listing(
        seller_address=creator_address,
        token_id=test_nfts[1],
        listing_type=ListingType.AUCTION,
        price=Decimal('0.5'),  # Starting bid
        currency="ETH",
        auction_duration=7 * 24 * 3600,  # 7 days
        reserve_price=Decimal('1.0')
    )
    if listing_id:
        listings.append(listing_id)
        print(f"  Auction listing: {listing_id} (Starting: 0.5 ETH, Reserve: 1.0 ETH)")
    
    # Dutch auction listing
    listing_id = await marketplace.create_listing(
        seller_address=creator_address,
        token_id=test_nfts[2],
        listing_type=ListingType.DUTCH_AUCTION,
        price=Decimal('3.0'),  # Starting price
        currency="ETH",
        auction_duration=3 * 24 * 3600,  # 3 days
        reserve_price=Decimal('1.0')  # Ending price
    )
    if listing_id:
        listings.append(listing_id)
        print(f"  Dutch auction listing: {listing_id} (3.0 ETH → 1.0 ETH)")
    
    # Test bidding
    print(f"\n🔨 Testing Bidding:")
    if len(listings) >= 2:
        bidder_address = "0x2345678901234567890123456789012345678901"
        
        # Place bids on auction
        auction_listing_id = listings[1]
        bid_amounts = [Decimal('0.6'), Decimal('0.8'), Decimal('1.2')]
        
        for i, amount in enumerate(bid_amounts):
            bidder = f"0x{str(i+2).zfill(40)}"
            success = await marketplace.place_bid(bidder, auction_listing_id, amount)
            print(f"  Bid {amount} ETH by {bidder[:10]}...: {'✅' if success else '❌'}")
    
    # Test buy now
    print(f"\n💰 Testing Buy Now:")
    if listings:
        buyer_address = "0x3456789012345678901234567890123456789012"
        fixed_price_listing = listings[0]
        
        success = await marketplace.buy_now(buyer_address, fixed_price_listing)
        print(f"  Buy now transaction: {'✅' if success else '❌'}")
    
    # Search NFTs
    print(f"\n🔍 Testing NFT Search:")
    search_results = await marketplace.search_nfts(
        query="Peper",
        category=NFTCategory.ART,
        sort_by="rarity",
        limit=10
    )
    
    print(f"  Found {len(search_results)} NFTs:")
    for result in search_results[:3]:
        print(f"    {result['metadata']['name']} - {result['rarity']['tier']} - {result['pricing']['current_price']} {result['pricing']['currency']}")
    
    # Get user portfolio
    print(f"\n👤 User Portfolio:")
    portfolio = await marketplace.get_user_portfolio(creator_address)
    print(f"  NFTs owned: {portfolio['portfolio']['total_count']}")
    print(f"  Portfolio value: {portfolio['portfolio']['total_value']} ETH")
    print(f"  Active listings: {len(portfolio['activity']['active_listings'])}")
    print(f"  Active bids: {len(portfolio['activity']['active_bids'])}")
    
    # Get marketplace stats
    print(f"\n📊 Marketplace Statistics:")
    stats = await marketplace.get_marketplace_stats()
    marketplace_stats = stats['marketplace_stats']
    
    print(f"  Total NFTs: {marketplace_stats['total_nfts']}")
    print(f"  Total Collections: {marketplace_stats['total_collections']}")
    print(f"  Total Listings: {marketplace_stats['total_listings']}")
    print(f"  Total Volume: {marketplace_stats['total_volume']} ETH")
    print(f"  Total Trades: {marketplace_stats['total_trades']}")
    print(f"  Active Users: {marketplace_stats['active_users']}")
    
    print(f"\n  Trending Collections:")
    for collection in stats['trending_collections'][:3]:
        print(f"    {collection['collection_slug']}: {collection['volume_7d']} ETH (7d)")
    
    # Get NFT details
    print(f"\n🎨 NFT Details:")
    if test_nfts:
        nft_details = await marketplace.get_nft_details(test_nfts[0])
        if nft_details:
            print(f"  Name: {nft_details['metadata']['name']}")
            print(f"  Owner: {nft_details['owner_address'][:10]}...")
            print(f"  Rarity: {nft_details['rarity']['tier']} (Score: {nft_details['rarity']['score']:.2f})")
            print(f"  Transfers: {nft_details['stats']['transfer_count']}")
            print(f"  Attributes: {len(nft_details['metadata']['attributes'])}")
    
    # Cleanup
    await marketplace.cleanup()
    
    print(f"\n✅ NFT trading system testing completed!")
    print(f"🎨 NFT marketplace ready for Phase 5!")

if __name__ == "__main__":
    asyncio.run(main())