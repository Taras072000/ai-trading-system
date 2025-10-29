"""
Global Load Balancer for Phase 5
Handles multi-regional traffic distribution and failover
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import yaml
import json
import geoip2.database
import geoip2.errors

class RegionStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"

@dataclass
class RegionEndpoint:
    region: str
    country_codes: List[str]
    endpoint_url: str
    capacity: int
    current_load: int
    latency_ms: float
    status: RegionStatus
    last_health_check: float

class GlobalLoadBalancer:
    """
    Global Load Balancer for multi-regional deployment
    Supports 50+ countries with intelligent routing
    """
    
    def __init__(self, config_path: str = "config/global_phase5_config.yaml"):
        self.config = self._load_config(config_path)
        self.regions: Dict[str, RegionEndpoint] = {}
        self.geoip_db = None
        self.health_check_interval = 30  # seconds
        self.running = False
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize regions
        self._initialize_regions()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load global configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    def _initialize_regions(self):
        """Initialize regional endpoints"""
        # North America
        self.regions["us-east"] = RegionEndpoint(
            region="us-east",
            country_codes=["US", "CA"],
            endpoint_url="https://us-east.peper-trading.com",
            capacity=100000,
            current_load=0,
            latency_ms=5.0,
            status=RegionStatus.HEALTHY,
            last_health_check=time.time()
        )
        
        self.regions["us-west"] = RegionEndpoint(
            region="us-west",
            country_codes=["US", "CA", "MX"],
            endpoint_url="https://us-west.peper-trading.com",
            capacity=80000,
            current_load=0,
            latency_ms=4.0,
            status=RegionStatus.HEALTHY,
            last_health_check=time.time()
        )
        
        # Europe
        self.regions["eu-west"] = RegionEndpoint(
            region="eu-west",
            country_codes=["GB", "IE", "FR", "ES", "PT"],
            endpoint_url="https://eu-west.peper-trading.com",
            capacity=120000,
            current_load=0,
            latency_ms=3.0,
            status=RegionStatus.HEALTHY,
            last_health_check=time.time()
        )
        
        self.regions["eu-central"] = RegionEndpoint(
            region="eu-central",
            country_codes=["DE", "AT", "CH", "NL", "BE", "LU"],
            endpoint_url="https://eu-central.peper-trading.com",
            capacity=100000,
            current_load=0,
            latency_ms=2.5,
            status=RegionStatus.HEALTHY,
            last_health_check=time.time()
        )
        
        self.regions["eu-north"] = RegionEndpoint(
            region="eu-north",
            country_codes=["SE", "NO", "DK", "FI", "IS"],
            endpoint_url="https://eu-north.peper-trading.com",
            capacity=60000,
            current_load=0,
            latency_ms=4.0,
            status=RegionStatus.HEALTHY,
            last_health_check=time.time()
        )
        
        # Asia Pacific
        self.regions["asia-east"] = RegionEndpoint(
            region="asia-east",
            country_codes=["JP", "KR", "TW"],
            endpoint_url="https://asia-east.peper-trading.com",
            capacity=150000,
            current_load=0,
            latency_ms=2.0,
            status=RegionStatus.HEALTHY,
            last_health_check=time.time()
        )
        
        self.regions["asia-southeast"] = RegionEndpoint(
            region="asia-southeast",
            country_codes=["SG", "MY", "TH", "ID", "PH", "VN"],
            endpoint_url="https://asia-southeast.peper-trading.com",
            capacity=100000,
            current_load=0,
            latency_ms=3.5,
            status=RegionStatus.HEALTHY,
            last_health_check=time.time()
        )
        
        self.regions["asia-south"] = RegionEndpoint(
            region="asia-south",
            country_codes=["IN", "BD", "LK", "PK"],
            endpoint_url="https://asia-south.peper-trading.com",
            capacity=80000,
            current_load=0,
            latency_ms=6.0,
            status=RegionStatus.HEALTHY,
            last_health_check=time.time()
        )
        
        # China (Special handling)
        self.regions["china"] = RegionEndpoint(
            region="china",
            country_codes=["CN"],
            endpoint_url="https://china.peper-trading.com",
            capacity=200000,
            current_load=0,
            latency_ms=1.5,
            status=RegionStatus.HEALTHY,
            last_health_check=time.time()
        )
        
        # Australia & Oceania
        self.regions["oceania"] = RegionEndpoint(
            region="oceania",
            country_codes=["AU", "NZ", "FJ", "PG"],
            endpoint_url="https://oceania.peper-trading.com",
            capacity=40000,
            current_load=0,
            latency_ms=8.0,
            status=RegionStatus.HEALTHY,
            last_health_check=time.time()
        )
        
        # Middle East & Africa
        self.regions["middle-east"] = RegionEndpoint(
            region="middle-east",
            country_codes=["AE", "SA", "QA", "KW", "BH", "OM", "IL", "TR"],
            endpoint_url="https://middle-east.peper-trading.com",
            capacity=60000,
            current_load=0,
            latency_ms=7.0,
            status=RegionStatus.HEALTHY,
            last_health_check=time.time()
        )
        
        self.regions["africa"] = RegionEndpoint(
            region="africa",
            country_codes=["ZA", "NG", "EG", "KE", "GH", "MA", "TN"],
            endpoint_url="https://africa.peper-trading.com",
            capacity=30000,
            current_load=0,
            latency_ms=12.0,
            status=RegionStatus.HEALTHY,
            last_health_check=time.time()
        )
        
        # South America
        self.regions["south-america"] = RegionEndpoint(
            region="south-america",
            country_codes=["BR", "AR", "CL", "CO", "PE", "UY", "VE"],
            endpoint_url="https://south-america.peper-trading.com",
            capacity=50000,
            current_load=0,
            latency_ms=10.0,
            status=RegionStatus.HEALTHY,
            last_health_check=time.time()
        )
        
        self.logger.info(f"Initialized {len(self.regions)} regional endpoints")
    
    def get_optimal_region(self, client_ip: str, user_country: Optional[str] = None) -> Optional[RegionEndpoint]:
        """
        Get optimal region for client based on geolocation and load
        """
        try:
            # Determine country from IP if not provided
            if not user_country and self.geoip_db:
                try:
                    response = self.geoip_db.country(client_ip)
                    user_country = response.country.iso_code
                except geoip2.errors.AddressNotFoundError:
                    user_country = "US"  # Default fallback
            
            # Find regions that serve this country
            candidate_regions = []
            for region in self.regions.values():
                if (user_country in region.country_codes and 
                    region.status == RegionStatus.HEALTHY and
                    region.current_load < region.capacity * 0.9):  # 90% capacity limit
                    candidate_regions.append(region)
            
            if not candidate_regions:
                # Fallback to any healthy region with capacity
                candidate_regions = [
                    region for region in self.regions.values()
                    if (region.status == RegionStatus.HEALTHY and
                        region.current_load < region.capacity * 0.9)
                ]
            
            if not candidate_regions:
                self.logger.warning("No healthy regions available!")
                return None
            
            # Select region with lowest load and latency score
            best_region = min(candidate_regions, key=lambda r: (
                r.current_load / r.capacity * 0.7 +  # Load factor (70% weight)
                r.latency_ms / 100 * 0.3  # Latency factor (30% weight)
            ))
            
            return best_region
            
        except Exception as e:
            self.logger.error(f"Error selecting optimal region: {e}")
            return list(self.regions.values())[0] if self.regions else None
    
    async def health_check_region(self, region: RegionEndpoint) -> bool:
        """
        Perform health check on a region
        """
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(f"{region.endpoint_url}/health") as response:
                    if response.status == 200:
                        latency = (time.time() - start_time) * 1000
                        region.latency_ms = latency
                        region.status = RegionStatus.HEALTHY
                        region.last_health_check = time.time()
                        
                        # Update load from response headers
                        if 'X-Current-Load' in response.headers:
                            region.current_load = int(response.headers['X-Current-Load'])
                        
                        return True
                    else:
                        region.status = RegionStatus.DEGRADED
                        return False
                        
        except Exception as e:
            self.logger.warning(f"Health check failed for {region.region}: {e}")
            region.status = RegionStatus.OFFLINE
            return False
    
    async def health_check_all_regions(self):
        """
        Perform health checks on all regions
        """
        tasks = [self.health_check_region(region) for region in self.regions.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        healthy_count = sum(1 for result in results if result is True)
        self.logger.info(f"Health check completed: {healthy_count}/{len(self.regions)} regions healthy")
    
    async def start_health_monitoring(self):
        """
        Start continuous health monitoring
        """
        self.running = True
        self.logger.info("Starting global health monitoring...")
        
        while self.running:
            await self.health_check_all_regions()
            await asyncio.sleep(self.health_check_interval)
    
    def stop_health_monitoring(self):
        """
        Stop health monitoring
        """
        self.running = False
        self.logger.info("Stopping global health monitoring...")
    
    def get_region_status(self) -> Dict:
        """
        Get status of all regions
        """
        return {
            region_name: {
                "status": region.status.value,
                "load_percentage": (region.current_load / region.capacity) * 100,
                "latency_ms": region.latency_ms,
                "last_health_check": region.last_health_check,
                "capacity": region.capacity,
                "current_load": region.current_load
            }
            for region_name, region in self.regions.items()
        }
    
    def route_request(self, client_ip: str, user_country: Optional[str] = None) -> Dict:
        """
        Route request to optimal region
        """
        optimal_region = self.get_optimal_region(client_ip, user_country)
        
        if not optimal_region:
            return {
                "success": False,
                "error": "No healthy regions available",
                "fallback_url": "https://global.peper-trading.com"
            }
        
        # Increment load
        optimal_region.current_load += 1
        
        return {
            "success": True,
            "region": optimal_region.region,
            "endpoint_url": optimal_region.endpoint_url,
            "estimated_latency_ms": optimal_region.latency_ms,
            "load_percentage": (optimal_region.current_load / optimal_region.capacity) * 100
        }
    
    def release_connection(self, region_name: str):
        """
        Release connection from region (decrement load)
        """
        if region_name in self.regions:
            self.regions[region_name].current_load = max(0, self.regions[region_name].current_load - 1)

# Global Load Balancer API
class GlobalLoadBalancerAPI:
    """
    REST API for Global Load Balancer
    """
    
    def __init__(self, load_balancer: GlobalLoadBalancer):
        self.load_balancer = load_balancer
    
    async def handle_route_request(self, request_data: Dict) -> Dict:
        """
        Handle routing request
        """
        client_ip = request_data.get("client_ip", "127.0.0.1")
        user_country = request_data.get("user_country")
        
        return self.load_balancer.route_request(client_ip, user_country)
    
    async def handle_status_request(self) -> Dict:
        """
        Handle status request
        """
        return {
            "global_status": "operational",
            "regions": self.load_balancer.get_region_status(),
            "total_regions": len(self.load_balancer.regions),
            "healthy_regions": len([
                r for r in self.load_balancer.regions.values() 
                if r.status == RegionStatus.HEALTHY
            ])
        }

# Example usage
async def main():
    """
    Example usage of Global Load Balancer
    """
    # Initialize load balancer
    load_balancer = GlobalLoadBalancer()
    
    # Start health monitoring
    health_task = asyncio.create_task(load_balancer.start_health_monitoring())
    
    # Simulate some routing requests
    test_requests = [
        {"client_ip": "203.0.113.1", "user_country": "JP"},  # Japan
        {"client_ip": "198.51.100.1", "user_country": "US"},  # USA
        {"client_ip": "192.0.2.1", "user_country": "DE"},     # Germany
        {"client_ip": "203.0.113.50", "user_country": "SG"},  # Singapore
        {"client_ip": "198.51.100.50", "user_country": "BR"}  # Brazil
    ]
    
    print("🌍 Global Load Balancer - Phase 5 Testing")
    print("=" * 50)
    
    for i, request in enumerate(test_requests, 1):
        result = load_balancer.route_request(
            request["client_ip"], 
            request["user_country"]
        )
        
        print(f"\n{i}. Request from {request['user_country']}:")
        print(f"   Routed to: {result.get('region', 'N/A')}")
        print(f"   Endpoint: {result.get('endpoint_url', 'N/A')}")
        print(f"   Latency: {result.get('estimated_latency_ms', 'N/A')}ms")
        print(f"   Load: {result.get('load_percentage', 'N/A'):.1f}%")
    
    # Show global status
    print(f"\n🔍 Global Status:")
    status = load_balancer.get_region_status()
    for region_name, region_status in status.items():
        print(f"   {region_name}: {region_status['status']} "
              f"({region_status['load_percentage']:.1f}% load, "
              f"{region_status['latency_ms']:.1f}ms)")
    
    # Stop monitoring
    await asyncio.sleep(2)
    load_balancer.stop_health_monitoring()
    health_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())