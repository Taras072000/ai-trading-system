"""
Venture Fund for Phase 5
Fintech investment fund with AI-powered deal sourcing and portfolio management
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timezone, timedelta
import uuid
import numpy as np
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

class InvestmentStage(Enum):
    PRE_SEED = "pre_seed"
    SEED = "seed"
    SERIES_A = "series_a"
    SERIES_B = "series_b"
    SERIES_C = "series_c"
    GROWTH = "growth"
    LATE_STAGE = "late_stage"

class InvestmentStatus(Enum):
    SOURCED = "sourced"
    SCREENING = "screening"
    DUE_DILIGENCE = "due_diligence"
    COMMITTEE_REVIEW = "committee_review"
    TERM_SHEET = "term_sheet"
    LEGAL_REVIEW = "legal_review"
    CLOSED = "closed"
    REJECTED = "rejected"
    EXITED = "exited"

class SectorFocus(Enum):
    TRADING_TECH = "trading_tech"
    BLOCKCHAIN = "blockchain"
    DEFI = "defi"
    PAYMENTS = "payments"
    LENDING = "lending"
    INSURTECH = "insurtech"
    REGTECH = "regtech"
    WEALTHTECH = "wealthtech"
    NEOBANKING = "neobanking"
    AI_FINTECH = "ai_fintech"

class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class ExitType(Enum):
    IPO = "ipo"
    ACQUISITION = "acquisition"
    MERGER = "merger"
    SECONDARY_SALE = "secondary_sale"
    BUYBACK = "buyback"
    WRITE_OFF = "write_off"

@dataclass
class Startup:
    """Startup company information"""
    startup_id: str
    name: str
    description: str
    
    # Company details
    founded_date: datetime
    country: str
    city: str
    website: str = ""
    
    # Business model
    sector: SectorFocus
    business_model: str = ""
    target_market: str = ""
    
    # Team
    founders: List[Dict[str, Any]] = field(default_factory=list)
    team_size: int = 0
    key_hires: List[Dict[str, Any]] = field(default_factory=list)
    
    # Product
    product_stage: str = "concept"  # concept, mvp, beta, launched, scaling
    technology_stack: List[str] = field(default_factory=list)
    competitive_advantages: List[str] = field(default_factory=list)
    
    # Financials
    current_revenue: float = 0.0
    revenue_growth_rate: float = 0.0
    burn_rate: float = 0.0
    runway_months: int = 0
    
    # Funding history
    total_funding_raised: float = 0.0
    last_valuation: float = 0.0
    previous_investors: List[str] = field(default_factory=list)
    
    # Metrics
    user_count: int = 0
    user_growth_rate: float = 0.0
    retention_rate: float = 0.0
    
    # AI scoring
    ai_score: float = 0.0
    ai_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.startup_id:
            self.startup_id = str(uuid.uuid4())

@dataclass
class Investment:
    """Investment opportunity or deal"""
    investment_id: str
    startup_id: str
    
    # Investment details
    stage: InvestmentStage
    investment_amount: float
    valuation: float
    equity_percentage: float
    
    # Terms
    liquidation_preference: str = "1x non-participating"
    anti_dilution: str = "weighted_average"
    board_seats: int = 0
    protective_provisions: List[str] = field(default_factory=list)
    
    # Status and timeline
    status: InvestmentStatus = InvestmentStatus.SOURCED
    source: str = ""  # How we found this deal
    
    # Due diligence
    dd_checklist: Dict[str, bool] = field(default_factory=dict)
    dd_notes: str = ""
    dd_score: float = 0.0
    
    # Risk assessment
    risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    
    # Committee decision
    committee_recommendation: str = ""
    committee_notes: str = ""
    
    # Legal and closing
    term_sheet_signed: bool = False
    legal_docs_complete: bool = False
    closing_date: Optional[datetime] = None
    
    # Performance tracking (post-investment)
    current_valuation: float = 0.0
    unrealized_return: float = 0.0
    
    # Timestamps
    sourced_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.investment_id:
            self.investment_id = str(uuid.uuid4())

@dataclass
class PortfolioCompany:
    """Portfolio company tracking"""
    portfolio_id: str
    startup_id: str
    investment_id: str
    
    # Investment details
    investment_date: datetime
    investment_amount: float
    equity_owned: float
    current_valuation: float
    
    # Performance metrics
    revenue_growth: List[Dict[str, Any]] = field(default_factory=list)
    user_metrics: List[Dict[str, Any]] = field(default_factory=list)
    key_milestones: List[Dict[str, Any]] = field(default_factory=list)
    
    # Board and governance
    board_member: str = ""
    board_meeting_frequency: str = "quarterly"
    last_board_meeting: Optional[datetime] = None
    
    # Support provided
    support_areas: List[str] = field(default_factory=list)
    introductions_made: List[Dict[str, Any]] = field(default_factory=list)
    
    # Exit tracking
    exit_opportunities: List[Dict[str, Any]] = field(default_factory=list)
    exit_valuation_estimate: float = 0.0
    
    # Risk monitoring
    risk_alerts: List[Dict[str, Any]] = field(default_factory=list)
    health_score: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.portfolio_id:
            self.portfolio_id = str(uuid.uuid4())

@dataclass
class ExitEvent:
    """Investment exit event"""
    exit_id: str
    portfolio_id: str
    
    # Exit details
    exit_type: ExitType
    exit_date: datetime
    exit_valuation: float
    proceeds: float
    
    # Returns
    multiple_of_money: float = 0.0
    irr: float = 0.0
    holding_period_years: float = 0.0
    
    # Exit details
    acquirer: str = ""
    exit_notes: str = ""
    
    # Distribution
    distributed_to_lps: float = 0.0
    carried_interest: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if not self.exit_id:
            self.exit_id = str(uuid.uuid4())

class AIStartupAnalyzer:
    """AI-powered startup analysis and scoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Scoring weights
        self.scoring_weights = {
            'team': 0.25,
            'market': 0.20,
            'product': 0.20,
            'traction': 0.15,
            'financials': 0.10,
            'technology': 0.10
        }
        
        # Sector multipliers
        self.sector_multipliers = {
            SectorFocus.AI_FINTECH: 1.2,
            SectorFocus.TRADING_TECH: 1.15,
            SectorFocus.BLOCKCHAIN: 1.1,
            SectorFocus.DEFI: 1.1,
            SectorFocus.PAYMENTS: 1.0,
            SectorFocus.WEALTHTECH: 1.0,
            SectorFocus.REGTECH: 0.95,
            SectorFocus.INSURTECH: 0.9,
            SectorFocus.LENDING: 0.85,
            SectorFocus.NEOBANKING: 0.8
        }
    
    async def analyze_startup(self, startup: Startup) -> Dict[str, Any]:
        """Comprehensive AI analysis of startup"""
        try:
            analysis = {}
            
            # Team analysis
            team_score = self._analyze_team(startup)
            analysis['team'] = {
                'score': team_score,
                'strengths': self._get_team_strengths(startup),
                'concerns': self._get_team_concerns(startup)
            }
            
            # Market analysis
            market_score = self._analyze_market(startup)
            analysis['market'] = {
                'score': market_score,
                'size_estimate': self._estimate_market_size(startup),
                'competition_level': self._assess_competition(startup)
            }
            
            # Product analysis
            product_score = self._analyze_product(startup)
            analysis['product'] = {
                'score': product_score,
                'differentiation': self._assess_differentiation(startup),
                'scalability': self._assess_scalability(startup)
            }
            
            # Traction analysis
            traction_score = self._analyze_traction(startup)
            analysis['traction'] = {
                'score': traction_score,
                'growth_trajectory': self._assess_growth_trajectory(startup),
                'user_engagement': self._assess_user_engagement(startup)
            }
            
            # Financial analysis
            financial_score = self._analyze_financials(startup)
            analysis['financials'] = {
                'score': financial_score,
                'unit_economics': self._analyze_unit_economics(startup),
                'capital_efficiency': self._assess_capital_efficiency(startup)
            }
            
            # Technology analysis
            tech_score = self._analyze_technology(startup)
            analysis['technology'] = {
                'score': tech_score,
                'innovation_level': self._assess_innovation(startup),
                'technical_risk': self._assess_technical_risk(startup)
            }
            
            # Calculate overall score
            overall_score = (
                team_score * self.scoring_weights['team'] +
                market_score * self.scoring_weights['market'] +
                product_score * self.scoring_weights['product'] +
                traction_score * self.scoring_weights['traction'] +
                financial_score * self.scoring_weights['financials'] +
                tech_score * self.scoring_weights['technology']
            )
            
            # Apply sector multiplier
            sector_multiplier = self.sector_multipliers.get(startup.sector, 1.0)
            overall_score *= sector_multiplier
            
            # Ensure score is between 0 and 100
            overall_score = max(0, min(100, overall_score))
            
            analysis['overall'] = {
                'score': overall_score,
                'recommendation': self._get_recommendation(overall_score),
                'key_risks': self._identify_key_risks(startup),
                'investment_thesis': self._generate_investment_thesis(startup, analysis)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Startup analysis failed: {e}")
            return {}
    
    def _analyze_team(self, startup: Startup) -> float:
        """Analyze team quality"""
        score = 50.0  # Base score
        
        # Founder experience
        for founder in startup.founders:
            if founder.get('previous_exits', 0) > 0:
                score += 15
            if founder.get('domain_experience', 0) > 5:
                score += 10
            if founder.get('technical_background', False):
                score += 5
        
        # Team size appropriateness
        if startup.team_size >= 5:
            score += 10
        elif startup.team_size >= 3:
            score += 5
        
        return min(100, score)
    
    def _analyze_market(self, startup: Startup) -> float:
        """Analyze market opportunity"""
        score = 50.0  # Base score
        
        # Sector scoring
        high_growth_sectors = [SectorFocus.AI_FINTECH, SectorFocus.DEFI, SectorFocus.TRADING_TECH]
        if startup.sector in high_growth_sectors:
            score += 20
        
        # Market timing
        if startup.sector in [SectorFocus.AI_FINTECH, SectorFocus.BLOCKCHAIN]:
            score += 15  # Hot sectors
        
        return min(100, score)
    
    def _analyze_product(self, startup: Startup) -> float:
        """Analyze product quality"""
        score = 30.0  # Base score
        
        # Product stage
        stage_scores = {
            'concept': 10,
            'mvp': 30,
            'beta': 50,
            'launched': 70,
            'scaling': 90
        }
        score += stage_scores.get(startup.product_stage, 0)
        
        # Competitive advantages
        score += len(startup.competitive_advantages) * 5
        
        return min(100, score)
    
    def _analyze_traction(self, startup: Startup) -> float:
        """Analyze traction metrics"""
        score = 20.0  # Base score
        
        # User growth
        if startup.user_growth_rate > 20:
            score += 30
        elif startup.user_growth_rate > 10:
            score += 20
        elif startup.user_growth_rate > 5:
            score += 10
        
        # User base size
        if startup.user_count > 100000:
            score += 25
        elif startup.user_count > 10000:
            score += 15
        elif startup.user_count > 1000:
            score += 10
        
        # Retention
        if startup.retention_rate > 80:
            score += 15
        elif startup.retention_rate > 60:
            score += 10
        
        return min(100, score)
    
    def _analyze_financials(self, startup: Startup) -> float:
        """Analyze financial metrics"""
        score = 40.0  # Base score
        
        # Revenue
        if startup.current_revenue > 1000000:
            score += 30
        elif startup.current_revenue > 100000:
            score += 20
        elif startup.current_revenue > 10000:
            score += 10
        
        # Revenue growth
        if startup.revenue_growth_rate > 100:
            score += 20
        elif startup.revenue_growth_rate > 50:
            score += 15
        elif startup.revenue_growth_rate > 20:
            score += 10
        
        # Runway
        if startup.runway_months > 18:
            score += 10
        elif startup.runway_months > 12:
            score += 5
        
        return min(100, score)
    
    def _analyze_technology(self, startup: Startup) -> float:
        """Analyze technology stack"""
        score = 50.0  # Base score
        
        # Modern tech stack
        modern_techs = ['ai', 'ml', 'blockchain', 'cloud', 'microservices']
        for tech in startup.technology_stack:
            if any(modern in tech.lower() for modern in modern_techs):
                score += 10
        
        return min(100, score)
    
    def _get_team_strengths(self, startup: Startup) -> List[str]:
        """Identify team strengths"""
        strengths = []
        
        if len(startup.founders) > 1:
            strengths.append("Strong founding team")
        
        for founder in startup.founders:
            if founder.get('previous_exits', 0) > 0:
                strengths.append("Experienced founders with exits")
                break
        
        if startup.team_size >= 10:
            strengths.append("Well-staffed team")
        
        return strengths
    
    def _get_team_concerns(self, startup: Startup) -> List[str]:
        """Identify team concerns"""
        concerns = []
        
        if len(startup.founders) == 1:
            concerns.append("Single founder risk")
        
        if startup.team_size < 3:
            concerns.append("Very small team")
        
        return concerns
    
    def _estimate_market_size(self, startup: Startup) -> str:
        """Estimate market size"""
        market_sizes = {
            SectorFocus.TRADING_TECH: "$50B+",
            SectorFocus.AI_FINTECH: "$100B+",
            SectorFocus.PAYMENTS: "$200B+",
            SectorFocus.BLOCKCHAIN: "$30B+",
            SectorFocus.DEFI: "$20B+",
            SectorFocus.WEALTHTECH: "$40B+",
            SectorFocus.LENDING: "$150B+",
            SectorFocus.INSURTECH: "$80B+",
            SectorFocus.REGTECH: "$25B+",
            SectorFocus.NEOBANKING: "$60B+"
        }
        
        return market_sizes.get(startup.sector, "$10B+")
    
    def _assess_competition(self, startup: Startup) -> str:
        """Assess competition level"""
        competitive_sectors = [SectorFocus.PAYMENTS, SectorFocus.LENDING, SectorFocus.NEOBANKING]
        
        if startup.sector in competitive_sectors:
            return "High"
        elif startup.sector in [SectorFocus.AI_FINTECH, SectorFocus.TRADING_TECH]:
            return "Medium"
        else:
            return "Low"
    
    def _assess_differentiation(self, startup: Startup) -> str:
        """Assess product differentiation"""
        if len(startup.competitive_advantages) >= 3:
            return "Strong"
        elif len(startup.competitive_advantages) >= 1:
            return "Moderate"
        else:
            return "Weak"
    
    def _assess_scalability(self, startup: Startup) -> str:
        """Assess product scalability"""
        scalable_models = ['saas', 'platform', 'marketplace', 'api']
        
        if any(model in startup.business_model.lower() for model in scalable_models):
            return "High"
        else:
            return "Medium"
    
    def _assess_growth_trajectory(self, startup: Startup) -> str:
        """Assess growth trajectory"""
        if startup.user_growth_rate > 20:
            return "Accelerating"
        elif startup.user_growth_rate > 10:
            return "Strong"
        elif startup.user_growth_rate > 0:
            return "Steady"
        else:
            return "Declining"
    
    def _assess_user_engagement(self, startup: Startup) -> str:
        """Assess user engagement"""
        if startup.retention_rate > 80:
            return "Excellent"
        elif startup.retention_rate > 60:
            return "Good"
        elif startup.retention_rate > 40:
            return "Average"
        else:
            return "Poor"
    
    def _analyze_unit_economics(self, startup: Startup) -> str:
        """Analyze unit economics"""
        # Simplified analysis
        if startup.current_revenue > startup.burn_rate * 12:
            return "Positive"
        elif startup.revenue_growth_rate > 50:
            return "Improving"
        else:
            return "Needs work"
    
    def _assess_capital_efficiency(self, startup: Startup) -> str:
        """Assess capital efficiency"""
        if startup.total_funding_raised > 0:
            revenue_per_dollar = startup.current_revenue / startup.total_funding_raised
            if revenue_per_dollar > 0.5:
                return "Excellent"
            elif revenue_per_dollar > 0.2:
                return "Good"
            else:
                return "Poor"
        return "Unknown"
    
    def _assess_innovation(self, startup: Startup) -> str:
        """Assess innovation level"""
        innovative_techs = ['ai', 'ml', 'blockchain', 'quantum']
        
        if any(tech in ' '.join(startup.technology_stack).lower() for tech in innovative_techs):
            return "High"
        else:
            return "Medium"
    
    def _assess_technical_risk(self, startup: Startup) -> str:
        """Assess technical risk"""
        high_risk_techs = ['quantum', 'experimental']
        
        if any(tech in ' '.join(startup.technology_stack).lower() for tech in high_risk_techs):
            return "High"
        else:
            return "Low"
    
    def _get_recommendation(self, score: float) -> str:
        """Get investment recommendation"""
        if score >= 80:
            return "Strong Buy"
        elif score >= 70:
            return "Buy"
        elif score >= 60:
            return "Consider"
        elif score >= 50:
            return "Pass"
        else:
            return "Strong Pass"
    
    def _identify_key_risks(self, startup: Startup) -> List[str]:
        """Identify key investment risks"""
        risks = []
        
        if len(startup.founders) == 1:
            risks.append("Single founder dependency")
        
        if startup.runway_months < 12:
            risks.append("Short runway")
        
        if startup.user_growth_rate < 5:
            risks.append("Slow user growth")
        
        if startup.retention_rate < 50:
            risks.append("Poor user retention")
        
        return risks
    
    def _generate_investment_thesis(self, startup: Startup, analysis: Dict[str, Any]) -> str:
        """Generate investment thesis"""
        thesis_parts = []
        
        # Market opportunity
        market_size = analysis['market']['size_estimate']
        thesis_parts.append(f"Large market opportunity ({market_size})")
        
        # Team strength
        if analysis['team']['score'] > 70:
            thesis_parts.append("Strong founding team")
        
        # Traction
        if analysis['traction']['score'] > 70:
            thesis_parts.append("Proven traction and growth")
        
        # Technology
        if analysis['technology']['score'] > 70:
            thesis_parts.append("Innovative technology platform")
        
        return "; ".join(thesis_parts)

class VentureFund:
    """Venture fund for fintech investments"""
    
    def __init__(self, fund_size: float = 100000000.0):  # $100M fund
        self.logger = logging.getLogger(__name__)
        
        # Fund details
        self.fund_size = fund_size
        self.committed_capital = 0.0
        self.deployed_capital = 0.0
        self.available_capital = fund_size
        
        # Components
        self.ai_analyzer = AIStartupAnalyzer()
        
        # Data storage
        self.startups: Dict[str, Startup] = {}
        self.investments: Dict[str, Investment] = {}
        self.portfolio_companies: Dict[str, PortfolioCompany] = {}
        self.exit_events: Dict[str, ExitEvent] = {}
        
        # Deal flow
        self.deal_pipeline: List[str] = []  # investment_ids
        
        # Fund metrics
        self.fund_metrics: Dict[str, Any] = {}
        
        # Initialize sample data
        asyncio.create_task(self._initialize_sample_data())
        
        # Start background tasks
        asyncio.create_task(self._portfolio_monitoring())
        asyncio.create_task(self._metrics_updater())
    
    async def _initialize_sample_data(self):
        """Initialize with sample startups and investments"""
        try:
            # Sample startups
            sample_startups = [
                {
                    'name': 'TradingAI Pro',
                    'description': 'AI-powered trading platform for retail investors',
                    'founded_date': datetime(2022, 3, 15, tzinfo=timezone.utc),
                    'country': 'USA',
                    'city': 'San Francisco',
                    'sector': SectorFocus.AI_FINTECH,
                    'product_stage': 'launched',
                    'current_revenue': 250000.0,
                    'revenue_growth_rate': 85.0,
                    'user_count': 15000,
                    'user_growth_rate': 25.0,
                    'retention_rate': 78.0,
                    'team_size': 12,
                    'technology_stack': ['Python', 'TensorFlow', 'React', 'AWS'],
                    'competitive_advantages': ['Proprietary AI models', 'Real-time execution', 'Low fees']
                },
                {
                    'name': 'DeFi Bridge',
                    'description': 'Cross-chain DeFi protocol for seamless asset transfers',
                    'founded_date': datetime(2021, 8, 20, tzinfo=timezone.utc),
                    'country': 'Singapore',
                    'city': 'Singapore',
                    'sector': SectorFocus.DEFI,
                    'product_stage': 'scaling',
                    'current_revenue': 500000.0,
                    'revenue_growth_rate': 120.0,
                    'user_count': 8000,
                    'user_growth_rate': 35.0,
                    'retention_rate': 85.0,
                    'team_size': 18,
                    'technology_stack': ['Solidity', 'Rust', 'Node.js', 'Ethereum'],
                    'competitive_advantages': ['Multi-chain support', 'Low gas fees', 'Security audited']
                },
                {
                    'name': 'PayFlow',
                    'description': 'B2B payment automation platform',
                    'founded_date': datetime(2023, 1, 10, tzinfo=timezone.utc),
                    'country': 'UK',
                    'city': 'London',
                    'sector': SectorFocus.PAYMENTS,
                    'product_stage': 'beta',
                    'current_revenue': 50000.0,
                    'revenue_growth_rate': 200.0,
                    'user_count': 500,
                    'user_growth_rate': 45.0,
                    'retention_rate': 92.0,
                    'team_size': 8,
                    'technology_stack': ['Go', 'PostgreSQL', 'React', 'GCP'],
                    'competitive_advantages': ['API-first design', 'Real-time reconciliation']
                }
            ]
            
            for startup_data in sample_startups:
                startup = Startup(startup_id="", **startup_data)
                
                # Add sample founders
                startup.founders = [
                    {
                        'name': f'Founder {i+1}',
                        'role': 'CEO' if i == 0 else 'CTO',
                        'previous_exits': np.random.randint(0, 3),
                        'domain_experience': np.random.randint(3, 15),
                        'technical_background': i > 0
                    }
                    for i in range(np.random.randint(1, 4))
                ]
                
                # Analyze startup
                analysis = await self.ai_analyzer.analyze_startup(startup)
                startup.ai_score = analysis.get('overall', {}).get('score', 0)
                startup.ai_analysis = analysis
                
                self.startups[startup.startup_id] = startup
            
            self.logger.info(f"Initialized {len(self.startups)} sample startups")
            
        except Exception as e:
            self.logger.error(f"Sample data initialization failed: {e}")
    
    async def source_startup(self, startup_data: Dict[str, Any]) -> Optional[str]:
        """Source new startup for evaluation"""
        try:
            startup = Startup(**startup_data)
            
            # AI analysis
            analysis = await self.ai_analyzer.analyze_startup(startup)
            startup.ai_score = analysis.get('overall', {}).get('score', 0)
            startup.ai_analysis = analysis
            
            self.startups[startup.startup_id] = startup
            
            self.logger.info(f"Startup sourced: {startup.name} (Score: {startup.ai_score:.1f})")
            return startup.startup_id
            
        except Exception as e:
            self.logger.error(f"Startup sourcing failed: {e}")
            return None
    
    async def create_investment_opportunity(self, startup_id: str, 
                                          investment_data: Dict[str, Any]) -> Optional[str]:
        """Create investment opportunity"""
        try:
            if startup_id not in self.startups:
                return None
            
            investment = Investment(
                investment_id="",
                startup_id=startup_id,
                **investment_data
            )
            
            # Initialize due diligence checklist
            investment.dd_checklist = {
                'financial_review': False,
                'legal_review': False,
                'technical_review': False,
                'market_analysis': False,
                'team_interviews': False,
                'reference_checks': False,
                'competitive_analysis': False,
                'customer_interviews': False
            }
            
            self.investments[investment.investment_id] = investment
            self.deal_pipeline.append(investment.investment_id)
            
            self.logger.info(f"Investment opportunity created: {investment.investment_id}")
            return investment.investment_id
            
        except Exception as e:
            self.logger.error(f"Investment opportunity creation failed: {e}")
            return None
    
    async def conduct_due_diligence(self, investment_id: str, 
                                  checklist_updates: Dict[str, bool]) -> bool:
        """Update due diligence checklist"""
        try:
            if investment_id not in self.investments:
                return False
            
            investment = self.investments[investment_id]
            
            # Update checklist
            investment.dd_checklist.update(checklist_updates)
            
            # Calculate DD score
            completed_items = sum(investment.dd_checklist.values())
            total_items = len(investment.dd_checklist)
            investment.dd_score = (completed_items / total_items) * 100
            
            # Update status if DD is complete
            if all(investment.dd_checklist.values()):
                investment.status = InvestmentStatus.COMMITTEE_REVIEW
            
            investment.updated_at = datetime.now(timezone.utc)
            
            self.logger.info(f"Due diligence updated: {investment_id} ({investment.dd_score:.1f}% complete)")
            return True
            
        except Exception as e:
            self.logger.error(f"Due diligence update failed: {e}")
            return False
    
    async def make_investment_decision(self, investment_id: str, 
                                     decision: str, notes: str = "") -> bool:
        """Make investment committee decision"""
        try:
            if investment_id not in self.investments:
                return False
            
            investment = self.investments[investment_id]
            
            investment.committee_recommendation = decision
            investment.committee_notes = notes
            
            if decision.lower() == "approve":
                investment.status = InvestmentStatus.TERM_SHEET
            else:
                investment.status = InvestmentStatus.REJECTED
            
            investment.updated_at = datetime.now(timezone.utc)
            
            self.logger.info(f"Investment decision: {investment_id} - {decision}")
            return True
            
        except Exception as e:
            self.logger.error(f"Investment decision failed: {e}")
            return False
    
    async def close_investment(self, investment_id: str) -> Optional[str]:
        """Close investment and add to portfolio"""
        try:
            if investment_id not in self.investments:
                return None
            
            investment = self.investments[investment_id]
            
            if investment.status != InvestmentStatus.TERM_SHEET:
                return None
            
            # Update investment status
            investment.status = InvestmentStatus.CLOSED
            investment.closing_date = datetime.now(timezone.utc)
            investment.term_sheet_signed = True
            investment.legal_docs_complete = True
            
            # Update fund metrics
            self.deployed_capital += investment.investment_amount
            self.available_capital -= investment.investment_amount
            
            # Create portfolio company
            portfolio_company = PortfolioCompany(
                portfolio_id="",
                startup_id=investment.startup_id,
                investment_id=investment_id,
                investment_date=investment.closing_date,
                investment_amount=investment.investment_amount,
                equity_owned=investment.equity_percentage,
                current_valuation=investment.valuation
            )
            
            self.portfolio_companies[portfolio_company.portfolio_id] = portfolio_company
            
            # Remove from deal pipeline
            if investment_id in self.deal_pipeline:
                self.deal_pipeline.remove(investment_id)
            
            self.logger.info(f"Investment closed: {investment_id} (${investment.investment_amount:,.0f})")
            return portfolio_company.portfolio_id
            
        except Exception as e:
            self.logger.error(f"Investment closing failed: {e}")
            return None
    
    async def update_portfolio_company(self, portfolio_id: str, 
                                     updates: Dict[str, Any]) -> bool:
        """Update portfolio company metrics"""
        try:
            if portfolio_id not in self.portfolio_companies:
                return False
            
            portfolio_company = self.portfolio_companies[portfolio_id]
            
            # Update metrics
            if 'current_valuation' in updates:
                portfolio_company.current_valuation = updates['current_valuation']
            
            if 'revenue_data' in updates:
                portfolio_company.revenue_growth.append({
                    'date': datetime.now(timezone.utc).isoformat(),
                    'revenue': updates['revenue_data']['revenue'],
                    'growth_rate': updates['revenue_data'].get('growth_rate', 0)
                })
            
            if 'user_data' in updates:
                portfolio_company.user_metrics.append({
                    'date': datetime.now(timezone.utc).isoformat(),
                    'user_count': updates['user_data']['user_count'],
                    'growth_rate': updates['user_data'].get('growth_rate', 0)
                })
            
            if 'milestone' in updates:
                portfolio_company.key_milestones.append({
                    'date': datetime.now(timezone.utc).isoformat(),
                    'milestone': updates['milestone'],
                    'description': updates.get('milestone_description', '')
                })
            
            portfolio_company.updated_at = datetime.now(timezone.utc)
            
            self.logger.info(f"Portfolio company updated: {portfolio_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Portfolio company update failed: {e}")
            return False
    
    async def execute_exit(self, portfolio_id: str, exit_data: Dict[str, Any]) -> Optional[str]:
        """Execute portfolio company exit"""
        try:
            if portfolio_id not in self.portfolio_companies:
                return None
            
            portfolio_company = self.portfolio_companies[portfolio_id]
            
            # Create exit event
            exit_event = ExitEvent(
                exit_id="",
                portfolio_id=portfolio_id,
                **exit_data
            )
            
            # Calculate returns
            investment_amount = portfolio_company.investment_amount
            exit_event.multiple_of_money = exit_event.proceeds / investment_amount
            
            # Calculate holding period
            holding_period = (exit_event.exit_date - portfolio_company.investment_date).days / 365.25
            exit_event.holding_period_years = holding_period
            
            # Calculate IRR (simplified)
            if holding_period > 0:
                exit_event.irr = ((exit_event.proceeds / investment_amount) ** (1 / holding_period) - 1) * 100
            
            # Update fund metrics
            self.deployed_capital -= investment_amount
            
            self.exit_events[exit_event.exit_id] = exit_event
            
            self.logger.info(f"Exit executed: {portfolio_id} - {exit_event.multiple_of_money:.1f}x return")
            return exit_event.exit_id
            
        except Exception as e:
            self.logger.error(f"Exit execution failed: {e}")
            return None
    
    async def get_deal_pipeline(self) -> List[Dict[str, Any]]:
        """Get current deal pipeline"""
        try:
            pipeline = []
            
            for investment_id in self.deal_pipeline:
                if investment_id in self.investments:
                    investment = self.investments[investment_id]
                    startup = self.startups.get(investment.startup_id)
                    
                    if startup:
                        pipeline.append({
                            'investment_id': investment_id,
                            'startup_name': startup.name,
                            'sector': startup.sector.value,
                            'stage': investment.stage.value,
                            'status': investment.status.value,
                            'investment_amount': investment.investment_amount,
                            'valuation': investment.valuation,
                            'ai_score': startup.ai_score,
                            'dd_score': investment.dd_score,
                            'sourced_at': investment.sourced_at.isoformat()
                        })
            
            # Sort by AI score descending
            pipeline.sort(key=lambda x: x['ai_score'], reverse=True)
            
            return pipeline
            
        except Exception as e:
            self.logger.error(f"Deal pipeline retrieval failed: {e}")
            return []
    
    async def get_portfolio_overview(self) -> Dict[str, Any]:
        """Get portfolio overview"""
        try:
            portfolio_data = []
            total_invested = 0.0
            total_current_value = 0.0
            
            for portfolio_company in self.portfolio_companies.values():
                startup = self.startups.get(portfolio_company.startup_id)
                
                if startup:
                    total_invested += portfolio_company.investment_amount
                    total_current_value += portfolio_company.current_valuation * (portfolio_company.equity_owned / 100)
                    
                    portfolio_data.append({
                        'portfolio_id': portfolio_company.portfolio_id,
                        'startup_name': startup.name,
                        'sector': startup.sector.value,
                        'investment_date': portfolio_company.investment_date.isoformat(),
                        'investment_amount': portfolio_company.investment_amount,
                        'equity_owned': portfolio_company.equity_owned,
                        'current_valuation': portfolio_company.current_valuation,
                        'unrealized_return': ((portfolio_company.current_valuation * portfolio_company.equity_owned / 100) / portfolio_company.investment_amount - 1) * 100,
                        'health_score': portfolio_company.health_score
                    })
            
            # Calculate portfolio metrics
            unrealized_return = ((total_current_value / total_invested) - 1) * 100 if total_invested > 0 else 0
            
            overview = {
                'portfolio_companies': portfolio_data,
                'summary': {
                    'total_companies': len(self.portfolio_companies),
                    'total_invested': total_invested,
                    'total_current_value': total_current_value,
                    'unrealized_return': unrealized_return,
                    'deployed_capital': self.deployed_capital,
                    'available_capital': self.available_capital,
                    'fund_utilization': (self.deployed_capital / self.fund_size) * 100
                }
            }
            
            return overview
            
        except Exception as e:
            self.logger.error(f"Portfolio overview generation failed: {e}")
            return {}
    
    async def _portfolio_monitoring(self):
        """Background task for portfolio monitoring"""
        while True:
            try:
                for portfolio_company in self.portfolio_companies.values():
                    # Simulate health score calculation
                    portfolio_company.health_score = np.random.uniform(60, 95)
                    
                    # Check for risk alerts
                    if portfolio_company.health_score < 70:
                        alert = {
                            'date': datetime.now(timezone.utc).isoformat(),
                            'type': 'health_score_low',
                            'message': f'Health score dropped to {portfolio_company.health_score:.1f}',
                            'severity': 'medium'
                        }
                        portfolio_company.risk_alerts.append(alert)
                        
                        # Keep only recent alerts
                        if len(portfolio_company.risk_alerts) > 10:
                            portfolio_company.risk_alerts = portfolio_company.risk_alerts[-10:]
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Portfolio monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _metrics_updater(self):
        """Background task to update fund metrics"""
        while True:
            try:
                # Calculate fund performance metrics
                total_invested = sum(pc.investment_amount for pc in self.portfolio_companies.values())
                total_current_value = sum(
                    pc.current_valuation * (pc.equity_owned / 100) 
                    for pc in self.portfolio_companies.values()
                )
                
                # Calculate realized returns from exits
                total_realized = sum(exit.proceeds for exit in self.exit_events.values())
                total_exit_cost = sum(
                    self.portfolio_companies[exit.portfolio_id].investment_amount 
                    for exit in self.exit_events.values()
                    if exit.portfolio_id in self.portfolio_companies
                )
                
                self.fund_metrics = {
                    'fund_size': self.fund_size,
                    'deployed_capital': self.deployed_capital,
                    'available_capital': self.available_capital,
                    'portfolio_companies': len(self.portfolio_companies),
                    'exits': len(self.exit_events),
                    'total_invested': total_invested,
                    'total_current_value': total_current_value,
                    'total_realized': total_realized,
                    'unrealized_return': ((total_current_value / total_invested) - 1) * 100 if total_invested > 0 else 0,
                    'realized_multiple': total_realized / total_exit_cost if total_exit_cost > 0 else 0,
                    'fund_utilization': (self.deployed_capital / self.fund_size) * 100,
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }
                
                await asyncio.sleep(1800)  # Update every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Metrics updater error: {e}")
                await asyncio.sleep(300)
    
    async def get_fund_metrics(self) -> Dict[str, Any]:
        """Get fund performance metrics"""
        return self.fund_metrics.copy()

# Example usage
async def main():
    """
    Example usage of Venture Fund
    """
    print("💰 Venture Fund - Phase 5 Testing")
    print("=" * 50)
    
    # Initialize fund
    fund = VentureFund(fund_size=100000000.0)  # $100M fund
    
    # Wait for initialization
    await asyncio.sleep(4)
    
    # Test startup sourcing
    print(f"\n🔍 Testing Startup Sourcing:")
    
    new_startup_data = {
        'name': 'QuantTrade AI',
        'description': 'Quantum-enhanced algorithmic trading platform',
        'founded_date': datetime(2023, 6, 1, tzinfo=timezone.utc),
        'country': 'USA',
        'city': 'New York',
        'sector': SectorFocus.AI_FINTECH,
        'product_stage': 'mvp',
        'current_revenue': 75000.0,
        'revenue_growth_rate': 150.0,
        'user_count': 2500,
        'user_growth_rate': 40.0,
        'retention_rate': 85.0,
        'team_size': 15,
        'technology_stack': ['Python', 'Qiskit', 'TensorFlow', 'React'],
        'competitive_advantages': ['Quantum algorithms', 'Real-time processing', 'Low latency']
    }
    
    startup_id = await fund.source_startup(new_startup_data)
    if startup_id:
        startup = fund.startups[startup_id]
        print(f"  ✅ Startup sourced: {startup.name}")
        print(f"  AI Score: {startup.ai_score:.1f}/100")
        print(f"  Recommendation: {startup.ai_analysis.get('overall', {}).get('recommendation', 'Unknown')}")
    
    # Test investment opportunity creation
    print(f"\n💼 Testing Investment Opportunity:")
    
    if startup_id:
        investment_data = {
            'stage': InvestmentStage.SEED,
            'investment_amount': 2000000.0,  # $2M
            'valuation': 10000000.0,  # $10M pre-money
            'equity_percentage': 20.0,
            'source': 'AI Deal Sourcing'
        }
        
        investment_id = await fund.create_investment_opportunity(startup_id, investment_data)
        if investment_id:
            print(f"  ✅ Investment opportunity created: {investment_id}")
            investment = fund.investments[investment_id]
            print(f"  Stage: {investment.stage.value}")
            print(f"  Amount: ${investment.investment_amount:,.0f}")
            print(f"  Valuation: ${investment.valuation:,.0f}")
            print(f"  Equity: {investment.equity_percentage}%")
    
    # Test due diligence
    print(f"\n🔍 Testing Due Diligence:")
    
    if investment_id:
        dd_updates = {
            'financial_review': True,
            'technical_review': True,
            'market_analysis': True,
            'team_interviews': True
        }
        
        success = await fund.conduct_due_diligence(investment_id, dd_updates)
        if success:
            investment = fund.investments[investment_id]
            print(f"  ✅ Due diligence updated")
            print(f"  DD Score: {investment.dd_score:.1f}%")
            print(f"  Status: {investment.status.value}")
    
    # Test investment decision
    print(f"\n✅ Testing Investment Decision:")
    
    if investment_id:
        success = await fund.make_investment_decision(
            investment_id, "approve", 
            "Strong team, large market, innovative technology"
        )
        if success:
            investment = fund.investments[investment_id]
            print(f"  ✅ Investment decision made: {investment.committee_recommendation}")
            print(f"  Status: {investment.status.value}")
    
    # Test investment closing
    print(f"\n🤝 Testing Investment Closing:")
    
    if investment_id:
        portfolio_id = await fund.close_investment(investment_id)
        if portfolio_id:
            portfolio_company = fund.portfolio_companies[portfolio_id]
            print(f"  ✅ Investment closed: {portfolio_id}")
            print(f"  Investment amount: ${portfolio_company.investment_amount:,.0f}")
            print(f"  Equity owned: {portfolio_company.equity_owned}%")
    
    # Test portfolio update
    print(f"\n📈 Testing Portfolio Update:")
    
    if portfolio_id:
        updates = {
            'current_valuation': 15000000.0,  # $15M (50% increase)
            'revenue_data': {
                'revenue': 125000.0,
                'growth_rate': 67.0
            },
            'milestone': 'Series A funding round completed'
        }
        
        success = await fund.update_portfolio_company(portfolio_id, updates)
        if success:
            print(f"  ✅ Portfolio company updated")
            portfolio_company = fund.portfolio_companies[portfolio_id]
            unrealized_return = ((portfolio_company.current_valuation * portfolio_company.equity_owned / 100) / portfolio_company.investment_amount - 1) * 100
            print(f"  Current valuation: ${portfolio_company.current_valuation:,.0f}")
            print(f"  Unrealized return: {unrealized_return:.1f}%")
    
    # Test deal pipeline
    print(f"\n📋 Testing Deal Pipeline:")
    
    pipeline = await fund.get_deal_pipeline()
    print(f"  Active deals: {len(pipeline)}")
    
    for deal in pipeline[:3]:  # Show top 3
        print(f"    {deal['startup_name']} - {deal['sector']} - Score: {deal['ai_score']:.1f}")
    
    # Test portfolio overview
    print(f"\n📊 Testing Portfolio Overview:")
    
    overview = await fund.get_portfolio_overview()
    if overview:
        summary = overview['summary']
        print(f"  Portfolio companies: {summary['total_companies']}")
        print(f"  Total invested: ${summary['total_invested']:,.0f}")
        print(f"  Current value: ${summary['total_current_value']:,.0f}")
        print(f"  Unrealized return: {summary['unrealized_return']:.1f}%")
        print(f"  Fund utilization: {summary['fund_utilization']:.1f}%")
    
    # Get fund metrics
    print(f"\n💰 Fund Metrics:")
    
    metrics = await fund.get_fund_metrics()
    if metrics:
        print(f"  Fund size: ${metrics['fund_size']:,.0f}")
        print(f"  Deployed capital: ${metrics['deployed_capital']:,.0f}")
        print(f"  Available capital: ${metrics['available_capital']:,.0f}")
        print(f"  Portfolio companies: {metrics['portfolio_companies']}")
        print(f"  Fund utilization: {metrics['fund_utilization']:.1f}%")
        print(f"  Unrealized return: {metrics['unrealized_return']:.1f}%")
    
    print(f"\n✅ Venture Fund testing completed!")

if __name__ == "__main__":
    asyncio.run(main())