"""
DAO Governance System for Phase 5
Implements decentralized governance with PEPER token
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

class ProposalType(Enum):
    PARAMETER_CHANGE = "parameter_change"
    STRATEGY_UPDATE = "strategy_update"
    TREASURY_ALLOCATION = "treasury_allocation"
    PROTOCOL_UPGRADE = "protocol_upgrade"
    PARTNERSHIP = "partnership"
    EMERGENCY_ACTION = "emergency_action"
    COMMUNITY_FUND = "community_fund"
    TOKENOMICS_CHANGE = "tokenomics_change"

class ProposalStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PASSED = "passed"
    REJECTED = "rejected"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class VoteType(Enum):
    FOR = "for"
    AGAINST = "against"
    ABSTAIN = "abstain"

class GovernanceRole(Enum):
    MEMBER = "member"
    DELEGATE = "delegate"
    COUNCIL_MEMBER = "council_member"
    ADMIN = "admin"
    FOUNDER = "founder"

@dataclass
class PEPERToken:
    """PEPER token configuration and utilities"""
    name: str = "PEPER"
    symbol: str = "PEPER"
    decimals: int = 18
    total_supply: Decimal = Decimal('1000000000')  # 1 billion tokens
    
    # Token distribution
    community_allocation: Decimal = Decimal('400000000')  # 40%
    team_allocation: Decimal = Decimal('200000000')       # 20%
    treasury_allocation: Decimal = Decimal('200000000')   # 20%
    liquidity_allocation: Decimal = Decimal('100000000')  # 10%
    ecosystem_allocation: Decimal = Decimal('100000000')  # 10%
    
    # Governance parameters
    min_proposal_threshold: Decimal = Decimal('10000')    # Min tokens to create proposal
    quorum_threshold: Decimal = Decimal('50000000')       # Min tokens for quorum (5%)
    voting_period: int = 7 * 24 * 3600                    # 7 days in seconds
    execution_delay: int = 2 * 24 * 3600                  # 2 days delay after passing
    
    # Staking rewards
    base_staking_apy: float = 0.08  # 8% base APY
    governance_bonus_apy: float = 0.02  # 2% bonus for active governance

@dataclass
class GovernanceMember:
    """DAO member with governance rights"""
    address: str
    token_balance: Decimal
    staked_balance: Decimal
    voting_power: Decimal
    role: GovernanceRole
    joined_at: datetime
    last_activity: datetime
    proposals_created: int = 0
    votes_cast: int = 0
    delegation_count: int = 0
    delegated_to: Optional[str] = None
    reputation_score: float = 100.0
    
    def __post_init__(self):
        self.voting_power = self._calculate_voting_power()
    
    def _calculate_voting_power(self) -> Decimal:
        """Calculate voting power based on staked tokens and reputation"""
        base_power = self.staked_balance
        
        # Reputation multiplier (0.5x to 2.0x)
        reputation_multiplier = max(0.5, min(2.0, self.reputation_score / 100.0))
        
        # Role multiplier
        role_multipliers = {
            GovernanceRole.MEMBER: 1.0,
            GovernanceRole.DELEGATE: 1.1,
            GovernanceRole.COUNCIL_MEMBER: 1.2,
            GovernanceRole.ADMIN: 1.3,
            GovernanceRole.FOUNDER: 1.5
        }
        
        role_multiplier = role_multipliers.get(self.role, 1.0)
        
        return base_power * Decimal(str(reputation_multiplier * role_multiplier))

@dataclass
class Vote:
    """Individual vote on a proposal"""
    voter_address: str
    proposal_id: str
    vote_type: VoteType
    voting_power: Decimal
    timestamp: datetime
    reason: Optional[str] = None
    delegated_from: List[str] = field(default_factory=list)

@dataclass
class Proposal:
    """DAO governance proposal"""
    id: str
    title: str
    description: str
    proposal_type: ProposalType
    proposer_address: str
    created_at: datetime
    voting_starts_at: datetime
    voting_ends_at: datetime
    execution_eta: Optional[datetime] = None
    
    # Voting results
    votes_for: Decimal = Decimal('0')
    votes_against: Decimal = Decimal('0')
    votes_abstain: Decimal = Decimal('0')
    total_votes: Decimal = Decimal('0')
    
    # Status and execution
    status: ProposalStatus = ProposalStatus.DRAFT
    execution_data: Dict[str, Any] = field(default_factory=dict)
    execution_result: Optional[Dict[str, Any]] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    required_quorum: Decimal = Decimal('50000000')
    required_majority: float = 0.51  # 51% majority
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique proposal ID"""
        data = f"{self.title}{self.proposer_address}{self.created_at.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def get_voting_results(self) -> Dict[str, Any]:
        """Get current voting results"""
        total_votes = self.votes_for + self.votes_against + self.votes_abstain
        
        if total_votes > 0:
            for_percentage = float(self.votes_for / total_votes * 100)
            against_percentage = float(self.votes_against / total_votes * 100)
            abstain_percentage = float(self.votes_abstain / total_votes * 100)
        else:
            for_percentage = against_percentage = abstain_percentage = 0.0
        
        return {
            'votes_for': self.votes_for,
            'votes_against': self.votes_against,
            'votes_abstain': self.votes_abstain,
            'total_votes': total_votes,
            'for_percentage': for_percentage,
            'against_percentage': against_percentage,
            'abstain_percentage': abstain_percentage,
            'quorum_reached': total_votes >= self.required_quorum,
            'majority_reached': for_percentage >= (self.required_majority * 100)
        }
    
    def can_execute(self) -> bool:
        """Check if proposal can be executed"""
        if self.status != ProposalStatus.PASSED:
            return False
        
        if self.execution_eta and datetime.now(timezone.utc) < self.execution_eta:
            return False
        
        return True

class DelegationManager:
    """Manages vote delegation system"""
    
    def __init__(self):
        self.delegations: Dict[str, str] = {}  # delegator -> delegate
        self.delegate_power: Dict[str, Decimal] = defaultdict(Decimal)
        self.logger = logging.getLogger(__name__)
    
    def delegate_votes(self, delegator: str, delegate: str, power: Decimal) -> bool:
        """Delegate voting power to another member"""
        try:
            # Remove previous delegation
            if delegator in self.delegations:
                old_delegate = self.delegations[delegator]
                self.delegate_power[old_delegate] -= power
            
            # Add new delegation
            self.delegations[delegator] = delegate
            self.delegate_power[delegate] += power
            
            self.logger.info(f"Delegation: {delegator} -> {delegate} ({power} PEPER)")
            return True
            
        except Exception as e:
            self.logger.error(f"Delegation failed: {e}")
            return False
    
    def revoke_delegation(self, delegator: str, power: Decimal) -> bool:
        """Revoke vote delegation"""
        try:
            if delegator in self.delegations:
                delegate = self.delegations[delegator]
                self.delegate_power[delegate] -= power
                del self.delegations[delegator]
                
                self.logger.info(f"Delegation revoked: {delegator}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Delegation revocation failed: {e}")
            return False
    
    def get_delegated_power(self, delegate: str) -> Decimal:
        """Get total delegated voting power for a delegate"""
        return self.delegate_power.get(delegate, Decimal('0'))
    
    def get_delegation_info(self, address: str) -> Dict[str, Any]:
        """Get delegation information for an address"""
        return {
            'delegated_to': self.delegations.get(address),
            'delegated_power': self.delegate_power.get(address, Decimal('0')),
            'is_delegate': address in self.delegate_power and self.delegate_power[address] > 0
        }

class ProposalExecutor:
    """Executes passed proposals"""
    
    def __init__(self, dao_governance):
        self.dao = dao_governance
        self.logger = logging.getLogger(__name__)
        
        # Execution handlers for different proposal types
        self.execution_handlers = {
            ProposalType.PARAMETER_CHANGE: self._execute_parameter_change,
            ProposalType.STRATEGY_UPDATE: self._execute_strategy_update,
            ProposalType.TREASURY_ALLOCATION: self._execute_treasury_allocation,
            ProposalType.PROTOCOL_UPGRADE: self._execute_protocol_upgrade,
            ProposalType.PARTNERSHIP: self._execute_partnership,
            ProposalType.EMERGENCY_ACTION: self._execute_emergency_action,
            ProposalType.COMMUNITY_FUND: self._execute_community_fund,
            ProposalType.TOKENOMICS_CHANGE: self._execute_tokenomics_change
        }
    
    async def execute_proposal(self, proposal: Proposal) -> Dict[str, Any]:
        """Execute a passed proposal"""
        try:
            if not proposal.can_execute():
                return {
                    'success': False,
                    'error': 'Proposal cannot be executed',
                    'status': proposal.status.value
                }
            
            handler = self.execution_handlers.get(proposal.proposal_type)
            if not handler:
                return {
                    'success': False,
                    'error': f'No handler for proposal type: {proposal.proposal_type.value}'
                }
            
            # Execute the proposal
            result = await handler(proposal)
            
            # Update proposal status
            proposal.status = ProposalStatus.EXECUTED
            proposal.execution_result = result
            
            self.logger.info(f"Proposal executed: {proposal.id}")
            
            return {
                'success': True,
                'proposal_id': proposal.id,
                'execution_result': result
            }
            
        except Exception as e:
            self.logger.error(f"Proposal execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'proposal_id': proposal.id
            }
    
    async def _execute_parameter_change(self, proposal: Proposal) -> Dict[str, Any]:
        """Execute parameter change proposal"""
        execution_data = proposal.execution_data
        
        # Update system parameters
        for param_name, new_value in execution_data.get('parameters', {}).items():
            # This would update actual system parameters
            self.logger.info(f"Parameter updated: {param_name} = {new_value}")
        
        return {
            'type': 'parameter_change',
            'parameters_updated': len(execution_data.get('parameters', {})),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _execute_strategy_update(self, proposal: Proposal) -> Dict[str, Any]:
        """Execute strategy update proposal"""
        execution_data = proposal.execution_data
        
        # Update trading strategies
        strategy_name = execution_data.get('strategy_name')
        new_parameters = execution_data.get('parameters', {})
        
        # This would update actual trading strategy parameters
        self.logger.info(f"Strategy updated: {strategy_name}")
        
        return {
            'type': 'strategy_update',
            'strategy_name': strategy_name,
            'parameters_updated': len(new_parameters),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _execute_treasury_allocation(self, proposal: Proposal) -> Dict[str, Any]:
        """Execute treasury allocation proposal"""
        execution_data = proposal.execution_data
        
        amount = Decimal(str(execution_data.get('amount', 0)))
        recipient = execution_data.get('recipient')
        purpose = execution_data.get('purpose')
        
        # This would execute actual treasury transfer
        self.logger.info(f"Treasury allocation: {amount} PEPER to {recipient} for {purpose}")
        
        return {
            'type': 'treasury_allocation',
            'amount': amount,
            'recipient': recipient,
            'purpose': purpose,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _execute_protocol_upgrade(self, proposal: Proposal) -> Dict[str, Any]:
        """Execute protocol upgrade proposal"""
        execution_data = proposal.execution_data
        
        upgrade_version = execution_data.get('version')
        upgrade_components = execution_data.get('components', [])
        
        # This would execute actual protocol upgrade
        self.logger.info(f"Protocol upgrade to version {upgrade_version}")
        
        return {
            'type': 'protocol_upgrade',
            'version': upgrade_version,
            'components': upgrade_components,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _execute_partnership(self, proposal: Proposal) -> Dict[str, Any]:
        """Execute partnership proposal"""
        execution_data = proposal.execution_data
        
        partner_name = execution_data.get('partner_name')
        partnership_type = execution_data.get('partnership_type')
        
        # This would initiate actual partnership
        self.logger.info(f"Partnership initiated with {partner_name}")
        
        return {
            'type': 'partnership',
            'partner_name': partner_name,
            'partnership_type': partnership_type,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _execute_emergency_action(self, proposal: Proposal) -> Dict[str, Any]:
        """Execute emergency action proposal"""
        execution_data = proposal.execution_data
        
        action_type = execution_data.get('action_type')
        
        # This would execute emergency action
        self.logger.info(f"Emergency action executed: {action_type}")
        
        return {
            'type': 'emergency_action',
            'action_type': action_type,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _execute_community_fund(self, proposal: Proposal) -> Dict[str, Any]:
        """Execute community fund proposal"""
        execution_data = proposal.execution_data
        
        fund_name = execution_data.get('fund_name')
        allocation = Decimal(str(execution_data.get('allocation', 0)))
        
        # This would create/fund community initiative
        self.logger.info(f"Community fund created: {fund_name} with {allocation} PEPER")
        
        return {
            'type': 'community_fund',
            'fund_name': fund_name,
            'allocation': allocation,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _execute_tokenomics_change(self, proposal: Proposal) -> Dict[str, Any]:
        """Execute tokenomics change proposal"""
        execution_data = proposal.execution_data
        
        change_type = execution_data.get('change_type')
        new_parameters = execution_data.get('parameters', {})
        
        # This would update tokenomics parameters
        self.logger.info(f"Tokenomics updated: {change_type}")
        
        return {
            'type': 'tokenomics_change',
            'change_type': change_type,
            'parameters': new_parameters,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

class DAOGovernance:
    """Main DAO governance system"""
    
    def __init__(self, config_path: str = "config/dao_config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Token configuration
        self.peper_token = PEPERToken()
        
        # Core components
        self.members: Dict[str, GovernanceMember] = {}
        self.proposals: Dict[str, Proposal] = {}
        self.votes: Dict[str, List[Vote]] = defaultdict(list)
        
        # Delegation system
        self.delegation_manager = DelegationManager()
        
        # Proposal executor
        self.executor = ProposalExecutor(self)
        
        # Treasury management
        self.treasury_balance = self.peper_token.treasury_allocation
        self.treasury_history: List[Dict[str, Any]] = []
        
        # Governance metrics
        self.metrics = {
            'total_members': 0,
            'total_proposals': 0,
            'active_proposals': 0,
            'total_votes_cast': 0,
            'treasury_balance': self.treasury_balance,
            'total_staked': Decimal('0'),
            'governance_participation_rate': 0.0
        }
        
        # Background tasks
        self.background_tasks = []
        self.is_running = False
    
    def _load_config(self, config_path: str) -> Dict:
        """Load DAO configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default DAO configuration"""
        return {
            'governance': {
                'voting_period_days': 7,
                'execution_delay_days': 2,
                'quorum_percentage': 5.0,
                'majority_threshold': 51.0,
                'min_proposal_tokens': 10000
            },
            'staking': {
                'base_apy': 8.0,
                'governance_bonus_apy': 2.0,
                'min_stake_amount': 100,
                'unstaking_period_days': 14
            },
            'reputation': {
                'base_score': 100.0,
                'proposal_bonus': 5.0,
                'vote_bonus': 1.0,
                'delegation_bonus': 2.0,
                'max_score': 200.0,
                'min_score': 10.0
            }
        }
    
    async def initialize(self):
        """Initialize DAO governance system"""
        self.logger.info("Initializing DAO governance system...")
        
        # Start background tasks
        self.is_running = True
        self.background_tasks = [
            asyncio.create_task(self._proposal_monitor()),
            asyncio.create_task(self._staking_rewards_distributor()),
            asyncio.create_task(self._reputation_updater())
        ]
        
        self.logger.info("DAO governance system initialized")
    
    async def register_member(self, address: str, initial_stake: Decimal) -> bool:
        """Register new DAO member"""
        try:
            if address in self.members:
                return False  # Already registered
            
            member = GovernanceMember(
                address=address,
                token_balance=initial_stake,
                staked_balance=initial_stake,
                voting_power=initial_stake,
                role=GovernanceRole.MEMBER,
                joined_at=datetime.now(timezone.utc),
                last_activity=datetime.now(timezone.utc)
            )
            
            self.members[address] = member
            self.metrics['total_members'] += 1
            self.metrics['total_staked'] += initial_stake
            
            self.logger.info(f"New member registered: {address} with {initial_stake} PEPER")
            return True
            
        except Exception as e:
            self.logger.error(f"Member registration failed: {e}")
            return False
    
    async def stake_tokens(self, address: str, amount: Decimal) -> bool:
        """Stake PEPER tokens for governance"""
        try:
            if address not in self.members:
                return False
            
            member = self.members[address]
            
            if member.token_balance < amount:
                return False  # Insufficient balance
            
            member.token_balance -= amount
            member.staked_balance += amount
            member.voting_power = member._calculate_voting_power()
            member.last_activity = datetime.now(timezone.utc)
            
            self.metrics['total_staked'] += amount
            
            self.logger.info(f"Tokens staked: {address} staked {amount} PEPER")
            return True
            
        except Exception as e:
            self.logger.error(f"Token staking failed: {e}")
            return False
    
    async def create_proposal(self, proposer_address: str, title: str, 
                            description: str, proposal_type: ProposalType,
                            execution_data: Dict[str, Any],
                            tags: List[str] = None) -> Optional[str]:
        """Create new governance proposal"""
        try:
            if proposer_address not in self.members:
                return None
            
            member = self.members[proposer_address]
            
            # Check minimum token requirement
            if member.staked_balance < self.peper_token.min_proposal_threshold:
                return None
            
            # Create proposal
            now = datetime.now(timezone.utc)
            voting_starts = now + timedelta(hours=1)  # 1 hour delay
            voting_ends = voting_starts + timedelta(seconds=self.peper_token.voting_period)
            execution_eta = voting_ends + timedelta(seconds=self.peper_token.execution_delay)
            
            proposal = Proposal(
                id="",  # Will be generated
                title=title,
                description=description,
                proposal_type=proposal_type,
                proposer_address=proposer_address,
                created_at=now,
                voting_starts_at=voting_starts,
                voting_ends_at=voting_ends,
                execution_eta=execution_eta,
                execution_data=execution_data,
                tags=tags or [],
                required_quorum=self.peper_token.quorum_threshold
            )
            
            self.proposals[proposal.id] = proposal
            
            # Update member stats
            member.proposals_created += 1
            member.last_activity = now
            
            # Update metrics
            self.metrics['total_proposals'] += 1
            self.metrics['active_proposals'] += 1
            
            self.logger.info(f"Proposal created: {proposal.id} by {proposer_address}")
            return proposal.id
            
        except Exception as e:
            self.logger.error(f"Proposal creation failed: {e}")
            return None
    
    async def vote_on_proposal(self, voter_address: str, proposal_id: str, 
                             vote_type: VoteType, reason: str = None) -> bool:
        """Vote on a proposal"""
        try:
            if voter_address not in self.members or proposal_id not in self.proposals:
                return False
            
            proposal = self.proposals[proposal_id]
            member = self.members[voter_address]
            
            # Check if voting is active
            now = datetime.now(timezone.utc)
            if now < proposal.voting_starts_at or now > proposal.voting_ends_at:
                return False
            
            # Check if already voted
            existing_votes = [v for v in self.votes[proposal_id] if v.voter_address == voter_address]
            if existing_votes:
                return False  # Already voted
            
            # Calculate voting power (including delegated power)
            voting_power = member.voting_power + self.delegation_manager.get_delegated_power(voter_address)
            
            # Create vote
            vote = Vote(
                voter_address=voter_address,
                proposal_id=proposal_id,
                vote_type=vote_type,
                voting_power=voting_power,
                timestamp=now,
                reason=reason
            )
            
            self.votes[proposal_id].append(vote)
            
            # Update proposal vote counts
            if vote_type == VoteType.FOR:
                proposal.votes_for += voting_power
            elif vote_type == VoteType.AGAINST:
                proposal.votes_against += voting_power
            else:  # ABSTAIN
                proposal.votes_abstain += voting_power
            
            proposal.total_votes += voting_power
            
            # Update member stats
            member.votes_cast += 1
            member.last_activity = now
            
            # Update metrics
            self.metrics['total_votes_cast'] += 1
            
            self.logger.info(f"Vote cast: {voter_address} voted {vote_type.value} on {proposal_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Voting failed: {e}")
            return False
    
    async def delegate_voting_power(self, delegator_address: str, 
                                  delegate_address: str) -> bool:
        """Delegate voting power to another member"""
        try:
            if (delegator_address not in self.members or 
                delegate_address not in self.members):
                return False
            
            delegator = self.members[delegator_address]
            delegate = self.members[delegate_address]
            
            # Delegate voting power
            success = self.delegation_manager.delegate_votes(
                delegator_address, delegate_address, delegator.voting_power
            )
            
            if success:
                delegator.delegated_to = delegate_address
                delegate.delegation_count += 1
                
                # Update last activity
                delegator.last_activity = datetime.now(timezone.utc)
                delegate.last_activity = datetime.now(timezone.utc)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Vote delegation failed: {e}")
            return False
    
    async def get_proposal(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get proposal details"""
        if proposal_id not in self.proposals:
            return None
        
        proposal = self.proposals[proposal_id]
        voting_results = proposal.get_voting_results()
        
        return {
            'id': proposal.id,
            'title': proposal.title,
            'description': proposal.description,
            'type': proposal.proposal_type.value,
            'proposer': proposal.proposer_address,
            'created_at': proposal.created_at.isoformat(),
            'voting_starts_at': proposal.voting_starts_at.isoformat(),
            'voting_ends_at': proposal.voting_ends_at.isoformat(),
            'execution_eta': proposal.execution_eta.isoformat() if proposal.execution_eta else None,
            'status': proposal.status.value,
            'voting_results': voting_results,
            'tags': proposal.tags,
            'execution_data': proposal.execution_data,
            'execution_result': proposal.execution_result
        }
    
    async def get_member_info(self, address: str) -> Optional[Dict[str, Any]]:
        """Get member information"""
        if address not in self.members:
            return None
        
        member = self.members[address]
        delegation_info = self.delegation_manager.get_delegation_info(address)
        
        return {
            'address': member.address,
            'token_balance': member.token_balance,
            'staked_balance': member.staked_balance,
            'voting_power': member.voting_power,
            'role': member.role.value,
            'joined_at': member.joined_at.isoformat(),
            'last_activity': member.last_activity.isoformat(),
            'proposals_created': member.proposals_created,
            'votes_cast': member.votes_cast,
            'reputation_score': member.reputation_score,
            'delegation_info': delegation_info
        }
    
    async def get_active_proposals(self) -> List[Dict[str, Any]]:
        """Get all active proposals"""
        active_proposals = []
        now = datetime.now(timezone.utc)
        
        for proposal in self.proposals.values():
            if (proposal.status == ProposalStatus.ACTIVE or 
                (proposal.voting_starts_at <= now <= proposal.voting_ends_at)):
                proposal_data = await self.get_proposal(proposal.id)
                if proposal_data:
                    active_proposals.append(proposal_data)
        
        return active_proposals
    
    async def get_dao_metrics(self) -> Dict[str, Any]:
        """Get DAO governance metrics"""
        # Update participation rate
        if self.metrics['total_members'] > 0:
            active_members = len([m for m in self.members.values() 
                                if m.last_activity > datetime.now(timezone.utc) - timedelta(days=30)])
            self.metrics['governance_participation_rate'] = active_members / self.metrics['total_members']
        
        return {
            'dao_metrics': self.metrics,
            'token_info': {
                'name': self.peper_token.name,
                'symbol': self.peper_token.symbol,
                'total_supply': self.peper_token.total_supply,
                'treasury_balance': self.treasury_balance
            },
            'governance_parameters': {
                'min_proposal_threshold': self.peper_token.min_proposal_threshold,
                'quorum_threshold': self.peper_token.quorum_threshold,
                'voting_period_days': self.peper_token.voting_period / (24 * 3600),
                'execution_delay_days': self.peper_token.execution_delay / (24 * 3600)
            }
        }
    
    async def _proposal_monitor(self):
        """Background task to monitor proposal status"""
        while self.is_running:
            try:
                now = datetime.now(timezone.utc)
                
                for proposal in self.proposals.values():
                    # Start voting for proposals
                    if (proposal.status == ProposalStatus.DRAFT and 
                        now >= proposal.voting_starts_at):
                        proposal.status = ProposalStatus.ACTIVE
                        self.logger.info(f"Voting started for proposal: {proposal.id}")
                    
                    # End voting and determine outcome
                    elif (proposal.status == ProposalStatus.ACTIVE and 
                          now >= proposal.voting_ends_at):
                        results = proposal.get_voting_results()
                        
                        if (results['quorum_reached'] and 
                            results['majority_reached']):
                            proposal.status = ProposalStatus.PASSED
                            self.logger.info(f"Proposal passed: {proposal.id}")
                        else:
                            proposal.status = ProposalStatus.REJECTED
                            self.metrics['active_proposals'] -= 1
                            self.logger.info(f"Proposal rejected: {proposal.id}")
                    
                    # Execute passed proposals
                    elif (proposal.status == ProposalStatus.PASSED and 
                          proposal.can_execute()):
                        await self.executor.execute_proposal(proposal)
                        self.metrics['active_proposals'] -= 1
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Proposal monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _staking_rewards_distributor(self):
        """Background task to distribute staking rewards"""
        while self.is_running:
            try:
                # Distribute rewards every hour
                await asyncio.sleep(3600)
                
                for member in self.members.values():
                    if member.staked_balance > 0:
                        # Calculate hourly reward
                        base_reward = (member.staked_balance * 
                                     Decimal(str(self.peper_token.base_staking_apy)) / 
                                     Decimal('8760'))  # Hours in year
                        
                        # Governance participation bonus
                        if member.votes_cast > 0:
                            bonus_reward = (member.staked_balance * 
                                          Decimal(str(self.peper_token.governance_bonus_apy)) / 
                                          Decimal('8760'))
                            base_reward += bonus_reward
                        
                        # Add rewards to balance
                        member.token_balance += base_reward
                
            except Exception as e:
                self.logger.error(f"Staking rewards error: {e}")
    
    async def _reputation_updater(self):
        """Background task to update member reputation scores"""
        while self.is_running:
            try:
                # Update reputation daily
                await asyncio.sleep(24 * 3600)
                
                config = self.config.get('reputation', {})
                
                for member in self.members.values():
                    # Base reputation decay
                    days_inactive = (datetime.now(timezone.utc) - member.last_activity).days
                    if days_inactive > 30:
                        decay_rate = 0.01 * days_inactive  # 1% per day after 30 days
                        member.reputation_score *= (1 - decay_rate)
                    
                    # Bonus for activity
                    if member.votes_cast > 0:
                        member.reputation_score += config.get('vote_bonus', 1.0)
                    
                    if member.proposals_created > 0:
                        member.reputation_score += config.get('proposal_bonus', 5.0)
                    
                    if member.delegation_count > 0:
                        member.reputation_score += config.get('delegation_bonus', 2.0)
                    
                    # Clamp reputation score
                    member.reputation_score = max(
                        config.get('min_score', 10.0),
                        min(config.get('max_score', 200.0), member.reputation_score)
                    )
                    
                    # Recalculate voting power
                    member.voting_power = member._calculate_voting_power()
                
            except Exception as e:
                self.logger.error(f"Reputation update error: {e}")
    
    async def cleanup(self):
        """Cleanup DAO governance system"""
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.logger.info("DAO governance system cleaned up")

# Example usage and testing
async def main():
    """
    Example usage of DAO Governance System
    """
    print("🏛️ DAO Governance System - Phase 5 Testing")
    print("=" * 50)
    
    # Initialize DAO
    dao = DAOGovernance()
    await dao.initialize()
    
    # Register test members
    print("\n👥 Registering DAO Members:")
    test_members = [
        ("0x1234567890123456789012345678901234567890", Decimal('100000')),
        ("0x2345678901234567890123456789012345678901", Decimal('50000')),
        ("0x3456789012345678901234567890123456789012", Decimal('75000')),
        ("0x4567890123456789012345678901234567890123", Decimal('200000')),
        ("0x5678901234567890123456789012345678901234", Decimal('30000'))
    ]
    
    for address, stake in test_members:
        success = await dao.register_member(address, stake)
        print(f"  {address[:10]}...: {stake} PEPER - {'✅' if success else '❌'}")
    
    # Create test proposal
    print(f"\n📝 Creating Test Proposal:")
    proposer = test_members[0][0]
    proposal_id = await dao.create_proposal(
        proposer_address=proposer,
        title="Increase Trading Strategy Allocation",
        description="Proposal to allocate 100,000 PEPER from treasury for new AI trading strategy development",
        proposal_type=ProposalType.TREASURY_ALLOCATION,
        execution_data={
            'amount': 100000,
            'recipient': '0x9999999999999999999999999999999999999999',
            'purpose': 'AI Trading Strategy Development'
        },
        tags=['treasury', 'ai', 'strategy']
    )
    
    if proposal_id:
        print(f"  Proposal created: {proposal_id}")
        
        # Get proposal details
        proposal_details = await dao.get_proposal(proposal_id)
        print(f"  Title: {proposal_details['title']}")
        print(f"  Type: {proposal_details['type']}")
        print(f"  Status: {proposal_details['status']}")
    
    # Test voting
    print(f"\n🗳️ Testing Voting:")
    if proposal_id:
        # Simulate voting period start
        dao.proposals[proposal_id].voting_starts_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        dao.proposals[proposal_id].status = ProposalStatus.ACTIVE
        
        votes = [
            (test_members[0][0], VoteType.FOR, "Support AI development"),
            (test_members[1][0], VoteType.FOR, "Good for platform growth"),
            (test_members[2][0], VoteType.AGAINST, "Too much allocation"),
            (test_members[3][0], VoteType.FOR, "Necessary investment"),
            (test_members[4][0], VoteType.ABSTAIN, "Need more information")
        ]
        
        for voter, vote_type, reason in votes:
            success = await dao.vote_on_proposal(voter, proposal_id, vote_type, reason)
            print(f"  {voter[:10]}... voted {vote_type.value}: {'✅' if success else '❌'}")
        
        # Check voting results
        proposal_details = await dao.get_proposal(proposal_id)
        results = proposal_details['voting_results']
        print(f"\n📊 Voting Results:")
        print(f"  For: {results['for_percentage']:.1f}% ({results['votes_for']} PEPER)")
        print(f"  Against: {results['against_percentage']:.1f}% ({results['votes_against']} PEPER)")
        print(f"  Abstain: {results['abstain_percentage']:.1f}% ({results['votes_abstain']} PEPER)")
        print(f"  Quorum Reached: {'✅' if results['quorum_reached'] else '❌'}")
        print(f"  Majority Reached: {'✅' if results['majority_reached'] else '❌'}")
    
    # Test delegation
    print(f"\n🤝 Testing Vote Delegation:")
    delegator = test_members[4][0]
    delegate = test_members[3][0]
    success = await dao.delegate_voting_power(delegator, delegate)
    print(f"  {delegator[:10]}... delegated to {delegate[:10]}...: {'✅' if success else '❌'}")
    
    # Get member info
    print(f"\n👤 Member Information:")
    for address, _ in test_members[:2]:
        member_info = await dao.get_member_info(address)
        if member_info:
            print(f"  {address[:10]}...:")
            print(f"    Staked: {member_info['staked_balance']} PEPER")
            print(f"    Voting Power: {member_info['voting_power']} PEPER")
            print(f"    Proposals Created: {member_info['proposals_created']}")
            print(f"    Votes Cast: {member_info['votes_cast']}")
            print(f"    Reputation: {member_info['reputation_score']:.1f}")
    
    # Get DAO metrics
    print(f"\n📈 DAO Metrics:")
    metrics = await dao.get_dao_metrics()
    dao_metrics = metrics['dao_metrics']
    print(f"  Total Members: {dao_metrics['total_members']}")
    print(f"  Total Proposals: {dao_metrics['total_proposals']}")
    print(f"  Active Proposals: {dao_metrics['active_proposals']}")
    print(f"  Total Votes Cast: {dao_metrics['total_votes_cast']}")
    print(f"  Treasury Balance: {dao_metrics['treasury_balance']} PEPER")
    print(f"  Total Staked: {dao_metrics['total_staked']} PEPER")
    print(f"  Participation Rate: {dao_metrics['governance_participation_rate']:.1%}")
    
    # Cleanup
    await dao.cleanup()
    
    print(f"\n✅ DAO governance testing completed!")
    print(f"🏛️ Decentralized governance ready for Phase 5!")

if __name__ == "__main__":
    asyncio.run(main())