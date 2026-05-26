"""
Human-in-the-Loop Approval System for Hermes Agent OSS.

Manages approval workflows for critical operations,
cost limits, and human oversight of agent decisions.
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    """Approval status enum."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class OperationType(str, Enum):
    """Types of operations requiring approval."""
    HIGH_COST = "high_cost"
    SENSITIVE_ACTION = "sensitive_action"
    EXTERNAL_API_CALL = "external_api_call"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_CONFIG_CHANGE = "system_config_change"
    CODE_EXECUTION = "code_execution"
    CUSTOM = "custom"


@dataclass
class ApprovalRequest:
    """Represents an approval request."""
    id: str
    operation_type: OperationType
    title: str
    description: str
    requested_at: datetime
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_at: Optional[datetime] = None
    rejected_at: Optional[datetime] = None
    approver_id: Optional[str] = None
    approver_notes: Optional[str] = None
    context: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    expires_at: Optional[datetime] = None
    auto_timeout_minutes: int = 30

    def is_expired(self) -> bool:
        """Check if approval request has expired."""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['requested_at'] = self.requested_at.isoformat()
        if self.approved_at:
            data['approved_at'] = self.approved_at.isoformat()
        if self.rejected_at:
            data['rejected_at'] = self.rejected_at.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data


class ApprovalRule:
    """
    Rule for determining if operation requires approval.
    """

    def __init__(self, operation_type: OperationType, 
                 condition_func: Callable, 
                 description: str = ""):
        """
        Initialize approval rule.

        Args:
            operation_type: Type of operation
            condition_func: Function that returns True if approval needed
            description: Rule description
        """
        self.operation_type = operation_type
        self.condition_func = condition_func
        self.description = description

    def requires_approval(self, context: Dict[str, Any]) -> bool:
        """
        Check if operation requires approval based on context.

        Args:
            context: Operation context

        Returns:
            True if approval required
        """
        try:
            return self.condition_func(context)
        except Exception as e:
            logger.error(f"Error in approval condition: {str(e)}")
            return True


class ApprovalManager:
    """
    Manages approval workflows and human-in-the-loop operations.
    
    Handles:
    - Approval request creation and tracking
    - Cost limits and budget enforcement
    - Risk assessment and escalation
    - Audit trail of all decisions
    """

    def __init__(self, approval_timeout_minutes: int = 30):
        """
        Initialize approval manager.

        Args:
            approval_timeout_minutes: Auto-timeout for pending approvals
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.approval_timeout_minutes = approval_timeout_minutes
        self.pending_requests: Dict[str, ApprovalRequest] = {}
        self.request_history: List[ApprovalRequest] = []
        self.rules: Dict[OperationType, List[ApprovalRule]] = {}
        self._cleanup_task = None
        
        # Cost tracking
        self.cost_limits: Dict[str, float] = {}
        self.current_costs: Dict[str, float] = {}
        self.cost_reset_time: Dict[str, datetime] = {}
        
        # Audit trail
        self.audit_trail: List[Dict[str, Any]] = []

    async def initialize(self) -> None:
        """
        Initialize the approval manager.
        Starts background cleanup task.
        """
        self.logger.info("Initializing approval manager")
        self._setup_default_rules()
        await self._start_cleanup_task()

    def _setup_default_rules(self) -> None:
        """
        Setup default approval rules.
        """
        # High cost operations
        self.add_rule(
            OperationType.HIGH_COST,
            lambda ctx: ctx.get('estimated_cost', 0) > ctx.get('cost_threshold', 10),
            "Cost exceeds threshold"
        )
        
        # Code execution
        self.add_rule(
            OperationType.CODE_EXECUTION,
            lambda ctx: ctx.get('allow_code_execution', False),
            "Code execution requested"
        )
        
        # Sensitive actions
        self.add_rule(
            OperationType.SENSITIVE_ACTION,
            lambda ctx: ctx.get('is_sensitive', False),
            "Sensitive operation detected"
        )
        
        self.logger.info("Default approval rules configured")

    def add_rule(self, operation_type: OperationType, 
                 condition_func: Callable, description: str = "") -> None:
        """
        Add an approval rule.

        Args:
            operation_type: Type of operation
            condition_func: Condition function
            description: Rule description
        """
        if operation_type not in self.rules:
            self.rules[operation_type] = []
        
        rule = ApprovalRule(operation_type, condition_func, description)
        self.rules[operation_type].append(rule)
        self.logger.debug(f"Added approval rule: {description}")

    def set_cost_limit(self, limit_name: str, limit_amount: float,
                      reset_interval_hours: int = 24) -> None:
        """
        Set a cost limit for tracking.

        Args:
            limit_name: Name of the limit
            limit_amount: Maximum amount
            reset_interval_hours: How often to reset the counter
        """
        self.cost_limits[limit_name] = limit_amount
        self.current_costs[limit_name] = 0.0
        self.cost_reset_time[limit_name] = datetime.now() + timedelta(
            hours=reset_interval_hours
        )
        self.logger.info(f"Set cost limit '{limit_name}': ${limit_amount}")

    def check_cost_limit(self, limit_name: str, additional_cost: float) -> bool:
        """
        Check if cost is within limit.

        Args:
            limit_name: Name of the limit
            additional_cost: Cost to add

        Returns:
            True if within limit, False if would exceed
        """
        if limit_name not in self.cost_limits:
            return True
        
        # Reset if interval expired
        if datetime.now() > self.cost_reset_time.get(limit_name, datetime.now()):
            self.current_costs[limit_name] = 0.0
            self.cost_reset_time[limit_name] = datetime.now() + timedelta(
                hours=24
            )
        
        current = self.current_costs.get(limit_name, 0.0)
        limit = self.cost_limits[limit_name]
        
        within_limit = (current + additional_cost) <= limit
        
        if not within_limit:
            self.logger.warning(
                f"Cost limit '{limit_name}' exceeded: "
                f"${current + additional_cost} > ${limit}"
            )
        
        return within_limit

    async def _check_approval_needed(self, operation_type: OperationType,
                                     context: Dict[str, Any]) -> bool:
        """
        Check if operation needs approval.

        Args:
            operation_type: Type of operation
            context: Operation context

        Returns:
            True if approval needed
        """
        if operation_type not in self.rules:
            return False
        
        rules = self.rules[operation_type]
        for rule in rules:
            if rule.requires_approval(context):
                return True
        
        return False

    async def request_approval(self, operation_type: OperationType,
                              title: str, description: str,
                              context: Optional[Dict[str, Any]] = None,
                              metadata: Optional[Dict[str, Any]] = None
                              ) -> ApprovalRequest:
        """
        Create an approval request.

        Args:
            operation_type: Type of operation
            title: Operation title
            description: Operation description
            context: Operation context
            metadata: Additional metadata

        Returns:
            Approval request
        """
        # Check if approval is needed
        context = context or {}
        needs_approval = await self._check_approval_needed(operation_type, context)
        
        if not needs_approval:
            self.logger.debug(f"Operation '{title}' does not require approval")
            return None
        
        # Create request
        request_id = str(uuid.uuid4())
        request = ApprovalRequest(
            id=request_id,
            operation_type=operation_type,
            title=title,
            description=description,
            requested_at=datetime.now(),
            context=context,
            metadata=metadata or {},
            expires_at=datetime.now() + timedelta(
                minutes=self.approval_timeout_minutes
            )
        )
        
        self.pending_requests[request_id] = request
        self._log_audit("approval_requested", request.to_dict())
        
        self.logger.info(f"Created approval request: {request_id} - {title}")
        return request

    async def approve(self, request_id: str, approver_id: str,
                     notes: Optional[str] = None) -> bool:
        """
        Approve an operation.

        Args:
            request_id: Approval request ID
            approver_id: ID of approver
            notes: Approval notes

        Returns:
            True if approved successfully
        """
        if request_id not in self.pending_requests:
            self.logger.warning(f"Approval request not found: {request_id}")
            return False
        
        request = self.pending_requests[request_id]
        
        if request.is_expired():
            request.status = ApprovalStatus.EXPIRED
            self.logger.warning(f"Approval request expired: {request_id}")
            return False
        
        request.status = ApprovalStatus.APPROVED
        request.approved_at = datetime.now()
        request.approver_id = approver_id
        request.approver_notes = notes
        
        self.request_history.append(request)
        del self.pending_requests[request_id]
        
        self._log_audit("approval_approved", {
            'request_id': request_id,
            'approver_id': approver_id,
            'notes': notes
        })
        
        self.logger.info(f"Approved request: {request_id}")
        return True

    async def reject(self, request_id: str, approver_id: str,
                    reason: str) -> bool:
        """
        Reject an operation.

        Args:
            request_id: Approval request ID
            approver_id: ID of rejector
            reason: Rejection reason

        Returns:
            True if rejected successfully
        """
        if request_id not in self.pending_requests:
            self.logger.warning(f"Approval request not found: {request_id}")
            return False
        
        request = self.pending_requests[request_id]
        request.status = ApprovalStatus.REJECTED
        request.rejected_at = datetime.now()
        request.approver_id = approver_id
        request.approver_notes = reason
        
        self.request_history.append(request)
        del self.pending_requests[request_id]
        
        self._log_audit("approval_rejected", {
            'request_id': request_id,
            'approver_id': approver_id,
            'reason': reason
        })
        
        self.logger.warning(f"Rejected request: {request_id} - {reason}")
        return True

    async def get_pending_approvals(self) -> List[ApprovalRequest]:
        """
        Get all pending approval requests.

        Returns:
            List of pending requests
        """
        return list(self.pending_requests.values())

    async def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """
        Get a specific approval request.

        Args:
            request_id: Request ID

        Returns:
            Approval request or None
        """
        if request_id in self.pending_requests:
            return self.pending_requests[request_id]
        
        for req in self.request_history:
            if req.id == request_id:
                return req
        
        return None

    def add_cost(self, limit_name: str, amount: float) -> None:
        """
        Add cost to a limit.

        Args:
            limit_name: Name of limit
            amount: Amount to add
        """
        if limit_name not in self.current_costs:
            return
        
        self.current_costs[limit_name] += amount
        self.logger.debug(
            f"Added ${amount} to '{limit_name}' "
            f"(total: ${self.current_costs[limit_name]})"
        )

    def get_cost_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get cost limit status.

        Returns:
            Cost status for all limits
        """
        status = {}
        for limit_name in self.cost_limits:
            current = self.current_costs.get(limit_name, 0.0)
            limit = self.cost_limits[limit_name]
            status[limit_name] = {
                'current': current,
                'limit': limit,
                'remaining': limit - current,
                'percentage_used': (current / limit * 100) if limit > 0 else 0,
                'reset_at': self.cost_reset_time.get(limit_name).isoformat()
            }
        return status

    async def _start_cleanup_task(self) -> None:
        """
        Start background task to cleanup expired requests.
        """
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(60)
                    expired_ids = [
                        req_id for req_id, req in self.pending_requests.items()
                        if req.is_expired()
                    ]
                    
                    for req_id in expired_ids:
                        req = self.pending_requests.pop(req_id)
                        req.status = ApprovalStatus.EXPIRED
                        self.request_history.append(req)
                        self.logger.warning(f"Request expired: {req_id}")
                
                except Exception as e:
                    self.logger.error(f"Error in cleanup task: {str(e)}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        self.logger.info("Started approval cleanup task")

    def _log_audit(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log an audit trail event.

        Args:
            event_type: Type of event
            data: Event data
        """
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': data
        }
        self.audit_trail.append(audit_entry)

    async def get_audit_trail(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get audit trail.

        Args:
            limit: Max entries to return

        Returns:
            Audit trail entries
        """
        return self.audit_trail[-limit:]

    async def shutdown(self) -> None:
        """
        Shutdown the approval manager.
        """
        if self._cleanup_task:
            self._cleanup_task.cancel()
        self.logger.info("Approval manager shutdown")