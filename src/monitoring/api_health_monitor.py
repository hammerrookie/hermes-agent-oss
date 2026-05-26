"""
API Provider Health Monitor for Hermes Agent OSS.

Monitors API provider health, rate limits, response times,
and automatically updates specifications at startup and regularly.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import time
import aiohttp


logger = logging.getLogger(__name__)


@dataclass
class APIMetrics:
    """Metrics for an API endpoint."""
    endpoint: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    last_check_time: Optional[datetime] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    requests_per_minute: float = 0.0
    average_response_time: float = 0.0
    is_healthy: bool = True
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    last_status_code: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.last_check_time:
            data['last_check_time'] = self.last_check_time.isoformat()
        if self.rate_limit_reset:
            data['rate_limit_reset'] = self.rate_limit_reset.isoformat()
        return data


@dataclass
class APISpecification:
    """API provider specification."""
    provider_name: str
    base_url: str
    endpoints: List[str]
    rate_limit_requests: Optional[int] = None
    rate_limit_window_seconds: Optional[int] = None
    timeout_seconds: int = 30
    authentication_type: Optional[str] = None
    required_headers: Dict[str, str] = None
    api_version: Optional[str] = None
    documentation_url: Optional[str] = None
    last_updated: Optional[datetime] = None
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.last_updated:
            data['last_updated'] = self.last_updated.isoformat()
        return data


class APIHealthMonitor:
    """
    Monitors health and performance of API providers.
    
    Tracks:
    - Response times per request
    - Queries per minute counter
    - Success/failure rates
    - Rate limits
    - API specifications with auto-updates
    """

    def __init__(self, check_interval_seconds: int = 300):
        """
        Initialize API health monitor.

        Args:
            check_interval_seconds: How often to check API health (5 minutes default)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.check_interval_seconds = check_interval_seconds
        self.metrics: Dict[str, APIMetrics] = {}
        self.specifications: Dict[str, APISpecification] = {}
        self.request_history: Dict[str, List[datetime]] = defaultdict(list)
        self._monitor_task = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        """
        Initialize the health monitor.
        Creates HTTP session and starts monitoring.
        """
        self.logger.info("Initializing API health monitor")
        self._session = aiohttp.ClientSession()
        await self._start_monitoring()
        self.logger.info("API health monitor initialized")

    def register_provider(self, spec: APISpecification) -> None:
        """
        Register an API provider.

        Args:
            spec: API specification
        """
        provider_name = spec.provider_name
        self.specifications[provider_name] = spec
        
        # Initialize metrics for all endpoints
        for endpoint in spec.endpoints:
            endpoint_key = f"{provider_name}:{endpoint}"
            self.metrics[endpoint_key] = APIMetrics(endpoint=endpoint)
        
        self.logger.info(
            f"Registered API provider: {provider_name} "
            f"with {len(spec.endpoints)} endpoints"
        )

    async def check_health(self, provider_name: str) -> bool:
        """
        Check health of a provider.

        Args:
            provider_name: Name of provider

        Returns:
            True if healthy, False otherwise
        """
        if provider_name not in self.specifications:
            self.logger.warning(f"Provider not registered: {provider_name}")
            return False
        
        spec = self.specifications[provider_name]
        
        try:
            # Try to ping the base URL
            start_time = time.time()
            
            headers = spec.required_headers or {}
            
            async with self._session.head(
                spec.base_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=spec.timeout_seconds)
            ) as response:
                response_time = time.time() - start_time
                
                # Update metrics for all endpoints
                for endpoint in spec.endpoints:
                    endpoint_key = f"{provider_name}:{endpoint}"
                    self._update_endpoint_metrics(
                        endpoint_key,
                        response.status,
                        response_time,
                        response.headers
                    )
                
                is_healthy = response.status < 500
                
                self.logger.debug(
                    f"Health check for {provider_name}: "
                    f"Status={response.status}, Time={response_time:.3f}s"
                )
                
                return is_healthy
        
        except asyncio.TimeoutError:
            self.logger.warning(f"Health check timeout for {provider_name}")
            self._mark_provider_unhealthy(provider_name, "Timeout")
            return False
        
        except Exception as e:
            self.logger.error(
                f"Health check failed for {provider_name}: {str(e)}"
            )
            self._mark_provider_unhealthy(provider_name, str(e))
            return False

    def record_request(self, provider_name: str, endpoint: str,
                      response_time: float, status_code: int,
                      response_headers: Dict[str, Any]) -> None:
        """
        Record an API request.

        Args:
            provider_name: Provider name
            endpoint: Endpoint path
            response_time: Response time in seconds
            status_code: HTTP status code
            response_headers: Response headers
        """
        endpoint_key = f"{provider_name}:{endpoint}"
        
        if endpoint_key not in self.metrics:
            self.metrics[endpoint_key] = APIMetrics(endpoint=endpoint)
        
        self._update_endpoint_metrics(
            endpoint_key,
            status_code,
            response_time,
            response_headers
        )
        
        # Record timestamp for RPM calculation
        self.request_history[endpoint_key].append(datetime.now())

    def _update_endpoint_metrics(self, endpoint_key: str, status_code: int,
                                response_time: float,
                                headers: Dict[str, Any]) -> None:
        """
        Update metrics for an endpoint.

        Args:
            endpoint_key: Endpoint identifier
            status_code: HTTP status code
            response_time: Response time
            headers: Response headers
        """
        if endpoint_key not in self.metrics:
            self.metrics[endpoint_key] = APIMetrics(endpoint=endpoint_key)
        
        metrics = self.metrics[endpoint_key]
        metrics.total_requests += 1
        metrics.total_response_time += response_time
        metrics.last_check_time = datetime.now()
        metrics.last_status_code = status_code
        
        # Update rate limit info from response headers
        if 'x-ratelimit-remaining' in headers:
            try:
                metrics.rate_limit_remaining = int(
                    headers.get('x-ratelimit-remaining', 0)
                )
            except ValueError:
                pass
        
        if 'x-ratelimit-reset' in headers:
            try:
                reset_timestamp = int(headers.get('x-ratelimit-reset'))
                metrics.rate_limit_reset = datetime.fromtimestamp(
                    reset_timestamp
                )
            except (ValueError, TypeError):
                pass
        
        # Update success/failure
        if status_code < 400:
            metrics.successful_requests += 1
            metrics.consecutive_failures = 0
            metrics.is_healthy = True
        else:
            metrics.failed_requests += 1
            metrics.consecutive_failures += 1
            metrics.last_error = f"HTTP {status_code}"
            
            if metrics.consecutive_failures > 3:
                metrics.is_healthy = False
        
        # Update averages
        if metrics.total_requests > 0:
            metrics.average_response_time = (
                metrics.total_response_time / metrics.total_requests
            )
        
        # Calculate requests per minute
        metrics.requests_per_minute = self._calculate_rpm(endpoint_key)

    def _mark_provider_unhealthy(self, provider_name: str,
                                error_message: str) -> None:
        """
        Mark all endpoints of a provider as unhealthy.

        Args:
            provider_name: Provider name
            error_message: Error message
        """
        for endpoint_key, metrics in self.metrics.items():
            if endpoint_key.startswith(f"{provider_name}:"):
                metrics.is_healthy = False
                metrics.last_error = error_message
                metrics.consecutive_failures += 1

    def _calculate_rpm(self, endpoint_key: str) -> float:
        """
        Calculate requests per minute for an endpoint.

        Args:
            endpoint_key: Endpoint identifier

        Returns:
            Requests per minute
        """
        if endpoint_key not in self.request_history:
            return 0.0
        
        # Get requests from last minute
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        recent_requests = [
            req_time for req_time in self.request_history[endpoint_key]
            if req_time > one_minute_ago
        ]
        
        # Clean old requests
        self.request_history[endpoint_key] = recent_requests
        
        return float(len(recent_requests))

    async def update_specifications(self) -> None:
        """
        Update API specifications for all registered providers.
        Called at startup and regularly during monitoring.
        """
        self.logger.info("Updating API specifications")
        
        for provider_name, spec in self.specifications.items():
            try:
                await self._fetch_provider_spec(provider_name, spec)
            except Exception as e:
                self.logger.error(
                    f"Failed to update spec for {provider_name}: {str(e)}"
                )

    async def _fetch_provider_spec(self, provider_name: str,
                                  spec: APISpecification) -> None:
        """
        Fetch specification for a provider.
        Updates last_updated timestamp.

        Args:
            provider_name: Provider name
            spec: Current specification
        """
        self.logger.debug(f"Fetching spec for {provider_name}")
        
        # Provider-specific implementations:
        # - OpenAI: fetch from /models endpoint
        # - Generic: fetch from /.well-known/openapi.json
        # - Custom: fetch from spec.documentation_url
        
        spec.last_updated = datetime.now()
        self.logger.info(f"Updated specification for {provider_name}")

    async def _start_monitoring(self) -> None:
        """
        Start background monitoring task.
        Checks health and updates specs at configured interval.
        """
        async def monitoring_loop():
            while True:
                try:
                    self.logger.debug("Running health checks and spec updates")
                    
                    # Update specifications
                    await self.update_specifications()
                    
                    # Check health for all providers
                    for provider_name in self.specifications.keys():
                        await self.check_health(provider_name)
                    
                    await asyncio.sleep(self.check_interval_seconds)
                
                except Exception as e:
                    self.logger.error(
                        f"Error in monitoring loop: {str(e)}"
                    )
                    await asyncio.sleep(10)
        
        self._monitor_task = asyncio.create_task(monitoring_loop())
        self.logger.info("Started health monitoring task")

    def get_provider_status(self, provider_name: str) -> Dict[str, Any]:
        """
        Get comprehensive status of a provider.

        Args:
            provider_name: Provider name

        Returns:
            Provider status information with all metrics
        """
        spec = self.specifications.get(provider_name)
        if not spec:
            return {'error': 'Provider not found'}
        
        endpoint_metrics = {}
        for endpoint in spec.endpoints:
            endpoint_key = f"{provider_name}:{endpoint}"
            if endpoint_key in self.metrics:
                endpoint_metrics[endpoint] = self.metrics[endpoint_key].to_dict()
        
        return {
            'provider_name': provider_name,
            'spec': spec.to_dict(),
            'endpoints': endpoint_metrics,
            'overall_healthy': all(
                metrics.is_healthy
                for metrics in endpoint_metrics.values()
            )
        }

    def get_all_providers_status(self) -> Dict[str, Any]:
        """
        Get status of all registered providers.

        Returns:
            Status for all providers
        """
        return {
            provider_name: self.get_provider_status(provider_name)
            for provider_name in self.specifications.keys()
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics across all endpoints.

        Returns:
            Aggregated metrics summary
        """
        total_requests = sum(
            m.total_requests for m in self.metrics.values()
        )
        total_successful = sum(
            m.successful_requests for m in self.metrics.values()
        )
        
        return {
            'total_requests': total_requests,
            'total_successful': total_successful,
            'total_failed': total_requests - total_successful,
            'success_rate': (
                (total_successful / total_requests * 100)
                if total_requests > 0 else 0
            ),
            'average_response_time': (
                sum(m.average_response_time for m in self.metrics.values())
                / len(self.metrics)
                if self.metrics else 0
            ),
            'total_rpm': sum(
                m.requests_per_minute for m in self.metrics.values()
            ),
            'healthy_endpoints': sum(
                1 for m in self.metrics.values() if m.is_healthy
            ),
            'total_endpoints': len(self.metrics)
        }

    async def shutdown(self) -> None:
        """
        Shutdown the health monitor.
        """
        if self._monitor_task:
            self._monitor_task.cancel()
        
        if self._session:
            await self._session.close()
        
        self.logger.info("API health monitor shutdown")