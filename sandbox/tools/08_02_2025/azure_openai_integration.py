"""
Azure OpenAI Integration for NewbornAI 2.0 IIT 4.0 Production Deployment
Phase 2B: Enterprise-grade Azure OpenAI integration with cost optimization

This module provides comprehensive Azure OpenAI integration for production
deployment of the IIT 4.0 consciousness detection system, including
authentication, rate limiting, error recovery, and cost optimization.

Key Features:
- Azure OpenAI API integration with authentication and key management
- Intelligent rate limiting and request throttling
- Comprehensive error handling and automatic retry with exponential backoff
- Cost optimization strategies and usage monitoring
- Production monitoring, alerting, and health checks
- Multi-region deployment support for high availability
- Secure credential management and rotation

Author: LLM Systems Architect (Hirosato Gamo's expertise)
Date: 2025-08-03
Version: 2.0.0
"""

import asyncio
import aiohttp
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import uuid
from datetime import datetime, timedelta
import hashlib
import statistics
from collections import deque, defaultdict
import math
import ssl
import certifi

# Import our implementations
from production_phi_calculator import ProductionPhiCalculator, CalculationRequest, CalculationResponse, CalculationPriority
from pyphi_iit4_bridge import PhiCalculationResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureRegion(Enum):
    """Supported Azure regions for OpenAI deployment"""
    EAST_US = "eastus"
    WEST_US = "westus"
    WEST_US_2 = "westus2"
    CENTRAL_US = "centralus"
    NORTH_EUROPE = "northeurope"
    WEST_EUROPE = "westeurope"
    SOUTHEAST_ASIA = "southeastasia"
    JAPAN_EAST = "japaneast"


class ServiceTier(Enum):
    """Azure OpenAI service tiers"""
    STANDARD = "standard"
    PREMIUM = "premium"
    DEDICATED = "dedicated"


class RequestPriority(Enum):
    """Request priority levels for rate limiting"""
    CRITICAL = 1      # Real-time consciousness detection
    HIGH = 2          # Interactive applications
    NORMAL = 3        # Standard processing
    LOW = 4           # Batch processing
    BACKGROUND = 5    # Maintenance tasks


@dataclass
class AzureOpenAIConfig:
    """Configuration for Azure OpenAI integration"""
    # Primary connection
    endpoint: str
    api_key: str
    api_version: str = "2023-12-01-preview"
    deployment_name: str = "consciousness-analyzer"
    
    # Failover configuration
    failover_endpoints: List[Dict[str, str]] = field(default_factory=list)
    
    # Rate limiting
    requests_per_minute: int = 100
    tokens_per_minute: int = 10000
    concurrent_requests: int = 10
    
    # Retry configuration
    max_retries: int = 3
    base_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    
    # Cost optimization
    enable_cost_optimization: bool = True
    max_daily_cost_usd: float = 100.0
    cost_alert_threshold: float = 0.8  # Alert at 80% of daily limit
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_retention_hours: int = 24
    health_check_interval: int = 300  # 5 minutes
    
    # Security
    enable_ssl_verification: bool = True
    certificate_path: Optional[str] = None
    timeout_seconds: float = 30.0


@dataclass
class RateLimitState:
    """Rate limiting state tracker"""
    requests_this_minute: int = 0
    tokens_this_minute: int = 0
    current_minute: int = 0
    last_request_time: float = 0.0
    active_requests: int = 0
    
    def reset_if_new_minute(self):
        """Reset counters if we're in a new minute"""
        current_minute = int(time.time() // 60)
        if current_minute != self.current_minute:
            self.requests_this_minute = 0
            self.tokens_this_minute = 0
            self.current_minute = current_minute


@dataclass
class CostTracker:
    """Cost tracking for Azure OpenAI usage"""
    daily_cost_usd: float = 0.0
    monthly_cost_usd: float = 0.0
    total_requests: int = 0
    total_tokens: int = 0
    last_reset_date: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d'))
    
    # Cost rates (example rates - actual rates vary)
    cost_per_1k_tokens: float = 0.002  # $0.002 per 1K tokens
    cost_per_request: float = 0.0001   # $0.0001 per request
    
    def add_usage(self, tokens: int, requests: int = 1):
        """Add usage and calculate cost"""
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Reset daily cost if new day
        if current_date != self.last_reset_date:
            self.daily_cost_usd = 0.0
            self.last_reset_date = current_date
        
        # Calculate incremental cost
        token_cost = (tokens / 1000.0) * self.cost_per_1k_tokens
        request_cost = requests * self.cost_per_request
        total_cost = token_cost + request_cost
        
        # Update counters
        self.daily_cost_usd += total_cost
        self.monthly_cost_usd += total_cost
        self.total_requests += requests
        self.total_tokens += tokens
    
    def is_over_budget(self, max_daily_cost: float) -> bool:
        """Check if over daily budget"""
        return self.daily_cost_usd >= max_daily_cost


@dataclass
class MonitoringMetrics:
    """Monitoring metrics for Azure OpenAI integration"""
    timestamp: float
    requests_per_minute: float
    average_latency: float
    error_rate: float
    cost_per_hour: float
    active_connections: int
    queue_size: int
    success_rate: float
    failover_events: int = 0


class CircuitBreaker:
    """Circuit breaker for Azure OpenAI resilience"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 300.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = asyncio.Lock()
    
    async def can_execute(self) -> bool:
        """Check if execution is allowed"""
        async with self._lock:
            if self.state == 'CLOSED':
                return True
            elif self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'HALF_OPEN'
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    async def on_success(self):
        """Record successful execution"""
        async with self._lock:
            self.failure_count = 0
            self.state = 'CLOSED'
    
    async def on_failure(self):
        """Record failed execution"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
    
    def get_state(self) -> str:
        """Get current circuit breaker state"""
        return self.state


class RateLimiter:
    """Intelligent rate limiter with priority-based queuing"""
    
    def __init__(self, config: AzureOpenAIConfig):
        self.config = config
        self.state = RateLimitState()
        self.priority_queues = {priority: asyncio.Queue() for priority in RequestPriority}
        self.semaphore = asyncio.Semaphore(config.concurrent_requests)
        self.processing_task = None
        self._stop_event = asyncio.Event()
    
    async def start(self):
        """Start rate limiter processing"""
        self.processing_task = asyncio.create_task(self._process_queue())
    
    async def stop(self):
        """Stop rate limiter processing"""
        self._stop_event.set()
        if self.processing_task:
            await self.processing_task
    
    async def acquire(self, priority: RequestPriority, estimated_tokens: int = 100) -> bool:
        """Acquire rate limit permit"""
        # Check if we can proceed immediately
        if await self._can_proceed(estimated_tokens):
            await self.semaphore.acquire()
            self.state.active_requests += 1
            self.state.last_request_time = time.time()
            return True
        
        # Add to priority queue
        future = asyncio.Future()
        await self.priority_queues[priority].put((future, estimated_tokens))
        
        # Wait for permit
        return await future
    
    async def release(self, actual_tokens: int):
        """Release rate limit permit"""
        self.state.requests_this_minute += 1
        self.state.tokens_this_minute += actual_tokens
        self.state.active_requests -= 1
        self.semaphore.release()
    
    async def _can_proceed(self, estimated_tokens: int) -> bool:
        """Check if request can proceed immediately"""
        self.state.reset_if_new_minute()
        
        # Check request limit
        if self.state.requests_this_minute >= self.config.requests_per_minute:
            return False
        
        # Check token limit
        if self.state.tokens_this_minute + estimated_tokens > self.config.tokens_per_minute:
            return False
        
        # Check concurrent requests
        if self.state.active_requests >= self.config.concurrent_requests:
            return False
        
        return True
    
    async def _process_queue(self):
        """Process priority queue for rate limiting"""
        while not self._stop_event.is_set():
            try:
                # Process queues in priority order
                for priority in RequestPriority:
                    queue = self.priority_queues[priority]
                    
                    if not queue.empty():
                        future, estimated_tokens = await asyncio.wait_for(
                            queue.get(), timeout=0.1
                        )
                        
                        if await self._can_proceed(estimated_tokens):
                            await self.semaphore.acquire()
                            self.state.active_requests += 1
                            self.state.last_request_time = time.time()
                            future.set_result(True)
                            break
                        else:
                            # Put back in queue if can't proceed
                            await queue.put((future, estimated_tokens))
                
                await asyncio.sleep(0.1)  # Small delay between checks
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Rate limiter processing error: {e}")
                await asyncio.sleep(1.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            'requests_this_minute': self.state.requests_this_minute,
            'tokens_this_minute': self.state.tokens_this_minute,
            'active_requests': self.state.active_requests,
            'queue_sizes': {
                priority.name: queue.qsize() 
                for priority, queue in self.priority_queues.items()
            }
        }


class AzureOpenAIClient:
    """Production-ready Azure OpenAI client with enterprise features"""
    
    def __init__(self, config: AzureOpenAIConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config)
        self.circuit_breaker = CircuitBreaker()
        self.cost_tracker = CostTracker()
        
        # HTTP session management
        self.session = None
        self.session_lock = asyncio.Lock()
        
        # Monitoring and metrics
        self.metrics_history = deque(maxlen=1000)
        self.error_history = deque(maxlen=100)
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_latency': 0.0,
            'total_cost': 0.0
        }
        
        # Failover state
        self.current_endpoint_index = 0
        self.failover_history = []
        
        logger.info("AzureOpenAIClient initialized")
    
    async def start(self):
        """Start the Azure OpenAI client"""
        await self.rate_limiter.start()
        await self._create_session()
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("AzureOpenAI client started")
    
    async def stop(self):
        """Stop the Azure OpenAI client"""
        await self.rate_limiter.stop()
        
        if hasattr(self, 'monitoring_task'):
            self.monitoring_task.cancel()
        
        if self.session:
            await self.session.close()
        
        logger.info("AzureOpenAI client stopped")
    
    async def _create_session(self):
        """Create HTTP session with proper SSL configuration"""
        async with self.session_lock:
            if self.session and not self.session.closed:
                return
            
            # SSL context configuration
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            if not self.config.enable_ssl_verification:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            
            # Connection configuration
            connector = aiohttp.TCPConnector(
                ssl=ssl_context,
                limit=100,  # Total connection pool size
                limit_per_host=10,  # Connections per host
                keepalive_timeout=30.0,
                enable_cleanup_closed=True
            )
            
            # Timeout configuration
            timeout = aiohttp.ClientTimeout(
                total=self.config.timeout_seconds,
                connect=10.0,
                sock_read=self.config.timeout_seconds
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'Content-Type': 'application/json',
                    'api-key': self.config.api_key,
                    'User-Agent': 'NewbornAI-IIT4/2.0'
                }
            )
    
    async def analyze_consciousness_data(self, 
                                       phi_results: List[PhiCalculationResult],
                                       priority: RequestPriority = RequestPriority.NORMAL) -> Dict[str, Any]:
        """
        Analyze consciousness data using Azure OpenAI
        
        Args:
            phi_results: List of φ calculation results
            priority: Request priority for rate limiting
            
        Returns:
            Analysis results from Azure OpenAI
        """
        # Check circuit breaker
        if not await self.circuit_breaker.can_execute():
            raise RuntimeError("Circuit breaker open - Azure OpenAI service unavailable")
        
        # Check cost budget
        if (self.config.enable_cost_optimization and 
            self.cost_tracker.is_over_budget(self.config.max_daily_cost_usd)):
            raise RuntimeError("Daily cost budget exceeded")
        
        # Prepare consciousness analysis prompt
        analysis_prompt = self._create_consciousness_analysis_prompt(phi_results)
        estimated_tokens = len(analysis_prompt.split()) * 1.3  # Rough token estimation
        
        # Acquire rate limit
        await self.rate_limiter.acquire(priority, int(estimated_tokens))
        
        try:
            # Make API request with retry logic
            result = await self._make_api_request_with_retry(analysis_prompt)
            
            # Update metrics and cost tracking
            actual_tokens = result.get('usage', {}).get('total_tokens', int(estimated_tokens))
            await self._update_metrics(True, actual_tokens)
            
            return result
            
        except Exception as e:
            await self._update_metrics(False, 0)
            await self.circuit_breaker.on_failure()
            raise e
        finally:
            await self.rate_limiter.release(int(estimated_tokens))
    
    def _create_consciousness_analysis_prompt(self, phi_results: List[PhiCalculationResult]) -> str:
        """Create consciousness analysis prompt for Azure OpenAI"""
        # Summarize φ results
        phi_summary = {
            'total_results': len(phi_results),
            'phi_values': [r.phi_value for r in phi_results if r.phi_value is not None],
            'node_counts': [r.node_count for r in phi_results],
            'calculation_times': [r.calculation_time for r in phi_results],
            'modes_used': [r.mode_used.value for r in phi_results if r.mode_used]
        }
        
        # Calculate summary statistics
        if phi_summary['phi_values']:
            phi_stats = {
                'mean': statistics.mean(phi_summary['phi_values']),
                'median': statistics.median(phi_summary['phi_values']),
                'max': max(phi_summary['phi_values']),
                'min': min(phi_summary['phi_values'])
            }
        else:
            phi_stats = {'mean': 0, 'median': 0, 'max': 0, 'min': 0}
        
        prompt = f"""
        Analyze the following consciousness detection results based on Integrated Information Theory (IIT) 4.0:
        
        Summary Statistics:
        - Total φ calculations: {phi_summary['total_results']}
        - φ value statistics: Mean={phi_stats['mean']:.3f}, Median={phi_stats['median']:.3f}, Range=[{phi_stats['min']:.3f}, {phi_stats['max']:.3f}]
        - Average system size: {statistics.mean(phi_summary['node_counts']) if phi_summary['node_counts'] else 0:.1f} nodes
        - Average calculation time: {statistics.mean(phi_summary['calculation_times']) if phi_summary['calculation_times'] else 0:.3f} seconds
        
        Individual Results:
        {json.dumps([{
            'phi_value': r.phi_value,
            'node_count': r.node_count,
            'concept_count': r.concept_count,
            'calculation_time': r.calculation_time,
            'cache_hit': r.cache_hit
        } for r in phi_results[:10]], indent=2)}  # Limit to first 10 for prompt size
        
        Please provide:
        1. Consciousness level assessment (scale 0-1)
        2. Key patterns or anomalies in the φ values
        3. Recommendations for system optimization
        4. Confidence level in the analysis (scale 0-1)
        5. Potential issues or concerns
        
        Respond in JSON format with clear structured analysis.
        """
        
        return prompt
    
    async def _make_api_request_with_retry(self, prompt: str) -> Dict[str, Any]:
        """Make API request with exponential backoff retry"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Try current endpoint
                endpoint = self._get_current_endpoint()
                
                request_data = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert consciousness researcher specializing in Integrated Information Theory (IIT) 4.0 analysis."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.3,
                    "top_p": 0.9
                }
                
                start_time = time.time()
                
                async with self.session.post(
                    f"{endpoint}/openai/deployments/{self.config.deployment_name}/chat/completions",
                    params={"api-version": self.config.api_version},
                    json=request_data
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        latency = time.time() - start_time
                        
                        # Extract and structure response
                        analysis_result = {
                            'analysis': result['choices'][0]['message']['content'],
                            'usage': result.get('usage', {}),
                            'latency': latency,
                            'endpoint': endpoint,
                            'attempt': attempt + 1
                        }
                        
                        await self.circuit_breaker.on_success()
                        return analysis_result
                    
                    elif response.status == 429:  # Rate limited
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limited, waiting {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    elif response.status in [500, 502, 503, 504]:  # Server errors
                        error_text = await response.text()
                        logger.warning(f"Server error {response.status}: {error_text}")
                        
                        # Try failover if available
                        if self.config.failover_endpoints and attempt < len(self.config.failover_endpoints):
                            await self._failover_to_next_endpoint()
                        
                        raise aiohttp.ClientError(f"Server error: {response.status}")
                    
                    else:  # Other client errors
                        error_text = await response.text()
                        logger.error(f"Client error {response.status}: {error_text}")
                        raise aiohttp.ClientError(f"Client error: {response.status}")
            
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    # Calculate exponential backoff delay
                    delay = min(
                        self.config.base_retry_delay * (2 ** attempt),
                        self.config.max_retry_delay
                    )
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Request failed after {self.config.max_retries + 1} attempts: {e}")
        
        # All retries exhausted
        raise last_exception or RuntimeError("Request failed after all retries")
    
    def _get_current_endpoint(self) -> str:
        """Get current endpoint (primary or failover)"""
        if self.current_endpoint_index == 0:
            return self.config.endpoint
        else:
            failover_index = self.current_endpoint_index - 1
            if failover_index < len(self.config.failover_endpoints):
                return self.config.failover_endpoints[failover_index]['endpoint']
            else:
                # Fallback to primary
                self.current_endpoint_index = 0
                return self.config.endpoint
    
    async def _failover_to_next_endpoint(self):
        """Failover to next available endpoint"""
        old_endpoint = self._get_current_endpoint()
        
        # Try next endpoint
        if self.current_endpoint_index < len(self.config.failover_endpoints):
            self.current_endpoint_index += 1
        else:
            self.current_endpoint_index = 0  # Wrap around to primary
        
        new_endpoint = self._get_current_endpoint()
        
        # Update session with new API key if different
        if self.current_endpoint_index > 0:
            failover_config = self.config.failover_endpoints[self.current_endpoint_index - 1]
            new_api_key = failover_config.get('api_key', self.config.api_key)
            
            if self.session:
                self.session.headers.update({'api-key': new_api_key})
        
        # Record failover event
        failover_event = {
            'timestamp': time.time(),
            'from_endpoint': old_endpoint,
            'to_endpoint': new_endpoint,
            'reason': 'automatic_failover'
        }
        self.failover_history.append(failover_event)
        
        logger.warning(f"Failed over from {old_endpoint} to {new_endpoint}")
    
    async def _update_metrics(self, success: bool, tokens: int):
        """Update performance metrics and cost tracking"""
        self.performance_stats['total_requests'] += 1
        
        if success:
            self.performance_stats['successful_requests'] += 1
            self.cost_tracker.add_usage(tokens)
        else:
            self.performance_stats['failed_requests'] += 1
        
        # Update cost tracking
        if success:
            self.performance_stats['total_cost'] = self.cost_tracker.daily_cost_usd
    
    async def _monitoring_loop(self):
        """Monitoring loop for metrics collection"""
        logger.info("Azure OpenAI monitoring started")
        
        while True:
            try:
                # Collect current metrics
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)
                
                # Check alerts
                await self._check_alerts(metrics)
                
                # Log periodic summary
                if len(self.metrics_history) % 12 == 0:  # Every hour
                    self._log_performance_summary()
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60.0)
    
    def _collect_current_metrics(self) -> MonitoringMetrics:
        """Collect current performance metrics"""
        total_requests = self.performance_stats['total_requests']
        successful_requests = self.performance_stats['successful_requests']
        
        # Calculate rates and averages
        if len(self.metrics_history) > 0:
            time_window = 300.0  # 5 minute window
            recent_requests = total_requests - (self.metrics_history[-1].timestamp if self.metrics_history else 0)
            requests_per_minute = (recent_requests / time_window) * 60.0
        else:
            requests_per_minute = 0.0
        
        error_rate = ((total_requests - successful_requests) / max(total_requests, 1)) * 100.0
        success_rate = (successful_requests / max(total_requests, 1)) * 100.0
        
        # Rate limiter stats
        rate_limiter_stats = self.rate_limiter.get_stats()
        
        return MonitoringMetrics(
            timestamp=time.time(),
            requests_per_minute=requests_per_minute,
            average_latency=0.0,  # Would need to track latencies
            error_rate=error_rate,
            cost_per_hour=self.cost_tracker.daily_cost_usd / 24.0,  # Rough estimate
            active_connections=rate_limiter_stats['active_requests'],
            queue_size=sum(rate_limiter_stats['queue_sizes'].values()),
            success_rate=success_rate,
            failover_events=len(self.failover_history)
        )
    
    async def _check_alerts(self, metrics: MonitoringMetrics):
        """Check for alert conditions"""
        alerts = []
        
        # High error rate
        if metrics.error_rate > 10.0:
            alerts.append(f"High error rate: {metrics.error_rate:.1f}%")
        
        # Circuit breaker open
        if self.circuit_breaker.get_state() == 'OPEN':
            alerts.append("Circuit breaker OPEN - service degraded")
        
        # Cost alerts
        cost_ratio = self.cost_tracker.daily_cost_usd / self.config.max_daily_cost_usd
        if cost_ratio > self.config.cost_alert_threshold:
            alerts.append(f"Cost alert: {cost_ratio:.1%} of daily budget used")
        
        # Large queue size
        if metrics.queue_size > 50:
            alerts.append(f"Large request queue: {metrics.queue_size}")
        
        # Failover events
        recent_failovers = sum(1 for event in self.failover_history 
                             if time.time() - event['timestamp'] < 3600)  # Last hour
        if recent_failovers > 2:
            alerts.append(f"Multiple failovers in last hour: {recent_failovers}")
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"AZURE OPENAI ALERT: {alert}")
    
    def _log_performance_summary(self):
        """Log performance summary"""
        if not self.metrics_history:
            return
        
        recent_metrics = list(self.metrics_history)[-12:]  # Last hour
        
        avg_error_rate = statistics.mean(m.error_rate for m in recent_metrics)
        avg_success_rate = statistics.mean(m.success_rate for m in recent_metrics)
        avg_queue_size = statistics.mean(m.queue_size for m in recent_metrics)
        total_cost = self.cost_tracker.daily_cost_usd
        
        logger.info(f"Azure OpenAI Performance Summary - "
                   f"Success Rate: {avg_success_rate:.1f}%, "
                   f"Error Rate: {avg_error_rate:.1f}%, "
                   f"Queue Size: {avg_queue_size:.1f}, "
                   f"Daily Cost: ${total_cost:.3f}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        start_time = time.time()
        
        try:
            # Simple test request
            test_prompt = "Perform a basic consciousness analysis test."
            
            # Use minimal resources for health check
            test_data = {
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 50,
                "temperature": 0.0
            }
            
            endpoint = self._get_current_endpoint()
            
            async with self.session.post(
                f"{endpoint}/openai/deployments/{self.config.deployment_name}/chat/completions",
                params={"api-version": self.config.api_version},
                json=test_data
            ) as response:
                
                health_check_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    
                    return {
                        'status': 'healthy',
                        'health_check_time': health_check_time,
                        'endpoint': endpoint,
                        'circuit_breaker_state': self.circuit_breaker.get_state(),
                        'cost_status': {
                            'daily_cost_usd': self.cost_tracker.daily_cost_usd,
                            'budget_usage': self.cost_tracker.daily_cost_usd / self.config.max_daily_cost_usd,
                            'over_budget': self.cost_tracker.is_over_budget(self.config.max_daily_cost_usd)
                        },
                        'rate_limiter_stats': self.rate_limiter.get_stats(),
                        'performance_stats': self.performance_stats.copy()
                    }
                else:
                    error_text = await response.text()
                    return {
                        'status': 'unhealthy',
                        'health_check_time': health_check_time,
                        'error': f"HTTP {response.status}: {error_text}",
                        'endpoint': endpoint
                    }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'health_check_time': time.time() - start_time,
                'error': str(e),
                'circuit_breaker_state': self.circuit_breaker.get_state()
            }
    
    def get_cost_report(self) -> Dict[str, Any]:
        """Get comprehensive cost report"""
        return {
            'daily_cost_usd': self.cost_tracker.daily_cost_usd,
            'monthly_cost_usd': self.cost_tracker.monthly_cost_usd,
            'total_requests': self.cost_tracker.total_requests,
            'total_tokens': self.cost_tracker.total_tokens,
            'cost_per_request': self.cost_tracker.daily_cost_usd / max(self.cost_tracker.total_requests, 1),
            'cost_per_token': self.cost_tracker.daily_cost_usd / max(self.cost_tracker.total_tokens, 1),
            'budget_status': {
                'max_daily_cost_usd': self.config.max_daily_cost_usd,
                'remaining_budget': max(0, self.config.max_daily_cost_usd - self.cost_tracker.daily_cost_usd),
                'budget_usage_percent': (self.cost_tracker.daily_cost_usd / self.config.max_daily_cost_usd) * 100
            },
            'projected_monthly_cost': self.cost_tracker.daily_cost_usd * 30  # Rough projection
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        if not self.metrics_history:
            return {'error': 'No metrics available'}
        
        recent_metrics = list(self.metrics_history)[-24:]  # Last 24 data points
        
        return {
            'current_stats': self.performance_stats.copy(),
            'recent_performance': {
                'average_error_rate': statistics.mean(m.error_rate for m in recent_metrics),
                'average_success_rate': statistics.mean(m.success_rate for m in recent_metrics),
                'average_requests_per_minute': statistics.mean(m.requests_per_minute for m in recent_metrics),
                'average_queue_size': statistics.mean(m.queue_size for m in recent_metrics)
            },
            'circuit_breaker': {
                'state': self.circuit_breaker.get_state(),
                'failure_count': self.circuit_breaker.failure_count
            },
            'failover_history': self.failover_history[-10:],  # Last 10 failover events
            'rate_limiter_stats': self.rate_limiter.get_stats()
        }


class AzureOpenAIIntegration:
    """Main integration class combining Azure OpenAI with IIT 4.0 consciousness detection"""
    
    def __init__(self, 
                 azure_config: AzureOpenAIConfig,
                 phi_calculator: ProductionPhiCalculator):
        self.azure_config = azure_config
        self.phi_calculator = phi_calculator
        self.azure_client = AzureOpenAIClient(azure_config)
        
        # Integration metrics
        self.integration_stats = {
            'consciousness_analyses': 0,
            'successful_analyses': 0,
            'phi_calculations_analyzed': 0,
            'average_analysis_time': 0.0
        }
        
        logger.info("AzureOpenAI Integration initialized")
    
    async def start(self):
        """Start the integration"""
        await self.azure_client.start()
        logger.info("AzureOpenAI Integration started")
    
    async def stop(self):
        """Stop the integration"""
        await self.azure_client.stop()
        logger.info("AzureOpenAI Integration stopped")
    
    async def analyze_consciousness_batch(self, 
                                        calculation_requests: List[CalculationRequest],
                                        priority: RequestPriority = RequestPriority.NORMAL) -> Dict[str, Any]:
        """
        Analyze consciousness for a batch of calculations
        
        Args:
            calculation_requests: List of φ calculation requests
            priority: Azure OpenAI request priority
            
        Returns:
            Comprehensive consciousness analysis
        """
        start_time = time.time()
        
        try:
            # Perform φ calculations
            phi_responses = await self.phi_calculator.calculate_phi_batch(calculation_requests)
            
            # Extract successful results
            successful_results = [
                response.result for response in phi_responses 
                if response.status == 'success' and response.result
            ]
            
            if not successful_results:
                return {
                    'status': 'error',
                    'error': 'No successful φ calculations to analyze',
                    'phi_results_count': 0
                }
            
            # Analyze with Azure OpenAI
            azure_analysis = await self.azure_client.analyze_consciousness_data(
                successful_results, priority
            )
            
            # Parse analysis result
            try:
                parsed_analysis = json.loads(azure_analysis['analysis'])
            except json.JSONDecodeError:
                # Fallback if not valid JSON
                parsed_analysis = {'raw_analysis': azure_analysis['analysis']}
            
            # Compile comprehensive result
            result = {
                'status': 'success',
                'analysis_time': time.time() - start_time,
                'phi_results_count': len(successful_results),
                'phi_summary': self._summarize_phi_results(successful_results),
                'azure_analysis': parsed_analysis,
                'usage_stats': azure_analysis.get('usage', {}),
                'azure_latency': azure_analysis.get('latency', 0.0),
                'endpoint_used': azure_analysis.get('endpoint', ''),
                'phi_calculation_times': [r.calculation_time for r in successful_results]
            }
            
            # Update integration stats
            self.integration_stats['consciousness_analyses'] += 1
            self.integration_stats['successful_analyses'] += 1
            self.integration_stats['phi_calculations_analyzed'] += len(successful_results)
            
            # Update average analysis time
            total_analyses = self.integration_stats['consciousness_analyses']
            current_avg = self.integration_stats['average_analysis_time']
            self.integration_stats['average_analysis_time'] = (
                (current_avg * (total_analyses - 1) + result['analysis_time']) / total_analyses
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Consciousness analysis batch error: {e}")
            
            self.integration_stats['consciousness_analyses'] += 1
            
            return {
                'status': 'error',
                'error': str(e),
                'analysis_time': time.time() - start_time,
                'phi_results_count': 0
            }
    
    def _summarize_phi_results(self, phi_results: List[PhiCalculationResult]) -> Dict[str, Any]:
        """Summarize φ calculation results"""
        phi_values = [r.phi_value for r in phi_results if r.phi_value is not None]
        node_counts = [r.node_count for r in phi_results]
        calc_times = [r.calculation_time for r in phi_results]
        
        summary = {
            'total_results': len(phi_results),
            'valid_phi_values': len(phi_values)
        }
        
        if phi_values:
            summary.update({
                'phi_statistics': {
                    'mean': statistics.mean(phi_values),
                    'median': statistics.median(phi_values),
                    'min': min(phi_values),
                    'max': max(phi_values),
                    'std': statistics.stdev(phi_values) if len(phi_values) > 1 else 0.0
                }
            })
        
        if node_counts:
            summary.update({
                'system_statistics': {
                    'average_nodes': statistics.mean(node_counts),
                    'min_nodes': min(node_counts),
                    'max_nodes': max(node_counts)
                }
            })
        
        if calc_times:
            summary.update({
                'performance_statistics': {
                    'average_calc_time': statistics.mean(calc_times),
                    'total_calc_time': sum(calc_times),
                    'min_calc_time': min(calc_times),
                    'max_calc_time': max(calc_times)
                }
            })
        
        return summary
    
    async def real_time_consciousness_monitoring(self, 
                                               calculation_stream: AsyncIterator[CalculationRequest],
                                               analysis_interval: float = 60.0) -> AsyncIterator[Dict[str, Any]]:
        """
        Real-time consciousness monitoring with periodic Azure OpenAI analysis
        
        Args:
            calculation_stream: Stream of φ calculation requests
            analysis_interval: Interval between Azure analyses (seconds)
            
        Yields:
            Consciousness analysis results
        """
        buffer = []
        last_analysis_time = time.time()
        
        async for request in calculation_stream:
            try:
                # Perform φ calculation
                response = await self.phi_calculator.calculate_phi_async(request)
                
                if response.status == 'success' and response.result:
                    buffer.append(response.result)
                
                # Check if it's time for analysis
                current_time = time.time()
                if (current_time - last_analysis_time >= analysis_interval and 
                    len(buffer) > 0):
                    
                    # Analyze accumulated results
                    dummy_requests = [
                        CalculationRequest(
                            request_id=f"analysis_{i}",
                            tpm=np.zeros((2, 2)),  # Dummy TPM
                            state=np.zeros(2)      # Dummy state
                        ) for i in range(len(buffer))
                    ]
                    
                    # Create fake responses with our real results
                    fake_responses = [
                        CalculationResponse(
                            request_id=req.request_id,
                            result=result,
                            status='success'
                        ) for req, result in zip(dummy_requests, buffer)
                    ]
                    
                    # Analyze with Azure OpenAI
                    analysis_result = await self.azure_client.analyze_consciousness_data(
                        buffer, RequestPriority.HIGH
                    )
                    
                    # Yield analysis result
                    yield {
                        'timestamp': current_time,
                        'analysis_result': analysis_result,
                        'phi_results_analyzed': len(buffer),
                        'analysis_interval': current_time - last_analysis_time
                    }
                    
                    # Reset for next interval
                    buffer = []
                    last_analysis_time = current_time
                    
            except Exception as e:
                logger.error(f"Real-time monitoring error: {e}")
                continue
    
    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for the entire integration"""
        start_time = time.time()
        
        health_results = {}
        
        # Check φ calculator health
        try:
            phi_health = await self.phi_calculator.health_check()
            health_results['phi_calculator'] = phi_health
        except Exception as e:
            health_results['phi_calculator'] = {'status': 'unhealthy', 'error': str(e)}
        
        # Check Azure OpenAI health
        try:
            azure_health = await self.azure_client.health_check()
            health_results['azure_openai'] = azure_health
        except Exception as e:
            health_results['azure_openai'] = {'status': 'unhealthy', 'error': str(e)}
        
        # Overall integration health
        phi_healthy = health_results['phi_calculator'].get('status') == 'healthy'
        azure_healthy = health_results['azure_openai'].get('status') == 'healthy'
        
        overall_status = 'healthy' if phi_healthy and azure_healthy else 'degraded'
        if not phi_healthy and not azure_healthy:
            overall_status = 'unhealthy'
        
        return {
            'overall_status': overall_status,
            'health_check_time': time.time() - start_time,
            'components': health_results,
            'integration_stats': self.integration_stats.copy(),
            'cost_report': self.azure_client.get_cost_report()
        }
    
    def get_integration_report(self) -> Dict[str, Any]:
        """Get comprehensive integration report"""
        return {
            'integration_stats': self.integration_stats.copy(),
            'azure_performance': self.azure_client.get_performance_report(),
            'cost_report': self.azure_client.get_cost_report(),
            'phi_calculator_status': self.phi_calculator.get_system_status()
        }


# Example usage and testing
async def test_azure_openai_integration():
    """Test Azure OpenAI integration"""
    logger.info("Testing Azure OpenAI Integration")
    
    # Configure Azure OpenAI (use environment variables in production)
    azure_config = AzureOpenAIConfig(
        endpoint="https://your-resource.openai.azure.com",
        api_key="your-api-key-here",
        deployment_name="gpt-4",
        requests_per_minute=10,  # Conservative for testing
        max_daily_cost_usd=10.0,
        enable_cost_optimization=True
    )
    
    # Create production calculator
    phi_calculator = ProductionPhiCalculator(max_workers=2)
    await phi_calculator.start()
    
    try:
        # Create integration
        integration = AzureOpenAIIntegration(azure_config, phi_calculator)
        await integration.start()
        
        # Test health check
        health = await integration.comprehensive_health_check()
        print(f"Integration health: {health['overall_status']}")
        
        # Test consciousness analysis (with dummy data)
        test_requests = []
        for i in range(3):
            tpm = np.random.rand(8, 3)
            tpm = tpm / np.sum(tpm, axis=1, keepdims=True)
            state = np.array([1, 0, 1])
            
            request = CalculationRequest(
                request_id=f"test_{i}",
                tpm=tpm,
                state=state,
                priority=CalculationPriority.NORMAL
            )
            test_requests.append(request)
        
        # Note: This will fail without valid Azure OpenAI credentials
        try:
            analysis_result = await integration.analyze_consciousness_batch(
                test_requests, RequestPriority.HIGH
            )
            print(f"Analysis result: {analysis_result['status']}")
            if analysis_result['status'] == 'success':
                print(f"φ results analyzed: {analysis_result['phi_results_count']}")
        except Exception as e:
            print(f"Analysis failed (expected without valid credentials): {e}")
        
        # Get reports
        integration_report = integration.get_integration_report()
        print(f"Integration stats: {integration_report['integration_stats']}")
        
        await integration.stop()
        
    finally:
        await phi_calculator.stop()


if __name__ == "__main__":
    # Note: This test requires valid Azure OpenAI credentials
    asyncio.run(test_azure_openai_integration())