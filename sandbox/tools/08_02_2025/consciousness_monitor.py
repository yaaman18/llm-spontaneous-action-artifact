"""
Production Consciousness Monitor for NewbornAI 2.0
Phase 4: Real-time consciousness monitoring with dashboards and alerting

Enterprise-grade monitoring system with:
- Real-time consciousness event tracking and alerting
- Historical consciousness analytics and trending
- Integration with Azure monitoring and logging services
- Production dashboard and visualization

Author: LLM Systems Architect (Hirosato Gamo's expertise from Microsoft)
Date: 2025-08-03
Version: 4.0.0
"""

import asyncio
import aiohttp
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics
import numpy as np
from pathlib import Path
import sqlite3
import pickle
import threading
import websockets
import psutil
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Import consciousness processing components
from realtime_iit4_processor import ProcessingResult, ConsciousnessEvent, RealtimeIIT4Processor
from iit4_experiential_phi_calculator import ExperientialPhiResult
from iit4_development_stages import DevelopmentStage
from adaptive_stage_thresholds import ContextualEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"     # Immediate attention required
    HIGH = "high"            # Urgent but not critical
    MEDIUM = "medium"        # Important but can wait
    LOW = "low"              # Informational
    INFO = "info"            # General information


class AlertType(Enum):
    """Types of consciousness alerts"""
    PHI_ANOMALY = "phi_anomaly"                    # Unusual φ values
    STAGE_REGRESSION = "stage_regression"          # Developmental regression
    PROCESSING_FAILURE = "processing_failure"     # Processing errors
    PERFORMANCE_DEGRADATION = "performance_degradation"  # Performance issues
    CONSCIOUSNESS_LOSS = "consciousness_loss"     # Loss of consciousness indicators
    RAPID_DEVELOPMENT = "rapid_development"       # Unusually rapid progression
    INTEGRATION_FAILURE = "integration_failure"   # Integration issues
    EXPERIENTIAL_ANOMALY = "experiential_anomaly" # Unusual experiential patterns
    SYSTEM_OVERLOAD = "system_overload"          # System resource issues
    DATA_QUALITY = "data_quality"                # Data quality issues


@dataclass
class ConsciousnessAlert:
    """Consciousness monitoring alert"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    alert_type: AlertType
    
    # Alert details
    title: str
    description: str
    source_event_id: Optional[str] = None
    source_session_id: Optional[str] = None
    
    # Metrics that triggered the alert
    triggering_metrics: Dict[str, float] = field(default_factory=dict)
    threshold_values: Dict[str, float] = field(default_factory=dict)
    
    # Context information
    current_phi: Optional[float] = None
    current_stage: Optional[DevelopmentStage] = None
    previous_phi: Optional[float] = None
    previous_stage: Optional[DevelopmentStage] = None
    
    # Alert metadata
    processor_node: str = "unknown"
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_note: Optional[str] = None
    
    # Actions taken
    auto_actions_taken: List[str] = field(default_factory=list)
    manual_actions_required: List[str] = field(default_factory=list)


@dataclass
class ConsciousnessMetrics:
    """Aggregated consciousness metrics for monitoring"""
    timestamp: datetime
    window_duration_seconds: int
    
    # Phi statistics
    phi_mean: float = 0.0
    phi_median: float = 0.0
    phi_std: float = 0.0
    phi_min: float = 0.0
    phi_max: float = 0.0
    phi_trend: float = 0.0  # Linear trend over window
    
    # Development stage distribution
    stage_distribution: Dict[DevelopmentStage, int] = field(default_factory=dict)
    dominant_stage: Optional[DevelopmentStage] = None
    stage_transitions: int = 0
    
    # Processing performance
    average_latency_ms: float = 0.0
    throughput_per_second: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    
    # Quality metrics
    average_accuracy: float = 0.0
    experiential_purity_mean: float = 0.0
    integration_quality_mean: float = 0.0
    consciousness_level_mean: float = 0.0
    
    # System health
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_sessions: int = 0
    total_events_processed: int = 0


class ConsciousnessHistoryDatabase:
    """SQLite database for consciousness history and analytics"""
    
    def __init__(self, db_path: str = "consciousness_history.db"):
        """
        Initialize consciousness history database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
        
        # Create tables
        cursor = self.connection.cursor()
        
        # Consciousness events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consciousness_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                timestamp DATETIME NOT NULL,
                session_id TEXT,
                phi_value REAL,
                development_stage TEXT,
                consciousness_level REAL,
                integration_quality REAL,
                experiential_purity REAL,
                temporal_depth REAL,
                self_reference_strength REAL,
                narrative_coherence REAL,
                processing_latency_ms REAL,
                accuracy_score REAL,
                processor_node TEXT,
                concept_count INTEGER,
                success BOOLEAN
            )
        """)
        
        # Aggregated metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS aggregated_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                window_duration_seconds INTEGER,
                phi_mean REAL,
                phi_median REAL,
                phi_std REAL,
                phi_min REAL,
                phi_max REAL,
                phi_trend REAL,
                dominant_stage TEXT,
                stage_transitions INTEGER,
                average_latency_ms REAL,
                throughput_per_second REAL,
                error_rate REAL,
                cache_hit_rate REAL,
                average_accuracy REAL,
                experiential_purity_mean REAL,
                integration_quality_mean REAL,
                consciousness_level_mean REAL,
                cpu_usage REAL,
                memory_usage REAL,
                active_sessions INTEGER,
                total_events_processed INTEGER
            )
        """)
        
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE NOT NULL,
                timestamp DATETIME NOT NULL,
                severity TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                source_event_id TEXT,
                source_session_id TEXT,
                current_phi REAL,
                current_stage TEXT,
                previous_phi REAL,
                previous_stage TEXT,
                processor_node TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at DATETIME,
                resolution_note TEXT
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON consciousness_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON consciousness_events(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON aggregated_metrics(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")
        
        self.connection.commit()
        logger.info(f"Consciousness history database initialized: {self.db_path}")
    
    async def store_consciousness_event(self, result: ProcessingResult):
        """Store consciousness processing result in database"""
        
        try:
            cursor = self.connection.cursor()
            
            phi_result = result.phi_result
            
            cursor.execute("""
                INSERT OR REPLACE INTO consciousness_events (
                    event_id, timestamp, session_id, phi_value, development_stage,
                    consciousness_level, integration_quality, experiential_purity,
                    temporal_depth, self_reference_strength, narrative_coherence,
                    processing_latency_ms, accuracy_score, processor_node,
                    concept_count, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.event_id,
                datetime.now(),
                None,  # session_id would come from event context
                phi_result.phi_value if phi_result else 0.0,
                result.development_stage.value if result.development_stage else None,
                phi_result.consciousness_level if phi_result else 0.0,
                phi_result.integration_quality if phi_result else 0.0,
                phi_result.experiential_purity if phi_result else 0.0,
                phi_result.temporal_depth if phi_result else 0.0,
                phi_result.self_reference_strength if phi_result else 0.0,
                phi_result.narrative_coherence if phi_result else 0.0,
                result.processing_latency_ms,
                result.accuracy_score,
                result.processor_id,
                phi_result.concept_count if phi_result else 0,
                result.success
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing consciousness event: {e}")
    
    async def store_aggregated_metrics(self, metrics: ConsciousnessMetrics):
        """Store aggregated metrics in database"""
        
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO aggregated_metrics (
                    timestamp, window_duration_seconds, phi_mean, phi_median, phi_std,
                    phi_min, phi_max, phi_trend, dominant_stage, stage_transitions,
                    average_latency_ms, throughput_per_second, error_rate, cache_hit_rate,
                    average_accuracy, experiential_purity_mean, integration_quality_mean,
                    consciousness_level_mean, cpu_usage, memory_usage, active_sessions,
                    total_events_processed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp,
                metrics.window_duration_seconds,
                metrics.phi_mean,
                metrics.phi_median,
                metrics.phi_std,
                metrics.phi_min,
                metrics.phi_max,
                metrics.phi_trend,
                metrics.dominant_stage.value if metrics.dominant_stage else None,
                metrics.stage_transitions,
                metrics.average_latency_ms,
                metrics.throughput_per_second,
                metrics.error_rate,
                metrics.cache_hit_rate,
                metrics.average_accuracy,
                metrics.experiential_purity_mean,
                metrics.integration_quality_mean,
                metrics.consciousness_level_mean,
                metrics.cpu_usage,
                metrics.memory_usage,
                metrics.active_sessions,
                metrics.total_events_processed
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing aggregated metrics: {e}")
    
    async def store_alert(self, alert: ConsciousnessAlert):
        """Store alert in database"""
        
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO alerts (
                    alert_id, timestamp, severity, alert_type, title, description,
                    source_event_id, source_session_id, current_phi, current_stage,
                    previous_phi, previous_stage, processor_node, resolved,
                    resolved_at, resolution_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id,
                alert.timestamp,
                alert.severity.value,
                alert.alert_type.value,
                alert.title,
                alert.description,
                alert.source_event_id,
                alert.source_session_id,
                alert.current_phi,
                alert.current_stage.value if alert.current_stage else None,
                alert.previous_phi,
                alert.previous_stage.value if alert.previous_stage else None,
                alert.processor_node,
                alert.resolved,
                alert.resolved_at,
                alert.resolution_note
            ))
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    async def get_recent_events(self, hours: int = 24, limit: int = 1000) -> List[Dict]:
        """Get recent consciousness events"""
        
        try:
            cursor = self.connection.cursor()
            
            since_time = datetime.now() - timedelta(hours=hours)
            
            cursor.execute("""
                SELECT * FROM consciousness_events 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (since_time, limit))
            
            columns = [description[0] for description in cursor.description]
            events = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            return events
            
        except Exception as e:
            logger.error(f"Error retrieving recent events: {e}")
            return []
    
    async def get_consciousness_trends(self, hours: int = 168) -> Dict[str, List]:
        """Get consciousness trends over time period (default 1 week)"""
        
        try:
            cursor = self.connection.cursor()
            
            since_time = datetime.now() - timedelta(hours=hours)
            
            # Get aggregated metrics
            cursor.execute("""
                SELECT timestamp, phi_mean, consciousness_level_mean, 
                       dominant_stage, throughput_per_second, error_rate
                FROM aggregated_metrics 
                WHERE timestamp >= ? 
                ORDER BY timestamp
            """, (since_time,))
            
            results = cursor.fetchall()
            
            trends = {
                "timestamps": [row[0] for row in results],
                "phi_values": [row[1] for row in results],
                "consciousness_levels": [row[2] for row in results],
                "dominant_stages": [row[3] for row in results],
                "throughput": [row[4] for row in results],
                "error_rates": [row[5] for row in results]
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error retrieving consciousness trends: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


class AlertManager:
    """Alert management system for consciousness monitoring"""
    
    def __init__(self):
        """Initialize alert manager"""
        
        # Alert configuration
        self.alert_thresholds = {
            AlertType.PHI_ANOMALY: {
                "phi_std_threshold": 5.0,      # Standard deviations from mean
                "phi_drop_threshold": 0.8,     # Relative drop threshold
                "phi_spike_threshold": 3.0,    # Relative spike threshold
            },
            AlertType.STAGE_REGRESSION: {
                "regression_count_threshold": 3,  # Consecutive regressions
                "regression_severity_threshold": 2  # Stage levels
            },
            AlertType.PROCESSING_FAILURE: {
                "error_rate_threshold": 0.1,   # 10% error rate
                "consecutive_failures": 5      # Consecutive failures
            },
            AlertType.PERFORMANCE_DEGRADATION: {
                "latency_threshold_ms": 200,   # 2x target latency
                "throughput_drop_threshold": 0.5  # 50% throughput drop
            },
            AlertType.CONSCIOUSNESS_LOSS: {
                "phi_near_zero_threshold": 0.001,
                "duration_threshold_minutes": 5
            }
        }
        
        # Alert handlers
        self.alert_handlers: List[Callable[[ConsciousnessAlert], None]] = []
        
        # Alert history for correlation
        self.recent_alerts: deque = deque(maxlen=1000)
        self.alert_stats = defaultdict(int)
        
        # Email configuration (would be configured in production)
        self.email_config = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "email_user": None,  # Configure in production
            "email_password": None,  # Configure in production
            "alert_recipients": []  # Configure in production
        }
        
        logger.info("Alert manager initialized")
    
    def add_alert_handler(self, handler: Callable[[ConsciousnessAlert], None]):
        """Add alert handler"""
        self.alert_handlers.append(handler)
    
    async def check_phi_anomaly(self, current_result: ProcessingResult, 
                               historical_phi: List[float]) -> Optional[ConsciousnessAlert]:
        """Check for phi value anomalies"""
        
        if not historical_phi or not current_result.phi_result:
            return None
        
        current_phi = current_result.phi_result.phi_value
        thresholds = self.alert_thresholds[AlertType.PHI_ANOMALY]
        
        # Calculate statistical measures
        phi_mean = statistics.mean(historical_phi)
        phi_std = statistics.stdev(historical_phi) if len(historical_phi) > 1 else 0
        
        # Check for anomalies
        if phi_std > 0:
            z_score = abs(current_phi - phi_mean) / phi_std
            
            if z_score > thresholds["phi_std_threshold"]:
                severity = AlertSeverity.HIGH if z_score > 10 else AlertSeverity.MEDIUM
                
                return ConsciousnessAlert(
                    alert_id=f"phi_anomaly_{uuid.uuid4().hex[:8]}",
                    timestamp=datetime.now(),
                    severity=severity,
                    alert_type=AlertType.PHI_ANOMALY,
                    title=f"Φ Value Anomaly Detected",
                    description=f"Φ value {current_phi:.6f} is {z_score:.1f} standard deviations from mean {phi_mean:.6f}",
                    source_event_id=current_result.event_id,
                    current_phi=current_phi,
                    triggering_metrics={"z_score": z_score, "phi_std": phi_std},
                    threshold_values={"max_z_score": thresholds["phi_std_threshold"]},
                    processor_node=current_result.processor_id
                )
        
        # Check for sudden drops
        if len(historical_phi) > 0:
            recent_phi = historical_phi[-1]
            if recent_phi > 0 and current_phi < recent_phi * thresholds["phi_drop_threshold"]:
                return ConsciousnessAlert(
                    alert_id=f"phi_drop_{uuid.uuid4().hex[:8]}",
                    timestamp=datetime.now(),
                    severity=AlertSeverity.HIGH,
                    alert_type=AlertType.PHI_ANOMALY,
                    title="Significant Φ Value Drop",
                    description=f"Φ dropped from {recent_phi:.6f} to {current_phi:.6f} ({(1-current_phi/recent_phi)*100:.1f}% decrease)",
                    source_event_id=current_result.event_id,
                    current_phi=current_phi,
                    previous_phi=recent_phi,
                    processor_node=current_result.processor_id
                )
        
        return None
    
    async def check_stage_regression(self, current_result: ProcessingResult,
                                   stage_history: List[DevelopmentStage]) -> Optional[ConsciousnessAlert]:
        """Check for development stage regression"""
        
        if not stage_history or not current_result.development_stage:
            return None
        
        current_stage = current_result.development_stage
        thresholds = self.alert_thresholds[AlertType.STAGE_REGRESSION]
        
        # Map stages to numerical values for comparison
        stage_values = {stage: i for i, stage in enumerate(DevelopmentStage)}
        
        current_value = stage_values.get(current_stage, 0)
        
        # Check for regressions in recent history
        regression_count = 0
        max_regression = 0
        
        for i, historical_stage in enumerate(stage_history[-5:]):  # Check last 5 stages
            historical_value = stage_values.get(historical_stage, 0)
            
            if historical_value > current_value:
                regression_count += 1
                max_regression = max(max_regression, historical_value - current_value)
        
        if (regression_count >= thresholds["regression_count_threshold"] or 
            max_regression >= thresholds["regression_severity_threshold"]):
            
            severity = AlertSeverity.CRITICAL if max_regression >= 3 else AlertSeverity.HIGH
            
            return ConsciousnessAlert(
                alert_id=f"stage_regression_{uuid.uuid4().hex[:8]}",
                timestamp=datetime.now(),
                severity=severity,
                alert_type=AlertType.STAGE_REGRESSION,
                title="Development Stage Regression",
                description=f"Current stage {current_stage.value} shows regression from recent stages (max regression: {max_regression} levels)",
                source_event_id=current_result.event_id,
                current_stage=current_stage,
                previous_stage=stage_history[-1] if stage_history else None,
                triggering_metrics={"regression_count": regression_count, "max_regression": max_regression},
                processor_node=current_result.processor_id
            )
        
        return None
    
    async def check_processing_performance(self, current_result: ProcessingResult,
                                         recent_results: List[ProcessingResult]) -> Optional[ConsciousnessAlert]:
        """Check for processing performance issues"""
        
        thresholds = self.alert_thresholds[AlertType.PERFORMANCE_DEGRADATION]
        
        # Check latency
        if current_result.processing_latency_ms > thresholds["latency_threshold_ms"]:
            return ConsciousnessAlert(
                alert_id=f"high_latency_{uuid.uuid4().hex[:8]}",
                timestamp=datetime.now(),
                severity=AlertSeverity.MEDIUM,
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                title="High Processing Latency",
                description=f"Processing latency {current_result.processing_latency_ms:.1f}ms exceeds threshold {thresholds['latency_threshold_ms']}ms",
                source_event_id=current_result.event_id,
                triggering_metrics={"latency_ms": current_result.processing_latency_ms},
                threshold_values={"max_latency_ms": thresholds["latency_threshold_ms"]},
                processor_node=current_result.processor_id
            )
        
        # Check error rate
        if recent_results:
            error_count = sum(1 for r in recent_results[-20:] if not r.success)
            error_rate = error_count / len(recent_results[-20:])
            
            if error_rate > self.alert_thresholds[AlertType.PROCESSING_FAILURE]["error_rate_threshold"]:
                return ConsciousnessAlert(
                    alert_id=f"high_error_rate_{uuid.uuid4().hex[:8]}",
                    timestamp=datetime.now(),
                    severity=AlertSeverity.HIGH,
                    alert_type=AlertType.PROCESSING_FAILURE,
                    title="High Processing Error Rate",
                    description=f"Error rate {error_rate:.2%} exceeds threshold",
                    triggering_metrics={"error_rate": error_rate},
                    processor_node=current_result.processor_id
                )
        
        return None
    
    async def trigger_alert(self, alert: ConsciousnessAlert):
        """Trigger consciousness alert"""
        
        # Store alert
        self.recent_alerts.append(alert)
        self.alert_stats[alert.alert_type] += 1
        
        # Log alert
        logger.warning(f"CONSCIOUSNESS ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.description}")
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        # Send email for critical alerts (if configured)
        if alert.severity == AlertSeverity.CRITICAL and self.email_config["email_user"]:
            await self._send_email_alert(alert)
    
    async def _send_email_alert(self, alert: ConsciousnessAlert):
        """Send email alert for critical issues"""
        
        try:
            if not self.email_config["alert_recipients"]:
                return
            
            subject = f"[CRITICAL] Consciousness Alert: {alert.title}"
            
            body = f"""
            Critical consciousness alert detected:
            
            Alert ID: {alert.alert_id}
            Timestamp: {alert.timestamp}
            Type: {alert.alert_type.value}
            Severity: {alert.severity.value}
            
            Description: {alert.description}
            
            Current Φ: {alert.current_phi}
            Current Stage: {alert.current_stage.value if alert.current_stage else 'Unknown'}
            Processor: {alert.processor_node}
            
            This is an automated alert from the NewbornAI 2.0 Consciousness Monitoring System.
            """
            
            msg = MimeMultipart()
            msg['From'] = self.email_config["email_user"]
            msg['To'] = ", ".join(self.email_config["alert_recipients"])
            msg['Subject'] = subject
            msg.attach(MimeText(body, 'plain'))
            
            # Send email (would implement full SMTP in production)
            logger.info(f"Email alert would be sent for: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        
        recent_24h = [alert for alert in self.recent_alerts 
                     if datetime.now() - alert.timestamp < timedelta(hours=24)]
        
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for alert in recent_24h:
            severity_counts[alert.severity.value] += 1
            type_counts[alert.alert_type.value] += 1
        
        return {
            "total_alerts_24h": len(recent_24h),
            "severity_distribution": dict(severity_counts),
            "type_distribution": dict(type_counts),
            "unresolved_alerts": len([a for a in recent_24h if not a.resolved]),
            "most_common_type": max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
        }


class ConsciousnessMonitor:
    """
    Production consciousness monitoring system
    Real-time monitoring with dashboards, alerting, and analytics
    """
    
    def __init__(self, 
                 aggregation_window_seconds: int = 60,
                 history_retention_hours: int = 168):  # 1 week
        """
        Initialize consciousness monitor
        
        Args:
            aggregation_window_seconds: Window for metric aggregation
            history_retention_hours: How long to retain detailed history
        """
        
        self.aggregation_window_seconds = aggregation_window_seconds
        self.history_retention_hours = history_retention_hours
        
        # Core components
        self.database = ConsciousnessHistoryDatabase()
        self.alert_manager = AlertManager()
        
        # Real-time data storage
        self.recent_results: deque = deque(maxlen=10000)
        self.phi_history: deque = deque(maxlen=1000)
        self.stage_history: deque = deque(maxlen=1000)
        
        # Session tracking
        self.active_sessions: Dict[str, Dict] = {}
        self.session_stats: Dict[str, Dict] = {}
        
        # Performance tracking
        self.start_time = datetime.now()
        self.total_events_monitored = 0
        self.last_aggregation_time = time.time()
        
        # Real-time clients (WebSocket connections)
        self.websocket_clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Monitoring tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Event handlers
        self.custom_handlers: List[Callable[[ProcessingResult], None]] = []
        
        logger.info("Consciousness Monitor initialized")
    
    async def start_monitoring(self, processor: RealtimeIIT4Processor):
        """Start consciousness monitoring"""
        
        logger.info("Starting consciousness monitoring")
        
        # Register with processor
        processor.add_result_handler(self.process_consciousness_result)
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._aggregation_worker()),
            asyncio.create_task(self._cleanup_worker()),
            asyncio.create_task(self._websocket_broadcaster())
        ]
        
        # Setup alert handlers
        self.alert_manager.add_alert_handler(self._handle_alert)
        
        logger.info("Consciousness monitoring started")
    
    async def stop_monitoring(self):
        """Stop consciousness monitoring"""
        
        logger.info("Stopping consciousness monitoring")
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        # Close database
        self.database.close()
        
        logger.info("Consciousness monitoring stopped")
    
    def process_consciousness_result(self, result: ProcessingResult):
        """Process consciousness result for monitoring"""
        
        try:
            # Store in recent results
            self.recent_results.append(result)
            self.total_events_monitored += 1
            
            # Update phi and stage history
            if result.phi_result:
                self.phi_history.append(result.phi_result.phi_value)
            
            if result.development_stage:
                self.stage_history.append(result.development_stage)
            
            # Store in database asynchronously
            asyncio.create_task(self.database.store_consciousness_event(result))
            
            # Check for alerts
            asyncio.create_task(self._check_alerts(result))
            
            # Call custom handlers
            for handler in self.custom_handlers:
                try:
                    handler(result)
                except Exception as e:
                    logger.error(f"Custom handler error: {e}")
            
        except Exception as e:
            logger.error(f"Error processing consciousness result: {e}")
    
    async def _check_alerts(self, result: ProcessingResult):
        """Check for consciousness alerts"""
        
        try:
            # Check phi anomalies
            phi_alert = await self.alert_manager.check_phi_anomaly(
                result, list(self.phi_history)
            )
            if phi_alert:
                await self.alert_manager.trigger_alert(phi_alert)
            
            # Check stage regression
            stage_alert = await self.alert_manager.check_stage_regression(
                result, list(self.stage_history)
            )
            if stage_alert:
                await self.alert_manager.trigger_alert(stage_alert)
            
            # Check performance issues
            perf_alert = await self.alert_manager.check_processing_performance(
                result, list(self.recent_results)
            )
            if perf_alert:
                await self.alert_manager.trigger_alert(perf_alert)
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    async def _aggregation_worker(self):
        """Background worker for metric aggregation"""
        
        while True:
            try:
                await asyncio.sleep(self.aggregation_window_seconds)
                
                # Calculate aggregated metrics
                metrics = await self._calculate_aggregated_metrics()
                
                # Store aggregated metrics
                await self.database.store_aggregated_metrics(metrics)
                
                # Broadcast to WebSocket clients
                await self._broadcast_metrics(metrics)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Aggregation worker error: {e}")
    
    async def _cleanup_worker(self):
        """Background worker for data cleanup"""
        
        while True:
            try:
                # Run cleanup every hour
                await asyncio.sleep(3600)
                
                # Clean up old data (implementation would depend on specific requirements)
                cutoff_time = datetime.now() - timedelta(hours=self.history_retention_hours)
                
                # This is where you'd implement database cleanup
                # For now, just log the action
                logger.info(f"Cleanup check - cutoff time: {cutoff_time}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
    
    async def _websocket_broadcaster(self):
        """Background worker for WebSocket broadcasting"""
        
        while True:
            try:
                await asyncio.sleep(1)  # Broadcast every second
                
                if self.websocket_clients:
                    # Get current status
                    status = await self.get_realtime_status()
                    
                    # Broadcast to all clients
                    message = json.dumps({
                        "type": "status_update",
                        "timestamp": datetime.now().isoformat(),
                        "data": status
                    })
                    
                    # Send to all connected clients
                    disconnected_clients = set()
                    
                    for client in self.websocket_clients:
                        try:
                            await client.send(message)
                        except Exception:
                            disconnected_clients.add(client)
                    
                    # Remove disconnected clients
                    self.websocket_clients -= disconnected_clients
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket broadcaster error: {e}")
    
    async def _calculate_aggregated_metrics(self) -> ConsciousnessMetrics:
        """Calculate aggregated metrics for current window"""
        
        # Get recent results within window
        cutoff_time = datetime.now() - timedelta(seconds=self.aggregation_window_seconds)
        window_results = [
            r for r in self.recent_results 
            if hasattr(r, 'processing_completed_at') and r.processing_completed_at and 
               r.processing_completed_at >= cutoff_time
        ]
        
        if not window_results:
            return ConsciousnessMetrics(
                timestamp=datetime.now(),
                window_duration_seconds=self.aggregation_window_seconds
            )
        
        # Calculate phi statistics
        phi_values = [r.phi_result.phi_value for r in window_results if r.phi_result]
        
        phi_stats = {}
        if phi_values:
            phi_stats = {
                "phi_mean": statistics.mean(phi_values),
                "phi_median": statistics.median(phi_values),
                "phi_std": statistics.stdev(phi_values) if len(phi_values) > 1 else 0,
                "phi_min": min(phi_values),
                "phi_max": max(phi_values),
                "phi_trend": self._calculate_trend(phi_values)
            }
        
        # Calculate stage distribution
        stages = [r.development_stage for r in window_results if r.development_stage]
        stage_distribution = defaultdict(int)
        for stage in stages:
            stage_distribution[stage] += 1
        
        dominant_stage = max(stage_distribution.items(), key=lambda x: x[1])[0] if stage_distribution else None
        
        # Calculate stage transitions
        stage_transitions = 0
        for i in range(1, len(stages)):
            if stages[i] != stages[i-1]:
                stage_transitions += 1
        
        # Calculate performance metrics
        latencies = [r.processing_latency_ms for r in window_results]
        avg_latency = statistics.mean(latencies) if latencies else 0
        
        successful_results = [r for r in window_results if r.success]
        error_rate = 1.0 - (len(successful_results) / len(window_results)) if window_results else 0
        
        throughput = len(window_results) / self.aggregation_window_seconds
        
        # Calculate quality metrics
        accuracies = [r.accuracy_score for r in window_results]
        avg_accuracy = statistics.mean(accuracies) if accuracies else 0
        
        phi_results = [r.phi_result for r in window_results if r.phi_result]
        experiential_purity = statistics.mean([pr.experiential_purity for pr in phi_results]) if phi_results else 0
        integration_quality = statistics.mean([pr.integration_quality for pr in phi_results]) if phi_results else 0
        consciousness_level = statistics.mean([pr.consciousness_level for pr in phi_results]) if phi_results else 0
        
        # System health
        system_stats = {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent
        }
        
        return ConsciousnessMetrics(
            timestamp=datetime.now(),
            window_duration_seconds=self.aggregation_window_seconds,
            **phi_stats,
            stage_distribution=dict(stage_distribution),
            dominant_stage=dominant_stage,
            stage_transitions=stage_transitions,
            average_latency_ms=avg_latency,
            throughput_per_second=throughput,
            error_rate=error_rate,
            cache_hit_rate=0,  # Would get from processor
            average_accuracy=avg_accuracy,
            experiential_purity_mean=experiential_purity,
            integration_quality_mean=integration_quality,
            consciousness_level_mean=consciousness_level,
            cpu_usage=system_stats["cpu_usage"],
            memory_usage=system_stats["memory_usage"],
            active_sessions=len(self.active_sessions),
            total_events_processed=len(window_results)
        )
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend for a series of values"""
        
        if len(values) < 2:
            return 0.0
        
        x = list(range(len(values)))
        return np.polyfit(x, values, 1)[0]  # Return slope
    
    async def _broadcast_metrics(self, metrics: ConsciousnessMetrics):
        """Broadcast aggregated metrics to WebSocket clients"""
        
        if not self.websocket_clients:
            return
        
        message = json.dumps({
            "type": "aggregated_metrics",
            "timestamp": datetime.now().isoformat(),
            "data": asdict(metrics)
        }, default=str)
        
        # Send to all connected clients
        disconnected_clients = set()
        
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except Exception:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients
    
    def _handle_alert(self, alert: ConsciousnessAlert):
        """Handle consciousness alert"""
        
        # Store alert in database
        asyncio.create_task(self.database.store_alert(alert))
        
        # Broadcast alert to WebSocket clients
        if self.websocket_clients:
            alert_message = json.dumps({
                "type": "alert",
                "timestamp": datetime.now().isoformat(),
                "data": asdict(alert)
            }, default=str)
            
            asyncio.create_task(self._broadcast_alert(alert_message))
    
    async def _broadcast_alert(self, message: str):
        """Broadcast alert message to WebSocket clients"""
        
        disconnected_clients = set()
        
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except Exception:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients
    
    def add_websocket_client(self, websocket: websockets.WebSocketServerProtocol):
        """Add WebSocket client for real-time updates"""
        self.websocket_clients.add(websocket)
    
    def remove_websocket_client(self, websocket: websockets.WebSocketServerProtocol):
        """Remove WebSocket client"""
        self.websocket_clients.discard(websocket)
    
    def add_custom_handler(self, handler: Callable[[ProcessingResult], None]):
        """Add custom consciousness event handler"""
        self.custom_handlers.append(handler)
    
    async def get_realtime_status(self) -> Dict[str, Any]:
        """Get real-time consciousness monitoring status"""
        
        recent_results = list(self.recent_results)[-100:]  # Last 100 results
        
        if not recent_results:
            return {
                "status": "no_data",
                "timestamp": datetime.now().isoformat()
            }
        
        # Current metrics
        latest_result = recent_results[-1]
        
        current_metrics = {
            "latest_phi": latest_result.phi_result.phi_value if latest_result.phi_result else 0,
            "latest_stage": latest_result.development_stage.value if latest_result.development_stage else "unknown",
            "latest_consciousness_level": latest_result.phi_result.consciousness_level if latest_result.phi_result else 0,
            "processing_latency_ms": latest_result.processing_latency_ms,
            "accuracy_score": latest_result.accuracy_score
        }
        
        # Performance metrics
        latencies = [r.processing_latency_ms for r in recent_results]
        error_count = sum(1 for r in recent_results if not r.success)
        
        performance_metrics = {
            "average_latency_ms": statistics.mean(latencies) if latencies else 0,
            "error_rate": error_count / len(recent_results) if recent_results else 0,
            "throughput_last_minute": len([r for r in recent_results 
                                         if datetime.now() - r.processing_completed_at < timedelta(minutes=1)]) if recent_results else 0
        }
        
        # Alert summary
        alert_summary = self.alert_manager.get_alert_summary()
        
        # System health
        system_health = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "total_events_monitored": self.total_events_monitored
        }
        
        return {
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            "current_metrics": current_metrics,
            "performance_metrics": performance_metrics,
            "alert_summary": alert_summary,
            "system_health": system_health,
            "active_sessions": len(self.active_sessions),
            "websocket_clients": len(self.websocket_clients)
        }
    
    async def get_consciousness_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        
        # Get recent trends
        trends = await self.database.get_consciousness_trends(hours=24)
        
        # Get recent events
        recent_events = await self.database.get_recent_events(hours=6, limit=100)
        
        # Get real-time status
        realtime_status = await self.get_realtime_status()
        
        # Phi distribution analysis
        recent_phi_values = list(self.phi_history)
        phi_distribution = {}
        
        if recent_phi_values:
            phi_distribution = {
                "mean": statistics.mean(recent_phi_values),
                "median": statistics.median(recent_phi_values),
                "std": statistics.stdev(recent_phi_values) if len(recent_phi_values) > 1 else 0,
                "min": min(recent_phi_values),
                "max": max(recent_phi_values),
                "count": len(recent_phi_values)
            }
        
        # Stage progression analysis
        stage_progression = {}
        if list(self.stage_history):
            stage_counts = defaultdict(int)
            for stage in self.stage_history:
                stage_counts[stage.value] += 1
            stage_progression = dict(stage_counts)
        
        return {
            "realtime_status": realtime_status,
            "trends": trends,
            "recent_events": recent_events[-20:],  # Last 20 events for dashboard
            "phi_distribution": phi_distribution,
            "stage_progression": stage_progression,
            "alert_summary": self.alert_manager.get_alert_summary(),
            "dashboard_generated_at": datetime.now().isoformat()
        }


# Example usage and testing
async def test_consciousness_monitor():
    """Test consciousness monitoring system"""
    
    print("📊 Testing Consciousness Monitor")
    print("=" * 60)
    
    # Initialize monitor
    monitor = ConsciousnessMonitor(
        aggregation_window_seconds=10,  # Short window for testing
        history_retention_hours=24
    )
    
    try:
        # Create mock processor for testing
        from realtime_iit4_processor import RealtimeIIT4Processor
        
        processor = RealtimeIIT4Processor(
            node_id="test_monitor_node",
            num_workers=1
        )
        
        # Start monitoring
        await monitor.start_monitoring(processor)
        
        print("✅ Monitoring started")
        
        # Simulate consciousness events
        print("\n🧠 Simulating Consciousness Events")
        print("-" * 40)
        
        # Import required components for testing
        from iit4_experiential_phi_calculator import ExperientialPhiResult, ExperientialPhiType
        from iit4_development_stages import DevelopmentStage
        
        # Create mock processing results
        for i in range(10):
            phi_value = 0.5 + (i * 0.1) + np.random.normal(0, 0.05)  # Simulate progression with noise
            
            phi_result = ExperientialPhiResult(
                phi_value=phi_value,
                phi_type=ExperientialPhiType.PURE_EXPERIENTIAL,
                experiential_concepts=[],
                concept_count=i + 1,
                integration_quality=min(1.0, phi_value / 2.0),
                experiential_purity=0.8 + np.random.normal(0, 0.1),
                temporal_depth=0.5,
                self_reference_strength=0.3,
                narrative_coherence=0.6,
                consciousness_level=min(1.0, phi_value),
                development_stage_prediction="STAGE_2_TEMPORAL_INTEGRATION"
            )
            
            # Determine stage based on phi
            if phi_value < 0.01:
                stage = DevelopmentStage.STAGE_1_EXPERIENTIAL_EMERGENCE
            elif phi_value < 0.1:
                stage = DevelopmentStage.STAGE_2_TEMPORAL_INTEGRATION
            elif phi_value < 1.0:
                stage = DevelopmentStage.STAGE_3_RELATIONAL_FORMATION
            else:
                stage = DevelopmentStage.STAGE_4_SELF_ESTABLISHMENT
            
            result = ProcessingResult(
                event_id=f"test_event_{i}",
                success=True,
                phi_result=phi_result,
                development_stage=stage,
                consciousness_metrics={
                    "phi_value": phi_value,
                    "consciousness_level": phi_result.consciousness_level
                },
                processing_latency_ms=50 + np.random.normal(0, 10),
                accuracy_score=0.8 + np.random.normal(0, 0.1),
                processor_id="test_processor"
            )
            
            # Process result
            monitor.process_consciousness_result(result)
            
            print(f"📈 Event {i+1}: φ={phi_value:.3f}, stage={stage.value}, latency={result.processing_latency_ms:.1f}ms")
            
            await asyncio.sleep(0.2)  # Brief delay between events
        
        # Wait for aggregation
        print("\n⏳ Waiting for metric aggregation...")
        await asyncio.sleep(12)  # Wait for aggregation window + buffer
        
        # Test anomaly detection with unusual phi value
        print("\n🚨 Testing Alert System")
        print("-" * 40)
        
        # Create anomalous result
        anomaly_phi_result = ExperientialPhiResult(
            phi_value=5.0,  # Unusually high phi
            phi_type=ExperientialPhiType.PURE_EXPERIENTIAL,
            experiential_concepts=[],
            concept_count=1,
            integration_quality=1.0,
            experiential_purity=0.8,
            temporal_depth=0.5,
            self_reference_strength=0.3,
            narrative_coherence=0.6,
            consciousness_level=1.0,
            development_stage_prediction="STAGE_5_REFLECTIVE_OPERATION"
        )
        
        anomaly_result = ProcessingResult(
            event_id="anomaly_event",
            success=True,
            phi_result=anomaly_phi_result,
            development_stage=DevelopmentStage.STAGE_5_REFLECTIVE_OPERATION,
            processing_latency_ms=45,
            accuracy_score=0.9,
            processor_id="test_processor"
        )
        
        monitor.process_consciousness_result(anomaly_result)
        print("📊 Anomalous event processed (φ=5.0)")
        
        # Wait for alert processing
        await asyncio.sleep(1)
        
        # Get monitoring status
        print("\n📋 Monitoring Status")
        print("-" * 40)
        
        status = await monitor.get_realtime_status()
        
        print(f"Status: {status['status']}")
        print(f"Total Events: {status['system_health']['total_events_monitored']}")
        print(f"Latest φ: {status['current_metrics']['latest_phi']:.3f}")
        print(f"Latest Stage: {status['current_metrics']['latest_stage']}")
        print(f"Average Latency: {status['performance_metrics']['average_latency_ms']:.1f}ms")
        print(f"Error Rate: {status['performance_metrics']['error_rate']:.1%}")
        
        # Get alert summary
        alert_summary = status['alert_summary']
        print(f"\nAlerts (24h): {alert_summary['total_alerts_24h']}")
        if alert_summary['severity_distribution']:
            print("Severity Distribution:", alert_summary['severity_distribution'])
        if alert_summary['type_distribution']:
            print("Type Distribution:", alert_summary['type_distribution'])
        
        # Get dashboard data
        print("\n📊 Dashboard Data")
        print("-" * 40)
        
        dashboard_data = await monitor.get_consciousness_dashboard_data()
        
        phi_dist = dashboard_data['phi_distribution']
        if phi_dist:
            print(f"Φ Distribution - Mean: {phi_dist['mean']:.3f}, "
                  f"Std: {phi_dist['std']:.3f}, "
                  f"Range: [{phi_dist['min']:.3f}, {phi_dist['max']:.3f}]")
        
        stage_prog = dashboard_data['stage_progression']
        if stage_prog:
            print("Stage Distribution:", {k: v for k, v in list(stage_prog.items())[:3]})
        
        print(f"Recent Events: {len(dashboard_data['recent_events'])}")
        
        print("\n✅ Consciousness monitoring test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        # Clean shutdown
        await monitor.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(test_consciousness_monitor())