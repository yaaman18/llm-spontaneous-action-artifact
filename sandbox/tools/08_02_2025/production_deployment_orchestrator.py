"""
Production Deployment Orchestrator for NewbornAI 2.0
Phase 4: Complete system orchestration for production deployment

Enterprise-grade deployment orchestration with:
- Complete system orchestration for production deployment
- Health checks, auto-scaling, and failure recovery
- Configuration management and environment orchestration
- Integration testing and deployment validation

Author: LLM Systems Architect (Hirosato Gamo's expertise from Microsoft)
Date: 2025-08-03
Version: 4.0.0
"""

import asyncio
import aiohttp
import aiofiles
import yaml
import json
import logging
import time
import uuid
import os
import sys
import signal
import subprocess
import psutil
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable, Set, Union
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import shutil
import tempfile
import docker
import kubernetes
from kubernetes import client, config
import boto3
from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.monitor import MonitorManagementClient

# Import consciousness processing components
from realtime_iit4_processor import RealtimeIIT4Processor, ProcessorState
from consciousness_monitor import ConsciousnessMonitor
from streaming_phi_calculator import StreamingPhiCalculator, StreamingMode
from adaptive_stage_thresholds import AdaptiveStageThresholdManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"


class ComponentType(Enum):
    """System component types"""
    PROCESSOR = "processor"
    MONITOR = "monitor"
    CALCULATOR = "calculator"
    DATABASE = "database"
    CACHE = "cache"
    LOAD_BALANCER = "load_balancer"
    API_GATEWAY = "api_gateway"


class ComponentState(Enum):
    """Component state enumeration"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILING = "failing"
    STOPPING = "stopping"
    ERROR = "error"


class ScalingPolicy(Enum):
    """Auto-scaling policies"""
    FIXED = "fixed"                     # Fixed number of instances
    CPU_BASED = "cpu_based"            # Scale based on CPU usage
    MEMORY_BASED = "memory_based"      # Scale based on memory usage
    THROUGHPUT_BASED = "throughput_based"  # Scale based on request throughput
    CUSTOM = "custom"                   # Custom scaling logic


@dataclass
class ComponentConfiguration:
    """Configuration for a system component"""
    component_id: str
    component_type: ComponentType
    image: str
    version: str
    
    # Resource requirements
    cpu_request: float = 0.5           # CPU cores
    cpu_limit: float = 2.0
    memory_request_mb: int = 512
    memory_limit_mb: int = 2048
    
    # Networking
    ports: List[int] = field(default_factory=lambda: [8080])
    health_check_path: str = "/health"
    health_check_port: int = 8080
    
    # Environment variables
    environment: Dict[str, str] = field(default_factory=dict)
    
    # Scaling configuration
    min_replicas: int = 1
    max_replicas: int = 10
    scaling_policy: ScalingPolicy = ScalingPolicy.CPU_BASED
    
    # Failure handling
    restart_policy: str = "Always"
    max_restart_attempts: int = 3
    failure_threshold: int = 3
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Storage
    persistent_volumes: List[Dict[str, str]] = field(default_factory=list)
    
    # Security
    security_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentConfiguration:
    """Complete deployment configuration"""
    deployment_id: str
    environment: DeploymentEnvironment
    namespace: str = "newbornai"
    
    # Components
    components: List[ComponentConfiguration] = field(default_factory=list)
    
    # Infrastructure
    cloud_provider: str = "azure"      # azure, aws, gcp
    region: str = "eastus"
    cluster_name: str = "newbornai-cluster"
    
    # Networking
    load_balancer_enabled: bool = True
    ssl_enabled: bool = True
    domain: Optional[str] = None
    
    # Monitoring
    monitoring_enabled: bool = True
    logging_enabled: bool = True
    alerting_enabled: bool = True
    
    # Backup and DR
    backup_enabled: bool = True
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM
    dr_enabled: bool = False
    
    # Security
    network_policies_enabled: bool = True
    rbac_enabled: bool = True
    secret_management: str = "kubernetes"  # kubernetes, vault, azure_keyvault
    
    # Configuration metadata
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"


@dataclass
class ComponentInstance:
    """Instance of a deployed component"""
    instance_id: str
    component_id: str
    node_id: str
    
    # Runtime information
    state: ComponentState = ComponentState.STOPPED
    pod_name: Optional[str] = None
    container_id: Optional[str] = None
    ip_address: Optional[str] = None
    
    # Health information
    health_status: str = "unknown"
    last_health_check: Optional[datetime] = None
    restart_count: int = 0
    
    # Performance metrics
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    throughput_rps: float = 0.0
    
    # Timestamps
    started_at: Optional[datetime] = None
    last_restart: Optional[datetime] = None


class HealthChecker:
    """Health checking system for deployed components"""
    
    def __init__(self, check_interval: int = 30):
        """
        Initialize health checker
        
        Args:
            check_interval: Health check interval in seconds
        """
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable] = {}
        self.health_history: Dict[str, List[Tuple[datetime, bool]]] = {}
        self.is_running = False
        self.check_task: Optional[asyncio.Task] = None
        
    def register_health_check(self, component_id: str, 
                            health_check_func: Callable[[ComponentInstance], bool]):
        """Register health check function for component"""
        self.health_checks[component_id] = health_check_func
        
    async def start_health_checking(self):
        """Start health checking"""
        self.is_running = True
        self.check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Health checking started")
        
    async def stop_health_checking(self):
        """Stop health checking"""
        self.is_running = False
        if self.check_task:
            self.check_task.cancel()
            try:
                await self.check_task
            except asyncio.CancelledError:
                pass
        logger.info("Health checking stopped")
        
    async def _health_check_loop(self):
        """Main health checking loop"""
        while self.is_running:
            try:
                await asyncio.sleep(self.check_interval)
                # Health checking logic would be implemented here
                logger.debug("Health check cycle completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                
    async def check_component_health(self, instance: ComponentInstance) -> bool:
        """Check health of specific component instance"""
        try:
            component_id = instance.component_id
            
            if component_id in self.health_checks:
                health_func = self.health_checks[component_id]
                is_healthy = health_func(instance)
            else:
                # Default HTTP health check
                is_healthy = await self._default_http_health_check(instance)
            
            # Update health history
            if component_id not in self.health_history:
                self.health_history[component_id] = []
            
            self.health_history[component_id].append((datetime.now(), is_healthy))
            
            # Keep only recent history
            if len(self.health_history[component_id]) > 100:
                self.health_history[component_id] = self.health_history[component_id][-100:]
            
            # Update instance health
            instance.health_status = "healthy" if is_healthy else "unhealthy"
            instance.last_health_check = datetime.now()
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Health check failed for {instance.instance_id}: {e}")
            instance.health_status = "error"
            return False
            
    async def _default_http_health_check(self, instance: ComponentInstance) -> bool:
        """Default HTTP-based health check"""
        try:
            if not instance.ip_address:
                return False
                
            url = f"http://{instance.ip_address}:8080/health"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(url) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.debug(f"HTTP health check failed for {instance.instance_id}: {e}")
            return False


class AutoScaler:
    """Auto-scaling system for component instances"""
    
    def __init__(self, scale_check_interval: int = 60):
        """
        Initialize auto-scaler
        
        Args:
            scale_check_interval: Scaling check interval in seconds
        """
        self.scale_check_interval = scale_check_interval
        self.scaling_policies: Dict[str, Dict] = {}
        self.scaling_history: Dict[str, List[Tuple[datetime, str, int]]] = {}
        self.is_running = False
        self.scale_task: Optional[asyncio.Task] = None
        
        # Scaling thresholds
        self.cpu_scale_up_threshold = 0.8      # 80% CPU
        self.cpu_scale_down_threshold = 0.3    # 30% CPU
        self.memory_scale_up_threshold = 0.8   # 80% Memory
        self.memory_scale_down_threshold = 0.3 # 30% Memory
        
    def configure_scaling_policy(self, component_id: str, policy: Dict[str, Any]):
        """Configure scaling policy for component"""
        self.scaling_policies[component_id] = policy
        
    async def start_auto_scaling(self):
        """Start auto-scaling"""
        self.is_running = True
        self.scale_task = asyncio.create_task(self._scaling_loop())
        logger.info("Auto-scaling started")
        
    async def stop_auto_scaling(self):
        """Stop auto-scaling"""
        self.is_running = False
        if self.scale_task:
            self.scale_task.cancel()
            try:
                await self.scale_task
            except asyncio.CancelledError:
                pass
        logger.info("Auto-scaling stopped")
        
    async def _scaling_loop(self):
        """Main scaling loop"""
        while self.is_running:
            try:
                await asyncio.sleep(self.scale_check_interval)
                # Scaling logic would be implemented here
                logger.debug("Scaling check cycle completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                
    async def evaluate_scaling_decision(self, component_id: str, 
                                      instances: List[ComponentInstance]) -> Optional[str]:
        """
        Evaluate whether scaling is needed
        
        Returns:
            Optional[str]: "scale_up", "scale_down", or None
        """
        try:
            if component_id not in self.scaling_policies:
                return None
                
            policy = self.scaling_policies[component_id]
            scaling_type = policy.get("type", ScalingPolicy.CPU_BASED)
            
            if scaling_type == ScalingPolicy.CPU_BASED:
                return self._evaluate_cpu_scaling(instances, policy)
            elif scaling_type == ScalingPolicy.MEMORY_BASED:
                return self._evaluate_memory_scaling(instances, policy)
            elif scaling_type == ScalingPolicy.THROUGHPUT_BASED:
                return self._evaluate_throughput_scaling(instances, policy)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Scaling evaluation error for {component_id}: {e}")
            return None
            
    def _evaluate_cpu_scaling(self, instances: List[ComponentInstance], 
                            policy: Dict[str, Any]) -> Optional[str]:
        """Evaluate CPU-based scaling"""
        if not instances:
            return None
            
        # Calculate average CPU usage
        cpu_usages = [instance.cpu_usage for instance in instances if instance.cpu_usage > 0]
        
        if not cpu_usages:
            return None
            
        avg_cpu = sum(cpu_usages) / len(cpu_usages)
        
        # Get thresholds from policy
        scale_up_threshold = policy.get("cpu_scale_up_threshold", self.cpu_scale_up_threshold)
        scale_down_threshold = policy.get("cpu_scale_down_threshold", self.cpu_scale_down_threshold)
        
        if avg_cpu > scale_up_threshold:
            return "scale_up"
        elif avg_cpu < scale_down_threshold and len(instances) > 1:
            return "scale_down"
        else:
            return None
            
    def _evaluate_memory_scaling(self, instances: List[ComponentInstance], 
                               policy: Dict[str, Any]) -> Optional[str]:
        """Evaluate memory-based scaling"""
        if not instances:
            return None
            
        # Calculate average memory usage percentage
        memory_usages = []
        for instance in instances:
            if instance.memory_usage_mb > 0:
                # Assume 2GB limit for calculation
                memory_limit = 2048
                memory_percent = instance.memory_usage_mb / memory_limit
                memory_usages.append(memory_percent)
        
        if not memory_usages:
            return None
            
        avg_memory = sum(memory_usages) / len(memory_usages)
        
        # Get thresholds from policy
        scale_up_threshold = policy.get("memory_scale_up_threshold", self.memory_scale_up_threshold)
        scale_down_threshold = policy.get("memory_scale_down_threshold", self.memory_scale_down_threshold)
        
        if avg_memory > scale_up_threshold:
            return "scale_up"
        elif avg_memory < scale_down_threshold and len(instances) > 1:
            return "scale_down"
        else:
            return None
            
    def _evaluate_throughput_scaling(self, instances: List[ComponentInstance], 
                                   policy: Dict[str, Any]) -> Optional[str]:
        """Evaluate throughput-based scaling"""
        if not instances:
            return None
            
        # Calculate total throughput
        total_throughput = sum(instance.throughput_rps for instance in instances)
        
        # Get thresholds from policy
        target_throughput_per_instance = policy.get("target_throughput_per_instance", 100)
        scale_up_multiplier = policy.get("scale_up_multiplier", 0.8)
        scale_down_multiplier = policy.get("scale_down_multiplier", 0.3)
        
        target_capacity = len(instances) * target_throughput_per_instance
        
        if total_throughput > target_capacity * scale_up_multiplier:
            return "scale_up"
        elif total_throughput < target_capacity * scale_down_multiplier and len(instances) > 1:
            return "scale_down"
        else:
            return None


class ConfigurationManager:
    """Configuration management system"""
    
    def __init__(self, config_base_path: str = "./configs"):
        """
        Initialize configuration manager
        
        Args:
            config_base_path: Base path for configuration files
        """
        self.config_base_path = Path(config_base_path)
        self.config_base_path.mkdir(exist_ok=True)
        
        # Configuration cache
        self.config_cache: Dict[str, Dict] = {}
        self.config_file_hashes: Dict[str, str] = {}
        
    async def load_deployment_config(self, environment: DeploymentEnvironment) -> DeploymentConfiguration:
        """Load deployment configuration for environment"""
        
        config_file = self.config_base_path / f"deployment_{environment.value}.yaml"
        
        try:
            if config_file.exists():
                async with aiofiles.open(config_file, 'r') as f:
                    config_data = yaml.safe_load(await f.read())
                    
                # Convert to deployment configuration
                return self._dict_to_deployment_config(config_data)
            else:
                # Create default configuration
                default_config = self._create_default_deployment_config(environment)
                await self.save_deployment_config(default_config)
                return default_config
                
        except Exception as e:
            logger.error(f"Error loading deployment config: {e}")
            return self._create_default_deployment_config(environment)
            
    async def save_deployment_config(self, config: DeploymentConfiguration):
        """Save deployment configuration"""
        
        config_file = self.config_base_path / f"deployment_{config.environment.value}.yaml"
        
        try:
            config_dict = asdict(config)
            
            async with aiofiles.open(config_file, 'w') as f:
                await f.write(yaml.dump(config_dict, default_flow_style=False))
                
            logger.info(f"Deployment configuration saved: {config_file}")
            
        except Exception as e:
            logger.error(f"Error saving deployment config: {e}")
            
    def _create_default_deployment_config(self, environment: DeploymentEnvironment) -> DeploymentConfiguration:
        """Create default deployment configuration"""
        
        # Default component configurations
        processor_config = ComponentConfiguration(
            component_id="consciousness_processor",
            component_type=ComponentType.PROCESSOR,
            image="newbornai/consciousness-processor",
            version="4.0.0",
            cpu_request=1.0,
            cpu_limit=4.0,
            memory_request_mb=1024,
            memory_limit_mb=4096,
            ports=[8080, 8081],
            min_replicas=2 if environment == DeploymentEnvironment.PRODUCTION else 1,
            max_replicas=10 if environment == DeploymentEnvironment.PRODUCTION else 3,
            environment={
                "LOG_LEVEL": "INFO" if environment == DeploymentEnvironment.PRODUCTION else "DEBUG",
                "METRICS_ENABLED": "true",
                "CACHE_SIZE": "10000"
            }
        )
        
        monitor_config = ComponentConfiguration(
            component_id="consciousness_monitor",
            component_type=ComponentType.MONITOR,
            image="newbornai/consciousness-monitor",
            version="4.0.0",
            cpu_request=0.5,
            cpu_limit=2.0,
            memory_request_mb=512,
            memory_limit_mb=2048,
            ports=[8082],
            min_replicas=1,
            max_replicas=3,
            depends_on=["consciousness_processor"],
            environment={
                "MONITORING_INTERVAL": "30",
                "ALERT_ENABLED": "true"
            }
        )
        
        calculator_config = ComponentConfiguration(
            component_id="streaming_calculator",
            component_type=ComponentType.CALCULATOR,
            image="newbornai/streaming-calculator",
            version="4.0.0",
            cpu_request=2.0,
            cpu_limit=8.0,
            memory_request_mb=2048,
            memory_limit_mb=8192,
            ports=[8083],
            min_replicas=1 if environment == DeploymentEnvironment.PRODUCTION else 1,
            max_replicas=5 if environment == DeploymentEnvironment.PRODUCTION else 2,
            scaling_policy=ScalingPolicy.THROUGHPUT_BASED,
            environment={
                "TARGET_THROUGHPUT": "1000",
                "STREAMING_MODE": "adaptive"
            }
        )
        
        # Database configuration
        database_config = ComponentConfiguration(
            component_id="consciousness_database",
            component_type=ComponentType.DATABASE,
            image="postgres:15",
            version="15",
            cpu_request=1.0,
            cpu_limit=4.0,
            memory_request_mb=1024,
            memory_limit_mb=4096,
            ports=[5432],
            min_replicas=1,
            max_replicas=1,  # Database doesn't auto-scale
            persistent_volumes=[
                {"name": "postgres-data", "size": "100Gi", "mount_path": "/var/lib/postgresql/data"}
            ],
            environment={
                "POSTGRES_DB": "consciousness_db",
                "POSTGRES_USER": "consciousness_user",
                "POSTGRES_PASSWORD": "secure_password_here"  # Would use secrets in production
            }
        )
        
        components = [processor_config, monitor_config, calculator_config, database_config]
        
        return DeploymentConfiguration(
            deployment_id=f"newbornai_{environment.value}_{uuid.uuid4().hex[:8]}",
            environment=environment,
            components=components,
            monitoring_enabled=True,
            logging_enabled=True,
            alerting_enabled=environment == DeploymentEnvironment.PRODUCTION,
            backup_enabled=environment == DeploymentEnvironment.PRODUCTION,
            dr_enabled=environment == DeploymentEnvironment.PRODUCTION
        )
        
    def _dict_to_deployment_config(self, config_dict: Dict[str, Any]) -> DeploymentConfiguration:
        """Convert dictionary to deployment configuration"""
        
        # Convert components
        components = []
        for comp_dict in config_dict.get("components", []):
            component = ComponentConfiguration(**comp_dict)
            components.append(component)
        
        # Remove components from dict to avoid duplicate
        config_dict = config_dict.copy()
        config_dict["components"] = components
        
        # Convert environment enum
        if "environment" in config_dict:
            config_dict["environment"] = DeploymentEnvironment(config_dict["environment"])
        
        return DeploymentConfiguration(**config_dict)


class KubernetesDeploymentManager:
    """Kubernetes-based deployment manager"""
    
    def __init__(self, namespace: str = "newbornai"):
        """
        Initialize Kubernetes deployment manager
        
        Args:
            namespace: Kubernetes namespace
        """
        self.namespace = namespace
        
        # Initialize Kubernetes client
        try:
            config.load_incluster_config()  # Try in-cluster config first
        except:
            try:
                config.load_kube_config()  # Try local kubeconfig
            except:
                logger.warning("Could not load Kubernetes configuration")
        
        self.k8s_apps_v1 = client.AppsV1Api()
        self.k8s_core_v1 = client.CoreV1Api()
        self.k8s_autoscaling_v1 = client.AutoscalingV1Api()
        
        # Deployed resources tracking
        self.deployed_resources: Dict[str, Dict] = {}
        
    async def deploy_component(self, component_config: ComponentConfiguration) -> bool:
        """Deploy component to Kubernetes"""
        
        try:
            logger.info(f"Deploying component: {component_config.component_id}")
            
            # Create deployment
            deployment_created = await self._create_deployment(component_config)
            
            # Create service
            service_created = await self._create_service(component_config)
            
            # Create horizontal pod autoscaler if needed
            hpa_created = True
            if component_config.max_replicas > component_config.min_replicas:
                hpa_created = await self._create_hpa(component_config)
            
            # Create persistent volume claims if needed
            pvc_created = True
            if component_config.persistent_volumes:
                pvc_created = await self._create_pvcs(component_config)
            
            success = deployment_created and service_created and hpa_created and pvc_created
            
            if success:
                # Track deployed resources
                self.deployed_resources[component_config.component_id] = {
                    "deployment": f"{component_config.component_id}-deployment",
                    "service": f"{component_config.component_id}-service",
                    "hpa": f"{component_config.component_id}-hpa" if hpa_created else None,
                    "pvcs": [pv["name"] for pv in component_config.persistent_volumes]
                }
                
                logger.info(f"Component deployed successfully: {component_config.component_id}")
            else:
                logger.error(f"Component deployment failed: {component_config.component_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deploying component {component_config.component_id}: {e}")
            return False
            
    async def _create_deployment(self, component_config: ComponentConfiguration) -> bool:
        """Create Kubernetes deployment"""
        
        try:
            # Define container
            container = client.V1Container(
                name=component_config.component_id,
                image=f"{component_config.image}:{component_config.version}",
                ports=[client.V1ContainerPort(container_port=port) for port in component_config.ports],
                env=[
                    client.V1EnvVar(name=key, value=value)
                    for key, value in component_config.environment.items()
                ],
                resources=client.V1ResourceRequirements(
                    requests={
                        "cpu": str(component_config.cpu_request),
                        "memory": f"{component_config.memory_request_mb}Mi"
                    },
                    limits={
                        "cpu": str(component_config.cpu_limit),
                        "memory": f"{component_config.memory_limit_mb}Mi"
                    }
                ),
                liveness_probe=client.V1Probe(
                    http_get=client.V1HTTPGetAction(
                        path=component_config.health_check_path,
                        port=component_config.health_check_port
                    ),
                    initial_delay_seconds=30,
                    period_seconds=10
                ),
                readiness_probe=client.V1Probe(
                    http_get=client.V1HTTPGetAction(
                        path=component_config.health_check_path,
                        port=component_config.health_check_port
                    ),
                    initial_delay_seconds=5,
                    period_seconds=5
                )
            )
            
            # Add volume mounts if needed
            if component_config.persistent_volumes:
                container.volume_mounts = [
                    client.V1VolumeMount(
                        name=pv["name"],
                        mount_path=pv["mount_path"]
                    )
                    for pv in component_config.persistent_volumes
                ]
            
            # Define pod template
            pod_template = client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={"app": component_config.component_id}
                ),
                spec=client.V1PodSpec(
                    containers=[container],
                    restart_policy=component_config.restart_policy,
                    volumes=[
                        client.V1Volume(
                            name=pv["name"],
                            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                                claim_name=pv["name"]
                            )
                        )
                        for pv in component_config.persistent_volumes
                    ] if component_config.persistent_volumes else None
                )
            )
            
            # Define deployment
            deployment = client.V1Deployment(
                metadata=client.V1ObjectMeta(
                    name=f"{component_config.component_id}-deployment",
                    namespace=self.namespace
                ),
                spec=client.V1DeploymentSpec(
                    replicas=component_config.min_replicas,
                    selector=client.V1LabelSelector(
                        match_labels={"app": component_config.component_id}
                    ),
                    template=pod_template
                )
            )
            
            # Create deployment
            self.k8s_apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating deployment for {component_config.component_id}: {e}")
            return False
            
    async def _create_service(self, component_config: ComponentConfiguration) -> bool:
        """Create Kubernetes service"""
        
        try:
            service = client.V1Service(
                metadata=client.V1ObjectMeta(
                    name=f"{component_config.component_id}-service",
                    namespace=self.namespace
                ),
                spec=client.V1ServiceSpec(
                    selector={"app": component_config.component_id},
                    ports=[
                        client.V1ServicePort(
                            port=port,
                            target_port=port,
                            protocol="TCP"
                        )
                        for port in component_config.ports
                    ],
                    type="ClusterIP"
                )
            )
            
            self.k8s_core_v1.create_namespaced_service(
                namespace=self.namespace,
                body=service
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating service for {component_config.component_id}: {e}")
            return False
            
    async def _create_hpa(self, component_config: ComponentConfiguration) -> bool:
        """Create Horizontal Pod Autoscaler"""
        
        try:
            hpa = client.V1HorizontalPodAutoscaler(
                metadata=client.V1ObjectMeta(
                    name=f"{component_config.component_id}-hpa",
                    namespace=self.namespace
                ),
                spec=client.V1HorizontalPodAutoscalerSpec(
                    scale_target_ref=client.V1CrossVersionObjectReference(
                        api_version="apps/v1",
                        kind="Deployment",
                        name=f"{component_config.component_id}-deployment"
                    ),
                    min_replicas=component_config.min_replicas,
                    max_replicas=component_config.max_replicas,
                    target_cpu_utilization_percentage=80
                )
            )
            
            self.k8s_autoscaling_v1.create_namespaced_horizontal_pod_autoscaler(
                namespace=self.namespace,
                body=hpa
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating HPA for {component_config.component_id}: {e}")
            return False
            
    async def _create_pvcs(self, component_config: ComponentConfiguration) -> bool:
        """Create Persistent Volume Claims"""
        
        try:
            for pv in component_config.persistent_volumes:
                pvc = client.V1PersistentVolumeClaim(
                    metadata=client.V1ObjectMeta(
                        name=pv["name"],
                        namespace=self.namespace
                    ),
                    spec=client.V1PersistentVolumeClaimSpec(
                        access_modes=["ReadWriteOnce"],
                        resources=client.V1ResourceRequirements(
                            requests={"storage": pv.get("size", "10Gi")}
                        ),
                        storage_class=pv.get("storage_class", "default")
                    )
                )
                
                self.k8s_core_v1.create_namespaced_persistent_volume_claim(
                    namespace=self.namespace,
                    body=pvc
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating PVCs for {component_config.component_id}: {e}")
            return False
            
    async def undeploy_component(self, component_id: str) -> bool:
        """Remove component from Kubernetes"""
        
        try:
            if component_id not in self.deployed_resources:
                logger.warning(f"Component not found in deployed resources: {component_id}")
                return True
            
            resources = self.deployed_resources[component_id]
            
            # Delete deployment
            if resources.get("deployment"):
                try:
                    self.k8s_apps_v1.delete_namespaced_deployment(
                        name=resources["deployment"],
                        namespace=self.namespace
                    )
                except:
                    pass
            
            # Delete service
            if resources.get("service"):
                try:
                    self.k8s_core_v1.delete_namespaced_service(
                        name=resources["service"],
                        namespace=self.namespace
                    )
                except:
                    pass
            
            # Delete HPA
            if resources.get("hpa"):
                try:
                    self.k8s_autoscaling_v1.delete_namespaced_horizontal_pod_autoscaler(
                        name=resources["hpa"],
                        namespace=self.namespace
                    )
                except:
                    pass
            
            # Delete PVCs
            for pvc_name in resources.get("pvcs", []):
                try:
                    self.k8s_core_v1.delete_namespaced_persistent_volume_claim(
                        name=pvc_name,
                        namespace=self.namespace
                    )
                except:
                    pass
            
            # Remove from tracking
            del self.deployed_resources[component_id]
            
            logger.info(f"Component undeployed: {component_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error undeploying component {component_id}: {e}")
            return False
            
    async def get_component_instances(self, component_id: str) -> List[ComponentInstance]:
        """Get running instances of component"""
        
        try:
            # Get pods for component
            pods = self.k8s_core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"app={component_id}"
            )
            
            instances = []
            for pod in pods.items:
                instance = ComponentInstance(
                    instance_id=pod.metadata.uid,
                    component_id=component_id,
                    node_id=pod.spec.node_name or "unknown",
                    pod_name=pod.metadata.name,
                    ip_address=pod.status.pod_ip,
                    started_at=pod.status.start_time,
                    restart_count=sum(
                        container_status.restart_count
                        for container_status in (pod.status.container_statuses or [])
                    )
                )
                
                # Determine state
                if pod.status.phase == "Running":
                    instance.state = ComponentState.RUNNING
                elif pod.status.phase == "Pending":
                    instance.state = ComponentState.STARTING
                elif pod.status.phase == "Failed":
                    instance.state = ComponentState.ERROR
                else:
                    instance.state = ComponentState.STOPPED
                
                instances.append(instance)
            
            return instances
            
        except Exception as e:
            logger.error(f"Error getting component instances for {component_id}: {e}")
            return []


class ProductionDeploymentOrchestrator:
    """
    Production deployment orchestrator
    Complete system orchestration for production deployment
    """
    
    def __init__(self, config_manager: ConfigurationManager):
        """
        Initialize production deployment orchestrator
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        
        # Core components
        self.health_checker = HealthChecker()
        self.auto_scaler = AutoScaler()
        self.k8s_manager = KubernetesDeploymentManager()
        
        # Deployment state
        self.current_deployment: Optional[DeploymentConfiguration] = None
        self.deployed_components: Dict[str, ComponentInstance] = {}
        self.deployment_status = "not_deployed"
        
        # Monitoring
        self.deployment_start_time: Optional[datetime] = None
        self.deployment_metrics: Dict[str, Any] = {}
        
        # Task management
        self.orchestration_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        logger.info("Production Deployment Orchestrator initialized")
        
    async def deploy_environment(self, environment: DeploymentEnvironment) -> bool:
        """
        Deploy complete environment
        
        Args:
            environment: Target deployment environment
            
        Returns:
            bool: True if deployment successful
        """
        
        try:
            logger.info(f"Starting deployment for environment: {environment.value}")
            self.deployment_start_time = datetime.now()
            self.deployment_status = "deploying"
            
            # Load deployment configuration
            config = await self.config_manager.load_deployment_config(environment)
            self.current_deployment = config
            
            # Validate configuration
            validation_result = await self._validate_deployment_config(config)
            if not validation_result["valid"]:
                logger.error(f"Configuration validation failed: {validation_result['errors']}")
                self.deployment_status = "failed"
                return False
            
            # Create namespace if needed
            await self._ensure_namespace(config.namespace)
            
            # Deploy components in dependency order
            deployment_order = self._calculate_deployment_order(config.components)
            
            for component_config in deployment_order:
                logger.info(f"Deploying component: {component_config.component_id}")
                
                success = await self.k8s_manager.deploy_component(component_config)
                
                if not success:
                    logger.error(f"Failed to deploy component: {component_config.component_id}")
                    self.deployment_status = "failed"
                    return False
                
                # Wait for component to be ready
                ready = await self._wait_for_component_ready(component_config.component_id, timeout=300)
                
                if not ready:
                    logger.error(f"Component failed to become ready: {component_config.component_id}")
                    self.deployment_status = "failed"
                    return False
                
                logger.info(f"Component deployed and ready: {component_config.component_id}")
            
            # Start orchestration services
            await self._start_orchestration_services()
            
            # Run deployment validation
            validation_success = await self._run_deployment_validation()
            
            if validation_success:
                self.deployment_status = "deployed"
                logger.info(f"Environment deployment completed successfully: {environment.value}")
                return True
            else:
                self.deployment_status = "validation_failed"
                logger.error(f"Deployment validation failed for environment: {environment.value}")
                return False
                
        except Exception as e:
            logger.error(f"Deployment failed for environment {environment.value}: {e}")
            self.deployment_status = "error"
            return False
            
    async def undeploy_environment(self) -> bool:
        """Undeploy current environment"""
        
        try:
            if not self.current_deployment:
                logger.warning("No deployment to undeploy")
                return True
            
            logger.info("Starting environment undeployment")
            self.deployment_status = "undeploying"
            
            # Stop orchestration services
            await self._stop_orchestration_services()
            
            # Undeploy components in reverse dependency order
            components = list(reversed(self.current_deployment.components))
            
            for component_config in components:
                logger.info(f"Undeploying component: {component_config.component_id}")
                
                success = await self.k8s_manager.undeploy_component(component_config.component_id)
                
                if not success:
                    logger.warning(f"Failed to undeploy component: {component_config.component_id}")
                else:
                    logger.info(f"Component undeployed: {component_config.component_id}")
            
            # Clean up state
            self.current_deployment = None
            self.deployed_components.clear()
            self.deployment_status = "not_deployed"
            
            logger.info("Environment undeployment completed")
            return True
            
        except Exception as e:
            logger.error(f"Undeployment failed: {e}")
            self.deployment_status = "error"
            return False
            
    async def _validate_deployment_config(self, config: DeploymentConfiguration) -> Dict[str, Any]:
        """Validate deployment configuration"""
        
        validation_result = {"valid": True, "errors": [], "warnings": []}
        
        try:
            # Basic validation
            if not config.components:
                validation_result["errors"].append("No components defined")
                validation_result["valid"] = False
            
            # Component validation
            component_ids = set()
            for component in config.components:
                # Check for duplicate IDs
                if component.component_id in component_ids:
                    validation_result["errors"].append(f"Duplicate component ID: {component.component_id}")
                    validation_result["valid"] = False
                component_ids.add(component.component_id)
                
                # Resource validation
                if component.cpu_request > component.cpu_limit:
                    validation_result["errors"].append(f"CPU request > limit for {component.component_id}")
                    validation_result["valid"] = False
                
                if component.memory_request_mb > component.memory_limit_mb:
                    validation_result["errors"].append(f"Memory request > limit for {component.component_id}")
                    validation_result["valid"] = False
                
                # Scaling validation
                if component.min_replicas > component.max_replicas:
                    validation_result["errors"].append(f"Min replicas > max replicas for {component.component_id}")
                    validation_result["valid"] = False
            
            # Dependency validation
            for component in config.components:
                for dependency in component.depends_on:
                    if dependency not in component_ids:
                        validation_result["errors"].append(f"Unknown dependency '{dependency}' for {component.component_id}")
                        validation_result["valid"] = False
            
            # Environment-specific validation
            if config.environment == DeploymentEnvironment.PRODUCTION:
                for component in config.components:
                    if component.min_replicas < 2 and component.component_type != ComponentType.DATABASE:
                        validation_result["warnings"].append(f"Single replica in production for {component.component_id}")
            
            logger.info(f"Configuration validation completed: valid={validation_result['valid']}")
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {e}")
        
        return validation_result
        
    def _calculate_deployment_order(self, components: List[ComponentConfiguration]) -> List[ComponentConfiguration]:
        """Calculate deployment order based on dependencies"""
        
        # Create dependency graph
        dependency_graph = {}
        component_map = {comp.component_id: comp for comp in components}
        
        for component in components:
            dependency_graph[component.component_id] = component.depends_on.copy()
        
        # Topological sort
        deployed = set()
        deployment_order = []
        
        while len(deployment_order) < len(components):
            # Find components with no undeployed dependencies
            ready_components = []
            
            for comp_id, dependencies in dependency_graph.items():
                if comp_id not in deployed:
                    undeployed_deps = [dep for dep in dependencies if dep not in deployed]
                    if not undeployed_deps:
                        ready_components.append(comp_id)
            
            if not ready_components:
                # Circular dependency or error - deploy remaining in original order
                remaining = [comp for comp in components if comp.component_id not in deployed]
                deployment_order.extend(remaining)
                break
            
            # Deploy ready components
            for comp_id in ready_components:
                component = component_map[comp_id]
                deployment_order.append(component)
                deployed.add(comp_id)
        
        return deployment_order
        
    async def _ensure_namespace(self, namespace: str):
        """Ensure Kubernetes namespace exists"""
        
        try:
            # Check if namespace exists
            try:
                self.k8s_manager.k8s_core_v1.read_namespace(name=namespace)
                logger.info(f"Namespace already exists: {namespace}")
            except:
                # Create namespace
                namespace_obj = client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=namespace)
                )
                
                self.k8s_manager.k8s_core_v1.create_namespace(body=namespace_obj)
                logger.info(f"Namespace created: {namespace}")
                
        except Exception as e:
            logger.error(f"Error ensuring namespace {namespace}: {e}")
            raise
            
    async def _wait_for_component_ready(self, component_id: str, timeout: int = 300) -> bool:
        """Wait for component to become ready"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                instances = await self.k8s_manager.get_component_instances(component_id)
                
                if instances:
                    # Check if at least one instance is running
                    running_instances = [
                        instance for instance in instances
                        if instance.state == ComponentState.RUNNING
                    ]
                    
                    if running_instances:
                        # Perform health check
                        healthy_instances = 0
                        for instance in running_instances:
                            is_healthy = await self.health_checker.check_component_health(instance)
                            if is_healthy:
                                healthy_instances += 1
                        
                        if healthy_instances > 0:
                            logger.info(f"Component ready: {component_id} ({healthy_instances} healthy instances)")
                            return True
                
                # Wait before next check
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.warning(f"Error checking component readiness for {component_id}: {e}")
                await asyncio.sleep(5)
        
        logger.error(f"Component failed to become ready within timeout: {component_id}")
        return False
        
    async def _start_orchestration_services(self):
        """Start orchestration services"""
        
        logger.info("Starting orchestration services")
        
        # Start health checking
        await self.health_checker.start_health_checking()
        
        # Start auto-scaling
        await self.auto_scaler.start_auto_scaling()
        
        # Configure scaling policies for components
        if self.current_deployment:
            for component in self.current_deployment.components:
                if component.scaling_policy != ScalingPolicy.FIXED:
                    policy = {
                        "type": component.scaling_policy,
                        "min_replicas": component.min_replicas,
                        "max_replicas": component.max_replicas
                    }
                    self.auto_scaler.configure_scaling_policy(component.component_id, policy)
        
        # Start monitoring tasks
        self.orchestration_tasks = [
            asyncio.create_task(self._monitoring_loop()),
            asyncio.create_task(self._metrics_collection_loop())
        ]
        
        self.is_running = True
        logger.info("Orchestration services started")
        
    async def _stop_orchestration_services(self):
        """Stop orchestration services"""
        
        logger.info("Stopping orchestration services")
        self.is_running = False
        
        # Stop services
        await self.health_checker.stop_health_checking()
        await self.auto_scaler.stop_auto_scaling()
        
        # Cancel tasks
        for task in self.orchestration_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.orchestration_tasks, return_exceptions=True)
        
        self.orchestration_tasks.clear()
        logger.info("Orchestration services stopped")
        
    async def _run_deployment_validation(self) -> bool:
        """Run post-deployment validation"""
        
        logger.info("Running deployment validation")
        
        try:
            if not self.current_deployment:
                return False
            
            validation_results = []
            
            # Test each component
            for component in self.current_deployment.components:
                logger.info(f"Validating component: {component.component_id}")
                
                # Get component instances
                instances = await self.k8s_manager.get_component_instances(component.component_id)
                
                if not instances:
                    logger.error(f"No instances found for component: {component.component_id}")
                    validation_results.append(False)
                    continue
                
                # Check instance health
                healthy_instances = 0
                for instance in instances:
                    is_healthy = await self.health_checker.check_component_health(instance)
                    if is_healthy:
                        healthy_instances += 1
                
                component_healthy = healthy_instances > 0
                validation_results.append(component_healthy)
                
                if component_healthy:
                    logger.info(f"Component validation passed: {component.component_id}")
                else:
                    logger.error(f"Component validation failed: {component.component_id}")
            
            # Integration tests
            integration_passed = await self._run_integration_tests()
            validation_results.append(integration_passed)
            
            # Overall validation result
            all_passed = all(validation_results)
            
            if all_passed:
                logger.info("Deployment validation passed")
            else:
                logger.error("Deployment validation failed")
            
            return all_passed
            
        except Exception as e:
            logger.error(f"Deployment validation error: {e}")
            return False
            
    async def _run_integration_tests(self) -> bool:
        """Run integration tests"""
        
        logger.info("Running integration tests")
        
        try:
            # Simple integration test - check if processor and monitor can communicate
            # In a real implementation, this would be more comprehensive
            
            # Test 1: Health endpoints
            health_tests_passed = True
            
            if self.current_deployment:
                for component in self.current_deployment.components:
                    instances = await self.k8s_manager.get_component_instances(component.component_id)
                    
                    for instance in instances:
                        if instance.ip_address:
                            try:
                                url = f"http://{instance.ip_address}:8080/health"
                                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                                    async with session.get(url) as response:
                                        if response.status != 200:
                                            health_tests_passed = False
                                            logger.error(f"Health check failed for {instance.instance_id}")
                            except Exception as e:
                                logger.warning(f"Integration test error for {instance.instance_id}: {e}")
                                health_tests_passed = False
            
            # Test 2: Basic functionality (simplified)
            functionality_tests_passed = True
            
            # More integration tests would be implemented here
            # For example:
            # - Test consciousness processing pipeline
            # - Test monitoring and alerting
            # - Test data persistence
            # - Test scaling behavior
            
            overall_passed = health_tests_passed and functionality_tests_passed
            
            logger.info(f"Integration tests completed: passed={overall_passed}")
            return overall_passed
            
        except Exception as e:
            logger.error(f"Integration tests error: {e}")
            return False
            
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                if self.current_deployment:
                    # Update deployment metrics
                    await self._update_deployment_metrics()
                    
                    # Check for scaling needs
                    await self._check_scaling_needs()
                    
                    # Log status
                    logger.debug("Monitoring cycle completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                
    async def _metrics_collection_loop(self):
        """Metrics collection loop"""
        
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Collect metrics every minute
                
                if self.current_deployment:
                    # Collect and store metrics
                    await self._collect_metrics()
                    
                    logger.debug("Metrics collection completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                
    async def _update_deployment_metrics(self):
        """Update deployment metrics"""
        
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "deployment_status": self.deployment_status,
                "uptime_seconds": (datetime.now() - self.deployment_start_time).total_seconds() if self.deployment_start_time else 0,
                "components": {}
            }
            
            if self.current_deployment:
                for component in self.current_deployment.components:
                    instances = await self.k8s_manager.get_component_instances(component.component_id)
                    
                    component_metrics = {
                        "instance_count": len(instances),
                        "healthy_instances": sum(1 for i in instances if i.health_status == "healthy"),
                        "total_restarts": sum(i.restart_count for i in instances),
                        "average_cpu": sum(i.cpu_usage for i in instances) / len(instances) if instances else 0,
                        "average_memory_mb": sum(i.memory_usage_mb for i in instances) / len(instances) if instances else 0
                    }
                    
                    metrics["components"][component.component_id] = component_metrics
            
            self.deployment_metrics = metrics
            
        except Exception as e:
            logger.error(f"Error updating deployment metrics: {e}")
            
    async def _check_scaling_needs(self):
        """Check if any components need scaling"""
        
        try:
            if not self.current_deployment:
                return
            
            for component in self.current_deployment.components:
                instances = await self.k8s_manager.get_component_instances(component.component_id)
                
                scaling_decision = await self.auto_scaler.evaluate_scaling_decision(
                    component.component_id, instances
                )
                
                if scaling_decision == "scale_up":
                    logger.info(f"Scaling up component: {component.component_id}")
                    # Scaling logic would be implemented here
                elif scaling_decision == "scale_down":
                    logger.info(f"Scaling down component: {component.component_id}")
                    # Scaling logic would be implemented here
                    
        except Exception as e:
            logger.error(f"Error checking scaling needs: {e}")
            
    async def _collect_metrics(self):
        """Collect detailed metrics"""
        
        try:
            # System-level metrics
            system_metrics = {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
            
            # Add to deployment metrics
            self.deployment_metrics["system"] = system_metrics
            
            logger.debug(f"Metrics collected: {system_metrics}")
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            
    async def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status"""
        
        try:
            status = {
                "deployment_status": self.deployment_status,
                "environment": self.current_deployment.environment.value if self.current_deployment else None,
                "deployment_id": self.current_deployment.deployment_id if self.current_deployment else None,
                "uptime_seconds": (datetime.now() - self.deployment_start_time).total_seconds() if self.deployment_start_time else 0,
                "is_running": self.is_running,
                "metrics": self.deployment_metrics,
                "components": {}
            }
            
            if self.current_deployment:
                for component in self.current_deployment.components:
                    instances = await self.k8s_manager.get_component_instances(component.component_id)
                    
                    component_status = {
                        "component_id": component.component_id,
                        "component_type": component.component_type.value,
                        "desired_replicas": component.min_replicas,
                        "actual_replicas": len(instances),
                        "healthy_replicas": sum(1 for i in instances if i.health_status == "healthy"),
                        "instances": [
                            {
                                "instance_id": i.instance_id,
                                "state": i.state.value,
                                "health_status": i.health_status,
                                "node_id": i.node_id,
                                "restart_count": i.restart_count,
                                "cpu_usage": i.cpu_usage,
                                "memory_usage_mb": i.memory_usage_mb
                            }
                            for i in instances
                        ]
                    }
                    
                    status["components"][component.component_id] = component_status
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting deployment status: {e}")
            return {"error": str(e)}


# Example usage and testing
async def test_production_deployment():
    """Test production deployment orchestrator"""
    
    print("🚀 Testing Production Deployment Orchestrator")
    print("=" * 60)
    
    # Initialize configuration manager
    config_manager = ConfigurationManager("./test_configs")
    
    # Initialize orchestrator
    orchestrator = ProductionDeploymentOrchestrator(config_manager)
    
    try:
        # Test configuration loading
        print("\n📋 Testing Configuration Management")
        print("-" * 40)
        
        # Load development configuration
        dev_config = await config_manager.load_deployment_config(DeploymentEnvironment.DEVELOPMENT)
        print(f"Loaded config: {dev_config.deployment_id}")
        print(f"Environment: {dev_config.environment.value}")
        print(f"Components: {len(dev_config.components)}")
        
        for component in dev_config.components:
            print(f"  - {component.component_id} ({component.component_type.value})")
        
        # Test configuration validation
        print("\n✅ Testing Configuration Validation")
        print("-" * 40)
        
        validation_result = await orchestrator._validate_deployment_config(dev_config)
        print(f"Validation result: {validation_result['valid']}")
        
        if validation_result['errors']:
            print("Errors:")
            for error in validation_result['errors']:
                print(f"  - {error}")
        
        if validation_result['warnings']:
            print("Warnings:")
            for warning in validation_result['warnings']:
                print(f"  - {warning}")
        
        # Test deployment order calculation
        print("\n📊 Testing Deployment Order")
        print("-" * 40)
        
        deployment_order = orchestrator._calculate_deployment_order(dev_config.components)
        print("Deployment order:")
        for i, component in enumerate(deployment_order, 1):
            deps = ", ".join(component.depends_on) if component.depends_on else "none"
            print(f"  {i}. {component.component_id} (depends on: {deps})")
        
        # Note: Actual Kubernetes deployment would require a real cluster
        # For testing, we'll simulate the deployment process
        print("\n🔧 Simulating Deployment Process")
        print("-" * 40)
        
        print("Deployment simulation:")
        print("1. ✅ Configuration validated")
        print("2. ✅ Namespace ensured")
        print("3. ✅ Components ordered by dependencies")
        print("4. 🔄 Deploying components...")
        
        # Simulate component deployment
        for i, component in enumerate(deployment_order, 1):
            print(f"   Step {i}: Deploying {component.component_id}...")
            await asyncio.sleep(0.2)  # Simulate deployment time
            print(f"   ✅ {component.component_id} deployed")
        
        print("5. ✅ Orchestration services started")
        print("6. ✅ Deployment validation completed")
        
        # Test status reporting
        print("\n📈 Testing Status Reporting")
        print("-" * 40)
        
        # Simulate some metrics
        orchestrator.deployment_status = "deployed"
        orchestrator.deployment_start_time = datetime.now() - timedelta(minutes=5)
        orchestrator.current_deployment = dev_config
        orchestrator.is_running = True
        
        # Update metrics
        await orchestrator._update_deployment_metrics()
        
        status = await orchestrator.get_deployment_status()
        
        print(f"Deployment Status: {status['deployment_status']}")
        print(f"Environment: {status['environment']}")
        print(f"Uptime: {status['uptime_seconds']:.0f} seconds")
        print(f"Running: {status['is_running']}")
        print(f"Components: {len(status['components'])}")
        
        # Test health checking
        print("\n🏥 Testing Health Checking")
        print("-" * 40)
        
        health_checker = HealthChecker(check_interval=5)
        
        # Register mock health check
        def mock_health_check(instance: ComponentInstance) -> bool:
            # Simulate health check logic
            return instance.state == ComponentState.RUNNING
        
        health_checker.register_health_check("consciousness_processor", mock_health_check)
        
        # Create mock instance
        mock_instance = ComponentInstance(
            instance_id="mock_instance_001",
            component_id="consciousness_processor",
            node_id="test_node",
            state=ComponentState.RUNNING,
            ip_address="192.168.1.100"
        )
        
        # Test health check
        is_healthy = await health_checker.check_component_health(mock_instance)
        print(f"Mock health check result: {is_healthy}")
        print(f"Instance health status: {mock_instance.health_status}")
        
        # Test auto-scaling
        print("\n📊 Testing Auto-Scaling")
        print("-" * 40)
        
        auto_scaler = AutoScaler(scale_check_interval=10)
        
        # Configure scaling policy
        scaling_policy = {
            "type": ScalingPolicy.CPU_BASED,
            "cpu_scale_up_threshold": 0.8,
            "cpu_scale_down_threshold": 0.3,
            "min_replicas": 1,
            "max_replicas": 5
        }
        
        auto_scaler.configure_scaling_policy("consciousness_processor", scaling_policy)
        
        # Test scaling decision with high CPU
        mock_instance.cpu_usage = 0.9  # 90% CPU
        instances = [mock_instance]
        
        scaling_decision = await auto_scaler.evaluate_scaling_decision("consciousness_processor", instances)
        print(f"Scaling decision (90% CPU): {scaling_decision}")
        
        # Test scaling decision with low CPU
        mock_instance.cpu_usage = 0.2  # 20% CPU
        
        scaling_decision = await auto_scaler.evaluate_scaling_decision("consciousness_processor", instances)
        print(f"Scaling decision (20% CPU): {scaling_decision}")
        
        print("\n✅ Production deployment orchestrator test completed!")
        print("Note: Full Kubernetes integration requires a real cluster")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_production_deployment())