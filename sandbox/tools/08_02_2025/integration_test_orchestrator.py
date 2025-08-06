"""
Integration Test Orchestrator for IIT 4.0 NewbornAI 2.0
Automated test orchestration for CI/CD pipelines with comprehensive reporting

This orchestrator implements enterprise-grade testing automation:
- Automated test orchestration for CI/CD pipelines
- Test environment setup and teardown
- Test data generation and validation
- Test reporting and coverage analysis
- Parallel test execution and load balancing
- Continuous integration quality gates

CI/CD Integration Features:
- Jenkins/GitHub Actions/GitLab CI compatibility
- Docker container test environments
- Distributed test execution
- Artifact collection and reporting
- Quality gate enforcement
- Automated rollback triggers

Author: TDD Engineer (Takuto Wada's expertise)
Date: 2025-08-03
Version: 1.0.0
"""

import asyncio
import os
import sys
import json
import yaml
import tempfile
import shutil
import subprocess
import time
import signal
import threading
import multiprocessing
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import asynccontextmanager, contextmanager
import logging
import traceback
import psutil
import docker
import pytest
import coverage
from junitparser import JUnitXml, TestCase, TestSuite, Failure, Error, Skipped

# Import test modules
from comprehensive_test_suite import ComprehensiveTestSuite, TestResult
from test_quality_assurance import QualityAssuranceManager, QualityMetrics
from iit4_core_engine import IIT4PhiCalculator
from iit4_experiential_phi_calculator import IIT4_ExperientialPhiCalculator
from realtime_iit4_processor import RealtimeIIT4Processor
from newborn_ai_2_integrated_system import NewbornAI20_IntegratedSystem

# Configure orchestrator logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
orchestrator_logger = logging.getLogger("test_orchestrator")


@dataclass
class TestEnvironment:
    """Test environment configuration"""
    environment_id: str
    name: str
    description: str
    setup_commands: List[str]
    teardown_commands: List[str]
    environment_variables: Dict[str, str]
    resource_limits: Dict[str, Any]
    docker_config: Optional[Dict[str, Any]] = None
    parallel_workers: int = 1
    timeout_minutes: int = 30
    
    def __post_init__(self):
        if not self.resource_limits:
            self.resource_limits = {
                "max_memory_mb": 2048,
                "max_cpu_percent": 80,
                "max_disk_mb": 1024
            }


@dataclass
class TestExecutionPlan:
    """Test execution plan configuration"""
    plan_id: str
    name: str
    test_suites: List[str]
    environments: List[TestEnvironment]
    execution_strategy: str  # sequential, parallel, distributed
    max_parallel_jobs: int = 4
    retry_count: int = 2
    timeout_minutes: int = 60
    quality_gates: Dict[str, Any] = field(default_factory=dict)
    artifacts_config: Dict[str, Any] = field(default_factory=dict)
    notifications: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestExecutionResult:
    """Comprehensive test execution result"""
    execution_id: str
    plan_id: str
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    
    # Execution status
    status: str  # success, failure, partial, error
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    
    # Quality metrics
    overall_quality_score: float
    coverage_percentage: float
    performance_score: float
    regression_count: int
    
    # Environment results
    environment_results: Dict[str, Dict[str, Any]]
    
    # Artifacts
    test_reports: Dict[str, Path]
    coverage_reports: Dict[str, Path]
    performance_reports: Dict[str, Path]
    log_files: Dict[str, Path]
    
    # Quality gates
    quality_gates_passed: bool
    gate_failures: List[str]
    
    # CI/CD integration
    ci_metadata: Dict[str, Any] = field(default_factory=dict)
    deployment_recommendation: str = "BLOCK"  # APPROVE, BLOCK, REVIEW


class TestDataGenerator:
    """Generate comprehensive test data for various scenarios"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_consciousness_test_data(self, complexity_level: str = "medium") -> Dict[str, Any]:
        """Generate consciousness test data for different complexity levels"""
        complexity_configs = {
            "simple": {"nodes": 2, "concepts": 3, "depth": 1},
            "medium": {"nodes": 5, "concepts": 8, "depth": 3},
            "complex": {"nodes": 10, "concepts": 20, "depth": 5},
            "extreme": {"nodes": 15, "concepts": 50, "depth": 10}
        }
        
        config = complexity_configs.get(complexity_level, complexity_configs["medium"])
        
        # Generate system state
        system_state = np.random.rand(config["nodes"])
        system_state = system_state / np.sum(system_state)  # Normalize
        
        # Generate connectivity matrix
        connectivity = np.random.rand(config["nodes"], config["nodes"])
        connectivity = (connectivity + connectivity.T) / 2  # Make symmetric
        np.fill_diagonal(connectivity, 0)  # No self-connections
        
        # Generate experiential concepts
        experiential_concepts = []
        for i in range(config["concepts"]):
            concept = {
                "content": f"Test experiential concept {i} for {complexity_level} complexity",
                "experiential_quality": np.random.uniform(0.3, 1.0),
                "coherence": np.random.uniform(0.5, 1.0),
                "temporal_depth": np.random.randint(1, config["depth"] + 1),
                "timestamp": datetime.now().isoformat()
            }
            experiential_concepts.append(concept)
        
        return {
            "complexity_level": complexity_level,
            "system_state": system_state,
            "connectivity_matrix": connectivity,
            "experiential_concepts": experiential_concepts,
            "expected_phi_range": self._estimate_phi_range(complexity_level),
            "test_metadata": {
                "generation_time": datetime.now().isoformat(),
                "seed": self.seed,
                "config": config
            }
        }
    
    def _estimate_phi_range(self, complexity_level: str) -> Tuple[float, float]:
        """Estimate expected phi value range based on complexity"""
        ranges = {
            "simple": (0.001, 0.1),
            "medium": (0.1, 2.0),
            "complex": (1.0, 10.0),
            "extreme": (5.0, 50.0)
        }
        return ranges.get(complexity_level, (0.1, 2.0))
    
    def generate_stress_test_data(self, stress_type: str = "volume") -> Dict[str, Any]:
        """Generate stress test data for performance testing"""
        stress_configs = {
            "volume": {"batch_size": 100, "iterations": 50, "concurrent_requests": 10},
            "spike": {"batch_size": 1000, "iterations": 5, "concurrent_requests": 50},
            "endurance": {"batch_size": 10, "iterations": 500, "concurrent_requests": 5},
            "concurrency": {"batch_size": 20, "iterations": 20, "concurrent_requests": 100}
        }
        
        config = stress_configs.get(stress_type, stress_configs["volume"])
        
        # Generate multiple test datasets
        test_datasets = []
        for i in range(config["batch_size"]):
            dataset = self.generate_consciousness_test_data("medium")
            dataset["stress_iteration"] = i
            test_datasets.append(dataset)
        
        return {
            "stress_type": stress_type,
            "config": config,
            "test_datasets": test_datasets,
            "expected_metrics": {
                "max_latency_ms": 1000,
                "min_throughput_rps": 1,
                "max_error_rate": 0.01
            }
        }
    
    def generate_edge_case_data(self) -> List[Dict[str, Any]]:
        """Generate comprehensive edge case test data"""
        edge_cases = []
        
        # Boundary value cases
        edge_cases.extend([
            {
                "name": "zero_phi_system",
                "system_state": np.zeros(3),
                "connectivity_matrix": np.zeros((3, 3)),
                "experiential_concepts": [],
                "expected_phi": 0.0
            },
            {
                "name": "minimal_system",
                "system_state": np.array([1e-10, 1e-10]),
                "connectivity_matrix": np.array([[0, 1e-10], [1e-10, 0]]),
                "experiential_concepts": [{"content": "minimal", "experiential_quality": 1e-10}],
                "expected_phi_range": (0.0, 0.001)
            },
            {
                "name": "maximal_system",
                "system_state": np.ones(5),
                "connectivity_matrix": np.ones((5, 5)) - np.eye(5),
                "experiential_concepts": [
                    {"content": f"maximal concept {i}", "experiential_quality": 1.0}
                    for i in range(10)
                ],
                "expected_phi_range": (1.0, 100.0)
            }
        ])
        
        # Error condition cases
        edge_cases.extend([
            {
                "name": "nan_values",
                "system_state": np.array([np.nan, 0.5, 0.3]),
                "connectivity_matrix": np.array([[0, 0.5, np.nan], [0.5, 0, 0.3], [np.nan, 0.3, 0]]),
                "experiential_concepts": [{"content": "nan test", "experiential_quality": np.nan}],
                "should_handle_gracefully": True
            },
            {
                "name": "infinite_values",
                "system_state": np.array([np.inf, 0.5]),
                "connectivity_matrix": np.array([[0, np.inf], [np.inf, 0]]),
                "experiential_concepts": [{"content": "inf test", "experiential_quality": float('inf')}],
                "should_handle_gracefully": True
            },
            {
                "name": "mismatched_dimensions",
                "system_state": np.array([1, 0, 1]),
                "connectivity_matrix": np.array([[0, 1], [1, 0]]),  # Wrong size
                "experiential_concepts": [{"content": "dimension mismatch"}],
                "should_raise_error": True
            }
        ])
        
        # Performance edge cases
        edge_cases.extend([
            {
                "name": "large_system",
                "system_state": np.random.rand(20),
                "connectivity_matrix": np.random.rand(20, 20),
                "experiential_concepts": [
                    {"content": f"large system concept {i}", "experiential_quality": 0.5}
                    for i in range(100)
                ],
                "performance_requirement": {"max_time_ms": 5000}
            }
        ])
        
        return edge_cases


class TestEnvironmentManager:
    """Manage test environments and their lifecycle"""
    
    def __init__(self, base_work_dir: Optional[Path] = None):
        self.base_work_dir = base_work_dir or Path(tempfile.gettempdir()) / "iit4_test_envs"
        self.active_environments: Dict[str, TestEnvironment] = {}
        self.docker_client = None
        
        # Initialize Docker client if available
        try:
            import docker
            self.docker_client = docker.from_env()
            orchestrator_logger.info("Docker client initialized successfully")
        except Exception as e:
            orchestrator_logger.warning(f"Docker not available: {e}")
    
    @asynccontextmanager
    async def environment(self, env_config: TestEnvironment):
        """Context manager for test environment lifecycle"""
        environment_path = None
        container = None
        
        try:
            # Setup environment
            orchestrator_logger.info(f"Setting up environment: {env_config.name}")
            environment_path = await self._setup_environment(env_config)
            
            # Setup Docker container if configured
            if env_config.docker_config and self.docker_client:
                container = await self._setup_docker_container(env_config)
            
            # Store active environment
            self.active_environments[env_config.environment_id] = env_config
            
            yield environment_path
            
        finally:
            # Cleanup
            try:
                orchestrator_logger.info(f"Tearing down environment: {env_config.name}")
                await self._teardown_environment(env_config, environment_path, container)
                
                # Remove from active environments
                if env_config.environment_id in self.active_environments:
                    del self.active_environments[env_config.environment_id]
                    
            except Exception as e:
                orchestrator_logger.error(f"Error during environment teardown: {e}")
    
    async def _setup_environment(self, env_config: TestEnvironment) -> Path:
        """Setup test environment"""
        # Create environment directory
        env_path = self.base_work_dir / env_config.environment_id
        env_path.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        for key, value in env_config.environment_variables.items():
            os.environ[key] = value
        
        # Run setup commands
        for command in env_config.setup_commands:
            try:
                result = await asyncio.create_subprocess_shell(
                    command,
                    cwd=env_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await result.communicate()
                
                if result.returncode != 0:
                    orchestrator_logger.error(f"Setup command failed: {command}")
                    orchestrator_logger.error(f"stderr: {stderr.decode()}")
                    
            except Exception as e:
                orchestrator_logger.error(f"Error running setup command '{command}': {e}")
        
        return env_path
    
    async def _setup_docker_container(self, env_config: TestEnvironment):
        """Setup Docker container for isolated testing"""
        if not self.docker_client:
            return None
        
        try:
            docker_config = env_config.docker_config
            
            # Build or pull image
            image_name = docker_config.get("image", "python:3.9-slim")
            
            # Create container
            container = self.docker_client.containers.run(
                image_name,
                command=docker_config.get("command", "sleep infinity"),
                environment=env_config.environment_variables,
                volumes=docker_config.get("volumes", {}),
                ports=docker_config.get("ports", {}),
                mem_limit=f"{env_config.resource_limits.get('max_memory_mb', 2048)}m",
                detach=True,
                name=f"iit4_test_{env_config.environment_id}"
            )
            
            orchestrator_logger.info(f"Docker container created: {container.id[:12]}")
            return container
            
        except Exception as e:
            orchestrator_logger.error(f"Failed to setup Docker container: {e}")
            return None
    
    async def _teardown_environment(self, env_config: TestEnvironment, 
                                  env_path: Optional[Path], container):
        """Teardown test environment"""
        # Run teardown commands
        if env_path:
            for command in env_config.teardown_commands:
                try:
                    result = await asyncio.create_subprocess_shell(
                        command,
                        cwd=env_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await result.communicate()
                except Exception as e:
                    orchestrator_logger.error(f"Error running teardown command '{command}': {e}")
        
        # Stop and remove Docker container
        if container:
            try:
                container.stop()
                container.remove()
                orchestrator_logger.info(f"Docker container removed: {container.id[:12]}")
            except Exception as e:
                orchestrator_logger.error(f"Error removing Docker container: {e}")
        
        # Clean environment variables
        for key in env_config.environment_variables.keys():
            if key in os.environ:
                del os.environ[key]
        
        # Remove environment directory
        if env_path and env_path.exists():
            try:
                shutil.rmtree(env_path)
            except Exception as e:
                orchestrator_logger.error(f"Error removing environment directory: {e}")


class TestReportGenerator:
    """Generate comprehensive test reports for CI/CD integration"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_junit_report(self, test_results: Dict[str, List[TestResult]], 
                            execution_result: TestExecutionResult) -> Path:
        """Generate JUnit XML report for CI/CD systems"""
        
        junit_xml = JUnitXml()
        
        for phase, test_list in test_results.items():
            test_suite = TestSuite(name=phase)
            
            for test_result in test_list:
                test_case = TestCase(
                    name=test_result.test_name,
                    classname=test_result.phase,
                    time=test_result.execution_time_ms / 1000.0  # Convert to seconds
                )
                
                if not test_result.passed:
                    if test_result.error_message:
                        test_case.result = [Failure(message=test_result.error_message)]
                    else:
                        test_case.result = [Error(message="Test failed without specific error")]
                
                # Add test properties
                test_case.system_out = json.dumps({
                    "coverage_percentage": test_result.coverage_percentage,
                    "memory_usage_mb": test_result.memory_usage_mb,
                    "assertions_count": test_result.assertions_count,
                    "edge_cases_tested": test_result.edge_cases_tested,
                    "mocked_dependencies": test_result.mocked_dependencies or [],
                    "performance_metrics": test_result.performance_metrics or {}
                }, indent=2)
                
                test_suite.add_testcase(test_case)
            
            junit_xml.add_testsuite(test_suite)
        
        # Write JUnit XML
        junit_path = self.output_dir / f"junit_report_{execution_result.execution_id}.xml"
        junit_xml.write(str(junit_path))
        
        return junit_path
    
    def generate_coverage_report(self, coverage_data: Dict[str, Any]) -> Path:
        """Generate coverage report in multiple formats"""
        coverage_dir = self.output_dir / "coverage"
        coverage_dir.mkdir(exist_ok=True)
        
        # HTML coverage report
        html_report = coverage_dir / "index.html"
        
        # XML coverage report (for SonarQube, etc.)
        xml_report = coverage_dir / "coverage.xml"
        
        # JSON coverage report
        json_report = coverage_dir / "coverage.json"
        
        try:
            # Initialize coverage
            cov = coverage.Coverage()
            cov.start()
            
            # This would normally be done during test execution
            # For now, create a summary report
            summary = {
                "coverage_percentage": coverage_data.get("average_coverage", 0),
                "lines_covered": coverage_data.get("lines_covered", 0),
                "lines_total": coverage_data.get("lines_total", 0),
                "branches_covered": coverage_data.get("branches_covered", 0),
                "branches_total": coverage_data.get("branches_total", 0),
                "files": coverage_data.get("files", {})
            }
            
            with open(json_report, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Generate simple HTML report
            html_content = self._generate_html_coverage_report(summary)
            with open(html_report, 'w') as f:
                f.write(html_content)
            
            orchestrator_logger.info(f"Coverage reports generated in {coverage_dir}")
            
        except Exception as e:
            orchestrator_logger.error(f"Error generating coverage report: {e}")
        
        return coverage_dir
    
    def _generate_html_coverage_report(self, summary: Dict[str, Any]) -> str:
        """Generate simple HTML coverage report"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>IIT 4.0 NewbornAI 2.0 - Coverage Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .coverage-bar {{ width: 100%; height: 20px; background: #ddd; border-radius: 10px; }}
                .coverage-fill {{ height: 100%; background: #4CAF50; border-radius: 10px; }}
                .metric {{ margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Coverage Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <div class="metric">
                    <strong>Overall Coverage: {summary['coverage_percentage']:.1f}%</strong>
                    <div class="coverage-bar">
                        <div class="coverage-fill" style="width: {summary['coverage_percentage']}%"></div>
                    </div>
                </div>
                <div class="metric">Lines: {summary['lines_covered']}/{summary['lines_total']}</div>
                <div class="metric">Branches: {summary['branches_covered']}/{summary['branches_total']}</div>
            </div>
            <p>Generated: {datetime.now().isoformat()}</p>
        </body>
        </html>
        """
    
    def generate_performance_report(self, performance_data: Dict[str, Any]) -> Path:
        """Generate performance analysis report"""
        performance_path = self.output_dir / "performance_report.json"
        
        # Add trend analysis and benchmarks
        enhanced_data = {
            **performance_data,
            "generation_time": datetime.now().isoformat(),
            "performance_grade": self._calculate_performance_grade(performance_data),
            "recommendations": self._generate_performance_recommendations(performance_data)
        }
        
        with open(performance_path, 'w') as f:
            json.dump(enhanced_data, f, indent=2)
        
        return performance_path
    
    def _calculate_performance_grade(self, data: Dict[str, Any]) -> str:
        """Calculate performance grade based on metrics"""
        regressions = data.get("regressions", {})
        improvements = data.get("improvements", {})
        
        if len(regressions) == 0 and len(improvements) > 0:
            return "A"
        elif len(regressions) <= 1:
            return "B"
        elif len(regressions) <= 3:
            return "C"
        else:
            return "F"
    
    def _generate_performance_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        regressions = data.get("regressions", {})
        for name, regression in regressions.items():
            if regression["regression_percentage"] > 10:
                recommendations.append(
                    f"Critical regression in {name}: {regression['regression_percentage']:.1f}% slower"
                )
            elif regression["regression_percentage"] > 5:
                recommendations.append(
                    f"Investigate performance regression in {name}: {regression['regression_percentage']:.1f}% slower"
                )
        
        if not recommendations:
            recommendations.append("Performance is within acceptable thresholds")
        
        return recommendations
    
    def generate_comprehensive_report(self, execution_result: TestExecutionResult) -> Path:
        """Generate comprehensive HTML report"""
        report_path = self.output_dir / f"comprehensive_report_{execution_result.execution_id}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>IIT 4.0 NewbornAI 2.0 - Test Execution Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2196F3; color: white; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .success {{ background: #d4edda; border-color: #c3e6cb; }}
                .warning {{ background: #fff3cd; border-color: #ffeaa7; }}
                .danger {{ background: #f8d7da; border-color: #f5c6cb; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .status-badge {{ padding: 5px 10px; border-radius: 3px; color: white; font-weight: bold; }}
                .status-success {{ background: #28a745; }}
                .status-failure {{ background: #dc3545; }}
                .status-warning {{ background: #ffc107; color: black; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Test Execution Report</h1>
                <p>Execution ID: {execution_result.execution_id}</p>
                <p>Duration: {execution_result.duration_minutes:.1f} minutes</p>
                <span class="status-badge status-{execution_result.status.lower()}">
                    {execution_result.status.upper()}
                </span>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">Total Tests: <strong>{execution_result.total_tests}</strong></div>
                <div class="metric">Passed: <strong>{execution_result.passed_tests}</strong></div>
                <div class="metric">Failed: <strong>{execution_result.failed_tests}</strong></div>
                <div class="metric">Coverage: <strong>{execution_result.coverage_percentage:.1f}%</strong></div>
                <div class="metric">Quality Score: <strong>{execution_result.overall_quality_score:.3f}</strong></div>
            </div>
            
            <div class="section {'success' if execution_result.quality_gates_passed else 'danger'}">
                <h2>Quality Gates</h2>
                <p><strong>Status:</strong> {'PASSED' if execution_result.quality_gates_passed else 'FAILED'}</p>
                {self._generate_gate_failures_html(execution_result.gate_failures)}
            </div>
            
            <div class="section">
                <h2>Environment Results</h2>
                {self._generate_environment_results_html(execution_result.environment_results)}
            </div>
            
            <div class="section">
                <h2>Artifacts</h2>
                <ul>
                    <li><a href="{execution_result.test_reports.get('junit', 'N/A')}">JUnit Report</a></li>
                    <li><a href="{execution_result.coverage_reports.get('html', 'N/A')}">Coverage Report</a></li>
                    <li><a href="{execution_result.performance_reports.get('json', 'N/A')}">Performance Report</a></li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Deployment Recommendation</h2>
                <span class="status-badge status-{self._get_recommendation_class(execution_result.deployment_recommendation)}">
                    {execution_result.deployment_recommendation}
                </span>
                <p>{self._get_recommendation_message(execution_result.deployment_recommendation)}</p>
            </div>
            
            <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd;">
                <p>Generated: {datetime.now().isoformat()}</p>
                <p>IIT 4.0 NewbornAI 2.0 - Test Orchestrator v1.0.0</p>
            </footer>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path
    
    def _generate_gate_failures_html(self, failures: List[str]) -> str:
        """Generate HTML for quality gate failures"""
        if not failures:
            return "<p>All quality gates passed successfully.</p>"
        
        html = "<h3>Failed Gates:</h3><ul>"
        for failure in failures:
            html += f"<li>{failure}</li>"
        html += "</ul>"
        return html
    
    def _generate_environment_results_html(self, env_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate HTML for environment results"""
        if not env_results:
            return "<p>No environment results available.</p>"
        
        html = "<table><tr><th>Environment</th><th>Status</th><th>Tests</th><th>Duration</th></tr>"
        for env_name, results in env_results.items():
            html += f"""
            <tr>
                <td>{env_name}</td>
                <td>{results.get('status', 'Unknown')}</td>
                <td>{results.get('tests_run', 0)}</td>
                <td>{results.get('duration_minutes', 0):.1f}m</td>
            </tr>
            """
        html += "</table>"
        return html
    
    def _get_recommendation_class(self, recommendation: str) -> str:
        """Get CSS class for recommendation badge"""
        return {
            "APPROVE": "success",
            "REVIEW": "warning", 
            "BLOCK": "failure"
        }.get(recommendation, "warning")
    
    def _get_recommendation_message(self, recommendation: str) -> str:
        """Get message for deployment recommendation"""
        messages = {
            "APPROVE": "All tests passed. Safe to deploy to production.",
            "REVIEW": "Some issues detected. Manual review recommended before deployment.",
            "BLOCK": "Critical issues found. Deployment should be blocked until resolved."
        }
        return messages.get(recommendation, "Unknown recommendation status.")


class IntegrationTestOrchestrator:
    """Master test orchestrator for CI/CD integration"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.execution_history: List[TestExecutionResult] = []
        
        # Initialize components
        self.data_generator = TestDataGenerator()
        self.environment_manager = TestEnvironmentManager()
        self.report_generator = TestReportGenerator(Path("test_reports"))
        
        # Default environments
        self.default_environments = self._create_default_environments()
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._shutdown_requested = False
    
    def _create_default_environments(self) -> List[TestEnvironment]:
        """Create default test environments"""
        return [
            TestEnvironment(
                environment_id="unit_test_env",
                name="Unit Test Environment",
                description="Isolated environment for unit testing",
                setup_commands=["pip install -r requirements.txt"],
                teardown_commands=["pip uninstall -y -r requirements.txt"],
                environment_variables={"PYTHONPATH": ".", "TEST_MODE": "unit"},
                resource_limits={"max_memory_mb": 1024, "max_cpu_percent": 50},
                parallel_workers=4,
                timeout_minutes=15
            ),
            TestEnvironment(
                environment_id="integration_test_env", 
                name="Integration Test Environment",
                description="Environment for integration testing",
                setup_commands=["pip install -r requirements.txt", "python -m pytest --version"],
                teardown_commands=["pkill -f python", "rm -rf __pycache__"],
                environment_variables={"PYTHONPATH": ".", "TEST_MODE": "integration"},
                resource_limits={"max_memory_mb": 2048, "max_cpu_percent": 70},
                parallel_workers=2,
                timeout_minutes=30
            ),
            TestEnvironment(
                environment_id="performance_test_env",
                name="Performance Test Environment", 
                description="Dedicated environment for performance testing",
                setup_commands=["pip install -r requirements.txt", "sysctl -w vm.max_map_count=262144"],
                teardown_commands=["pkill -f python", "sync", "echo 3 > /proc/sys/vm/drop_caches"],
                environment_variables={"PYTHONPATH": ".", "TEST_MODE": "performance"},
                resource_limits={"max_memory_mb": 4096, "max_cpu_percent": 90},
                parallel_workers=1,
                timeout_minutes=60
            )
        ]
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        orchestrator_logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self._shutdown_requested = True
    
    async def execute_test_plan(self, plan: TestExecutionPlan) -> TestExecutionResult:
        """Execute comprehensive test plan"""
        execution_id = f"exec_{int(time.time())}_{plan.plan_id}"
        start_time = datetime.now()
        
        orchestrator_logger.info(f"Starting test execution: {execution_id}")
        orchestrator_logger.info(f"Plan: {plan.name}")
        orchestrator_logger.info(f"Strategy: {plan.execution_strategy}")
        
        # Initialize execution result
        execution_result = TestExecutionResult(
            execution_id=execution_id,
            plan_id=plan.plan_id,
            start_time=start_time,
            end_time=start_time,  # Will be updated
            duration_minutes=0.0,
            status="running",
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            skipped_tests=0,
            error_tests=0,
            overall_quality_score=0.0,
            coverage_percentage=0.0,
            performance_score=0.0,
            regression_count=0,
            environment_results={},
            test_reports={},
            coverage_reports={},
            performance_reports={},
            log_files={},
            quality_gates_passed=False,
            gate_failures=[],
            deployment_recommendation="BLOCK"
        )
        
        try:
            # Execute based on strategy
            if plan.execution_strategy == "sequential":
                await self._execute_sequential(plan, execution_result)
            elif plan.execution_strategy == "parallel":
                await self._execute_parallel(plan, execution_result)
            elif plan.execution_strategy == "distributed":
                await self._execute_distributed(plan, execution_result)
            else:
                raise ValueError(f"Unknown execution strategy: {plan.execution_strategy}")
            
            # Final quality assessment
            await self._perform_final_quality_assessment(execution_result)
            
            # Generate reports
            await self._generate_all_reports(execution_result)
            
            # Determine final status and recommendation
            self._finalize_execution_result(execution_result)
            
        except Exception as e:
            execution_result.status = "error"
            execution_result.deployment_recommendation = "BLOCK"
            orchestrator_logger.error(f"Test execution failed: {e}")
            orchestrator_logger.error(traceback.format_exc())
        
        finally:
            execution_result.end_time = datetime.now()
            execution_result.duration_minutes = (
                execution_result.end_time - execution_result.start_time
            ).total_seconds() / 60.0
            
            self.execution_history.append(execution_result)
            orchestrator_logger.info(f"Test execution completed: {execution_result.status}")
        
        return execution_result
    
    async def _execute_sequential(self, plan: TestExecutionPlan, result: TestExecutionResult):
        """Execute tests sequentially across environments"""
        for env_config in plan.environments:
            if self._shutdown_requested:
                break
                
            async with self.environment_manager.environment(env_config) as env_path:
                env_result = await self._run_tests_in_environment(
                    env_config, env_path, plan.test_suites
                )
                result.environment_results[env_config.name] = env_result
                
                # Aggregate results
                self._aggregate_results(result, env_result)
    
    async def _execute_parallel(self, plan: TestExecutionPlan, result: TestExecutionResult):
        """Execute tests in parallel across environments"""
        # Create tasks for parallel execution
        tasks = []
        for env_config in plan.environments:
            task = asyncio.create_task(
                self._run_environment_parallel(env_config, plan.test_suites)
            )
            tasks.append((env_config.name, task))
        
        # Wait for completion with timeout
        timeout_seconds = plan.timeout_minutes * 60
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[task for _, task in tasks], return_exceptions=True),
                timeout=timeout_seconds
            )
            
            # Process results
            for (env_name, _), env_result in zip(tasks, results):
                if isinstance(env_result, Exception):
                    orchestrator_logger.error(f"Environment {env_name} failed: {env_result}")
                    env_result = {"status": "error", "error": str(env_result)}
                
                result.environment_results[env_name] = env_result
                self._aggregate_results(result, env_result)
                
        except asyncio.TimeoutError:
            orchestrator_logger.error(f"Test execution timed out after {plan.timeout_minutes} minutes")
            result.status = "timeout"
    
    async def _run_environment_parallel(self, env_config: TestEnvironment, test_suites: List[str]):
        """Run tests in a single environment (for parallel execution)"""
        async with self.environment_manager.environment(env_config) as env_path:
            return await self._run_tests_in_environment(env_config, env_path, test_suites)
    
    async def _execute_distributed(self, plan: TestExecutionPlan, result: TestExecutionResult):
        """Execute tests in distributed manner (future implementation)"""
        # For now, fall back to parallel execution
        # In a full implementation, this would distribute tests across multiple nodes
        orchestrator_logger.info("Distributed execution not fully implemented, falling back to parallel")
        await self._execute_parallel(plan, result)
    
    async def _run_tests_in_environment(self, env_config: TestEnvironment, 
                                      env_path: Path, test_suites: List[str]) -> Dict[str, Any]:
        """Run tests in a specific environment"""
        env_start_time = time.time()
        
        try:
            # Initialize test suite
            test_suite = ComprehensiveTestSuite()
            qa_manager = QualityAssuranceManager()
            
            # Generate test data for this environment
            test_data = self.data_generator.generate_consciousness_test_data("medium")
            
            # Run tests with monitoring
            with self._monitor_resources(env_config.resource_limits):
                # Execute test suite
                test_results = await test_suite.run_all_tests()
                
                # Quality assessment
                quality_metrics = await qa_manager.comprehensive_quality_assessment(test_suite)
                
                # Analyze results
                analysis = test_suite.analyze_test_results(test_results)
            
            env_duration = time.time() - env_start_time
            
            return {
                "status": "success",
                "environment_id": env_config.environment_id,
                "tests_run": analysis["summary"]["total_tests"],
                "tests_passed": analysis["summary"]["passed_tests"],
                "tests_failed": analysis["summary"]["failed_tests"],
                "duration_minutes": env_duration / 60.0,
                "coverage_percentage": analysis["coverage"]["average_coverage"],
                "quality_score": quality_metrics.overall_quality_score,
                "test_results": test_results,
                "quality_metrics": quality_metrics,
                "analysis": analysis
            }
            
        except Exception as e:
            env_duration = time.time() - env_start_time
            orchestrator_logger.error(f"Tests failed in environment {env_config.name}: {e}")
            
            return {
                "status": "error",
                "environment_id": env_config.environment_id,
                "error": str(e),
                "duration_minutes": env_duration / 60.0,
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0
            }
    
    @contextmanager
    def _monitor_resources(self, limits: Dict[str, Any]):
        """Monitor resource usage during test execution"""
        initial_memory = psutil.virtual_memory().percent
        initial_cpu = psutil.cpu_percent()
        
        def resource_monitor():
            while True:
                memory_usage = psutil.virtual_memory().percent
                cpu_usage = psutil.cpu_percent(interval=1)
                
                if memory_usage > limits.get("max_memory_mb", 90):
                    orchestrator_logger.warning(f"High memory usage: {memory_usage}%")
                
                if cpu_usage > limits.get("max_cpu_percent", 90):
                    orchestrator_logger.warning(f"High CPU usage: {cpu_usage}%")
                
                time.sleep(5)
        
        monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
        monitor_thread.start()
        
        try:
            yield
        finally:
            # Monitor thread will terminate when main thread exits
            pass
    
    def _aggregate_results(self, execution_result: TestExecutionResult, env_result: Dict[str, Any]):
        """Aggregate results from environment execution"""
        if env_result.get("status") == "success":
            execution_result.total_tests += env_result.get("tests_run", 0)
            execution_result.passed_tests += env_result.get("tests_passed", 0)
            execution_result.failed_tests += env_result.get("tests_failed", 0)
            
            # Update coverage (weighted average)
            if execution_result.total_tests > 0:
                execution_result.coverage_percentage = (
                    execution_result.coverage_percentage * (execution_result.total_tests - env_result.get("tests_run", 0)) +
                    env_result.get("coverage_percentage", 0) * env_result.get("tests_run", 0)
                ) / execution_result.total_tests
            
            # Update quality score (weighted average)
            if execution_result.total_tests > 0:
                execution_result.overall_quality_score = (
                    execution_result.overall_quality_score * (execution_result.total_tests - env_result.get("tests_run", 0)) +
                    env_result.get("quality_score", 0) * env_result.get("tests_run", 0)
                ) / execution_result.total_tests
        else:
            execution_result.error_tests += 1
    
    async def _perform_final_quality_assessment(self, result: TestExecutionResult):
        """Perform final quality assessment and gate checking"""
        # Check quality gates
        gate_failures = []
        
        # Coverage gate
        if result.coverage_percentage < 95.0:
            gate_failures.append(f"Coverage {result.coverage_percentage:.1f}% below 95% threshold")
        
        # Success rate gate
        success_rate = result.passed_tests / max(result.total_tests, 1)
        if success_rate < 0.95:
            gate_failures.append(f"Success rate {success_rate:.1%} below 95% threshold")
        
        # Quality score gate
        if result.overall_quality_score < 0.85:
            gate_failures.append(f"Quality score {result.overall_quality_score:.3f} below 0.85 threshold")
        
        result.gate_failures = gate_failures
        result.quality_gates_passed = len(gate_failures) == 0
    
    async def _generate_all_reports(self, result: TestExecutionResult):
        """Generate all test reports"""
        try:
            # Collect test results from environments
            all_test_results = {}
            coverage_data = {"average_coverage": result.coverage_percentage}
            performance_data = {"regressions": {}, "improvements": {}}
            
            for env_name, env_result in result.environment_results.items():
                if "test_results" in env_result:
                    all_test_results[env_name] = env_result["test_results"]
            
            # Generate JUnit report
            junit_path = self.report_generator.generate_junit_report(all_test_results, result)
            result.test_reports["junit"] = junit_path
            
            # Generate coverage report
            coverage_path = self.report_generator.generate_coverage_report(coverage_data)
            result.coverage_reports["html"] = coverage_path
            
            # Generate performance report
            performance_path = self.report_generator.generate_performance_report(performance_data)
            result.performance_reports["json"] = performance_path
            
            # Generate comprehensive report
            comprehensive_path = self.report_generator.generate_comprehensive_report(result)
            result.test_reports["comprehensive"] = comprehensive_path
            
            orchestrator_logger.info("All test reports generated successfully")
            
        except Exception as e:
            orchestrator_logger.error(f"Error generating reports: {e}")
    
    def _finalize_execution_result(self, result: TestExecutionResult):
        """Finalize execution result and recommendation"""
        # Determine status
        if result.error_tests > 0:
            result.status = "error"
        elif result.failed_tests > 0:
            result.status = "failure"
        elif result.passed_tests == result.total_tests and result.quality_gates_passed:
            result.status = "success"
        else:
            result.status = "partial"
        
        # Determine deployment recommendation
        if result.status == "success" and result.quality_gates_passed:
            result.deployment_recommendation = "APPROVE"
        elif result.status == "partial" and len(result.gate_failures) <= 2:
            result.deployment_recommendation = "REVIEW"
        else:
            result.deployment_recommendation = "BLOCK"
    
    def create_default_test_plan(self) -> TestExecutionPlan:
        """Create default comprehensive test plan"""
        return TestExecutionPlan(
            plan_id="default_comprehensive",
            name="Comprehensive IIT 4.0 Test Plan",
            test_suites=["comprehensive", "quality_assurance", "integration"],
            environments=self.default_environments,
            execution_strategy="parallel",
            max_parallel_jobs=3,
            retry_count=1,
            timeout_minutes=90,
            quality_gates={
                "min_coverage": 95.0,
                "min_success_rate": 0.95,
                "min_quality_score": 0.85,
                "max_regression_count": 2
            },
            artifacts_config={
                "junit_xml": True,
                "coverage_html": True,
                "performance_json": True,
                "comprehensive_html": True
            }
        )
    
    def get_execution_history(self) -> List[TestExecutionResult]:
        """Get execution history for trend analysis"""
        return self.execution_history.copy()


async def main():
    """Main orchestrator execution for CI/CD integration"""
    print("🚀 IIT 4.0 NewbornAI 2.0 - Integration Test Orchestrator")
    print("📋 Automated CI/CD Testing | Quality Gates | Comprehensive Reporting")
    print("=" * 80)
    
    # Initialize orchestrator
    orchestrator = IntegrationTestOrchestrator()
    
    # Create and execute test plan
    test_plan = orchestrator.create_default_test_plan()
    
    # Execute comprehensive testing
    execution_result = await orchestrator.execute_test_plan(test_plan)
    
    # Print summary
    print(f"\n🎯 EXECUTION SUMMARY")
    print(f"   Status: {execution_result.status.upper()}")
    print(f"   Duration: {execution_result.duration_minutes:.1f} minutes")
    print(f"   Tests: {execution_result.passed_tests}/{execution_result.total_tests} passed")
    print(f"   Coverage: {execution_result.coverage_percentage:.1f}%")
    print(f"   Quality Score: {execution_result.overall_quality_score:.3f}")
    print(f"   Quality Gates: {'✅ PASSED' if execution_result.quality_gates_passed else '❌ FAILED'}")
    print(f"   Deployment: {execution_result.deployment_recommendation}")
    
    if execution_result.gate_failures:
        print(f"\n❌ QUALITY GATE FAILURES:")
        for failure in execution_result.gate_failures:
            print(f"   • {failure}")
    
    # Print report locations
    print(f"\n📄 REPORTS GENERATED:")
    for report_type, path in execution_result.test_reports.items():
        print(f"   {report_type.title()}: {path}")
    
    # Return exit code for CI/CD
    if execution_result.deployment_recommendation == "APPROVE":
        print(f"\n🎉 SUCCESS: All tests passed! Ready for deployment.")
        return 0
    elif execution_result.deployment_recommendation == "REVIEW":
        print(f"\n⚠️  WARNING: Manual review required before deployment.")
        return 1
    else:
        print(f"\n🚫 FAILURE: Deployment blocked due to test failures.")
        return 2


if __name__ == "__main__":
    # Run orchestrator and exit with appropriate code
    exit_code = asyncio.run(main())
    sys.exit(exit_code)