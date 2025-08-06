"""
Clean Architecture Compliance Report for IIT 4.0 NewbornAI 2.0
Comprehensive analysis and automated detection of SOLID violations and architectural issues

This module provides:
1. Automated SOLID principles analysis
2. Clean Architecture layer boundary validation
3. Dependency analysis and circular dependency detection
4. Code organization and module structure assessment
5. Detailed violation reporting with specific recommendations

Author: Clean Architecture Engineer (Uncle Bob's expertise)
Date: 2025-08-03
Version: 1.0.0
"""

import ast
import re
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import importlib.util
import inspect


class ViolationType(Enum):
    """Types of Clean Architecture violations"""
    SRP_VIOLATION = "Single Responsibility Principle Violation"
    OCP_VIOLATION = "Open/Closed Principle Violation"
    LSP_VIOLATION = "Liskov Substitution Principle Violation"
    ISP_VIOLATION = "Interface Segregation Principle Violation"
    DIP_VIOLATION = "Dependency Inversion Principle Violation"
    LAYER_BOUNDARY_VIOLATION = "Layer Boundary Violation"
    CIRCULAR_DEPENDENCY = "Circular Dependency"
    TIGHT_COUPLING = "Tight Coupling"
    LOW_COHESION = "Low Cohesion"
    FRAMEWORK_DEPENDENCY = "Framework Dependency in Business Logic"


class Severity(Enum):
    """Violation severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class Violation:
    """Clean Architecture violation record"""
    violation_type: ViolationType
    severity: Severity
    file_path: str
    line_number: int
    description: str
    recommendation: str
    code_snippet: str = ""
    related_files: List[str] = field(default_factory=list)
    

@dataclass
class ModuleAnalysis:
    """Analysis result for a single module"""
    file_path: str
    class_count: int
    function_count: int
    lines_of_code: int
    complexity_score: float
    responsibilities: List[str]
    dependencies: List[str]
    violations: List[Violation]
    

@dataclass
class LayerAnalysis:
    """Analysis result for architectural layers"""
    layer_name: str
    modules: List[str]
    inward_dependencies: List[str]
    outward_dependencies: List[str]
    boundary_violations: List[Violation]
    

@dataclass
class ComplianceReport:
    """Complete Clean Architecture compliance report"""
    project_name: str
    analysis_timestamp: str
    total_files: int
    total_violations: int
    violation_summary: Dict[ViolationType, int]
    severity_summary: Dict[Severity, int]
    module_analyses: List[ModuleAnalysis]
    layer_analyses: List[LayerAnalysis]
    dependency_graph: Dict[str, List[str]]
    circular_dependencies: List[List[str]]
    overall_score: float
    recommendations: List[str]


class DependencyAnalyzer:
    """Analyzes dependencies between modules"""
    
    def __init__(self):
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
        
    def analyze_imports(self, file_path: str, content: str) -> List[str]:
        """Extract import dependencies from file content"""
        dependencies = []
        
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append(node.module)
        except SyntaxError:
            # Handle syntax errors gracefully
            pass
            
        # Filter for project-internal dependencies
        project_deps = []
        for dep in dependencies:
            if not self._is_external_dependency(dep):
                project_deps.append(dep)
                
        return project_deps
    
    def _is_external_dependency(self, module_name: str) -> bool:
        """Check if dependency is external (not part of the project)"""
        external_patterns = [
            'asyncio', 'numpy', 'pandas', 'sklearn', 'torch', 
            'fastapi', 'pydantic', 'uvicorn', 'aiohttp', 'aioredis',
            'logging', 'json', 'time', 'datetime', 'pathlib',
            'typing', 'dataclasses', 'enum', 'collections',
            'concurrent', 'threading', 'multiprocessing'
        ]
        return any(module_name.startswith(pattern) for pattern in external_patterns)
    
    def add_dependency(self, from_module: str, to_module: str):
        """Add dependency relationship"""
        self.dependency_graph[from_module].add(to_module)
        self.reverse_graph[to_module].add(from_module)
    
    def find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies using DFS"""
        visited = set()
        rec_stack = set()
        cycles = []
        nodes = list(self.dependency_graph.keys())  # Create snapshot of keys
        
        def dfs(node: str, path: List[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path.copy()):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    try:
                        cycle_start = path.index(neighbor)
                        cycle = path[cycle_start:] + [neighbor]
                        cycles.append(cycle)
                    except ValueError:
                        pass  # Skip if neighbor not in path
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in nodes:
            if node not in visited:
                dfs(node, [])
        
        return cycles


class SOLIDAnalyzer:
    """Analyzes SOLID principles compliance"""
    
    def analyze_srp(self, file_path: str, content: str) -> List[Violation]:
        """Analyze Single Responsibility Principle compliance"""
        violations = []
        
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    violations.extend(self._check_class_srp(file_path, node, content))
                elif isinstance(node, ast.FunctionDef):
                    violations.extend(self._check_function_srp(file_path, node, content))
        except SyntaxError:
            pass
            
        return violations
    
    def _check_class_srp(self, file_path: str, class_node: ast.ClassDef, content: str) -> List[Violation]:
        """Check SRP compliance for a class"""
        violations = []
        
        # Count responsibilities by analyzing method types
        responsibilities = self._identify_class_responsibilities(class_node)
        
        if len(responsibilities) > 3:  # Threshold for too many responsibilities
            violation = Violation(
                violation_type=ViolationType.SRP_VIOLATION,
                severity=Severity.HIGH,
                file_path=file_path,
                line_number=class_node.lineno,
                description=f"Class '{class_node.name}' has {len(responsibilities)} responsibilities: {', '.join(responsibilities)}",
                recommendation="Split this class into smaller, focused classes each with a single responsibility",
                code_snippet=f"class {class_node.name}:"
            )
            violations.append(violation)
            
        return violations
    
    def _identify_class_responsibilities(self, class_node: ast.ClassDef) -> List[str]:
        """Identify different responsibilities in a class"""
        responsibilities = set()
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                method_name = node.name
                
                # Categorize methods by their purpose
                if method_name.startswith('calculate'):
                    responsibilities.add('Calculation')
                elif method_name.startswith('process'):
                    responsibilities.add('Processing')
                elif method_name.startswith('save') or method_name.startswith('load'):
                    responsibilities.add('Persistence')
                elif method_name.startswith('validate'):
                    responsibilities.add('Validation')
                elif method_name.startswith('format') or method_name.startswith('render'):
                    responsibilities.add('Presentation')
                elif method_name.startswith('send') or method_name.startswith('receive'):
                    responsibilities.add('Communication')
                elif 'file' in method_name or 'path' in method_name:
                    responsibilities.add('File Management')
                elif 'log' in method_name or 'debug' in method_name:
                    responsibilities.add('Logging')
                elif len(node.body) > 20:  # Large methods suggest multiple concerns
                    responsibilities.add('Complex Logic')
                    
        return list(responsibilities)
    
    def _check_function_srp(self, file_path: str, func_node: ast.FunctionDef, content: str) -> List[Violation]:
        """Check SRP compliance for a function"""
        violations = []
        
        # Check function length as indicator of multiple responsibilities
        func_length = len(func_node.body)
        
        if func_length > 30:  # Threshold for function length
            violation = Violation(
                violation_type=ViolationType.SRP_VIOLATION,
                severity=Severity.MEDIUM,
                file_path=file_path,
                line_number=func_node.lineno,
                description=f"Function '{func_node.name}' is too long ({func_length} statements) and likely has multiple responsibilities",
                recommendation="Break this function into smaller, focused functions each handling a single concern",
                code_snippet=f"def {func_node.name}():"
            )
            violations.append(violation)
            
        return violations
    
    def analyze_dip(self, file_path: str, content: str) -> List[Violation]:
        """Analyze Dependency Inversion Principle compliance"""
        violations = []
        
        # Check for direct instantiation of concrete classes
        concrete_instantiations = self._find_concrete_instantiations(content)
        
        for line_num, class_name in concrete_instantiations:
            if self._is_framework_or_external_class(class_name):
                violation = Violation(
                    violation_type=ViolationType.DIP_VIOLATION,
                    severity=Severity.HIGH,
                    file_path=file_path,
                    line_number=line_num,
                    description=f"Direct instantiation of concrete class '{class_name}' violates DIP",
                    recommendation="Use dependency injection or abstract factory pattern instead of direct instantiation",
                    code_snippet=f"{class_name}()"
                )
                violations.append(violation)
        
        return violations
    
    def _find_concrete_instantiations(self, content: str) -> List[Tuple[int, str]]:
        """Find direct instantiations of concrete classes"""
        instantiations = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Look for patterns like ClassName() or self.attr = ClassName()
            matches = re.findall(r'([A-Z][a-zA-Z0-9_]*)\s*\(', line)
            for match in matches:
                if not match.startswith('_'):  # Skip private classes
                    instantiations.append((i, match))
                    
        return instantiations
    
    def _is_framework_or_external_class(self, class_name: str) -> bool:
        """Check if class is likely a framework or external dependency"""
        framework_patterns = [
            'Calculator', 'Processor', 'Engine', 'Manager', 'Service',
            'Client', 'Session', 'Connection', 'Queue', 'Cache'
        ]
        return any(pattern in class_name for pattern in framework_patterns)


class LayerBoundaryAnalyzer:
    """Analyzes Clean Architecture layer boundaries"""
    
    LAYER_DEFINITIONS = {
        'Entities': ['entity', 'model', 'domain'],
        'Use Cases': ['use_case', 'service', 'application'],
        'Interface Adapters': ['controller', 'presenter', 'gateway', 'repository'],
        'Frameworks & Drivers': ['api', 'server', 'database', 'external', 'infrastructure']
    }
    
    def classify_module(self, file_path: str) -> str:
        """Classify module into Clean Architecture layer"""
        file_name = Path(file_path).stem.lower()
        
        for layer, keywords in self.LAYER_DEFINITIONS.items():
            if any(keyword in file_name for keyword in keywords):
                return layer
                
        # Default classification based on patterns
        if 'core' in file_name or 'domain' in file_name:
            return 'Entities'
        elif 'business' in file_name or 'logic' in file_name:
            return 'Use Cases'
        elif 'interface' in file_name or 'adapter' in file_name:
            return 'Interface Adapters'
        else:
            return 'Frameworks & Drivers'
    
    def check_dependency_direction(self, from_layer: str, to_layer: str) -> bool:
        """Check if dependency direction follows Clean Architecture rules"""
        layer_order = ['Frameworks & Drivers', 'Interface Adapters', 'Use Cases', 'Entities']
        
        try:
            from_index = layer_order.index(from_layer)
            to_index = layer_order.index(to_layer)
            
            # Dependencies should only flow inward (higher index to lower index)
            return from_index >= to_index
        except ValueError:
            return False  # Unknown layer
    
    def analyze_layer_violations(self, dependency_graph: Dict[str, List[str]]) -> List[Violation]:
        """Find layer boundary violations"""
        violations = []
        
        for from_module, dependencies in dependency_graph.items():
            from_layer = self.classify_module(from_module)
            
            for to_module in dependencies:
                to_layer = self.classify_module(to_module)
                
                if not self.check_dependency_direction(from_layer, to_layer):
                    violation = Violation(
                        violation_type=ViolationType.LAYER_BOUNDARY_VIOLATION,
                        severity=Severity.HIGH,
                        file_path=from_module,
                        line_number=1,
                        description=f"Invalid dependency from {from_layer} to {to_layer}",
                        recommendation=f"Invert this dependency using dependency injection or interface abstraction",
                        related_files=[to_module]
                    )
                    violations.append(violation)
        
        return violations


class CleanArchitectureAnalyzer:
    """Main Clean Architecture compliance analyzer"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.dependency_analyzer = DependencyAnalyzer()
        self.solid_analyzer = SOLIDAnalyzer()
        self.layer_analyzer = LayerBoundaryAnalyzer()
        
    def analyze_project(self) -> ComplianceReport:
        """Perform comprehensive Clean Architecture analysis"""
        print("🔍 Starting Clean Architecture compliance analysis...")
        start_time = time.time()
        
        # Find all Python files
        python_files = list(self.project_root.glob("*.py"))
        print(f"📁 Found {len(python_files)} Python files to analyze")
        
        # Analyze each module
        module_analyses = []
        all_violations = []
        dependency_graph = {}
        
        for file_path in python_files:
            print(f"🔬 Analyzing {file_path.name}...")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                analysis = self._analyze_module(str(file_path), content)
                module_analyses.append(analysis)
                all_violations.extend(analysis.violations)
                
                # Build dependency graph
                if analysis.dependencies:
                    dependency_graph[str(file_path)] = analysis.dependencies
                    for dep in analysis.dependencies:
                        self.dependency_analyzer.add_dependency(str(file_path), dep)
                        
            except Exception as e:
                print(f"⚠️  Error analyzing {file_path}: {e}")
                continue
        
        # Analyze layer boundaries
        layer_violations = self.layer_analyzer.analyze_layer_violations(dependency_graph)
        all_violations.extend(layer_violations)
        
        # Find circular dependencies
        circular_deps = self.dependency_analyzer.find_circular_dependencies()
        for cycle in circular_deps:
            violation = Violation(
                violation_type=ViolationType.CIRCULAR_DEPENDENCY,
                severity=Severity.CRITICAL,
                file_path=cycle[0],
                line_number=1,
                description=f"Circular dependency detected: {' -> '.join(cycle)}",
                recommendation="Break the circular dependency by introducing interfaces or dependency injection",
                related_files=cycle[1:]
            )
            all_violations.append(violation)
        
        # Generate layer analyses
        layer_analyses = self._generate_layer_analyses(module_analyses, dependency_graph)
        
        # Calculate summary statistics
        violation_summary = self._calculate_violation_summary(all_violations)
        severity_summary = self._calculate_severity_summary(all_violations)
        overall_score = self._calculate_overall_score(all_violations, len(python_files))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_violations, overall_score)
        
        analysis_time = time.time() - start_time
        print(f"✅ Analysis completed in {analysis_time:.2f} seconds")
        print(f"📊 Found {len(all_violations)} violations across {len(python_files)} files")
        
        return ComplianceReport(
            project_name="IIT 4.0 NewbornAI 2.0",
            analysis_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_files=len(python_files),
            total_violations=len(all_violations),
            violation_summary=violation_summary,
            severity_summary=severity_summary,
            module_analyses=module_analyses,
            layer_analyses=layer_analyses,
            dependency_graph=dependency_graph,
            circular_dependencies=circular_deps,
            overall_score=overall_score,
            recommendations=recommendations
        )
    
    def _analyze_module(self, file_path: str, content: str) -> ModuleAnalysis:
        """Analyze a single module"""
        violations = []
        
        # SOLID analysis
        violations.extend(self.solid_analyzer.analyze_srp(file_path, content))
        violations.extend(self.solid_analyzer.analyze_dip(file_path, content))
        
        # Additional analysis
        violations.extend(self._check_framework_dependencies(file_path, content))
        violations.extend(self._check_coupling_issues(file_path, content))
        
        # Extract module metrics
        metrics = self._extract_module_metrics(content)
        dependencies = self.dependency_analyzer.analyze_imports(file_path, content)
        responsibilities = self._identify_module_responsibilities(content)
        
        return ModuleAnalysis(
            file_path=file_path,
            class_count=metrics['class_count'],
            function_count=metrics['function_count'],
            lines_of_code=metrics['lines_of_code'],
            complexity_score=metrics['complexity_score'],
            responsibilities=responsibilities,
            dependencies=dependencies,
            violations=violations
        )
    
    def _check_framework_dependencies(self, file_path: str, content: str) -> List[Violation]:
        """Check for framework dependencies in business logic"""
        violations = []
        
        # Check if this is likely business logic
        if any(keyword in file_path.lower() for keyword in ['core', 'domain', 'entity', 'business']):
            # Look for framework imports
            framework_imports = [
                'fastapi', 'flask', 'django', 'aiohttp', 'uvicorn',
                'sqlalchemy', 'pymongo', 'redis', 'celery'
            ]
            
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                for framework in framework_imports:
                    if f'import {framework}' in line or f'from {framework}' in line:
                        violation = Violation(
                            violation_type=ViolationType.FRAMEWORK_DEPENDENCY,
                            severity=Severity.HIGH,
                            file_path=file_path,
                            line_number=i,
                            description=f"Business logic depends on framework '{framework}'",
                            recommendation="Use dependency injection to decouple business logic from frameworks",
                            code_snippet=line.strip()
                        )
                        violations.append(violation)
        
        return violations
    
    def _check_coupling_issues(self, file_path: str, content: str) -> List[Violation]:
        """Check for tight coupling issues"""
        violations = []
        
        # Count imports as coupling indicator
        import_count = len(re.findall(r'^(import|from)\s+', content, re.MULTILINE))
        
        if import_count > 15:  # Threshold for too many imports
            violation = Violation(
                violation_type=ViolationType.TIGHT_COUPLING,
                severity=Severity.MEDIUM,
                file_path=file_path,
                line_number=1,
                description=f"Module has {import_count} imports indicating tight coupling",
                recommendation="Reduce dependencies by using interfaces and dependency injection",
                code_snippet=""
            )
            violations.append(violation)
        
        return violations
    
    def _extract_module_metrics(self, content: str) -> Dict[str, Any]:
        """Extract metrics from module content"""
        metrics = {
            'class_count': 0,
            'function_count': 0,
            'lines_of_code': len(content.split('\n')),
            'complexity_score': 0.0
        }
        
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    metrics['class_count'] += 1
                elif isinstance(node, ast.FunctionDef):
                    metrics['function_count'] += 1
                    
            # Simple complexity calculation
            metrics['complexity_score'] = (
                metrics['class_count'] * 2 + 
                metrics['function_count'] + 
                metrics['lines_of_code'] / 100
            )
        except SyntaxError:
            pass
            
        return metrics
    
    def _identify_module_responsibilities(self, content: str) -> List[str]:
        """Identify module responsibilities"""
        responsibilities = set()
        
        # Look for common patterns in code
        patterns = {
            'Data Processing': ['process', 'transform', 'convert', 'parse'],
            'Calculation': ['calculate', 'compute', 'analyze', 'algorithm'],
            'Storage': ['save', 'load', 'store', 'persist', 'database'],
            'Communication': ['send', 'receive', 'request', 'response', 'api'],
            'Validation': ['validate', 'check', 'verify', 'ensure'],
            'Configuration': ['config', 'settings', 'options', 'parameter'],
            'Logging': ['log', 'debug', 'info', 'warn', 'error'],
            'Testing': ['test', 'mock', 'assert', 'expect']
        }
        
        content_lower = content.lower()
        for responsibility, keywords in patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                responsibilities.add(responsibility)
        
        return list(responsibilities)
    
    def _generate_layer_analyses(self, module_analyses: List[ModuleAnalysis], 
                                dependency_graph: Dict[str, List[str]]) -> List[LayerAnalysis]:
        """Generate layer-based analyses"""
        layers = defaultdict(list)
        
        # Group modules by layer
        for analysis in module_analyses:
            layer = self.layer_analyzer.classify_module(analysis.file_path)
            layers[layer].append(analysis.file_path)
        
        layer_analyses = []
        for layer_name, modules in layers.items():
            # Calculate layer dependencies
            inward_deps = []
            outward_deps = []
            boundary_violations = []
            
            for module in modules:
                if module in dependency_graph:
                    for dep in dependency_graph[module]:
                        dep_layer = self.layer_analyzer.classify_module(dep)
                        if dep_layer != layer_name:
                            outward_deps.append(dep_layer)
                            if not self.layer_analyzer.check_dependency_direction(layer_name, dep_layer):
                                violation = Violation(
                                    violation_type=ViolationType.LAYER_BOUNDARY_VIOLATION,
                                    severity=Severity.HIGH,
                                    file_path=module,
                                    line_number=1,
                                    description=f"Layer boundary violation: {layer_name} -> {dep_layer}",
                                    recommendation="Use dependency inversion to fix layer dependency",
                                    related_files=[dep]
                                )
                                boundary_violations.append(violation)
            
            layer_analyses.append(LayerAnalysis(
                layer_name=layer_name,
                modules=modules,
                inward_dependencies=list(set(inward_deps)),
                outward_dependencies=list(set(outward_deps)),
                boundary_violations=boundary_violations
            ))
        
        return layer_analyses
    
    def _calculate_violation_summary(self, violations: List[Violation]) -> Dict[ViolationType, int]:
        """Calculate violation summary by type"""
        summary = defaultdict(int)
        for violation in violations:
            summary[violation.violation_type] += 1
        return dict(summary)
    
    def _calculate_severity_summary(self, violations: List[Violation]) -> Dict[Severity, int]:
        """Calculate violation summary by severity"""
        summary = defaultdict(int)
        for violation in violations:
            summary[violation.severity] += 1
        return dict(summary)
    
    def _calculate_overall_score(self, violations: List[Violation], total_files: int) -> float:
        """Calculate overall Clean Architecture compliance score"""
        if total_files == 0:
            return 0.0
        
        # Weight violations by severity
        severity_weights = {
            Severity.CRITICAL: 10,
            Severity.HIGH: 5,
            Severity.MEDIUM: 2,
            Severity.LOW: 1,
            Severity.INFO: 0.5
        }
        
        total_penalty = sum(severity_weights[v.severity] for v in violations)
        max_possible_penalty = total_files * 20  # Assume max 20 points penalty per file
        
        # Score from 0 to 100
        score = max(0, 100 - (total_penalty / max_possible_penalty * 100))
        return min(100, score)
    
    def _generate_recommendations(self, violations: List[Violation], overall_score: float) -> List[str]:
        """Generate architectural improvement recommendations"""
        recommendations = []
        
        # Overall assessment
        if overall_score >= 80:
            recommendations.append("✅ Excellent Clean Architecture compliance! Minor improvements needed.")
        elif overall_score >= 60:
            recommendations.append("🟡 Good Clean Architecture foundation with some violations to address.")
        elif overall_score >= 40:
            recommendations.append("🟠 Moderate Clean Architecture compliance - significant improvements needed.")
        else:
            recommendations.append("🔴 Poor Clean Architecture compliance - major refactoring required.")
        
        # Specific recommendations based on violation types
        violation_counts = self._calculate_violation_summary(violations)
        
        if ViolationType.SRP_VIOLATION in violation_counts:
            count = violation_counts[ViolationType.SRP_VIOLATION]
            recommendations.append(
                f"📋 Address {count} Single Responsibility violations by breaking large classes/functions into smaller, focused units"
            )
        
        if ViolationType.DIP_VIOLATION in violation_counts:
            count = violation_counts[ViolationType.DIP_VIOLATION]
            recommendations.append(
                f"🔄 Resolve {count} Dependency Inversion violations by using interfaces and dependency injection"
            )
        
        if ViolationType.LAYER_BOUNDARY_VIOLATION in violation_counts:
            count = violation_counts[ViolationType.LAYER_BOUNDARY_VIOLATION]
            recommendations.append(
                f"🏗️  Fix {count} layer boundary violations by enforcing inward-only dependencies"
            )
        
        if ViolationType.CIRCULAR_DEPENDENCY in violation_counts:
            count = violation_counts[ViolationType.CIRCULAR_DEPENDENCY]
            recommendations.append(
                f"🔄 Break {count} circular dependencies using interface abstractions"
            )
        
        if ViolationType.FRAMEWORK_DEPENDENCY in violation_counts:
            count = violation_counts[ViolationType.FRAMEWORK_DEPENDENCY]
            recommendations.append(
                f"⚡ Remove {count} framework dependencies from business logic layers"
            )
        
        # Priority recommendations
        critical_violations = [v for v in violations if v.severity == Severity.CRITICAL]
        if critical_violations:
            recommendations.insert(1, f"🚨 PRIORITY: Address {len(critical_violations)} CRITICAL violations immediately")
        
        return recommendations
    
    def generate_detailed_report(self, report: ComplianceReport) -> str:
        """Generate detailed text report"""
        lines = []
        lines.append("=" * 80)
        lines.append("CLEAN ARCHITECTURE COMPLIANCE REPORT")
        lines.append("IIT 4.0 NewbornAI 2.0 Implementation")
        lines.append("=" * 80)
        lines.append("")
        
        # Summary
        lines.append(f"📊 PROJECT OVERVIEW")
        lines.append(f"   Project: {report.project_name}")
        lines.append(f"   Analysis Date: {report.analysis_timestamp}")
        lines.append(f"   Files Analyzed: {report.total_files}")
        lines.append(f"   Total Violations: {report.total_violations}")
        lines.append(f"   Overall Score: {report.overall_score:.1f}/100")
        lines.append("")
        
        # Severity Summary
        lines.append("🚨 VIOLATION SEVERITY BREAKDOWN")
        for severity, count in report.severity_summary.items():
            lines.append(f"   {severity.value}: {count}")
        lines.append("")
        
        # Violation Type Summary
        lines.append("📋 VIOLATION TYPE BREAKDOWN")
        for violation_type, count in report.violation_summary.items():
            lines.append(f"   {violation_type.value}: {count}")
        lines.append("")
        
        # Layer Analysis
        lines.append("🏗️  ARCHITECTURAL LAYER ANALYSIS")
        for layer_analysis in report.layer_analyses:
            lines.append(f"   Layer: {layer_analysis.layer_name}")
            lines.append(f"     Modules: {len(layer_analysis.modules)}")
            lines.append(f"     Boundary Violations: {len(layer_analysis.boundary_violations)}")
            if layer_analysis.outward_dependencies:
                lines.append(f"     Dependencies: {', '.join(layer_analysis.outward_dependencies)}")
        lines.append("")
        
        # Circular Dependencies
        if report.circular_dependencies:
            lines.append("🔄 CIRCULAR DEPENDENCIES")
            for cycle in report.circular_dependencies:
                lines.append(f"   {' -> '.join([Path(f).name for f in cycle])}")
        lines.append("")
        
        # Top Violations
        all_violations = []
        for module in report.module_analyses:
            all_violations.extend(module.violations)
        
        critical_violations = [v for v in all_violations if v.severity == Severity.CRITICAL]
        high_violations = [v for v in all_violations if v.severity == Severity.HIGH]
        
        if critical_violations:
            lines.append("🚨 CRITICAL VIOLATIONS (Fix Immediately)")
            for violation in critical_violations[:5]:  # Top 5
                lines.append(f"   {Path(violation.file_path).name}:{violation.line_number}")
                lines.append(f"     {violation.description}")
                lines.append(f"     → {violation.recommendation}")
                lines.append("")
        
        if high_violations:
            lines.append("🔴 HIGH PRIORITY VIOLATIONS")
            for violation in high_violations[:10]:  # Top 10
                lines.append(f"   {Path(violation.file_path).name}:{violation.line_number}")
                lines.append(f"     {violation.violation_type.value}")
                lines.append(f"     {violation.description}")
                lines.append("")
        
        # Recommendations
        lines.append("💡 RECOMMENDATIONS")
        for i, recommendation in enumerate(report.recommendations, 1):
            lines.append(f"   {i}. {recommendation}")
        lines.append("")
        
        # Module Quality Rankings
        lines.append("📈 MODULE QUALITY RANKING")
        modules_by_violations = sorted(
            report.module_analyses,
            key=lambda m: len([v for v in m.violations if v.severity in [Severity.CRITICAL, Severity.HIGH]])
        )
        
        lines.append("   Best Modules (Lowest Violations):")
        for module in modules_by_violations[:5]:
            violation_count = len([v for v in module.violations if v.severity in [Severity.CRITICAL, Severity.HIGH]])
            lines.append(f"     {Path(module.file_path).name}: {violation_count} critical/high violations")
        
        lines.append("")
        lines.append("   Modules Needing Attention (Highest Violations):")
        for module in modules_by_violations[-5:]:
            violation_count = len([v for v in module.violations if v.severity in [Severity.CRITICAL, Severity.HIGH]])
            lines.append(f"     {Path(module.file_path).name}: {violation_count} critical/high violations")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)


def main():
    """Run Clean Architecture compliance analysis"""
    project_root = "/Users/yamaguchimitsuyuki/omoikane-lab/sandbox/tools/08_02_2025"
    
    print("🏗️  Clean Architecture Compliance Analysis")
    print("=" * 60)
    
    analyzer = CleanArchitectureAnalyzer(project_root)
    report = analyzer.analyze_project()
    
    # Generate detailed report
    detailed_report = analyzer.generate_detailed_report(report)
    
    # Save report to file
    report_file = Path(project_root) / "clean_architecture_compliance_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(detailed_report)
    
    # Save JSON report for programmatic access
    json_report_file = Path(project_root) / "clean_architecture_compliance_report.json"
    
    # Convert report to JSON-serializable format
    json_data = {
        "project_name": report.project_name,
        "analysis_timestamp": report.analysis_timestamp,
        "total_files": report.total_files,
        "total_violations": report.total_violations,
        "overall_score": report.overall_score,
        "violation_summary": {vt.value: count for vt, count in report.violation_summary.items()},
        "severity_summary": {s.value: count for s, count in report.severity_summary.items()},
        "circular_dependencies": report.circular_dependencies,
        "recommendations": report.recommendations,
        "module_analyses": [
            {
                "file_path": m.file_path,
                "class_count": m.class_count,
                "function_count": m.function_count,
                "lines_of_code": m.lines_of_code,
                "complexity_score": m.complexity_score,
                "responsibilities": m.responsibilities,
                "violation_count": len(m.violations),
                "critical_violations": len([v for v in m.violations if v.severity == Severity.CRITICAL]),
                "high_violations": len([v for v in m.violations if v.severity == Severity.HIGH])
            }
            for m in report.module_analyses
        ]
    }
    
    with open(json_report_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\n📝 Reports generated:")
    print(f"   Detailed Report: {report_file}")
    print(f"   JSON Report: {json_report_file}")
    
    # Print summary to console
    print(f"\n📊 ANALYSIS SUMMARY")
    print(f"   Overall Score: {report.overall_score:.1f}/100")
    print(f"   Total Violations: {report.total_violations}")
    print(f"   Critical: {report.severity_summary.get(Severity.CRITICAL, 0)}")
    print(f"   High: {report.severity_summary.get(Severity.HIGH, 0)}")
    print(f"   Medium: {report.severity_summary.get(Severity.MEDIUM, 0)}")
    
    # Print top recommendations
    print(f"\n💡 TOP RECOMMENDATIONS:")
    for i, recommendation in enumerate(report.recommendations[:3], 1):
        print(f"   {i}. {recommendation}")
    
    return report


if __name__ == "__main__":
    main()