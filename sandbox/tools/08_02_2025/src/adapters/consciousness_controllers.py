"""
Consciousness Controllers - Interface Adapters Layer
Handle HTTP requests and coordinate with application use cases

Following Clean Architecture principles:
- Depends on application layer (use cases)
- Converts external requests to internal format
- Handles presentation concerns
- Implements dependency injection

Author: Clean Architecture Engineer (Uncle Bob's expertise)
Date: 2025-08-03
Version: 1.0.0
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import asdict
import logging

# Application layer imports (inward dependency)
from ..application.consciousness_use_cases import (
    ConsciousnessApplicationService,
    PhiCalculationResult, ConsciousnessAnalysisResult, DevelopmentProgressionResult
)
from ..domain.consciousness_entities import (
    SystemState, PhiValue, DevelopmentStage, ConsciousnessLevel
)

logger = logging.getLogger(__name__)


class ConsciousnessApiController:
    """
    REST API controller for consciousness operations
    Handles HTTP requests and coordinates with application services
    """
    
    def __init__(self, consciousness_service: ConsciousnessApplicationService):
        self._consciousness_service = consciousness_service
    
    async def calculate_phi(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle phi calculation API request
        
        Args:
            request_data: {
                "nodes": [1, 2, 3],
                "state_vector": [0.1, 0.8, 0.3],
                "connectivity_matrix": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                "track_events": true
            }
            
        Returns:
            API response with calculation results
        """
        try:
            # Validate request
            if not self._validate_phi_request(request_data):
                return {
                    "success": False,
                    "error": "Invalid request data",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Convert to domain entities
            system_state = self._convert_to_system_state(request_data)
            track_events = request_data.get("track_events", True)
            
            # Execute use case
            result = await self._consciousness_service.calculate_phi.execute(
                system_state, track_events
            )
            
            # Convert result to API response
            return self._format_phi_calculation_response(result)
            
        except Exception as e:
            logger.error(f"Phi calculation API error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_consciousness(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle consciousness analysis API request
        
        Args:
            request_data: {
                "nodes": [1, 2, 3],
                "state_vector": [0.1, 0.8, 0.3],
                "connectivity_matrix": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                "phi_history": [0.1, 0.15, 0.12, 0.18]
            }
            
        Returns:
            API response with consciousness analysis
        """
        try:
            # Validate request
            if not self._validate_analysis_request(request_data):
                return {
                    "success": False,
                    "error": "Invalid request data",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Convert to domain entities
            system_state = self._convert_to_system_state(request_data)
            phi_history = self._convert_to_phi_history(request_data.get("phi_history", []))
            
            # Execute use case
            result = await self._consciousness_service.analyze_consciousness.execute(
                system_state, phi_history
            )
            
            # Convert result to API response
            return self._format_consciousness_analysis_response(result)
            
        except Exception as e:
            logger.error(f"Consciousness analysis API error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def manage_development(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle development progression API request
        
        Args:
            request_data: {
                "nodes": [1, 2, 3],
                "state_vector": [0.1, 0.8, 0.3],
                "connectivity_matrix": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                "current_stage": "reflexive"
            }
            
        Returns:
            API response with development progression results
        """
        try:
            # Validate request
            if not self._validate_development_request(request_data):
                return {
                    "success": False,
                    "error": "Invalid request data",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Convert to domain entities
            system_state = self._convert_to_system_state(request_data)
            current_stage = self._convert_to_development_stage(
                request_data.get("current_stage", "reflexive")
            )
            
            # Execute use case
            result = await self._consciousness_service.manage_development.execute(
                system_state, current_stage
            )
            
            # Convert result to API response
            return self._format_development_response(result)
            
        except Exception as e:
            logger.error(f"Development management API error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def comprehensive_analysis(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle comprehensive consciousness analysis API request
        """
        try:
            # Validate request
            if not self._validate_comprehensive_request(request_data):
                return {
                    "success": False,
                    "error": "Invalid request data",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Convert to domain entities
            system_state = self._convert_to_system_state(request_data)
            phi_history = self._convert_to_phi_history(request_data.get("phi_history", []))
            current_stage = self._convert_to_development_stage(
                request_data.get("current_stage", "reflexive")
            )
            
            # Execute comprehensive use case
            result = await self._consciousness_service.comprehensive_analysis(
                system_state, phi_history, current_stage
            )
            
            # Format comprehensive response
            return self._format_comprehensive_response(result)
            
        except Exception as e:
            logger.error(f"Comprehensive analysis API error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # Private helper methods for request validation and conversion
    
    def _validate_phi_request(self, request_data: Dict[str, Any]) -> bool:
        """Validate phi calculation request data"""
        required_fields = ["nodes", "state_vector", "connectivity_matrix"]
        return all(field in request_data for field in required_fields)
    
    def _validate_analysis_request(self, request_data: Dict[str, Any]) -> bool:
        """Validate consciousness analysis request data"""
        return self._validate_phi_request(request_data)
    
    def _validate_development_request(self, request_data: Dict[str, Any]) -> bool:
        """Validate development management request data"""
        return self._validate_phi_request(request_data)
    
    def _validate_comprehensive_request(self, request_data: Dict[str, Any]) -> bool:
        """Validate comprehensive analysis request data"""
        return self._validate_phi_request(request_data)
    
    def _convert_to_system_state(self, request_data: Dict[str, Any]) -> SystemState:
        """Convert request data to SystemState domain entity"""
        nodes = frozenset(request_data["nodes"])
        state_vector = tuple(request_data["state_vector"])
        connectivity_matrix = tuple(tuple(row) for row in request_data["connectivity_matrix"])
        
        return SystemState(
            nodes=nodes,
            state_vector=state_vector,
            connectivity_matrix=connectivity_matrix,
            timestamp=datetime.now()
        )
    
    def _convert_to_phi_history(self, phi_values: List[float]) -> List[PhiValue]:
        """Convert phi value list to PhiValue domain entities"""
        return [PhiValue(value=val) for val in phi_values]
    
    def _convert_to_development_stage(self, stage_name: str) -> DevelopmentStage:
        """Convert stage name to DevelopmentStage enum"""
        stage_mapping = {
            "reflexive": DevelopmentStage.STAGE_0_REFLEXIVE,
            "reactive": DevelopmentStage.STAGE_1_REACTIVE,
            "adaptive": DevelopmentStage.STAGE_2_ADAPTIVE,
            "predictive": DevelopmentStage.STAGE_3_PREDICTIVE,
            "reflective": DevelopmentStage.STAGE_4_REFLECTIVE,
            "introspective": DevelopmentStage.STAGE_5_INTROSPECTIVE,
            "metacognitive": DevelopmentStage.STAGE_6_METACOGNITIVE
        }
        return stage_mapping.get(stage_name, DevelopmentStage.STAGE_0_REFLEXIVE)
    
    def _format_phi_calculation_response(self, result: PhiCalculationResult) -> Dict[str, Any]:
        """Format phi calculation result for API response"""
        response = {
            "success": result.success,
            "calculation_id": result.calculation_id,
            "execution_time_ms": result.execution_time_ms,
            "timestamp": datetime.now().isoformat()
        }
        
        if result.success and result.phi_structure:
            response.update({
                "phi_value": result.phi_structure.system_phi.value,
                "consciousness_level": result.phi_structure.consciousness_level.name,
                "development_stage": result.phi_structure.development_stage.name,
                "distinction_count": len(result.phi_structure.distinctions),
                "complexity": result.phi_structure.complexity,
                "is_conscious": result.phi_structure.is_conscious()
            })
        elif result.error_message:
            response["error"] = result.error_message
        
        return response
    
    def _format_consciousness_analysis_response(self, result: ConsciousnessAnalysisResult) -> Dict[str, Any]:
        """Format consciousness analysis result for API response"""
        return {
            "success": True,
            "analysis_id": result.analysis_id,
            "is_conscious": result.is_conscious,
            "consciousness_level": result.consciousness_level.name,
            "phi_value": result.phi_value.value,
            "development_stage": result.development_stage.name,
            "stability_score": result.stability_score,
            "timestamp": datetime.now().isoformat()
        }
    
    def _format_development_response(self, result: DevelopmentProgressionResult) -> Dict[str, Any]:
        """Format development progression result for API response"""
        return {
            "success": True,
            "previous_stage": result.previous_stage.name,
            "new_stage": result.new_stage.name,
            "progression_occurred": result.progression_occurred,
            "readiness_score": result.readiness_score,
            "next_milestone": result.next_milestone.name if result.next_milestone else None,
            "timestamp": datetime.now().isoformat()
        }
    
    def _format_comprehensive_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format comprehensive analysis result for API response"""
        if not result["success"]:
            return {
                "success": False,
                "error": result["error"],
                "timestamp": result["timestamp"].isoformat()
            }
        
        # Extract individual results
        phi_calc = result["phi_calculation"]
        consciousness = result["consciousness_analysis"]
        development = result["development_progression"]
        
        return {
            "success": True,
            "timestamp": result["timestamp"].isoformat(),
            "phi_calculation": {
                "calculation_id": phi_calc.calculation_id,
                "execution_time_ms": phi_calc.execution_time_ms,
                "phi_value": phi_calc.phi_structure.system_phi.value if phi_calc.phi_structure else 0,
                "complexity": phi_calc.phi_structure.complexity if phi_calc.phi_structure else 0
            },
            "consciousness_analysis": {
                "analysis_id": consciousness.analysis_id,
                "is_conscious": consciousness.is_conscious,
                "consciousness_level": consciousness.consciousness_level.name,
                "stability_score": consciousness.stability_score
            },
            "development_progression": {
                "previous_stage": development.previous_stage.name,
                "new_stage": development.new_stage.name,
                "progression_occurred": development.progression_occurred,
                "readiness_score": development.readiness_score,
                "next_milestone": development.next_milestone.name if development.next_milestone else None
            }
        }


class ConsciousnessStreamController:
    """
    WebSocket controller for real-time consciousness monitoring
    """
    
    def __init__(self, consciousness_service: ConsciousnessApplicationService):
        self._consciousness_service = consciousness_service
        self._active_streams = {}
    
    async def start_monitoring_stream(self, stream_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start real-time consciousness monitoring stream
        
        Args:
            stream_id: Unique identifier for the stream
            config: Stream configuration including monitoring interval
            
        Returns:
            Stream start confirmation
        """
        try:
            if stream_id in self._active_streams:
                return {
                    "success": False,
                    "error": f"Stream {stream_id} already active",
                    "timestamp": datetime.now().isoformat()
                }
            
            monitoring_interval = config.get("monitoring_interval", 0.1)
            
            # Create system state stream generator (mock for now)
            async def system_state_generator():
                # This would be connected to actual system state source
                import random
                while True:
                    yield SystemState(
                        nodes=frozenset([1, 2, 3]),
                        state_vector=(random.random(), random.random(), random.random()),
                        connectivity_matrix=((0, 1, 0), (1, 0, 1), (0, 1, 0)),
                        timestamp=datetime.now()
                    )
                    await asyncio.sleep(monitoring_interval)
            
            # Start monitoring in background task
            task = asyncio.create_task(
                self._consciousness_service.monitor_stream.start_monitoring(
                    system_state_generator(), monitoring_interval
                )
            )
            
            self._active_streams[stream_id] = {
                "task": task,
                "config": config,
                "start_time": datetime.now()
            }
            
            return {
                "success": True,
                "stream_id": stream_id,
                "monitoring_interval": monitoring_interval,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Stream monitoring start error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def stop_monitoring_stream(self, stream_id: str) -> Dict[str, Any]:
        """Stop consciousness monitoring stream"""
        try:
            if stream_id not in self._active_streams:
                return {
                    "success": False,
                    "error": f"Stream {stream_id} not found",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Stop monitoring
            self._consciousness_service.monitor_stream.stop_monitoring()
            
            # Cancel background task
            stream_info = self._active_streams[stream_id]
            stream_info["task"].cancel()
            
            # Calculate stream duration
            duration = datetime.now() - stream_info["start_time"]
            
            del self._active_streams[stream_id]
            
            return {
                "success": True,
                "stream_id": stream_id,
                "duration_seconds": duration.total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Stream monitoring stop error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_active_streams(self) -> Dict[str, Any]:
        """Get information about active monitoring streams"""
        return {
            "active_stream_count": len(self._active_streams),
            "streams": {
                stream_id: {
                    "start_time": info["start_time"].isoformat(),
                    "config": info["config"]
                }
                for stream_id, info in self._active_streams.items()
            },
            "timestamp": datetime.now().isoformat()
        }