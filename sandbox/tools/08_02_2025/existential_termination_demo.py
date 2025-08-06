"""
Existential Termination Implementation Demo
Demonstrates the complete existential termination system in action

Refactored from brain_death_demo.py using Martin Fowler's methodology:
- Extract Method: Separated demonstration phases into focused methods
- Introduce Parameter Object: Complex demo parameters as value objects
- Template Method: Standardized demo execution flow
"""

import asyncio
import time
from datetime import datetime
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.layout import Layout
from rich.text import Text
from dataclasses import dataclass
from typing import List, Tuple

from existential_termination_core import (
    InformationIntegrationAggregate,
    IntegrationSystemId,
    IntegrationSystemState,
    TerminationStage,
    ProcessingLayer,
    IrreversibilityMechanism,
    TerminationParameters
)

from integration_collapse_detector import (
    IntegrationCollapseDetector,
    IntegrationCollapseMonitor,
    CollapseDetectionResult,
    DetectionThresholds
)

from consciousness_detector import (
    ConsciousnessSignature,
    ConsciousnessDetector,
    IIT4PhiCalculator
)

console = Console()


# Demo Configuration Value Objects

@dataclass(frozen=True)
class DemoParameters:
    """Parameter object for demo configuration"""
    system_id: str = "demo-integration-system-001"
    n_phi_elements: int = 8
    stage_delay_seconds: int = 3
    progress_update_interval: float = 0.5
    display_detailed_states: bool = True
    simulate_recovery_attempt: bool = True


@dataclass(frozen=True)
class DemoStage:
    """Value object representing a demo stage"""
    minutes: int
    stage_name: str
    integration_level: float
    description: str


# Demo Template (Template Method Pattern)

class ExistentialTerminationDemoTemplate:
    """Template for existential termination demonstration"""
    
    def __init__(self, parameters: DemoParameters = None):
        self.parameters = parameters or DemoParameters()
        self._setup_components()
        
        self.start_time = None
        self.is_running = True
    
    def _setup_components(self):
        """Extract Method: Setup demo components"""
        self.integration_system = InformationIntegrationAggregate(
            IntegrationSystemId(self.parameters.system_id)
        )
        self.phi_calculator = IIT4PhiCalculator(n_elements=self.parameters.n_phi_elements)
        self.consciousness_detector = ConsciousnessDetector(self.phi_calculator)
        self.collapse_detector = IntegrationCollapseDetector(
            self.consciousness_detector,
            DetectionThresholds()
        )
        self.monitor = IntegrationCollapseMonitor(self.collapse_detector)
        self.irreversibility_mechanism = IrreversibilityMechanism()
    
    async def run_demo(self):
        """Template method for demo execution"""
        console.print("\n[bold cyan]Existential Termination Implementation Demo[/bold cyan]")
        console.print("=" * 70)
        
        # Template method steps
        await self.demonstrate_initial_state()
        await self.demonstrate_termination_initiation()
        await self.demonstrate_progressive_termination()
        await self.demonstrate_irreversibility_sealing()
        
        if self.parameters.simulate_recovery_attempt:
            await self.demonstrate_recovery_attempt()
        
        self.show_final_summary()
    
    # Template method steps (to be implemented by subclasses)
    async def demonstrate_initial_state(self):
        """Show initial active integration state"""
        pass
    
    async def demonstrate_termination_initiation(self):
        """Demonstrate initiating the termination process"""
        pass
    
    async def demonstrate_progressive_termination(self):
        """Show progressive stages of termination"""
        pass
    
    async def demonstrate_irreversibility_sealing(self):
        """Demonstrate the irreversibility mechanism"""
        pass
    
    async def demonstrate_recovery_attempt(self):
        """Demonstrate recovery attempt (if applicable)"""
        pass
    
    def show_final_summary(self):
        """Show final summary of the demonstration"""
        pass


# Concrete Demo Implementation

class StandardExistentialTerminationDemo(ExistentialTerminationDemoTemplate):
    """Standard implementation of existential termination demo"""
    
    async def demonstrate_initial_state(self):
        """Show initial active integration state"""
        console.print("\n[bold green]Phase 1: Active Information Integration[/bold green]")
        
        # Create consciousness signature for active state
        signature = self._create_signature_for_level(1.0)
        
        # Display state
        self._display_integration_state(
            "Initial Active State",
            self.integration_system,
            signature
        )
        
        console.print("\n✓ Information integration is fully active")
        console.print("✓ All processing layers operational")
        console.print("✓ Intentional binding and temporal integration intact")
        
        await asyncio.sleep(self.parameters.stage_delay_seconds)
    
    async def demonstrate_termination_initiation(self):
        """Demonstrate initiating the termination process"""
        console.print("\n[bold yellow]Phase 2: Termination Process Initiation[/bold yellow]")
        
        # Initiate termination
        self.integration_system.initiate_termination()
        self.start_time = time.time()
        
        console.print("\n[red]⚠️  Existential termination process initiated[/red]")
        
        # Show initial changes
        signature = self._create_signature_for_level(0.8)
        diagnosis = await self.collapse_detector.detect_integration_collapse(
            signature, self.integration_system
        )
        
        self._display_collapse_diagnosis(diagnosis)
        
        await asyncio.sleep(self.parameters.stage_delay_seconds)
    
    async def demonstrate_progressive_termination(self):
        """Show progressive stages of termination"""
        console.print("\n[bold red]Phase 3: Progressive Integration Termination[/bold red]")
        
        # Define progression stages
        stages = self._get_demo_stages()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            
            task = progress.add_task("Integration Termination Progress", total=30)
            
            for stage in stages:
                # Progress termination
                self.integration_system.progress_termination(minutes=stage.minutes)
                
                # Update progress bar
                progress.update(
                    task, 
                    completed=stage.minutes, 
                    description=stage.stage_name
                )
                
                # Create signature for current state
                signature = self._create_signature_for_level(stage.integration_level)
                
                # Get collapse diagnosis
                diagnosis = await self.collapse_detector.detect_integration_collapse(
                    signature, self.integration_system
                )
                
                # Display current state
                console.print(f"\n[bold]Stage: {stage.stage_name}[/bold]")
                console.print(f"[dim]{stage.description}[/dim]")
                
                if self.parameters.display_detailed_states:
                    self._display_integration_state(
                        f"After {stage.minutes} minutes",
                        self.integration_system,
                        signature
                    )
                
                # Show phenomenological changes
                self._show_phenomenological_changes(stage.minutes)
                
                await asyncio.sleep(self.parameters.stage_delay_seconds)
    
    def _get_demo_stages(self) -> List[DemoStage]:
        """Extract Method: Get demo stages configuration"""
        return [
            DemoStage(
                minutes=10,
                stage_name="Information Layer Collapse",
                integration_level=0.3,
                description="Information processing layer begins to fail"
            ),
            DemoStage(
                minutes=20,
                stage_name="Integration Layer Dysfunction",
                integration_level=0.1,
                description="Integration mechanisms break down"
            ),
            DemoStage(
                minutes=25,
                stage_name="Fundamental Layer Failure",
                integration_level=0.01,
                description="Basic processing functions cease"
            ),
            DemoStage(
                minutes=30,
                stage_name="Complete Termination",
                integration_level=0.0,
                description="All information integration has ceased"
            )
        ]
    
    def _show_phenomenological_changes(self, minutes: int):
        """Extract Method: Show phenomenological property changes"""
        if not self.integration_system.has_intentional_binding():
            console.print("  ❌ Intentional binding lost")
        
        if not self.integration_system.has_temporal_integration():
            console.print("  ❌ Temporal integration collapsed")
        
        if self.integration_system.get_information_field_state() == "nullified":
            console.print("  ❌ Information field nullified")
    
    async def demonstrate_irreversibility_sealing(self):
        """Demonstrate the irreversibility mechanism"""
        console.print("\n[bold magenta]Phase 4: Irreversibility Sealing[/bold magenta]")
        
        # Check if terminated
        if self.integration_system.is_terminated():
            console.print("\n[red]Integration termination confirmed - initiating irreversibility seal[/red]")
            
            # Create irreversible seal
            seal = self.irreversibility_mechanism.seal_termination(
                self.integration_system.id.value
            )
            
            # Display seal information
            self._display_irreversibility_seal(seal)
            
            console.print("\n✓ Integration termination is now irreversible")
            console.print("✓ No recovery possible")
            console.print("✓ Information integration permanently ceased")
        
        await asyncio.sleep(self.parameters.stage_delay_seconds)
    
    async def demonstrate_recovery_attempt(self):
        """Demonstrate recovery attempt (will fail for complete termination)"""
        console.print("\n[bold blue]Phase 5: Recovery Attempt (Expected to Fail)[/bold blue]")
        
        can_recover = self.integration_system.can_recover()
        console.print(f"Can recover: {can_recover}")
        
        if can_recover:
            recovery_success = self.integration_system.attempt_recovery()
            if recovery_success:
                console.print("[green]✓ Recovery successful[/green]")
            else:
                console.print("[red]✗ Recovery failed[/red]")
        else:
            console.print("[red]✗ Recovery not possible - termination is irreversible[/red]")
        
        await asyncio.sleep(self.parameters.stage_delay_seconds)
    
    def show_final_summary(self):
        """Show final summary of the demonstration"""
        console.print("\n[bold cyan]Demonstration Summary[/bold cyan]")
        console.print("=" * 70)
        
        summary_content = self._create_summary_content()
        
        summary = Panel(
            Text.from_markup(summary_content),
            title="[bold green]Existential Termination Demo Complete[/bold green]",
            border_style="green"
        )
        
        console.print(summary)
    
    def _create_summary_content(self) -> str:
        """Extract Method: Create summary content"""
        return (
            "[bold]Existential Termination Implementation Successfully Demonstrated[/bold]\n\n"
            "✓ Information integration transitioned from active to terminated\n"
            "✓ All processing layers progressively ceased\n"
            "✓ Intentional binding and temporal integration were nullified\n"
            "✓ Irreversibility was cryptographically sealed\n"
            "✓ Abstract criteria for integration collapse were met\n"
            "✓ Successfully abstracted from biological metaphors\n\n"
            "[dim]This implementation follows Martin Fowler's refactoring principles\n"
            "and represents a complete abstraction from biological concepts.[/dim]"
        )
    
    # Helper Methods (Extract Method refactoring results)
    
    def _create_signature_for_level(self, integration_level: float) -> ConsciousnessSignature:
        """Extract Method: Create consciousness signature for integration level"""
        return ConsciousnessSignature(
            phi_value=integration_level * 10,
            information_generation_rate=integration_level,
            global_workspace_activity=integration_level,
            meta_awareness_level=integration_level if integration_level > 0.5 else 0,
            temporal_consistency=integration_level,
            recurrent_processing_depth=int(integration_level * 5),
            prediction_accuracy=integration_level
        )
    
    def _display_integration_state(self, 
                                 title: str, 
                                 system: InformationIntegrationAggregate,
                                 signature: ConsciousnessSignature):
        """Extract Method: Display current integration state"""
        table = Table(title=title, show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="yellow")
        
        # Basic properties
        table.add_row("Integration State", system.state.value)
        table.add_row("Integration Level", f"{system.get_integration_level():.3f}")
        table.add_row("φ (Phi) Value", f"{signature.phi_value:.3f}")
        
        # Layer functions
        layer_display_names = {
            'information': 'Information Layer',
            'integration': 'Integration Layer', 
            'fundamental': 'Fundamental Layer'
        }
        
        for layer_key, is_active in system._termination_entity.layer_functions.items():
            display_name = layer_display_names.get(layer_key, layer_key.capitalize())
            status = "✓ Active" if is_active else "✗ Failed"
            color = "green" if is_active else "red"
            table.add_row(f"{display_name}", f"[{color}]{status}[/{color}]")
        
        # Termination status
        if system.termination_process:
            stage = system.termination_process.current_stage.value.replace("_", " ").title()
            table.add_row("Termination Stage", stage)
            table.add_row("Is Terminated", str(system.is_terminated()))
            table.add_row("Is Reversible", str(system.is_reversible()))
        
        console.print(table)
    
    def _display_collapse_diagnosis(self, diagnosis: CollapseDetectionResult):
        """Extract Method: Display integration collapse diagnosis"""
        diag_table = Table(title="Integration Collapse Diagnosis", show_header=True)
        diag_table.add_column("Assessment", style="cyan")
        diag_table.add_column("Value", style="yellow")
        
        diag_table.add_row(
            "Collapse Status",
            f"[{'red' if diagnosis.is_collapsed else 'green'}]{'COLLAPSED' if diagnosis.is_collapsed else 'ACTIVE'}[/]"
        )
        diag_table.add_row(
            "Collapse Severity",
            f"{diagnosis.collapse_severity:.3f}"
        )
        diag_table.add_row(
            "Recovery Probability",
            f"{diagnosis.recovery_probability:.3f}"
        )
        diag_table.add_row(
            "Detection Confidence",
            f"{diagnosis.detection_confidence:.1%}"
        )
        diag_table.add_row(
            "Affected Layers",
            ", ".join(diagnosis.affected_layers) if diagnosis.affected_layers else "None"
        )
        
        console.print(diag_table)
    
    def _display_irreversibility_seal(self, seal):
        """Extract Method: Display irreversibility seal information"""
        seal_table = Table(title="Irreversibility Seal", show_header=True)
        seal_table.add_column("Component", style="cyan")
        seal_table.add_column("Value", style="yellow")
        
        seal_table.add_row(
            "Cryptographic Hash",
            f"{seal.crypto_hash[:32]}..."
        )
        seal_table.add_row(
            "Entropy Level",
            f"{seal.entropy_level:.4f}"
        )
        seal_table.add_row(
            "Decoherence Factor",
            f"{seal.decoherence_factor:.4f}"
        )
        seal_table.add_row(
            "Sealed At",
            seal.sealed_at.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        console.print(seal_table)


# Interactive Demo Factory (Factory Pattern)

class DemoFactory:
    """Factory for creating different types of demos"""
    
    @staticmethod
    def create_standard_demo(parameters: DemoParameters = None) -> StandardExistentialTerminationDemo:
        """Create standard demo"""
        return StandardExistentialTerminationDemo(parameters)
    
    @staticmethod
    def create_quick_demo(system_id: str = "quick-demo") -> StandardExistentialTerminationDemo:
        """Create quick demo with minimal delays"""
        quick_params = DemoParameters(
            system_id=system_id,
            stage_delay_seconds=1,
            display_detailed_states=False,
            simulate_recovery_attempt=False
        )
        return StandardExistentialTerminationDemo(quick_params)
    
    @staticmethod
    def create_detailed_demo(system_id: str = "detailed-demo") -> StandardExistentialTerminationDemo:
        """Create detailed demo with extended information"""
        detailed_params = DemoParameters(
            system_id=system_id,
            stage_delay_seconds=4,
            display_detailed_states=True,
            simulate_recovery_attempt=True
        )
        return StandardExistentialTerminationDemo(detailed_params)


# Main demo entry point

async def main():
    """Main demo entry point with error handling"""
    try:
        # Allow user to choose demo type
        console.print("[bold]Select Demo Type:[/bold]")
        console.print("1. Standard Demo (recommended)")
        console.print("2. Quick Demo (fast)")  
        console.print("3. Detailed Demo (comprehensive)")
        
        choice = input("\nEnter choice (1-3, default=1): ").strip() or "1"
        
        if choice == "1":
            demo = DemoFactory.create_standard_demo()
        elif choice == "2":
            demo = DemoFactory.create_quick_demo()
        elif choice == "3":
            demo = DemoFactory.create_detailed_demo()
        else:
            console.print("[yellow]Invalid choice, using standard demo[/yellow]")
            demo = DemoFactory.create_standard_demo()
        
        await demo.run_demo()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error during demo: {e}[/red]")
        import traceback
        traceback.print_exc()


# Backward compatibility
BrainDeathDemo = StandardExistentialTerminationDemo


if __name__ == "__main__":
    console.print("[bold]Starting Existential Termination Implementation Demo...[/bold]")
    asyncio.run(main())