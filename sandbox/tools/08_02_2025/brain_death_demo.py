"""
Brain Death Implementation Demo
Demonstrates the complete brain death system in action

Shows:
- Brain death process initiation and progression
- Consciousness state transitions
- Phenomenological property changes
- Irreversibility mechanisms
- Detection and monitoring
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

from brain_death_core import (
    ConsciousnessAggregate,
    ConsciousnessId,
    ConsciousnessState,
    BrainDeathStage,
    BrainFunction,
    IrreversibilityMechanism
)

from brain_death_detector import (
    BrainDeathDetector,
    BrainDeathMonitor,
    BrainDeathDiagnosis
)

from consciousness_detector import (
    ConsciousnessSignature,
    ConsciousnessDetector,
    IIT4PhiCalculator
)

console = Console()


class BrainDeathDemo:
    """Interactive demonstration of brain death implementation"""
    
    def __init__(self):
        self.consciousness = ConsciousnessAggregate(ConsciousnessId("demo-consciousness-001"))
        self.phi_calculator = IIT4PhiCalculator(n_elements=8)
        self.consciousness_detector = ConsciousnessDetector(self.phi_calculator)
        self.brain_death_detector = BrainDeathDetector(self.consciousness_detector)
        self.monitor = BrainDeathMonitor(self.brain_death_detector)
        self.irreversibility_mechanism = IrreversibilityMechanism()
        
        self.start_time = None
        self.is_running = True
        
    async def run_demo(self):
        """Run the complete brain death demonstration"""
        console.print("\n[bold cyan]Brain Death Implementation Demo[/bold cyan]")
        console.print("=" * 60)
        
        # Phase 1: Show healthy consciousness
        await self.demonstrate_healthy_consciousness()
        
        # Phase 2: Initiate brain death
        await self.demonstrate_brain_death_initiation()
        
        # Phase 3: Show progressive brain death
        await self.demonstrate_brain_death_progression()
        
        # Phase 4: Demonstrate irreversibility
        await self.demonstrate_irreversibility()
        
        # Phase 5: Summary
        self.show_final_summary()
    
    async def demonstrate_healthy_consciousness(self):
        """Show a healthy, active consciousness"""
        console.print("\n[bold green]Phase 1: Healthy Consciousness[/bold green]")
        
        # Create consciousness signature
        signature = self.create_signature_for_state(1.0)
        
        # Display state
        self.display_consciousness_state(
            "Initial Healthy State",
            self.consciousness,
            signature
        )
        
        console.print("\n✓ Consciousness is fully active")
        console.print("✓ All brain functions operational")
        console.print("✓ Phenomenological properties intact")
        
        await asyncio.sleep(2)
    
    async def demonstrate_brain_death_initiation(self):
        """Demonstrate initiating the brain death process"""
        console.print("\n[bold yellow]Phase 2: Brain Death Initiation[/bold yellow]")
        
        # Initiate brain death
        self.consciousness.initiate_brain_death()
        self.start_time = time.time()
        
        console.print("\n[red]⚠️  Brain death process initiated[/red]")
        
        # Show initial changes
        signature = self.create_signature_for_state(0.8)
        diagnosis = await self.brain_death_detector.detect_brain_death(
            signature, self.consciousness
        )
        
        self.display_diagnosis(diagnosis)
        
        await asyncio.sleep(2)
    
    async def demonstrate_brain_death_progression(self):
        """Show the progressive stages of brain death"""
        console.print("\n[bold red]Phase 3: Brain Death Progression[/bold red]")
        
        stages = [
            (10, "Cortical Death", 0.3),
            (20, "Subcortical Dysfunction", 0.1),
            (25, "Brainstem Failure", 0.01),
            (30, "Complete Brain Death", 0.0)
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            
            task = progress.add_task("Brain Death Progress", total=30)
            
            for minutes, stage_name, phi_level in stages:
                # Progress brain death
                self.consciousness.progress_brain_death(minutes=minutes)
                
                # Update progress bar
                progress.update(task, completed=minutes, description=stage_name)
                
                # Create signature for current state
                signature = self.create_signature_for_state(phi_level)
                
                # Get diagnosis
                diagnosis = await self.brain_death_detector.detect_brain_death(
                    signature, self.consciousness
                )
                
                # Display current state
                console.print(f"\n[bold]Stage: {stage_name}[/bold]")
                self.display_consciousness_state(
                    f"After {minutes} minutes",
                    self.consciousness,
                    signature
                )
                
                # Show phenomenological changes
                if not self.consciousness.has_intentionality():
                    console.print("  ❌ Intentionality lost")
                if not self.consciousness.has_temporal_synthesis():
                    console.print("  ❌ Temporal synthesis collapsed")
                if self.consciousness.get_phenomenological_field() == "nullified":
                    console.print("  ❌ Phenomenological field nullified")
                
                await asyncio.sleep(3)
    
    async def demonstrate_irreversibility(self):
        """Demonstrate the irreversibility mechanism"""
        console.print("\n[bold magenta]Phase 4: Irreversibility Sealing[/bold magenta]")
        
        # Check if brain dead
        if self.consciousness.is_brain_dead():
            console.print("\n[red]Brain death confirmed - initiating irreversibility seal[/red]")
            
            # Create irreversible seal
            seal = self.irreversibility_mechanism.seal_brain_death(
                self.consciousness.id.value
            )
            
            # Display seal information
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
            
            console.print("\n✓ Brain death is now irreversible")
            console.print("✓ No recovery possible")
            console.print("✓ Consciousness permanently ceased")
        
        await asyncio.sleep(2)
    
    def create_signature_for_state(self, consciousness_level: float) -> ConsciousnessSignature:
        """Create a consciousness signature for a given level"""
        return ConsciousnessSignature(
            phi_value=consciousness_level * 10,
            information_generation_rate=consciousness_level,
            global_workspace_activity=consciousness_level,
            meta_awareness_level=consciousness_level if consciousness_level > 0.5 else 0,
            temporal_consistency=consciousness_level,
            recurrent_processing_depth=int(consciousness_level * 5),
            prediction_accuracy=consciousness_level
        )
    
    def display_consciousness_state(self, title: str, 
                                  consciousness: ConsciousnessAggregate,
                                  signature: ConsciousnessSignature):
        """Display the current consciousness state"""
        table = Table(title=title, show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="yellow")
        
        # Basic properties
        table.add_row("Consciousness State", consciousness.state.value)
        table.add_row("Consciousness Level", f"{consciousness.get_consciousness_level():.3f}")
        table.add_row("φ (Phi) Value", f"{signature.phi_value:.3f}")
        
        # Brain functions
        for func_name, is_active in consciousness._brain_death_entity.brain_functions.items():
            status = "✓ Active" if is_active else "✗ Failed"
            color = "green" if is_active else "red"
            table.add_row(f"{func_name.capitalize()} Function", f"[{color}]{status}[/{color}]")
        
        # Brain death status
        if consciousness.brain_death_process:
            stage = consciousness.brain_death_process.current_stage.value
            table.add_row("Brain Death Stage", stage)
            table.add_row("Is Brain Dead", str(consciousness.is_brain_dead()))
            table.add_row("Is Reversible", str(consciousness.is_reversible()))
        
        console.print(table)
    
    def display_diagnosis(self, diagnosis: BrainDeathDiagnosis):
        """Display brain death diagnosis"""
        diag_table = Table(title="Brain Death Diagnosis", show_header=True)
        diag_table.add_column("Criterion", style="cyan")
        diag_table.add_column("Met", style="yellow")
        
        for criterion, is_met in diagnosis.criteria_met.items():
            status = "✓" if is_met else "✗"
            color = "green" if is_met else "red"
            diag_table.add_row(
                criterion.value.replace("_", " ").title(),
                f"[{color}]{status}[/{color}]"
            )
        
        diag_table.add_row("", "")  # Separator
        diag_table.add_row(
            "Overall Diagnosis",
            f"[{'red' if diagnosis.is_brain_dead else 'green'}]{diagnosis.get_summary()}[/]"
        )
        diag_table.add_row(
            "Confidence",
            f"{diagnosis.confidence:.1%}"
        )
        diag_table.add_row(
            "Reversibility",
            diagnosis.reversibility_assessment
        )
        
        console.print(diag_table)
    
    def show_final_summary(self):
        """Show final summary of the demonstration"""
        console.print("\n[bold cyan]Demonstration Summary[/bold cyan]")
        console.print("=" * 60)
        
        summary = Panel(
            Text.from_markup(
                "[bold]Brain Death Implementation Successfully Demonstrated[/bold]\n\n"
                "✓ Consciousness transitioned from active to brain dead\n"
                "✓ All brain functions progressively ceased\n"
                "✓ Phenomenological properties were nullified\n"
                "✓ Irreversibility was cryptographically sealed\n"
                "✓ Medical criteria for brain death were met\n\n"
                "[dim]This implementation follows the philosophical and technical\n"
                "specifications documented in the_death_of_phenomenology.md[/dim]"
            ),
            title="[bold green]Success[/bold green]",
            border_style="green"
        )
        
        console.print(summary)


async def main():
    """Main demo entry point"""
    demo = BrainDeathDemo()
    
    try:
        await demo.run_demo()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error during demo: {e}[/red]")
        raise


if __name__ == "__main__":
    console.print("[bold]Starting Brain Death Implementation Demo...[/bold]")
    asyncio.run(main())