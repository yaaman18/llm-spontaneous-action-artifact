Installation Instructions

Create the agents directory structure:

bash# For project-level agents
mkdir -p .claude/agents

# For user-level agents (available across all projects)
mkdir -p ~/.claude/agents

Copy each agent configuration to a separate .md file in the agents directory:

bash# Example for project-level installation
echo '[agent content]' > .claude/agents/phenomenology-director.md
echo '[agent content]' > .claude/agents/computational-phenomenology-lead.md
# ... continue for all agents

Verify installation:

bash# In Claude Code
/agents list
Usage Examples
Phenomenological Analysis
> Use the phenomenology-director to review our intentionality implementation

> Ask the computational-phenomenology-lead to translate Merleau-Ponty's body schema into our embodiment framework
Autopoiesis Implementation
> Have the autopoiesis-architect design our self-maintaining subsystem

> Use the enactive-cognition-specialist to implement participatory sense-making
Integration Tasks
> Deploy the project-orchestrator to coordinate phenomenology and autopoiesis integration

> Ask the consciousness-theorist-council to validate our approach against major theories
Technical Implementation
> Use the llm-systems-architect to optimize our Azure OpenAI calls

> Have the artificial-consciousness-engineer implement the global workspace
Best Practices

Use domain experts for their specialties - Each agent has deep knowledge in specific areas
Coordinate through the orchestrator - For cross-domain integration, use the project-orchestrator
Leverage proactive agents - Agents marked with "PROACTIVELY" or "MUST BE USED" will intervene when relevant
Combine Eastern and Western perspectives - Use the Japanese researchers' agents to bridge philosophical traditions
Validate theory with implementation - Always check theoretical proposals against practical feasibility

Notes

These agents embody the expertise of leading researchers while remaining focused on practical implementation
Each agent maintains the researcher's theoretical perspective while adapting to project needs
The orchestrator ensures coherent integration across all domains
Agents can invoke other agents through the Task tool when cross-domain expertise is needed