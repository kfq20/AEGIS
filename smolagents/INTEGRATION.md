# Smolagents Integration with AEGIS

This directory contains the **smolagents** library, which provides powerful agent frameworks that can be used to enhance AEGIS's multi-agent system evaluation capabilities.

## Overview

Smolagents is a lightweight library for building AI agents that can:
- Execute code safely in sandboxed environments
- Use various tools and APIs
- Support multiple LLM providers
- Integrate with existing frameworks

## Integration with AEGIS

The smolagents library has been integrated into AEGIS to provide:

1. **Enhanced Agent Capabilities**: Use smolagents' CodeAgent and ToolCallingAgent for more sophisticated error injection and testing
2. **GAIA Benchmark Integration**: The `examples/open_deep_research/` directory contains tools for evaluating agents on the GAIA benchmark
3. **Multi-LLM Support**: Leverage smolagents' support for various LLM providers in AEGIS evaluations
4. **Tool Ecosystem**: Access to a rich set of tools for web browsing, file manipulation, and more

## Key Components

### Core Library (`src/smolagents/`)
- `agents.py`: Main agent implementations (CodeAgent, ToolCallingAgent)
- `tools/`: Collection of built-in tools
- `models/`: LLM model integrations

### GAIA Evaluation (`examples/open_deep_research/`)
- `run_gaia.py`: GAIA benchmark evaluation script
- `scripts/`: Specialized tools for GAIA tasks
- Can be used to evaluate AEGIS-generated error scenarios

## Usage Examples

### Basic Agent Integration
```python
from smolagents import CodeAgent, WebSearchTool
from aegis_core.malicious_factory import MaliciousFactory

# Create a smolagents-powered evaluation agent
agent = CodeAgent(tools=[WebSearchTool()], model=your_model)

# Use with AEGIS error injection
factory = MaliciousFactory()
# ... inject errors and evaluate with smolagents
```

### GAIA Benchmark Evaluation
```bash
cd smolagents/examples/open_deep_research
python run_gaia.py --model-id gpt-4o --concurrency 32
```

## Installation

If using smolagents independently:
```bash
cd smolagents
pip install -e .
```

## Security Considerations

- Smolagents supports sandboxed code execution via E2B, Docker, or WebAssembly
- Ensure proper API key management when using with AEGIS
- Review the SECURITY.md file for best practices

## Documentation

- Full documentation: https://huggingface.co/docs/smolagents
- Examples in the `examples/` directory
- Original README.md contains comprehensive usage information

## Contributing

This integration maintains the original smolagents codebase structure. For contributions to smolagents itself, refer to the original repository at https://github.com/huggingface/smolagents.