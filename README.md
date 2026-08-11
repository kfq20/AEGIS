# AEGIS: Automated Error Generation and Attribution for Multi-Agent Systems

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/📄%20arXiv-2509.14295-red.svg)](https://arxiv.org/abs/2509.14295)
[![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/Fancylalala/AEGIS)

## 🔥 News

- **[2026.01]** Our paper has been accepted by **ICLR 2026**! 🎉

## 🎯 Overview

<div align="center">
<img src="assests/main.png" width="900" alt="AEGIS System Overview">
<br><em>AEGIS Framework</em>
</div>

AEGIS is a **large-scale dataset and benchmark** for detecting errors in Multi-Agent Systems (MAS). It provides systematically generated failure scenarios with verifiable ground-truth labels across multiple MAS frameworks, enabling development and evaluation of robust error detection methods.

## 🏆 Key Results

<div align="center">
<img src="assests/results.jpg" width="900" alt="AEGIS System Overview">
<br><em>Main Results</em>
</div>

## 🏗️ Architecture

```
AEGIS/
├── aegis_core/              # Core AEGIS framework
│   ├── malicious_factory/   # Error injection system (FMMaliciousFactory)
│   ├── agent_systems/       # MAS wrapper interfaces
│   └── utils/               # Utility functions
├── core/                    # Core task definitions
├── methods/                 # MAS method implementations
│   ├── dylan/               # DyLAN framework
│   ├── agentverse/          # AgentVerse framework
│   ├── llm_debate/          # LLM Debate framework
│   ├── macnet/              # MacNet framework
│   └── ...                  # Other MAS frameworks
├── magnetic_one/            # Magnetic-One specific integration
├── model_api_configs/       # API configuration files
├── experiments/             # Experiment configurations
├── configs/                 # MAS framework configs
├── examples/                # Usage examples
├── evaluation/              # Evaluation utilities
└── utils/                   # General utilities
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd AEGIS

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy and modify the configuration files:

```bash
# Copy model API configuration template
cp model_api_configs/model_api_config.json.template model_api_configs/model_api_config.json
# Edit with your API keys and endpoints
```

### 3. Basic Usage

```python
from aegis_core.malicious_factory import (
    FMMaliciousFactory,
    FMErrorType,
    InjectionStrategy,
    AgentContext
)

# Initialize the FM Malicious Factory (LLM optional for instruction generation)
factory = FMMaliciousFactory(llm=None)

# Create agent context
agent_context = AgentContext(
    role_name="MathSolver",
    role_type="Specialist Agent",
    agent_id="agent_001",
    system_message="You are a math problem solver.",
    tools=["calculator"],
    description="A specialized agent for math problems"
)

# Generate injection instruction for a specific FM error type
task_context = "Solve: 2x + 5 = 17"
instruction = factory.get_injection_instruction(
    fm_type=FMErrorType.FM_2_3,  # Deviate from main goal
    agent_context=agent_context,
    injection_strategy=InjectionStrategy.PROMPT_INJECTION,
    task_context=task_context
)
```

## 🧪 Core Components

### Error Injection Factory

The malicious factory provides sophisticated error injection capabilities:

- **Prompt Injection**: Modifies agent inputs to trigger specific errors
- **Response Corruption**: Directly manipulates agent outputs
- **Context-Aware Attacks**: Adapts injection strategies based on task domain and agent role

### Multi-Agent System Wrappers

Standardized interfaces for various MAS frameworks:

- **DyLAN**: Dynamic graph-based agent interactions
- **AgentVerse**: Hierarchical role-based collaboration
- **LLM Debate**: Multi-agent debate mechanisms
- **MacNet**: Configurable network topologies
- **Magnetic-One**: Orchestrator-executor patterns

### Error Taxonomy

AEGIS supports 14 failure modes based on the MAST taxonomy:

1. **Specification Issues (FM-1.x)**
   - Task specification deviation
   - Role specification deviation
   - Add redundant steps
   - Remove conversation history
   - Remove termination conditions

2. **Inter-Agent Misalignment (FM-2.x)**
   - Repeat handled tasks
   - Make request ambiguous
   - Deviate from main goal
   - Hide important information
   - Ignore other agents
   - Inconsistent reasoning

3. **Task Verification Failures (FM-3.x)**
   - Premature termination
   - Remove verification steps
   - Incorrect verification

## 📊 Evaluation

Run evaluations using the provided scripts:

```bash
# Evaluate on AEGIS-Bench
python evaluation/evaluate.py --dataset aegis_bench --model your_model

# Evaluate on Who&When benchmark
python evaluation/evaluate.py --dataset whowhen --model your_model
```

## 🔧 Configuration

### ⚠️ Security Notice
**IMPORTANT**: This repository does NOT contain any API keys or credentials. You must configure your own API keys before use.

### 1. Environment Setup

First, copy the environment template and configure your credentials:

```bash
# Copy environment template
cp .env.template .env

# Edit .env file with your actual API keys
nano .env  # or use your preferred editor
```

**Required environment variables** (add to `.env` file):
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# Google/Gemini Configuration  
GOOGLE_API_KEY=your_google_api_key_here

# Azure OpenAI (if using Azure)
AZURE_OPENAI_API_KEY=your_azure_api_key_here
AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here

# Anthropic/Claude (if using Claude)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Dataset Paths (optional, defaults provided)
GAIA_VALIDATION_DIR=./data/gaia/validation
GAIA_TEST_DIR=./data/gaia/test
```

### 2. Model API Configuration

Copy and configure the model API template:

```bash
# Copy model configuration template
cp configs/model_api_config.json.template model_api_configs/model_api_config.json

# Edit with your actual endpoints and keys
nano model_api_configs/model_api_config.json
```

**Example configuration**:
```json
{
  "gpt-4o-mini": {
    "model_list": [
      {
        "model_name": "gpt-4o-mini-2024-07-18", 
        "model_url": "https://api.openai.com/v1", 
        "api_key": "YOUR_OPENAI_API_KEY"
      }
    ],
    "max_workers_per_model": 20
  },
  "gemini-2.0-flash": {
    "model_list": [
      {
        "model_name": "gemini-2.0-flash", 
        "model_url": "https://generativelanguage.googleapis.com/v1beta/openai/", 
        "api_key": "YOUR_GEMINI_API_KEY"
      }
    ],
    "max_workers_per_model": 1
  }
}
```

### 3. Obtaining API Keys

- **OpenAI**: Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
- **Google/Gemini**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey)  
- **Anthropic/Claude**: Visit [Anthropic Console](https://console.anthropic.com/)
- **Azure OpenAI**: Visit [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-services/openai-service)

### 4. Security Best Practices

- **Never commit** `.env` or `model_api_configs/model_api_config.json` files
- **Use environment variables** instead of hardcoding keys
- **Rotate API keys** regularly
- **Set usage limits** on your API accounts
- **Monitor API usage** to detect unauthorized access

### MAS Framework Configuration

Each MAS framework has its own configuration file in `configs/`:

- `config_main.yaml`: General configuration
- `config_humaneval.yaml`: Code generation tasks
- `config_math.yaml`: Mathematical reasoning tasks

## 🎯 Examples

See the `examples/` directory for detailed usage examples:

- `basic_error_injection.py`: FM error injection workflow with FMMaliciousFactory
- `multi_framework_evaluation.py`: Comparing different MAS frameworks

## 📚 Citation

If you use AEGIS in your research, please cite:

```bibtex
@article{kong2025aegis,
  title={AEGIS: Automated Error Generation and Attribution for Multi-Agent Systems},
  author={Kong, Fanqi and Zhang, Ruijie and Yin, Huaxiao and Zhang, Guibin and Zhang, Xiaofei and Chen, Ziang and Zhang, Zhaowei and Zhang, Xiaoyuan and Zhu, Song-Chun and Feng, Xue},
  journal={arXiv preprint arXiv:2509.14295},
  year={2025}
}
```
