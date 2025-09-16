# Contributing to AEGIS

Thank you for your interest in contributing to AEGIS! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- Virtual environment (recommended)

### Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/your-username/aegis.git
   cd aegis
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## 🛠️ Development Guidelines

### Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting
- **mypy**: Type checking

Run all checks:
```bash
# Format code
black aegis_core/ examples/ tests/
isort aegis_core/ examples/ tests/

# Check linting
flake8 aegis_core/ examples/ tests/

# Type checking
mypy aegis_core/
```

### Testing

We use pytest for testing. Tests are organized in the `tests/` directory:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=aegis_core

# Run specific test file
pytest tests/test_malicious_factory.py
```

### Documentation

- Use clear docstrings for all public functions and classes
- Follow Google-style docstring format
- Update README.md for significant changes
- Add examples for new features

## 🎯 Types of Contributions

### Bug Reports

When reporting bugs, please include:

- Python version and operating system
- AEGIS version
- Minimal code example that reproduces the issue
- Full error traceback
- Expected vs actual behavior

### Feature Requests

For feature requests, please:

- Describe the use case and motivation
- Provide examples of the desired API/behavior
- Consider backwards compatibility
- Discuss potential implementation approaches

### Code Contributions

#### Adding New MAS Frameworks

To add support for a new Multi-Agent System framework:

1. Create a new wrapper class in `aegis_core/agent_systems/`:
   ```python
   from .base_wrapper import BaseMASWrapper
   
   class NewFrameworkWrapper(BaseMASWrapper):
       def __init__(self, config):
           super().__init__(config)
           # Initialize framework-specific components
       
       async def run_task(self, task):
           # Implement task execution logic
           pass
   ```

2. Register the wrapper in `aegis_core/agent_systems/__init__.py`:
   ```python
   MAS_REGISTRY["new_framework"] = NewFrameworkWrapper
   ```

3. Add configuration templates in `configs/`
4. Create tests in `tests/test_agent_systems/`
5. Update documentation

#### Adding New Error Modes

To implement new error injection strategies:

1. Add error mode definitions in `aegis_core/malicious_factory/strategies/`
2. Update the error taxonomy documentation
3. Add corresponding injection templates
4. Create validation tests
5. Update examples

#### Performance Optimizations

- Profile code changes using `cProfile` or similar tools
- Include benchmark results in pull request descriptions
- Ensure optimizations don't break existing functionality

## 📋 Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clear, concise commit messages
   - Keep commits focused and atomic
   - Add tests for new functionality

3. **Ensure quality**:
   ```bash
   # Run all checks before submitting
   black aegis_core/ examples/ tests/
   isort aegis_core/ examples/ tests/
   flake8 aegis_core/ examples/ tests/
   pytest
   ```

4. **Update documentation**:
   - Add docstrings to new functions/classes
   - Update README.md if needed
   - Add examples for new features

5. **Submit pull request**:
   - Use a descriptive title
   - Reference relevant issues
   - Describe changes and motivation
   - Include test results and examples

### Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New code has appropriate test coverage
- [ ] Documentation is updated
- [ ] Changes are backwards compatible (or breaking changes are clearly noted)
- [ ] Commit messages are clear and descriptive

## 🔒 Security Considerations

AEGIS deals with error injection and system manipulation. Please:

- **Never commit real API keys or credentials**
- **Ensure new injection strategies are for defensive purposes only**
- **Document potential security implications**
- **Follow responsible disclosure for security issues**

## 🌟 Recognition

Contributors are recognized in:

- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- Project documentation

## 📞 Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord**: Real-time chat (link in README)

## 📄 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to abide by its terms.

## 📜 License

By contributing to AEGIS, you agree that your contributions will be licensed under the MIT License.