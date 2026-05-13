# Geiger

A modern async Python project for agent-based task execution and dataset generation.

## Installation

```bash
pip install -e .
```

## Usage

```python
from geiger import Config, ToolParser, AgentExecutor

config = Config()
parser = ToolParser(config)
executor = AgentExecutor(config)
```

## Features

- Async tool parsing and execution
- Agent-based task automation
- Dataset generation and formatting
- Review and validation workflows