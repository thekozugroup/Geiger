# Geiger

A modern async Python project for agent-based task execution and dataset generation.

## Installation

```bash
pip install -e .
```

## Usage

### Parse TOOLS.md

```python
from geiger import ToolParser

parser = ToolParser("path/to/TOOLS.md")
tools = parser.parse()
```

### Execute Agents with Tools

```python
import asyncio
from geiger import AgentConfig, AgentExecutor, ToolParser

config = AgentConfig(
    api_key="your-api-key",  # Set your OpenAI API key
    model="gpt-4",
)

executor = AgentExecutor(config)

tools = ToolParser("path/to/TOOLS.md").parse()

tool_executor = {
    "tool_name": lambda arg1, arg2: {"result": f"processed {arg1} with {arg2}"}
}

async def main():
    trace = await executor.execute_single(
        prompt="Use the tool to process some data",
        tools=tools,
        tool_executor=tool_executor,
    )
    print(f"Trace: {trace.session_id}")

asyncio.run(main())
```

### Generate Datasets

```python
import asyncio
from geiger import DatasetConfig, DatasetGenerator, DatasetTrace

config = DatasetConfig(output_dir="./output", min_grade_threshold=0.5)
generator = DatasetGenerator(config)

traces = [
    DatasetTrace(
        messages=[
            {"role": "human", "content": "Process this data"},
            {"role": "assistant", "content": "Using tool...", "tool_calls": [{"tool": "tool_name", "args": {"arg1": "value"}}]},
            {"role": "tool", "tool_result": '{"result": "success"}'},
        ],
        grade=0.85,
        tool_definitions=[{"name": "tool_name", "description": "..."}],
    ),
]

async def main():
    stats = await generator.generate(traces)
    print(f"Generated {stats.total_traces} traces, filtered to {stats.filtered_traces}")

asyncio.run(main())
```

## TOOLS.md Format

Geiger parses tool definitions from a `TOOLS.md` file with the following structure:

```markdown
## tool_name

Description of what the tool does.

### Arguments

- param1 (string): Description of param1
- param2 (integer): Description of param2

### Returns

Description of what the tool returns.
```

Example:

```markdown
## get_weather

Fetches current weather for a location.

### Arguments

- location (string): City name or coordinates
- units (string): Temperature units (celsius/fahrenheit)

### Returns

JSON object with temperature, conditions, and forecast.
```

## Features

- Async tool parsing and execution
- Agent-based task automation
- Dataset generation and formatting
- Review and validation workflows