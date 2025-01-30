# glimpse-llm

Interactive visualization tool for LLM interpretability

## Installation

```bash
pip install glimpse-llm
```

## Quick Start

```python
from glimpse_llm import launch

# Launch with default model
launch()

# Or specify a model
launch("gpt2-medium")
```

## Development

1. Clone the repository
2. Install development dependencies: `pip install -e ".[dev]"`
3. Install frontend dependencies: `cd frontend && npm install`
4. Build frontend: `npm run build`