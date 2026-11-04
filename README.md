# Artificial-Life Trading Ecosystem

This repository provides the foundational structure for an artificial-life trading ecosystem in which a shared, gradient-trained forecasting backbone supports a population of evolving trading agents.

## Core architectural principle

The recurrent forecasting model is shared and trained through gradient descent. The evolutionary genome is intentionally small and contains only the agent-specific decision head and trading parameters. The full forecasting backbone is not evolved.

## Repository layout

- `src/` contains the Python package and module skeletons.
- `configs/` stores project configuration and environment defaults.
- `data/` holds raw and processed datasets.
- `models/` stores forecasting and genome model artifacts.
- `app/` is reserved for the eventual dashboard or Streamlit app.
- `tests/` contains project validation and smoke tests.
- `docs/` contains architecture and design notes.

## Scope of this milestone

This repository foundation intentionally avoids implementing:

- data processing pipelines
- forecasting training logic
- trading strategies
- evolutionary algorithms
- real brokerage or market execution features

The focus here is a clean package layout, configuration system, utilities, and import-safe skeletons that can support Milestone 1 work.

## Quick start

```bash
cd artificial-life-trading-ecosystem
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

## Planned layers

- Data layer
- Forecasting layer
- Agent layer
- Trading simulation layer
- Evolutionary population layer
- Evaluation layer
- Lineage layer
- Dashboard layer
