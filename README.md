# VIKMO AI/ML Intern Assignment - Dealer Assistant

> **Status**: Implementation in progress - Phase 1

## Overview
This repository contains the implementation of the VIKMO AI/ML Intern Take-Home Assignment, featuring a conversational Dealer Assistant with RAG, tool-calling, and optional demand forecasting.

## Tech Stack
- **Language**: Python 3.10+
- **LLM Provider**: Google Gemini (Free Tier) - TBD
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Validation**: Pydantic
- **Data Processing**: Pandas

## Setup
```bash
# Clone the repository
git clone <repository-url>
cd VIKMO-AI-ML-Intern-Assignment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the assistant
python -m assistant.cli

# Run evaluation
python eval/run_eval.py
```

## Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key (recommended)
- `OPENAI_API_KEY`: OpenAI API key (alternative)
- `OLLAMA_BASE_URL`: Ollama server URL (for local models)

## Example Interactions
*To be populated during implementation*

## Assumptions
See DESIGN.md for detailed assumptions and design decisions.

## Project Structure
```
.
├── README.md                    # This file
├── DESIGN.md                    # Design decisions and methodology
├── requirements.txt             # Python dependencies
├── assistant/
│   ├── __init__.py
│   ├── retrieval.py            # RAG and retrieval system
│   ├── tools.py                # Tool implementations
│   ├── agent.py                # Agent loop and conversation
│   └── cli.py                  # Command-line interface
├── eval/
│   ├── __init__.py
│   ├── eval_set.jsonl           # Evaluation test cases
│   ├── run_eval.py             # Evaluation runner
│   └── results.json            # Evaluation results
└── forecasting/
    ├── __init__.py
    ├── forecast.py             # Forecasting model
    ├── baseline.py             # Baseline model
    └── results.json            # Forecasting results
```

## Evaluation Results
*To be populated after Phase 4*
