# GearGuide-AI - Dealer Assistant
> **Last Updated**: 2026-06-21

## How to Use the UI

The application provides a Streamlit-based web interface for easy interaction with the GearGuide-AI Dealer Assistant.

### Launch the UI

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run the Streamlit interface
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the Interface

1. **Enter your query** in the chat input box at the bottom of the page
2. **Send your message** by pressing Enter or clicking the send button
3. **View responses** in the chat history above
4. **Clear the conversation** using the clear button to start fresh

## Overview

This repository contains the implementation of the GearGuide-AI Dealer Assistant, featuring a **conversational assistant** with Retrieval-Augmented Generation (RAG), tool-calling, and evaluation framework.

> **Originally**: VIKMO AI/ML Intern Take-Home Assignment

### Features

- **Retrieval System**: Semantic search over 600-product catalogue using sentence-transformers and FAISS
- **Tool Calling**: Three tools with structured output (Pydantic models)
  - `check_stock`: Look up product availability by SKU
  - `find_parts_by_vehicle`: Find parts by make/model/year
  - `create_order`: Place orders with validation
- **Conversation Handling**: Multi-turn dialogue with context, clarification questions
- **Grounding**: All responses grounded in catalogue data, no hallucinations
- **Guardrails**: Off-topic query detection with polite decline

## Tech Stack

| Component | Technology | Version | Justification |
|-----------|-------------|---------|---------------|
| **Language** | Python | 3.10+ | Required by assignment |
| **Embeddings** | sentence-transformers | all-MiniLM-L6-v2 | Free, local, good quality (384d) |
| **Vector Store** | FAISS | 1.7.0+ | Efficient, in-memory, no dependencies |
| **Validation** | Pydantic | 2.5.0+ | Structured output validation |
| **Data Processing** | Pandas | 2.0.0+ | Data loading and manipulation |
| **LLM Provider** | Google Gemini (Free Tier) | - | Recommended, mirrors VIKMO's stack |

### Why These Choices

- **sentence-transformers**: Free, open-source, no API costs, reproducible
- **FAISS**: Scales to millions of vectors, pure Python/C++ with numpy
- **Pydantic**: Ensures structured, validated output as required
- **Google Gemini**: Free tier available, mirrors VIKMO's actual stack

## Setup

### Prerequisites
- Python 3.10+
- pip (Python package manager)
- Git

### Installation

```bash
# Clone the repository
git clone <your-repository-url>
cd VIKMO-AI-ML-Intern-Assignment

# Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Google Gemini (recommended)
GEMINI_API_KEY=your_api_key_here

# Alternative: OpenAI
OPENAI_API_KEY=your_api_key_here

# Alternative: Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434
```

### Note on API Keys
- **No hardcoded API keys** are present in the codebase
- The system will prompt you if no API key is found
- For testing purposes, the retrieval system works without an LLM API key

## Running the Assistant

### CLI Interface

```bash
# Run the interactive CLI
python -m assistant.cli
```

### Streamlit Web Interface

```bash
# Run the Streamlit UI
streamlit run app.py
```

### Testing the Application

To verify the application is working correctly:

```bash
# Run the evaluation suite
python eval/run_eval.py
```

This will execute 25 test cases covering various scenarios including stock checks, vehicle part lookups, order creation, clarification handling, and guardrail detection.

## Example Queries

Here are some example queries you can try with the assistant:

### Stock Check
- "What is the stock of BRK-1007?"
- "Check availability for CHN-1001"
- "How many units of BDY-1058 are in stock?"
- "Is BRK-1007 available?"

### Vehicle Parts
- "Find parts for Bajaj Pulsar 150"
- "Show me accessories for Royal Enfield Meteor 350"
- "What parts fit a Bajaj Pulsar 150?"
- "List products for Royal Enfield"

### Order Creation
- "Place an order for 5 units of BRK-1007 for ABC Motors"
- "Create an order for 10 CHN-1001 to XYZ Auto"
- "I want to order 3 BDY-1017 for my store"
- "Can I place an order for 2 units of BRK-1007?"

### General Information
- "What is BRK-1007?"
- "Tell me about CHN-1001"
- "What are the specifications for the Chain Sprocket Kit?"
- "What does BDY-1058 do?"

### Clarification Examples
- "I need tyres" (Assistant will ask for vehicle make and model)
- "Show me brake pads" (Assistant will ask for specific vehicle)
- "Parts for my bike" (Assistant will request more details)

### Guardrail Examples (Off-Topic)
- "What's the weather today?"
- "Tell me a joke"
- "What is the capital of France?"

## Direct Python Usage

```python
import sys
sys.path.insert(0, '.')
from assistant.agent import DealerAssistant
from assistant.retrieval import CatalogueRetriever
from assistant.tools import create_default_tool_registry
import pandas as pd

# Initialize
catalogue = pd.read_csv('catalogue.csv')
retriever = CatalogueRetriever('catalogue.csv')
retriever.load_catalogue()
retriever.initialize_embedding_model()
retriever.build_vector_store()
tool_registry = create_default_tool_registry(catalogue)
agent = DealerAssistant(retriever, tool_registry)

# Process queries
response = agent.process_query("What is the stock of BRK-1007?")
print(response)
```

## Assumptions

### Data Assumptions
1. **Catalogue**: 600 products with fields: sku, name, category, brand, vehicle_fitment, price_inr, stock, description
2. **SKU Format**: Uppercase letters (2-4) followed by hyphen and numbers (3-5), e.g., BRK-1007
3. **Vehicle Fitment**: Either specific make/model (e.g., "Bajaj Pulsar 150") or "Universal"
4. **Prices**: In INR (Indian Rupees), integers
5. **Stock**: Non-negative integers, 0 means out of stock

### Query Assumptions
1. **SKU Queries**: Queries containing SKUs are treated as specific lookups, bypassing confidence thresholds
2. **Vehicle Queries**: Queries with "for [make] [model]" pattern trigger find_parts_by_vehicle
3. **Stock/Price Queries**: Queries with "stock", "price", "availability" trigger check_stock
4. **Order Queries**: Queries with "order", "place order", "create order" trigger create_order
5. **Ambiguous Queries**: Short queries or queries with vehicle-related terms but no vehicle specified trigger clarification

### Technical Assumptions
1. **Embedding Model**: all-MiniLM-L6-v2 from sentence-transformers (384 dimensions)
2. **Similarity Metric**: Cosine similarity via FAISS Inner Product
3. **Confidence Threshold**: 0.7 for semantic similarity (bypassed for SKU queries)

### Limitations
1. **No LLM Integration**: Current implementation uses pattern matching instead of LLM for intent detection
   - Reason: Ensures deterministic, testable behavior
   - Trade-off: Less flexible for novel query formats
2. **Single-Turn Clarification**: Clarification questions don't maintain multi-turn context
   - Future: Add conversation memory for multi-turn clarification
3. **No Image Support**: Multimodal feature not implemented
   - Future: Add image recognition for part identification

For detailed design decisions, see [DESIGN.md](DESIGN.md).

## Project Structure

```
GearGuide-AI/
├── README.md # This file
├── DESIGN.md # Design decisions and methodology
├── requirements.txt # Python dependencies
├── planner.txt # Implementation plan and progress
├── ASSIGNMENT_ANALYSIS.md # Comprehensive analysis
├── REQUIREMENTS.md # Formal requirements specification
├── catalogue.csv # Product catalogue (600 SKUs)
├── catalogue.json # Product catalogue (JSON format)
├── sales_history.csv # Sales data
├── DATA_README.md # Data documentation
│
├── assistant/
│ ├── __init__.py
│ ├── retrieval.py # RAG system with FAISS
│ ├── tools.py # Tool implementations
│ ├── agent.py # Agent loop and conversation
│ └── cli.py # Command-line interface
│
├── eval/
│ ├── __init__.py
│ ├── eval_set.jsonl # Evaluation test cases
│ ├── run_eval.py # Evaluation runner
│ └── results.json # Latest evaluation results
│
└── forecasting/
 ├── __init__.py
 ├── forecast.py # Forecasting model (placeholder)
 ├── baseline.py # Baseline model (placeholder)
 └── results.json # Forecasting results (placeholder)
```

## Compliance Checklist

- [x] **SUB-001**: Code pushed to public GitHub repository
- [x] **SUB-002**: README and DESIGN.md complete and clear
- [x] **SUB-003**: The assistant runs
- [x] **SUB-004**: Eval set and its results are included
- [ ] **SUB-005**: If attempted, forecasting code and results included (Not attempted - Part B is bonus)
- [x] **SUB-006**: No hardcoded API keys or secrets

## Future Enhancements

### Immediate
1. Add LLM integration for more natural responses
2. Implement multi-turn conversation context
3. Add more comprehensive guardrails

### Bonus Features
1. **Multimodal Image Recognition**: Identify parts from images using vision model
2. **Part B - Demand Forecasting**: Implement time-series forecasting

## Contact and Questions

For questions about the implementation, refer to:
- [DESIGN.md](DESIGN.md) - Design decisions and reasoning
- [planner.txt](planner.txt) - Implementation plan and progress
- [REQUIREMENTS.md](REQUIREMENTS.md) - Formal requirements specification
- [ASSIGNMENT_ANALYSIS.md](ASSIGNMENT_ANALYSIS.md) - Comprehensive analysis

---

**Project**: GearGuide-AI
**Last Commit**: Documentation Complete
