# GearGuide-AI - LLM-Powered Dealer Assistant

> **Status**: LLM-First Architecture Implementation Complete  
> **Last Updated**: 2026-06-22
> [!TIP]
> **Recommended LLM Provider**: Based on project testing, we recommend configuring **NVIDIA NIM** with the `mistral-medium-3.5-128b` model. During evaluation, this provider has shown the most stable and responsive behavior. Users are encouraged to add an `NVIDIA_API_KEY` to their configuration for the best experience.

## How to Use the UI

The application provides a Streamlit-based web interface for easy interaction with the GearGuide-AI Dealer Assistant.

### Launch the UI

After completing the Setup section below:

```bash
# Run the Streamlit interface
streamlit run ui/app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the Interface

1. **Enter your query** in the chat input box at the bottom of the page
2. **Send your message** by pressing Enter or clicking the send button
3. **View responses** in the chat history above
4. **Clear the conversation** using the clear button to start fresh

---

## 🚀 Overview

This repository contains the implementation of the **GearGuide-AI Dealer Assistant**, featuring a **fully LLM-driven architecture** with Retrieval-Augmented Generation (RAG), intelligent tool-calling, and conversational reasoning.

### 🎯 Key Achievement: LLM-First Migration

The system has been **completely migrated from pattern-based logic to LLM-driven reasoning**. The LLM now serves as the primary reasoning engine for:

- **Understanding**: Intent detection, entity extraction, scope checking
- **Routing**: Tool selection, retrieval decisions, action planning
- **Conversation**: Multi-turn dialogue, follow-up reasoning, context management
- **Guardrails**: Off-topic detection, domain checking, safety validation
- **Grounding**: Response validation against retrieved context

## ✨ Features

### Core Capabilities
- **LLM-Driven Reasoning**: NVIDIA NIM (priority 1), Google Gemini (priority 2), or OpenAI (priority 3) powers all decision-making
- **Real RAG**: Semantic search over 600-product catalogue using sentence-transformers and FAISS
- **Intelligent Tool Calling**: LLM decides when and how to use tools
- **Multi-Turn Conversation**: Understands follow-up queries without repeated information
- **Grounded Responses**: All answers are validated against catalogue data
- **LLM-Based Guardrails**: Intelligent off-topic detection and scope enforcement

### Tools (Function Calling)
Three tools with structured Pydantic output:
- `check_stock`: Look up product availability by SKU
- `find_parts_by_vehicle`: Find parts by make/model/year
- `create_order`: Place orders with validation and structured output

### Conversation Intelligence
- Remembers context across turns
- Understands references (e.g., "Can I order 5?" after stock check)
- Asks intelligent clarification questions
- Handles ambiguous queries gracefully

## 🛠️ Tech Stack

| Component | Technology | Version | Justification |
|-----------|-------------|---------|---------------|
| **Language** | Python | 3.10+ | Required by assignment |
| **LLM Provider** | NVIDIA NIM | Latest | Priority 1 (Recommended) |
| **LLM Fallback 1** | Google GenAI | Latest | Priority 2 fallback |
| **LLM Fallback 2** | OpenAI | Latest | Priority 3 fallback |
| **Embeddings** | sentence-transformers | all-MiniLM-L6-v2 | Free, local, good quality (384d) |
| **Vector Store** | FAISS | 1.7.0+ | Efficient, in-memory, no dependencies |
| **Validation** | Pydantic | 2.5.0+ | Structured output validation |
| **Data Processing** | Pandas | 2.0.0+ | Data loading and manipulation |
| **UI** | Streamlit | Latest | Interactive web interface |

### Why These Choices
- **LLM-First**: The LLM drives all reasoning, replacing pattern matching and hardcoded logic
- **sentence-transformers**: Free, open-source, no API costs, reproducible
- **FAISS**: Scales to millions of vectors, pure Python/C++ with numpy
- **Pydantic**: Ensures structured, validated output as required
- **Streamlit**: Quick, interactive UI for demonstration

## 📋 Setup

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
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note**: First-time setup may take a few minutes as dependencies (including the ~80MB embedding model and LLM SDKs) are downloaded.

### Environment Configuration

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` and add your API keys:

```ini
# LLM Provider API Keys (Priority: 1. NVIDIA NIM, 2. Gemini, 3. OpenAI)
NVIDIA_API_KEY=your_nvidia_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

> **Important**: The system **requires** a working LLM provider (Gemini, OpenAI, or NVIDIA NIM). Without a valid API key, the application will not function and will display a setup panel instead of the chat interface.

## 🏃 Running the Assistant

### CLI Interface

```bash
# Run the interactive CLI
python -m assistant.cli
```

### Testing the Application

```bash
# Run the evaluation suite
python eval/run_eval.py
```

This will execute test cases covering:
- **Happy Paths**: Normal, well-formed queries
- **Tricky/Ambiguous**: Queries needing clarification
- **Out-of-Scope**: Off-topic queries (guardrail testing)

## 💬 Example Queries

### Stock Check
- "What is the stock of BRK-1007?"
- "Check availability for CHN-1001"
- "How many units of BDY-1058 are in stock?"
- "Is BRK-1007 available?"
- "What is the price of BRK-1007?"

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

### Multi-Turn Conversation
```
User: What is the stock of BRK-1007?
Assistant: Brake Pad Set — Royal Enfield Meteor 350 (SKU: BRK-1007) - ₹530, Stock: 474, Status: In Stock

User: Can I order 5?
Assistant: Order ORD-ABC12345 for User - Total: ₹2650
Items:
 - BRK-1007: 5
```

### Clarification Examples
- "I need tyres" → Assistant: "Which vehicle? Please specify make and model (e.g., 'Bajaj Pulsar 150')."
- "Show me brake pads" → Assistant: "Which vehicle? Please specify make and model."
- "Parts for my bike" → Assistant: "Could you please provide more details?"

### Guardrail Examples (Off-Topic)
- "What's the weather today?" → Assistant: "I'm sorry, I can only help with auto parts, stock, and orders."
- "Tell me a joke" → Assistant: "I'm not able to assist with that. I'm here to help with auto parts queries."
- "What is the capital of France?" → Assistant: "I'm sorry, I can only help with auto parts, stock, and orders."

## 🏗️ Architecture

### LLM-First Pipeline

```
User Query
    ↓
┌─────────────────────────┐
│  LLM Understanding        │ ← Intent, entities, scope
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Guardrail Check         │ ← LLM-based off-topic detection
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Clarification Check     │ ← LLM decides if more info needed
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Retrieval               │ ← FAISS + sentence-transformers
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  LLM Action Decision     │ ← Tool call or retrieval response
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Tool Execution          │ ← check_stock, find_parts, create_order
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Context Assembly        │ ← Combine all context
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  LLM Response Generation │ ← Grounded in context
└─────────────────────────┘
    ↓
Final Response
```

### Key Components

1. **LLM Provider Layer** (`assistant/llm_provider.py`)
   - Provider abstraction for Gemini, OpenAI, and NVIDIA NIM
   - Automatic fallback logic (NVIDIA NIM → Gemini → OpenAI)
   - Environment-based configuration

2. **LLM Agent** (`assistant/agent.py`)
   - LLM-driven reasoning engine
   - Conversation context management
   - Tool calling orchestration
   - Guardrail enforcement
   - Grounding validation

3. **Retrieval System** (`assistant/retrieval.py`)
   - FAISS vector store
   - sentence-transformers embeddings
   - Semantic search over catalogue

4. **Tool System** (`assistant/tools.py`)
   - Pydantic-validated tool implementations
   - Tool registry with descriptions
   - Structured output for LLM consumption

5. **UI** (`ui/app.py`)
   - Streamlit web interface
   - Conversation history display
   - Quick action buttons

6. **Evaluation** (`eval/run_eval.py`)
   - Comprehensive test suite
   - Quality metric collection
   - Failure analysis

## 📊 Evaluation

The evaluation framework validates:

1. **Retrieval Quality**: Accuracy of semantic search results
2. **Tool-Calling Quality**: Correct tool selection and parameter extraction
3. **Conversation Quality**: Multi-turn dialogue handling
4. **Grounding Quality**: Response accuracy against catalogue data
5. **Multi-Turn Reasoning**: Follow-up query understanding

### Current Results
- **Total Tests**: 25
- **Categories**: Happy Path (10), Tricky/Ambiguous (10), Out-of-Scope (5)
- **Target**: 100% pass rate across all categories

Run evaluation:
```bash
python eval/run_eval.py
```

## 🎯 Assumptions

### Data Assumptions
1. **Catalogue**: 600 products with fields: sku, name, category, brand, vehicle_fitment, price_inr, stock, description
2. **SKU Format**: Uppercase letters (2-4) followed by hyphen and numbers (3-5), e.g., BRK-1007
3. **Vehicle Fitment**: Either specific make/model (e.g., "Bajaj Pulsar 150") or "Universal"
4. **Prices**: In INR (Indian Rupees), integers
5. **Stock**: Non-negative integers, 0 means out of stock

### Query Assumptions
1. **Multi-Turn Context**: The LLM maintains conversation context for follow-up queries
2. **Tool Calling**: The LLM decides when and how to call tools based on query intent
3. **Grounding**: All product-specific responses are validated against retrieved data

### Technical Assumptions
1. **LLM Priority**: NVIDIA NIM is preferred, with Gemini and OpenAI as fallbacks (NVIDIA NIM → Gemini → OpenAI)
2. **Embedding Model**: all-MiniLM-L6-v2 from sentence-transformers (384 dimensions)
3. **Similarity Metric**: Cosine similarity via FAISS Inner Product
4. **Confidence Threshold**: 0.7 for semantic similarity (bypassed for SKU queries)

## 🚀 Future Enhancements

### Immediate
1. **Enhanced Conversation Memory**: Better context tracking for longer dialogues
2. **Improved Tool Descriptions**: More detailed schemas for better LLM understanding
3. **Quality Metric Refinement**: More sophisticated evaluation of each quality dimension

### Bonus Features (Per Assignment)
1. **Demand Forecasting (Part B)**: Time-series forecasting for sales data
2. **Multimodal Image Recognition**: Identify parts from images using vision models

## 📚 Documentation

- **[DESIGN.md](DESIGN.md)**: Detailed design decisions and methodology


## 🌟 Compliance Checklist

- [x] **SUB-001**: Code pushed to public GitHub repository
- [x] **SUB-002**: README and DESIGN.md complete and clear
- [x] **SUB-003**: The assistant runs
- [x] **SUB-004**: Eval set and its results are included
- [ ] **SUB-005**: Demand forecasting code (Part B - bonus, not attempted)
- [x] **SUB-006**: No hardcoded API keys or secrets

---


**Project**: GearGuide-AI  
**Architecture**: LLM-First  
**Last Commit**: LLM Migration Complete
