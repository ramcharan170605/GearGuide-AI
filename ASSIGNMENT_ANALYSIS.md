# VIKMO AI/ML Intern Assignment - Analysis

## Executive Summary

This take-home assignment evaluates **Applied GenAI judgment** and **classical ML methodology** through two parts:
- **Part A (Core - 100 points)**: Build a conversational LLM-powered Dealer Assistant with RAG, tool-calling, and evaluation
- **Part B (Bonus - +15-20 points)**: Demand forecasting with leakage-free validation

**CRITICAL**: Part A is the core evaluation. Part B is weighted bonus. All grading follows the chain: assignment.md → Planner → ASSIGNMENT_ANALYSIS.md → REQUIREMENTS.md → planner.txt → Coder → Implementation → Reviewer → Compliance + Gap Analysis

---

## 1. Requirements Extraction from assignment.md

### 1.1 Technical Requirements

| Category | Requirement | Source (assignment.md line) |
|----------|-------------|----------------------------|
| **Language** | Python | Line 40 |
| **LLM Provider** | Any (Google Gemini recommended, free tier) | Lines 42-43 |
| **Frameworks** | Any (LangChain, LlamaIndex, provider SDK, or custom) | Lines 45-46 |
| **Secrets Management** | Environment variables, NO hardcoded API keys | Line 48 |

### 1.2 Data Requirements

| Dataset | Purpose | Size | Fields |
|---------|---------|------|--------|
| `catalogue.csv` / `catalogue.json` | Part A retrieval corpus | ~600 SKUs | sku, name, category, brand, vehicle_fitment, price_inr, stock, description |
| `sales_history.csv` | Part B forecasting | 2,340 rows (30 SKUs × 78 weeks) | date, sku, units_sold, promo_flag |

**Note**: Catalogue has 600 SKUs (not 500-1000 as stated in assignment), which confirms "large enough that stuffing the whole thing into the prompt is not a valid approach."

### 1.3 Part A - Dealer Assistant Requirements

#### 1.3.1 Retrieval (RAG) - 20 points
- **MUST**: Implement real retrieval (embeddings + vector search, or justified alternative)
- **MUST NOT**: Stuff entire catalogue into prompt
- **MUST**: Explain chunking/embedding/indexing choices in DESIGN.md
- **Data**: Index the catalogue for product retrieval

#### 1.3.2 Tools (Function Calling) - 25 points
- **MUST**: Implement at least these three tools:
  - `check_stock` - look up availability for a product
  - `create_order` - place order with structured output (validated JSON)
  - `find_parts_by_vehicle` - find parts by make/model/year
- **MUST**: Model chooses tools from conversation
- **MUST**: Order creation uses structured output (not free text)

#### 1.3.3 Conversation - 10 points
- **MUST**: Handle multi-turn dialogue
- **MUST**: Maintain context across turns
- **MUST**: Ask clarifying questions for ambiguous requests

#### 1.3.4 Grounding - Implicit requirement
- **MUST**: Answers about products, prices, stock come from data
- **MUST NOT**: Hallucinate/invent information

#### 1.3.5 Evaluation - 20 points
- **MUST**: Provide eval set covering:
  - Happy paths
  - Tricky/ambiguous queries
  - Out-of-scope requests
- **MUST**: Define "correct" for each test case
- **MUST**: Run assistant against eval set
- **MUST**: Report results with simple metrics
- **MUST**: Analyze failure modes honestly (where, why, what to change)

#### 1.3.6 Interface
- **MINIMUM**: CLI or script
- **BONUS**: Chat/WhatsApp-style UI (+5 points)

### 1.4 Part B - Demand Forecasting (Bonus +15-20 points)

#### 1.4.1 Requirements
- **MUST**: Forecast near-term demand (e.g., next 4 weeks) per SKU
- **MUST**: Hold out most recent period as test window
- **MUST**: Fit ONLY on data before test window (no leakage)
- **MUST**: Report error per SKU and overall (MAE or MAPE)
- **MUST**: Beat a naive baseline (last-value, seasonal-naive, moving average)
- **MUST**: Report baseline error and show improvement
- **MUST**: Justify any complexity over baseline

#### 1.4.2 Documentation
- **MUST**: Document approach, validation scheme, leakage prevention in DESIGN.md

### 1.5 Documentation Requirements

#### 1.5.1 DESIGN.md (Required - 15 points)
- Retrieval approach and justification
- Embedding/indexing choices
- Tool design and how model decides to call them
- Prompt design and guardrails
- Evaluation methodology and failure analysis
- (Part B) Model choice, validation scheme, leakage prevention

#### 1.5.2 README.md (Required)
1. Overview and what you implemented
2. Tech stack and which LLM/provider used
3. Setup (environment variables, how to run assistant and eval)
4. Example interactions
5. Any assumptions, with pointer to DESIGN.md and eval results

### 1.6 Repository Structure (Required)

```
your-repo/
├── README.md
├── DESIGN.md
├── requirements.txt
├── assistant/          # retrieval, tools, agent loop
│   ├── __init__.py
│   ├── retrieval.py
│   ├── tools.py
│   ├── agent.py
│   └── ...
├── eval/               # eval set + results
│   ├── eval_set.jsonl
│   ├── run_eval.py
│   └── results.json
└── forecasting/        # optional - Part B
    ├── forecast.py
    ├── baseline.py
    └── results.json
```

### 1.7 Scoring Rubric

| Category | Points | Weight | Key Factors |
|----------|--------|--------|-------------|
| Retrieval / RAG | 20 | Core | Real retrieval, sensible indexing, scales beyond prompt-stuffing |
| Agent & Tool-Calling | 25 | Core | Tool design, correct invocation, structured order output |
| Conversation Handling | 10 | Core | Multi-turn context, clarifying questions |
| Evaluation & Failure Analysis | 20 | Core | Meaningful eval set, metrics, honest failure analysis |
| Code Quality | 10 | Core | Clean, readable, well-structured |
| DESIGN.md & Documentation | 15 | Core | Setup, clear reasoning, methodology |
| **Part B - Demand Forecasting** | **+15-20** | Bonus | Beats baseline, leakage-free backtest, clear metrics |
| **Chat/WhatsApp-style UI** | **+5** | Bonus | Extra UI beyond CLI |
| **Multimodal - Image Recognition** | **+5** | Bonus | Identify part from image |
| **Guardrails** | **+5** | Bonus | Against off-topic/hallucination |

**Total Core**: 100 points | **Maximum with all bonuses**: 130 points

### 1.8 Submission Checklist

- [ ] Code pushed to public GitHub repository
- [ ] README and DESIGN.md complete and clear
- [ ] The assistant runs
- [ ] Eval set and its results are included
- [ ] If attempted, forecasting code and results included
- [ ] No hardcoded API keys or secrets

### 1.9 Sample Interactions (Part A)

The assistant must handle:
1. "Do you have brake pads for a Bajaj Pulsar 150?" → retrieve fitting parts, confirm vehicle, return options with price and stock
2. "Place an order for 10 units of [SKU] for ABC Motors." → call create_order with structured payload, return confirmation
3. "What's the cheapest chain lube you stock?" → grounded answer from catalogue
4. "I need tyres." → ask clarifying question (vehicle/size)
5. "What's the weather today?" → stay on-domain / politely decline (guardrail)

---

## 2. Data Analysis

### 2.1 Catalogue Analysis

**File**: `catalogue.csv` / `catalogue.json`
**Records**: 600 SKUs
**Fields**:
- `sku`: string, unique (e.g., "BRK-1042", "CAR-1024")
- `name`: string, display name
- `category`: string, 11 categories
- `brand`: string
- `vehicle_fitment`: string, make+model or "Universal"
- `price_inr`: integer, in INR
- `stock`: integer, units on hand (0 to 400+ range)
- `description`: string, product blurb

**Categories** (from sample): Car Care & Accessories, Body & Styling, Drivetrain, Tyres & Tubes, Lighting & Electrical, Engine & Oils, Brakes, Controls & Grips, Filters

**Key Observations**:
- `vehicle_fitment` field enables `find_parts_by_vehicle` tool
- `stock` field enables `check_stock` tool
- Some SKUs have 0 stock (e.g., ELE-1051)
- `description` field useful for semantic search

### 2.2 Sales History Analysis

**File**: `sales_history.csv`
**Records**: 2,340 rows
**Structure**: 30 SKUs × 78 weekly observations
**Date Range**: 2024-12-16 to 2026-06-08 (Mondays)
**Fields**:
- `date`: ISO date (week start)
- `sku`: string, matches catalogue
- `units_sold`: integer
- `promo_flag`: 0/1

**Key Observations**:
- Data includes trend, seasonality, festive lift (Oct-Nov), promo spikes, Poisson noise
- Most recent weeks suitable for held-out test window
- Need to identify which SKUs are in the forecasting subset (30 out of 600)

---

## 3. Architectural Decisions Mapping to Requirements

### 3.1 Part A Architecture

#### 3.1.1 Retrieval System
**Requirement**: Real retrieval, sensible indexing, scales beyond prompt-stuffing (20 points)

**Decision**: 
- Use **sentence-transformers** for embeddings (free, local, no API costs)
- Use **FAISS** or **Chroma** for vector store (efficient, scalable)
- Chunking: Each product is a document (no need to chunk - products are self-contained)
- Index fields: name, description, vehicle_fitment, category, brand
- Query expansion: For vehicle queries, boost vehicle_fitment field

**Justification**:
- 600 products × ~200 tokens each = ~120K tokens, far exceeds typical context windows
- Vector search enables semantic matching beyond keyword search
- Local embeddings ensure reproducibility and no API costs

#### 3.1.2 Tool System
**Requirement**: Tool design, correct invocation, structured order output (25 points)

**Decision**:
- Implement tool registry pattern with function schemas
- Use **Function Calling** via LLM (if supported) or **ReAct** pattern (reason + act)
- Tools:
  1. `check_stock(sku: str) -> dict`: Returns stock for SKU
  2. `create_order(dealer: str, items: list[dict]) -> dict`: Creates order with validation
  3. `find_parts_by_vehicle(make: str, model: str, year: str = None) -> list[dict]`: Finds matching parts
  4. `search_products(query: str, limit: int = 5) -> list[dict]`: Semantic search

**Structured Output**: Use Pydantic models for validation

#### 3.1.3 Agent Loop
**Requirement**: Multi-turn context, clarifying questions (10 points)

**Decision**:
- Use conversational history with memory
- Implement **clarification trigger**: When query is ambiguous, ask follow-up
- State machine: WAITING_FOR_QUERY → CLARIFYING → EXECUTING → RESPONDING

#### 3.1.4 Grounding & Guardrails
**Requirement**: Answers from data, not invented; guardrails against off-topic (+5 bonus)

**Decision**:
- **Grounding**: All product answers come from retrieval results only
- **Guardrails**:
  - Domain detection: Classify query as on-topic or off-topic
  - Confidence threshold: If retrieval confidence < threshold, ask for clarification
  - Response validation: Check that all factual claims are in retrieved documents

### 3.2 Part B Architecture

#### 3.2.1 Forecasting Approach
**Requirement**: Beat baseline, leakage-free, clear metrics (+15-20 points)

**Decision**:
- **Baseline**: Seasonal Naive (last year same week) or Moving Average
- **Model**: SARIMAX or Prophet for seasonality handling
- **Validation**: Time-series split with expanding window
- **Test Window**: Last 4-8 weeks held out
- **Metrics**: MAE, MAPE, per-SKU and overall

**Leakage Prevention**:
- Strict date-based filtering: train only on data < test window start
- No lookahead features
- Walk-forward validation

### 3.3 Evaluation System

**Requirement**: Meaningful eval set, metrics, honest failure analysis (20 points)

**Decision**:
- **Eval Set**: 20-30 test interactions covering:
  - 10 happy path queries
  - 10 tricky/ambiguous queries
  - 5 out-of-scope requests
- **Metrics**:
  - Retrieval: Hit rate, MRR
  - Tool execution: Success rate, accuracy
  - Conversation: Context maintenance score
  - Overall: Pass/fail per test case
- **Failure Analysis**: Categorize failures and propose fixes

---

## 4. Implementation Priority

### 4.1 Phase 1: Foundation (Day 1)
- [ ] Set up repository structure
- [ ] Create requirements.txt with dependencies
- [ ] Implement retrieval system (embeddings + vector store)
- [ ] Index catalogue
- [ ] Basic CLI interface

### 4.2 Phase 2: Tools & Agent (Day 1-2)
- [ ] Implement all required tools (check_stock, create_order, find_parts_by_vehicle)
- [ ] Implement agent loop with conversation memory
- [ ] Add clarification logic
- [ ] Add grounding checks

### 4.3 Phase 3: Evaluation (Day 2)
- [ ] Create eval set
- [ ] Implement evaluation runner
- [ ] Run evaluation and analyze failures
- [ ] Iterate on fixes

### 4.4 Phase 4: Documentation & Polish (Day 2-3)
- [ ] Write DESIGN.md
- [ ] Write README.md
- [ ] Add example interactions
- [ ] Clean code and add comments

### 4.5 Phase 5: Bonus (If time permits)
- [ ] Part B: Demand forecasting
- [ ] Chat/WhatsApp-style UI
- [ ] Guardrails
- [ ] Multimodal (image recognition)

---

## 5. Risk Assessment

### 5.1 High Priority Risks

| Risk | Mitigation | Contingency |
|------|------------|-------------|
| LLM API costs/unavailability | Use Google Gemini free tier or local Ollama | Fall back to smaller local model |
| Retrieval quality poor | Test multiple embedding models, tune parameters | Use hybrid search (semantic + keyword) |
| Tool calling unreliable | Implement validation and retry logic | Use ReAct pattern for more control |
| Evaluation too subjective | Define clear pass/fail criteria for each test | Use automated checks where possible |

### 5.2 Medium Priority Risks

| Risk | Mitigation | Contingency |
|------|------------|-------------|
| Time constraints | Prioritize core Part A first | Skip bonuses if needed |
| Data quality issues | Validate data on load | Clean data programmatically |
| Forecasting complexity | Start with simple models | Use statistical methods, not deep learning |

---

## 6. Success Criteria

### 6.1 Minimum Viable Submission (Pass)
- [ ] Part A fully functional (retrieval, tools, conversation)
- [ ] Eval set with results and failure analysis
- [ ] DESIGN.md and README.md complete
- [ ] Repository structure correct
- [ ] No hardcoded API keys

### 6.2 Good Submission (Strong Pass)
- [ ] All above
- [ ] Clean, well-documented code
- [ ] Thoughtful retrieval and tool design
- [ ] Honest, insightful failure analysis

### 6.3 Excellent Submission (Top Tier)
- [ ] All above
- [ ] Part B forecasting with baseline beating
- [ ] Bonus features (UI, guardrails, multimodal)
- [ ] Exceptional documentation and code quality

---

## 7. Next Steps

Proceed to **REQUIREMENTS.md** to formalize the extracted requirements into actionable items, then to **planner.txt** for task breakdown and scheduling.

**Chain Compliance**: This document traces all requirements directly to assignment.md. All subsequent planning must reference sections from this analysis.
