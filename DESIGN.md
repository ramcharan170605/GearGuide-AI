# VIKMO AI/ML Intern Assignment - Design Document

> **Status**: Implementation in progress - Phase 1
> **Last Updated**: 2026-06-21

## Table of Contents
1. [Overview](#1-overview)
2. [Part A - Dealer Assistant](#2-part-a---dealer-assistant)
3. [Part B - Demand Forecasting](#3-part-b---demand-forecasting)
4. [Evaluation Methodology](#4-evaluation-methodology)
5. [Failure Analysis](#5-failure-analysis)

---

## 1. Overview
This document explains the key design decisions for the VIKMO AI/ML Intern Assignment implementation. All decisions trace back to requirements in assignment.md.

---

## 2. Part A - Dealer Assistant

### 2.1 Retrieval Approach
*To be populated after P1-T007*

**Embedding Model**: 
- **Choice**: all-MiniLM-L6-v2 from sentence-transformers
- **Justification**: 
  - Good quality (384 dimensions) with 83.1% accuracy on STS benchmark
  - Free and open-source, no API costs or rate limits
  - Local execution ensures reproducibility and privacy
  - Lightweight (80MB) and fast inference
- **Alternative considered**: Google's text-embedding-004 (higher quality but requires API, has costs)
- **Requirement Trace**: RAG-001, RAG-005, RAG-006

**Vector Store**:
- **Choice**: FAISS (Facebook AI Similarity Search) with IndexIDMap2
- **Justification**: 
  - Efficient in-memory vector search with O(1) lookup
  - No external dependencies (pure Python/C++ with numpy)
  - Scales to millions of vectors
  - IndexIDMap2 allows mapping integer IDs to vectors for metadata retrieval
  - Uses Inner Product (cosine similarity) which works well with normalized embeddings
- **Alternative considered**: Chroma (persistent storage but heavier, requires separate server)
- **Requirement Trace**: RAG-001, RAG-002, RAG-006

**Indexing Strategy**:
- Each product is a single document (no chunking needed - products are self-contained)
- Index fields: name, description, vehicle_fitment, category, brand, sku
- Embedding text: concatenate name + " " + description + " " + vehicle_fitment
- This combines the most semantically rich fields for matching user queries
- **Requirement Trace**: RAG-003, RAG-007

**Why this scales beyond prompt-stuffing**:
- 600 products × ~200 tokens each = ~120,000 tokens total
- Largest context windows (Claude 3: 200K, GPT-4: 128K) cannot fit all products
- Even with compression, semantic relationships would be lost
- Vector search enables O(log n) semantic matching without prompt limitations
- **Requirement Trace**: RAG-002, RAG-006

**Why this scales beyond prompt-stuffing**:
- 600 products × ~200 tokens = ~120,000 tokens
- Typical context window: 8,192-32,768 tokens
- Cannot fit all products in prompt even with largest context windows
- Vector search enables semantic matching without prompt limitations

### 2.2 Tool Design

**Tool Architecture**:
- **Pattern**: Registry pattern with function wrappers
- **Implementation**: ToolRegistry class that stores callable functions with schemas
- **Discovery**: Tools can be listed and retrieved by name at runtime
- **Requirement Trace**: TOOL-004, TOOL-006

**Required Tools Implemented**:

1. **check_stock** (TOOL-001, TOOL-005, TOOL-007)
   - Input: SKU (string)
   - Output: CheckStockResult (Pydantic model with sku, name, stock, in_stock, price_inr, category)
   - Behavior: Looks up product by SKU and returns structured stock information
   - Validation: Raises ValueError if SKU not found

2. **find_parts_by_vehicle** (TOOL-003, TOOL-005)
   - Input: make (string), model (string), year (optional string), limit (int = 10)
   - Output: list[dict] of matching products
   - Behavior: 
     - First tries exact match on vehicle_fitment field
     - Falls back to partial match (make and model in any order)
     - Returns metadata with all product fields
   - Validation: Returns empty list if no matches found

3. **create_order** (TOOL-002, TOOL-005, TOOL-007)
   - Input: dealer (string), items (list[OrderItem])
   - Output: CreateOrderResult (Pydantic model with order_id, dealer, items, total_inr, status)
   - Behavior:
     - Validates all SKUs exist in catalogue
     - Calculates total from price_inr × quantity for each item
     - Generates unique order ID using UUID
     - Returns structured confirmation
   - Validation: Raises ValueError if SKU not found or quantity <= 0

**Structured Output**:
- All tools use Pydantic models for output validation
- Ensures consistent structure across all tool responses
- Prevents hallucinated or malformed output
- **Requirement Trace**: TOOL-005, TOOL-007

**How the Model Decides to Call Tools**:
- The agent (agent.py) will parse user intent from the query
- Match intent to available tool names and descriptions
- Extract parameters from the query
- Call the appropriate tool via the registry
- **Requirement Trace**: TOOL-004, TOOL-006

**Task Trace**: P2-T001, P2-T002, P2-T003, P2-T004

### 2.3 Prompt Design and Guardrails

**Agent Loop Architecture**:
- **Pattern**: Modular agent with clear separation of concerns
- **Components**: 
  - ConversationContext: Maintains conversation state (history, retrieval results, tool calls)
  - DealerAssistant: Main agent class that orchestrates retrieval, tool calling, and response generation
- **Flow**: Query → Clarification Check → Retrieval → Action Decision → Tool/Retrieval → Grounding Check → Response
- **Requirement Trace**: CONV-001, CONV-002, TOOL-004, TOOL-006

**Clarification Logic** (CONV-003, CONV-004):
- **Detection**: Pattern-based detection of ambiguous queries
- **Triggers**: 
  - Queries with vehicle-related terms but no vehicle specification
  - Short queries (<= 2 words) without specific terms
  - Low-confidence retrieval results (score < 0.7) without SKU
- **Skip Conditions**: 
  - Queries containing specific SKUs
  - Queries with explicit actions (stock check, order creation)
- **Response**: Context-specific clarification questions
- **Task Trace**: P2-T006

**Tool Calling Decision** (TOOL-004, TOOL-006):
- **Pattern Matching**: Keyword-based detection of tool invocation patterns
- **Patterns**:
  - `check_stock`: "stock", "availability", "what is the stock", "what's the stock"
  - `create_order`: "place order", "create order", "order for"
  - `find_parts_by_vehicle`: "for [vehicle make]"
- **Fallback**: Retrieval-based response if no tool matches
- **Task Trace**: P2-T007

**Grounding Checks** (GROUND-001, GROUND-002):
- **Approach**: 
  - Tool responses: Exempt from grounding check (tools use catalogue data directly)
  - Retrieval responses: Verify all mentioned SKUs are in retrieval results
- **Implementation**: Regex-based extraction of SKUs from response, cross-reference with retrieval
- **Task Trace**: P3-T004

**Confidence Threshold**:
- **Value**: 0.7 for semantic similarity
- **Behavior**: 
  - If top retrieval score < 0.7 and no SKU in query: ask for clarification
  - If top retrieval score < 0.7 but SKU present: proceed with tool/action (SKUs are specific identifiers)
- **Justification**: SKU-based queries are exact lookups and don't require semantic similarity

**Task Trace**: P2-T005, P2-T006, P2-T007, P3-T004

---

## 3. Part B - Demand Forecasting

### 3.1 Model Choice
*To be populated if Part B is implemented*

### 3.2 Validation Scheme
*To be populated if Part B is implemented*

### 3.3 Leakage Prevention
*To be populated if Part B is implemented*

---

## 4. Evaluation Methodology
*To be populated after P4-T008*

---

## 5. Failure Analysis
*To be populated after P4-T008*
