# VIKMO AI/ML Intern Assignment - Formal Requirements Specification

> **CRITICAL**: This document is derived directly from `assignment.md`. All implementation must trace back to these requirements. If any conflict arises: assignment.md → ASSIGNMENT_ANALYSIS.md → REQUIREMENTS.md → planner.txt → Implementation

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Technical Constraints](#2-technical-constraints)
3. [Part A - Dealer Assistant Requirements](#3-part-a---dealer-assistant-requirements)
4. [Part B - Demand Forecasting Requirements](#4-part-b---demand-forecasting-requirements)
5. [Documentation Requirements](#5-documentation-requirements)
6. [Repository Structure Requirements](#6-repository-structure-requirements)
7. [Scoring and Evaluation](#7-scoring-and-evaluation)
8. [Submission Checklist](#8-submission-checklist)

---

## 1. Project Overview

### 1.1 Objective
Build a conversational LLM-powered Dealer Assistant (Part A - Core) with optional Demand Forecasting (Part B - Bonus) for an auto-parts catalogue.

### 1.2 Primary Focus Areas
- LLM / agent engineering, retrieval, evaluation, ML methodology
- Conversational and agentic AI layer (mirrors VIKMO's actual work)
- Reliability and evaluation rigor
- Clean ML methodology for forecasting

### 1.3 What VIKMO Evaluates
- ✅ Applied GenAI judgment: retrieval, tool/function calling, grounding, structured output
- ✅ Rigour about reliability: evaluation methodology, failure analysis
- ✅ Sound ML methodology: clean validation, no leakage, beating baselines
- ✅ Clean, readable, well-organised code and honest documentation

**Source**: assignment.md lines 19-24

---

## 2. Technical Constraints

### 2.1 Language
- **REQUIRED**: Python
- **Source**: assignment.md line 40

### 2.2 LLM Provider
- **RECOMMENDED**: Google Gemini (free tier, mirrors VIKMO's stack)
- **ACCEPTABLE**: OpenAI, Groq, Ollama, or any local model
- **Source**: assignment.md lines 42-43

### 2.3 Frameworks
- **OPTION**: LangChain, LlamaIndex, provider SDK directly, or custom orchestration
- **REQUIREMENT**: Must be able to explain your choice
- **Source**: assignment.md lines 45-46

### 2.4 Secrets Management
- **MUST NOT**: Hardcode API keys
- **MUST**: Read from environment variables
- **Source**: assignment.md line 48

---

## 3. Part A - Dealer Assistant Requirements

### 3.1 Retrieval (RAG) - 20 Points

#### 3.1.1 Core Requirements
- **RAG-001**: MUST implement real retrieval (embeddings + vector search, or justified alternative)
- **RAG-002**: MUST NOT stuff entire catalogue into prompt
- **RAG-003**: MUST index the catalogue for product retrieval
- **RAG-004**: Catalogue is ~600 SKUs (large enough to require real retrieval)
- **Source**: assignment.md lines 55-60

#### 3.1.2 Implementation Requirements
- **RAG-005**: MUST explain chunking/embedding/indexing choices in DESIGN.md
- **RAG-006**: Retrieval must be sensible and scale beyond prompt-stuffing
- **Source**: assignment.md lines 59-60, 104-106

#### 3.1.3 Data Fields to Index
- **RAG-007**: MUST index: sku, name, category, brand, vehicle_fitment, price_inr, stock, description
- **Source**: DATA_README.md lines 14-24

### 3.2 Tools (Function Calling) - 25 Points

#### 3.2.1 Required Tools
- **TOOL-001**: MUST implement `check_stock` - look up availability for a product
- **TOOL-002**: MUST implement `create_order` - place an order for a dealer with line items
- **TOOL-003**: MUST implement `find_parts_by_vehicle` - find parts that fit a given make/model/year
- **Source**: assignment.md lines 62-66

#### 3.2.2 Tool Invocation Requirements
- **TOOL-004**: Model MUST call tools to act, choosing them from the conversation
- **TOOL-005**: Order creation MUST use structured output (validated JSON, not free text)
- **Source**: assignment.md lines 62, 66-67

#### 3.2.3 Tool Design Requirements
- **TOOL-006**: Tool design must be sensible and well-architected
- **TOOL-007**: Tool invocation must be correct
- **Source**: assignment.md lines 108-110

### 3.3 Conversation - 10 Points

#### 3.3.1 Multi-Turn Dialogue
- **CONV-001**: MUST handle multi-turn dialogue
- **CONV-002**: MUST maintain context across turns
- **Source**: assignment.md line 68

#### 3.3.2 Clarification
- **CONV-003**: MUST ask clarifying questions when request is ambiguous
- **CONV-004**: Example: "I need brake pads" → "for which bike?"
- **Source**: assignment.md lines 69-70

### 3.4 Grounding

#### 3.4.1 Factual Accuracy
- **GROUND-001**: Answers about products, prices, and stock MUST come from the data
- **GROUND-002**: MUST NOT invent/hallucinate information
- **Source**: assignment.md line 72

### 3.5 Evaluation - 20 Points

#### 3.5.1 Eval Set Requirements
- **EVAL-001**: MUST provide an eval set
- **EVAL-002**: Eval set MUST cover happy paths
- **EVAL-003**: Eval set MUST cover tricky/ambiguous queries
- **EVAL-004**: Eval set MUST cover out-of-scope requests
- **Source**: assignment.md lines 74-75

#### 3.5.2 Evaluation Execution
- **EVAL-005**: MUST define what 'correct' means for each test case
- **EVAL-006**: MUST run assistant against the eval set
- **EVAL-007**: MUST report results with simple metrics
- **Source**: assignment.md lines 76-78

#### 3.5.3 Failure Analysis
- **EVAL-008**: MUST analyze failure modes honestly
- **EVAL-009**: MUST explain where it breaks
- **EVAL-010**: MUST explain why it breaks
- **EVAL-011**: MUST explain what would be changed to fix it
- **Source**: assignment.md lines 79-80

#### 3.5.4 Evaluation Weight
- **EVAL-012**: Evaluation carries real weight in grading
- **EVAL-013**: This (not a polished demo) is the clearest signal of an AI engineer
- **Source**: assignment.md line 80

### 3.6 Interface

#### 3.6.1 Minimum Requirements
- **UI-001**: CLI or script is acceptable
- **Source**: assignment.md line 81

#### 3.6.2 Bonus
- **UI-002**: Chat/WhatsApp-style UI (+5 bonus points)
- **Source**: assignment.md line 130

### 3.7 Sample Interactions

The assistant MUST handle interactions like:
- **SAMPLE-001**: "Do you have brake pads for a Bajaj Pulsar 150?" → retrieve fitting parts, confirm vehicle if needed, return options with price and stock
- **SAMPLE-002**: "Place an order for 10 units of [SKU] for ABC Motors." → call create_order with structured payload, return confirmation
- **SAMPLE-003**: "What's the cheapest chain lube you stock?" → grounded answer from catalogue
- **SAMPLE-004**: "I need tyres." → ask clarifying question (vehicle/size)
- **SAMPLE-005**: "What's the weather today?" → stay on-domain / politely decline (guardrail)
- **Source**: assignment.md lines 158-165

---

## 4. Part B - Demand Forecasting Requirements

### 4.1 Core Requirements
- **FC-001**: Using provided sales history, forecast near-term demand (e.g., next 4 weeks) per SKU
- **Source**: assignment.md line 83

### 4.2 Validation Requirements
- **FC-002**: MUST hold out most recent period as test window
- **FC-003**: MUST fit only on data before test window (NO LEAKAGE)
- **FC-004**: MUST report error per SKU and overall
- **Source**: assignment.md lines 84-86

### 4.3 Metrics
- **FC-005**: MUST use sensible metric (e.g., MAE or MAPE)
- **Source**: assignment.md line 86

### 4.4 Baseline Comparison
- **FC-006**: MUST beat a naive baseline (last-value, seasonal-naive, moving average)
- **FC-007**: MUST report baseline's error
- **FC-008**: MUST show approach beats baseline
- **FC-009**: MUST justify any complexity over baseline
- **FC-010**: If simple method wins, MUST say so
- **Source**: assignment.md lines 87-90

### 4.5 Documentation
- **FC-011**: MUST document approach, validation scheme, leakage prevention in DESIGN.md
- **Source**: assignment.md line 91

### 4.6 Data
- **FC-012**: Sales history: 2,340 rows (30 SKUs × 78 weeks)
- **FC-013**: Date range: 2024-12-16 to 2026-06-08 (Mondays)
- **FC-014**: Fields: date, sku, units_sold, promo_flag
- **FC-015**: Data includes: trend, seasonality, festive lift, promo spikes, Poisson noise
- **Source**: DATA_README.md lines 28-37

---

## 5. Documentation Requirements

### 5.1 DESIGN.md - 15 Points

#### 5.1.1 Required Sections
- **DESIGN-001**: Explain retrieval approach and why
- **DESIGN-002**: Explain embedding/indexing choices
- **DESIGN-003**: Explain tool design and how model decides to call them
- **DESIGN-004**: Explain prompt design and guardrails against hallucination/off-topic
- **DESIGN-005**: Explain evaluation methodology and what failures taught you
- **DESIGN-006**: (Part B) Explain model choice, validation scheme, leakage prevention
- **Source**: assignment.md lines 92-97

### 5.2 README.md - Required

#### 5.2.1 Required Sections
- **README-001**: Overview and what you implemented
- **README-002**: Tech stack and which LLM/provider used
- **README-003**: Setup (environment variables, how to run assistant and eval)
- **README-004**: Example interactions
- **README-005**: Any assumptions, with pointer to DESIGN.md and eval results
- **Source**: assignment.md lines 146-150

---

## 6. Repository Structure Requirements

### 6.1 Required Structure
```
your-repo/
├── README.md                    [REPO-001]
├── DESIGN.md                    [REPO-002]
├── requirements.txt             [REPO-003]
├── assistant/                   [REPO-004]
│   └── (retrieval, tools, agent loop)
├── eval/                        [REPO-005]
│   └── (eval set + results)
└── forecasting/                 [REPO-006] (optional - Part B)
```
- **Source**: assignment.md lines 138-144

### 6.2 File Requirements
- **REPO-001**: MUST have README.md at root
- **REPO-002**: MUST have DESIGN.md at root
- **REPO-003**: MUST have requirements.txt at root
- **REPO-004**: MUST have assistant/ directory with retrieval, tools, agent loop
- **REPO-005**: MUST have eval/ directory with eval set and results
- **REPO-006**: IF attempting Part B, MUST have forecasting/ directory

---

## 7. Scoring and Evaluation

### 7.1 Core Scoring (100 Points Total)

| Category | Points | Requirements | Source |
|----------|--------|--------------|--------|
| Retrieval / RAG | 20 | RAG-001 to RAG-007 | assignment.md lines 103-106 |
| Agent & Tool-Calling | 25 | TOOL-001 to TOOL-007 | assignment.md lines 108-110 |
| Conversation Handling | 10 | CONV-001 to CONV-004 | assignment.md lines 111-113 |
| Evaluation & Failure Analysis | 20 | EVAL-001 to EVAL-011 | assignment.md lines 114-117 |
| Code Quality | 10 | Clean, readable, well-structured | assignment.md lines 118-120 |
| DESIGN.md & Documentation | 15 | DESIGN-001 to DESIGN-006, README-001 to README-005 | assignment.md lines 121-123 |

### 7.2 Bonus Scoring

| Bonus Item | Points | Requirements | Source |
|------------|--------|--------------|--------|
| Demand Forecasting (Part B) | +15-20 | FC-001 to FC-015 | assignment.md lines 126-128 |
| Chat/WhatsApp-style UI | +5 | UI-002 | assignment.md line 130 |
| Multimodal - Image Recognition | +5 | Identify part from image | assignment.md line 132 |
| Guardrails | +5 | Against off-topic/hallucination | assignment.md line 134 |

**Maximum Score**: 130 points (100 core + 30 bonus)

---

## 8. Submission Checklist

### 8.1 Required Items
- [ ] **SUB-001**: Code pushed to public GitHub repository
- [ ] **SUB-002**: README and DESIGN.md complete and clear
- [ ] **SUB-003**: The assistant runs
- [ ] **SUB-004**: Eval set and its results are included
- [ ] **SUB-005**: If attempted, forecasting code and results included
- [ ] **SUB-006**: No hardcoded API keys or secrets
- **Source**: assignment.md lines 152-156

---

## 9. Traceability Matrix

All requirements in this document trace back to `assignment.md`. The following mapping shows the lineage:

```
assignment.md
    ├── Project Overview (lines 1-27) → Section 1
    ├── Technical Requirements (lines 37-48) → Section 2
    ├── Provided Data (lines 49-53) → Sections 3.1.3, 4.6
    ├── Part A Requirements (lines 55-81) → Section 3
    ├── Part B Requirements (lines 83-90) → Section 4
    ├── DESIGN.md Requirements (lines 92-97) → Section 5.1
    ├── Evaluation Criteria (lines 99-134) → Section 7
    ├── Submission Requirements (lines 136-156) → Section 8
    └── Sample Interactions (lines 158-165) → Section 3.7
```

---

## 10. Implementation Compliance Checklist

Before implementing any feature, verify:
- [ ] The requirement exists in this document
- [ ] The requirement traces back to assignment.md
- [ ] The implementation satisfies the requirement
- [ ] The implementation is documented in DESIGN.md (if applicable)
- [ ] The implementation is tested in the eval set (if applicable)

**CRITICAL RULE**: If planner.txt conflicts with this document, this document wins. If this document conflicts with assignment.md, assignment.md wins.

---

## Appendix A: Requirement IDs Quick Reference

### Part A Requirements (61 total)
- **RAG**: 7 requirements (RAG-001 to RAG-007)
- **TOOL**: 7 requirements (TOOL-001 to TOOL-007)
- **CONV**: 4 requirements (CONV-001 to CONV-004)
- **GROUND**: 2 requirements (GROUND-001 to GROUND-002)
- **EVAL**: 11 requirements (EVAL-001 to EVAL-011)
- **UI**: 2 requirements (UI-001 to UI-002)
- **SAMPLE**: 5 requirements (SAMPLE-001 to SAMPLE-005)

### Part B Requirements (15 total)
- **FC**: 15 requirements (FC-001 to FC-015)

### Documentation Requirements (11 total)
- **DESIGN**: 6 requirements (DESIGN-001 to DESIGN-006)
- **README**: 5 requirements (README-001 to README-005)

### Repository Requirements (6 total)
- **REPO**: 6 requirements (REPO-001 to REPO-006)

### Submission Requirements (6 total)
- **SUB**: 6 requirements (SUB-001 to SUB-006)

**Total Formal Requirements**: 104

---

*Next Step: Create planner.txt with task breakdown mapping each task to one or more requirements from this document.*
