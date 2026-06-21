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
*To be populated after P2-T004*

### 2.3 Prompt Design and Guardrails
*To be populated after P3-T004*

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
