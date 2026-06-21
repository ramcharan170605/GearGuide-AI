# VIKMO AI/ML Intern Assignment - Planning Summary

## ✅ Planning Complete

The complete planning chain has been established as required:

```
assignment.md (Authoritative Specification)
    ↓
ASSIGNMENT_ANALYSIS.md (Comprehensive Analysis)
    ↓
REQUIREMENTS.md (Formal Requirements with Traceability)
    ↓
planner.txt (Detailed Implementation Plan)
    ↓
Implementation (Next Step)
    ↓
Reviewer → Compliance + Gap Analysis
```

---

## 📋 Documents Created

### 1. ASSIGNMENT_ANALYSIS.md
- **Purpose**: Comprehensive analysis of assignment.md
- **Contents**:
  - Executive Summary
  - Requirements Extraction (mapped to assignment.md line numbers)
  - Data Analysis (catalogue and sales history)
  - Architectural Decisions with Justifications
  - Implementation Priority
  - Risk Assessment
  - Success Criteria
- **Traceability**: Every requirement traces to assignment.md

### 2. REQUIREMENTS.md
- **Purpose**: Formal specification of all requirements
- **Contents**:
  - 104 formal requirements organized by category
  - Requirement IDs (e.g., RAG-001, TOOL-002, EVAL-005)
  - Direct traceability to assignment.md
  - 8 main sections covering all aspects
- **Categories**:
  - Part A: 61 requirements (RAG, TOOL, CONV, GROUND, EVAL, UI, SAMPLE)
  - Part B: 15 requirements (FC)
  - Documentation: 11 requirements (DESIGN, README)
  - Repository: 6 requirements (REPO)
  - Submission: 6 requirements (SUB)

### 3. planner.txt
- **Purpose**: Detailed implementation roadmap
- **Contents**:
  - 8 Phases with 70+ tasks
  - Task IDs (e.g., P1-T001, P2-T004, P4-T008)
  - Duration estimates for each task
  - Priority levels (P0 = Must Have, P1 = Bonus)
  - Requirement mappings for each task
  - Dependency graph
  - Critical path identification
  - Milestones and checkpoints
  - Risk mitigation tasks
  - Quality checklist
- **Total Estimated Time**: ~33-34 hours (fits within 3-day window)

---

## 🎯 Implementation Roadmap

### Critical Path (20-22 hours) - MUST COMPLETE

```
Phase 1: Foundation (4h)
├── Set up repository and structure
├── Implement retrieval (embeddings + FAISS/Chroma)
└── Index catalogue

Phase 2: Agent Core (7.5h)
├── Implement all 3 required tools (check_stock, find_parts_by_vehicle, create_order)
├── Implement agent loop with conversation memory
├── Implement clarification logic
└── Create CLI interface

Phase 3: Grounding (2h)
├── Implement grounding checks
├── Implement confidence thresholds
└── Implement response validation

Phase 4: Evaluation (6h)
├── Create eval set (25 test cases: 10 happy, 10 tricky, 5 out-of-scope)
├── Implement eval runner
├── Run evaluation
└── Analyze and document failures

Phase 5: Documentation (5h)
├── Complete DESIGN.md (all required sections)
├── Complete README.md (all required sections)
└── Add code comments

Phase 8: Final Review (2.5h)
├── Run final evaluation
├── Verify submission checklist
├── Polish code
├── Push to GitHub
└── Verify no hardcoded API keys
```

### Bonus Path (Optional - 6-8 hours)

```
Phase 6: Part B Forecasting (3.5h) - +15-20 points
├── Analyze sales data
├── Implement naive baseline
├── Implement forecasting model (SARIMAX/Prophet)
├── Run leakage-free backtest
└── Document in DESIGN.md

Phase 7: Additional Bonuses (2.5h) - +5-15 points
├── Guardrails against hallucination (+5)
├── Chat/WhatsApp-style UI (+5)
└── Multimodal image recognition (+5)
```

---

## 📊 Scoring Strategy

### Core (100 points) - Guaranteed
- **Retrieval / RAG**: 20 points → Focus on solid implementation
- **Agent & Tool-Calling**: 25 points → Ensure all 3 tools work correctly
- **Conversation Handling**: 10 points → Multi-turn + clarification
- **Evaluation & Failure Analysis**: 20 points → **CRITICAL** - This is key signal
- **Code Quality**: 10 points → Clean, readable, well-structured
- **DESIGN.md & Documentation**: 15 points → Complete and clear

### Bonus (Up to 30 points)
- **Part B - Demand Forecasting**: +15-20 points → Highest ROI
- **Guardrails**: +5 points → Easy to implement
- **Chat UI**: +5 points → Moderate effort
- **Multimodal**: +5 points → Higher effort

---

## ⚡ Quick Start Guide

### To Begin Implementation:

1. **Review the chain**:
   - Read assignment.md (already done)
   - Review ASSIGNMENT_ANALYSIS.md for understanding
   - Review REQUIREMENTS.md for specific requirements
   - Review planner.txt for task details

2. **Start with Phase 1**:
   ```bash
   # Create GitHub repository
   # Clone locally
   cd VIKMO-AI-ML-Intern-Assignment
   
   # Create structure
   mkdir -p assistant eval forecasting
   touch README.md DESIGN.md requirements.txt
   touch assistant/__init__.py assistant/retrieval.py assistant/tools.py assistant/agent.py assistant/cli.py
   touch eval/__init__.py eval/eval_set.jsonl eval/run_eval.py eval/results.json
   ```

3. **Install dependencies** (from planner.txt P1-T003):
   ```bash
   # requirements.txt content:
   sentence-transformers>=2.2.0
   faiss-cpu>=1.7.0
   pydantic>=2.5.0
   google-generativeai>=0.3.0  # or openai, or both
   python-dotenv>=1.0.0
   pandas>=2.0.0
   numpy>=1.24.0
   scikit-learn>=1.3.0
   ```

4. **Implement in order**:
   - P1-T004: Embedding system
   - P1-T005: Vector store
   - P1-T006: Load and index catalogue
   - P1-T007: Test retrieval
   - Then continue to Phase 2...

---

## 🔍 Key Requirements Checklist

### Before Submission, Verify:

#### Repository Structure (REPO-001 to REPO-006)
- [ ] README.md at root
- [ ] DESIGN.md at root
- [ ] requirements.txt at root
- [ ] assistant/ directory with code
- [ ] eval/ directory with eval set and results
- [ ] forecasting/ directory (if doing Part B)

#### Part A Core (100 points)
- [ ] Real retrieval implemented (RAG-001, RAG-002)
- [ ] All 3 tools implemented (TOOL-001 to TOOL-003)
- [ ] Structured output for orders (TOOL-005, TOOL-007)
- [ ] Multi-turn conversation (CONV-001, CONV-002)
- [ ] Clarification questions (CONV-003, CONV-004)
- [ ] Grounding checks (GROUND-001, GROUND-002)
- [ ] Eval set with 25 test cases (EVAL-001 to EVAL-004)
- [ ] Eval runner and results (EVAL-005 to EVAL-007)
- [ ] Failure analysis (EVAL-008 to EVAL-011)

#### Documentation
- [ ] DESIGN.md complete (DESIGN-001 to DESIGN-006)
- [ ] README.md complete (README-001 to README-005)
- [ ] Code comments added

#### Submission
- [ ] Code pushed to public GitHub
- [ ] Assistant runs
- [ ] No hardcoded API keys (2.4, SUB-006)

---

## 💡 Pro Tips

### 1. Start Simple
- Use Google Gemini free tier (as recommended)
- Use FAISS for vector store (no external dependencies)
- Use `all-MiniLM-L6-v2` for embeddings (good quality, free)

### 2. Focus on Evaluation
- This carries **20 points** and is the "clearest signal of an AI engineer"
- Create meaningful test cases that actually test your system
- Be honest in failure analysis - VIKMO values this

### 3. Grounding is Critical
- Never let the LLM invent product information
- Always verify against the catalogue
- Implement confidence thresholds for retrieval

### 4. Documentation Matters
- DESIGN.md is worth **15 points**
- Explain your choices and reasoning
- VIKMO will ask you to defend your decisions in the technical round

### 5. Time Management
- Part A is **100 points** - prioritize this
- Part B is **bonus** - only if time permits
- Quality > Quantity - better to have solid Part A than rushed everything

---

## 📚 Data Files Available

| File | Size | Purpose | Status |
|------|------|---------|--------|
| assignment.md | - | Specification | ✅ Read |
| catalogue.csv | 600 SKUs | Part A retrieval corpus | ✅ Available |
| catalogue.json | 600 SKUs | Part A retrieval corpus | ✅ Available |
| sales_history.csv | 2,340 rows | Part B forecasting | ✅ Available |
| DATA_README.md | - | Data documentation | ✅ Read |

---

## 🚀 Next Steps

You are now ready to begin implementation. The recommended approach:

1. **Create the GitHub repository** (P1-T001)
2. **Set up the project structure** (P1-T002)
3. **Create requirements.txt** (P1-T003)
4. **Implement retrieval system** (P1-T004 to P1-T007)
5. **Continue following planner.txt**

**Estimated time to first milestone (Foundation Complete)**: 4-5 hours

---

## 📞 Need Help?

If anything is unclear about the requirements, refer back to:
1. assignment.md (the source of truth)
2. REQUIREMENTS.md (formal requirements with line references)
3. ASSIGNMENT_ANALYSIS.md (detailed analysis)
4. planner.txt (task breakdown)

All documents are in this directory and ready for reference during implementation.

---

**Planning Status**: ✅ COMPLETE
**Ready for Implementation**: ✅ YES
**Next Task**: P1-T001 (Set up GitHub repository)

*Last Updated: 2026-06-21*
