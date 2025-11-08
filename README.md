# HRM + Elect_Engin1: AI Electrical Engineering System

**HRM-First Architecture**: Advanced reasoning for professional electrical design

## Overview

This project integrates **Sapient's HRM (Hierarchical Reasoning Model)** with **Elect_Engin1** to create an AI-powered electrical engineering system where:

- **HRM** = Core reasoning engine (all engineering decisions)
- **LLMs (GPT-4)** = Language processing tools (text understanding & generation)

## Core Principle

> **HRM is the brain. LLMs are the language tools.**

```
User Input → HRM Plans → HRM Executes → Solution
                  ├─→ LLM (language)
                  ├─→ HRM (reasoning)
                  └─→ Rules (NEC codes)
```

## Why HRM-First?

### Traditional AI (LLM-First):
- ❌ LLMs struggle with multi-step technical reasoning
- ❌ Slow chain-of-thought processing
- ❌ Expensive API calls for calculations
- ❌ Unreliable for complex engineering

### HRM-First Approach:
- ✅ HRM excels at hierarchical technical reasoning
- ✅ 100x faster inference than LLM chain-of-thought
- ✅ 27M parameters (tiny, can run locally)
- ✅ Deterministic engineering decisions
- ✅ LLM only for language tasks (cheaper)

## What HRM Does

### Engineering Tasks (No LLM Needed):
- **Circuit Routing**: Optimal pathfinding through floor plans
- **Panel Optimization**: Phase balance and load distribution
- **NEC Validation**: Multi-step code compliance checking
- **Load Calculations**: Complex demand factor analysis
- **Conductor Sizing**: Based on NEC tables
- **Voltage Drop**: Calculations and validation
- **Design Review**: Technical error detection

### When HRM Uses LLM:
- Parse user intent from natural language
- Extract parameters from documents
- Generate human-readable reports
- Explain technical decisions in plain English

## Project Structure

```
.
├── app/                        # Elect_Engin1 application
│   ├── ai/
│   │   ├── hrm_orchestrator.py # HRM core orchestrator ⭐
│   │   ├── llm.py             # LLM support tool
│   │   └── gpt_preflight.py   # Technical review
│   ├── cad/                   # CAD generation
│   ├── schemas/               # Data models
│   └── main.py                # FastAPI routes
├── models/                     # HRM neural network
├── dataset/                    # HRM training builders
├── elect_engin_app/           # Full Elect_Engin1 app
├── hrm_wrapper.py             # HRM inference wrapper
├── ARCHITECTURE.md            # HRM-first design ⭐
├── INTEGRATION_PLAN.md        # Implementation roadmap
└── main.py                    # Demo & documentation
```

## Quick Start

### 1. Run the Demo

```bash
python main.py
```

This shows:
- HRM-first architecture diagram
- Capability breakdown (HRM vs LLM)
- Example: HRM orchestrating panel schedule task
- Integration status

### 2. Start Elect_Engin1 Server

```bash
cd elect_engin_app
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open: http://localhost:8000/static

### 3. Explore the Code

**Core Orchestrator**:
```python
from app.ai.hrm_orchestrator import execute_engineering_task, TaskType

# HRM decides and executes
result = execute_engineering_task(
    task_type=TaskType.PANEL_OPTIMIZATION,
    parameters={"voltage": "480Y/277V", "circuits": 42},
    session_id="demo"
)
```

## Example: Panel Schedule Task

### User Input:
```
"Create a 480V panel schedule for warehouse, 42 circuits"
```

### HRM Process:
1. **HRM analyzes** → identifies PANEL_OPTIMIZATION task
2. **HRM asks LLM** → extract voltage="480Y/277V", circuits=42
3. **HRM validates** → normalize parameters, check ranges
4. **HRM calculates** → optimal phase balance (no LLM)
5. **HRM validates** → NEC code compliance (no LLM)
6. **HRM generates** → panel IR with circuits (no LLM)
7. **HRM asks LLM** → format professional report
8. **HRM delivers** → engineering-sound solution

### Key Point:
- LLM used **twice** (language tasks only)
- HRM performed **all engineering** independently

## Integration Status

### ✅ Completed:
- [x] HRM repository integrated
- [x] Elect_Engin1 app integrated
- [x] HRM orchestrator created
- [x] Architecture documented
- [x] Task types defined
- [x] LLM assist system implemented

### 🔄 In Progress:
- [ ] Connect orchestrator to FastAPI routes
- [ ] Create training datasets
- [ ] Train HRM models

### 📋 Next Steps:
1. Create panel schedule dataset (1000 examples)
2. Train HRM on panel optimization
3. Benchmark vs current GPT approach
4. Expand to circuit routing & NEC validation

## Performance

### HRM Capabilities:
- **Parameters**: 27 million (vs billions for LLMs)
- **Training**: ~1,000 examples (vs millions for LLMs)
- **Speed**: 100x faster than chain-of-thought
- **Proven**: 55% Sudoku-Extreme, 74.5% Maze pathfinding
- **Deployment**: Can run locally (no API dependency)

### Elect_Engin1 Features:
- CAD generation (one-line, power, lighting)
- Panel schedule creation
- OCR for panel photos → Excel
- AI technical review (PE-level)
- Voice/text interface
- NEC code validation

## Documentation

- **ARCHITECTURE.md** - Detailed HRM-first design
- **INTEGRATION_PLAN.md** - Implementation phases
- **app/ai/hrm_orchestrator.py** - Core code with comments
- **main.py** - Demo and examples

## Benefits

### Immediate:
- Better technical reasoning
- Faster task execution
- Lower AI costs

### Medium-term:
- Professional-grade engineering
- Offline capability
- Custom domain training

### Long-term:
- PE-level design automation
- Multi-step code compliance
- Industry-standard accuracy

## Technologies

- **HRM**: Sapient's Hierarchical Reasoning Model (PyTorch)
- **Elect_Engin1**: FastAPI + OpenAI + ezdxf + pytesseract
- **Integration**: Python orchestrator layer

## Resources

- **HRM Official**: https://github.com/sapientinc/HRM
- **HRM Paper**: https://arxiv.org/abs/2506.21734  
- **Elect_Engin1**: https://github.com/jdoores16/Elect_Enginv1
- **Sapient**: https://sapient.inc

## Next Steps

To continue development:

1. **Create training data**: Generate 1000 panel schedule examples
2. **Train HRM**: Start with panel optimization task
3. **Integrate**: Connect to FastAPI routes
4. **Benchmark**: Compare with GPT-based approach
5. **Expand**: Add circuit routing and NEC validation

Run `python main.py` to see the full demo and architecture!
