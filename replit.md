# HRM Experiments - Sapient Integration with Elect_Engin1

## Overview
Integration of **Sapient's HRM (Hierarchical Reasoning Model)** with **Elect_Engin1** electrical design assistant.

**Architecture**: **HRM-First** - HRM is the core reasoning engine that uses LLMs like GPT-4 as language tools.

## Core Concept
- **HRM**: Core decision maker, performs all engineering reasoning
- **LLMs (GPT-4)**: Support tools for language processing and text generation
- **Integration**: HRM orchestrates; LLM assists with language tasks only

## Purpose
Create an AI-powered electrical engineering system where:
1. HRM performs multi-step technical reasoning (calculations, optimization, validation)
2. LLM handles language tasks (parsing user intent, generating reports)
3. Together they deliver professional-grade electrical designs

## Recent Changes
- **2025-11-08**: Initial HRM setup
  - Cloned Sapient HRM repository
  - Organized models, dataset builders, utilities
  - Created integration wrapper (`hrm_wrapper.py`)
  
- **2025-11-08**: Elect_Engin1 integration
  - Cloned Elect_Engin1 app from GitHub
  - Created HRM orchestrator (`app/ai/hrm_orchestrator.py`)
  - Defined HRM-first architecture (see ARCHITECTURE.md)
  - Set up task types for electrical engineering

## Project Architecture

### HRM-First Decision Flow
```
User Input → HRM Plans → HRM Executes → HRM Validates → Solution
                             ├─→ LLM (language)
                             ├─→ HRM (reasoning)
                             └─→ Rules (NEC)
```

### Engineering Tasks HRM Handles
1. **Circuit Routing**: Optimal pathfinding through floor plans
2. **Panel Optimization**: Phase balance and load distribution
3. **NEC Validation**: Multi-step code compliance checking
4. **Load Calculations**: Complex demand factor analysis
5. **Design Review**: Technical validation and optimization

### When HRM Uses LLM
- Parse user intent from natural language
- Extract parameters from text/documents
- Generate human-readable reports
- Explain technical decisions in plain English

### What HRM Does Alone
- All technical calculations
- All engineering decisions
- All code validation
- All optimization problems
- All pathfinding/routing

## Technology Stack
- **HRM**: 27M parameter reasoning model (PyTorch)
- **Elect_Engin1**: FastAPI + OpenAI + ezdxf (CAD)
- **Integration**: Python orchestrator layer

## File Structure
```
.
├── app/                        # Elect_Engin1 application
│   ├── ai/
│   │   ├── hrm_orchestrator.py # NEW: HRM core orchestrator
│   │   ├── llm.py             # Existing: LLM support tool
│   │   └── gpt_preflight.py   # Will be enhanced with HRM
│   ├── cad/                   # CAD generation (will use HRM routing)
│   ├── schemas/               # Panel IR, models
│   └── main.py                # FastAPI routes
├── models/                     # HRM neural network
├── dataset/                    # HRM training data builders
├── hrm_wrapper.py             # HRM inference wrapper
├── ARCHITECTURE.md            # HRM-first design
└── INTEGRATION_PLAN.md        # Implementation roadmap
```

## Key Features

### HRM Capabilities (27M parameters)
- Trains on ~1,000 examples (vs millions for LLMs)
- 100x faster inference than chain-of-thought
- Proven: 55% Sudoku-Extreme, 74.5% Maze-30x30
- No pre-training required

### Elect_Engin1 Capabilities
- CAD generation (one-line, power, lighting)
- Panel schedule creation with NEC validation
- OCR for panel photos → Excel
- AI technical review (PE-level)
- Voice/text interface

### Integration Benefits
- **Better reasoning**: Multi-step technical analysis
- **Faster**: HRM 100x faster than LLM reasoning
- **Cheaper**: Smaller model, fewer API calls
- **More reliable**: Deterministic engineering
- **Offline capable**: Run HRM locally

## Implementation Plan

### Phase 1: Core Orchestrator ✓
- [x] Create HRMOrchestrator class
- [x] Define task types and LLM request types
- [x] Set up HRM-LLM interaction pattern
- [ ] Integrate with Elect_Engin1 routes

### Phase 2: Panel Optimization (Next)
- [ ] Train HRM on panel schedule datasets
- [ ] Replace GPT planning with HRM reasoning
- [ ] Benchmark phase balance quality

### Phase 3: Circuit Routing
- [ ] Train HRM on floor plan datasets
- [ ] Integrate pathfinding with CAD
- [ ] Measure routing efficiency

### Phase 4: NEC Validation
- [ ] Train HRM on code scenarios
- [ ] Multi-step rule chaining
- [ ] Enhanced technical review

## Training Data Needed
1. **Panel Schedules**: 1,000 examples with optimal assignments
2. **Floor Plans**: 1,000 examples with optimal routing
3. **NEC Scenarios**: 1,000 examples with violations/corrections

## Resources
- **HRM Official**: https://github.com/sapientinc/HRM
- **HRM Paper**: https://arxiv.org/abs/2506.21734
- **Elect_Engin1**: https://github.com/jdoores16/Elect_Enginv1

## User Preferences
- **Architecture**: HRM-first (HRM as core decision maker, LLM as language tool)
- **Focus**: Engineering reasoning quality over language fluency
- **Integration**: Keep Elect_Engin1 functional, enhance with HRM capabilities

## Next Steps
1. Finish HRM orchestrator integration with FastAPI routes
2. Create training datasets for panel optimization
3. Train first HRM model (panel optimization)
4. Benchmark against current GPT-based approach
5. Expand to circuit routing and NEC validation
