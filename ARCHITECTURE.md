# HRM-First Architecture for Elect_Engin1

## Core Principle
**HRM is the brain. LLMs are the language tools.**

```
┌─────────────────────────────────────────────────────────────┐
│                         USER INPUT                          │
│                    (Text, Voice, Files)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    HRM ORCHESTRATOR                         │
│                  (Core Decision Maker)                      │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  HRM Hierarchical Reasoning:                         │  │
│  │  • Analyzes engineering problem                      │  │
│  │  • Creates multi-step execution plan                 │  │
│  │  • Performs technical calculations                   │  │
│  │  • Validates designs against NEC codes               │  │
│  │  • Optimizes circuit routing & load distribution     │  │
│  │  • Makes all engineering decisions                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│         ┌─────────────────┬──────────────────┐             │
│         ▼                 ▼                  ▼             │
│   ┌──────────┐     ┌───────────┐     ┌──────────────┐    │
│   │   LLM    │     │    HRM    │     │  Engineering │    │
│   │  Assist  │     │  Reasoning │     │    Rules    │    │
│   │ (GPT-4)  │     │  (27M)    │     │   Database   │    │
│   └──────────┘     └───────────┘     └──────────────┘    │
│   Used for:        Used for:         Used for:           │
│   • Language       • Planning        • NEC codes         │
│   • Reports        • Optimization    • Standards         │
│   • Extraction     • Validation      • Templates         │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   ENGINEERING SOLUTION                      │
│        (CAD Files, Panel Schedules, Reports)                │
└─────────────────────────────────────────────────────────────┘
```

## Decision Flow

### Traditional LLM-First (OLD):
```
User → LLM decides → LLM calls tools → LLM synthesizes → Result
           ↑
    (LLM is orchestrator)
```

### HRM-First (NEW):
```
User → HRM plans → HRM executes → HRM synthesizes → Result
                       ├─→ LLM (language only)
                       ├─→ HRM reasoning
                       └─→ Rules/Standards
           ↑
    (HRM is orchestrator)
```

## When HRM Uses LLM

HRM delegates to LLM **ONLY** for language-specific tasks:

### 1. **User Intent Parsing**
```python
# User says: "Create a 42-circuit panel for the warehouse"
# HRM uses LLM to extract:
{
    "task": "panel_schedule",
    "circuits": 42,
    "location": "warehouse"
}
# Then HRM does ALL the engineering
```

### 2. **Report Generation**
```python
# HRM completes panel optimization
# HRM asks LLM: "Format this technical data as a report"
# LLM generates human-readable text
# HRM validates and delivers
```

### 3. **Data Extraction**
```python
# HRM needs info from user's spec document
# HRM asks LLM: "Extract panel voltage from this PDF"
# LLM returns structured data
# HRM validates and continues reasoning
```

### 4. **Explanation**
```python
# HRM makes engineering decision
# User asks "Why did you choose 3/0 AWG?"
# HRM asks LLM: "Explain this technical rationale in plain English"
# LLM translates HRM's reasoning
```

## What HRM Does Independently

HRM performs **ALL engineering tasks** without LLM:

### ✅ Circuit Routing
- Pathfinding through floor plans
- Obstacle avoidance
- Conduit run optimization
- Code-compliant spacing

### ✅ Panel Optimization
- Phase balance calculations
- Load distribution
- Breaker sizing
- Demand factor application

### ✅ NEC Validation
- Multi-step code compliance
- Conductor sizing (Table 310.15)
- Voltage drop calculations
- Grounding verification
- Protection coordination

### ✅ Load Calculations
- KVA/Ampere conversions
- Coincident loads
- Diversity factors
- Future capacity planning

### ✅ Design Review
- Technical error detection
- Safety analysis
- Best practice validation
- Optimization suggestions

## Example: Panel Schedule Task

### Step-by-Step with HRM as Orchestrator:

```python
# User input
user_text = "Create a 480V panel schedule for warehouse, 42 circuits"

# Step 1: HRM decides this is a panel task (HRM reasoning)
task_type = TaskType.PANEL_OPTIMIZATION

# Step 2: HRM asks LLM to extract parameters (LLM assist)
llm_result = hrm.request_llm_assist(
    LLMRequest.EXTRACT_PARAMETERS,
    context=user_text
)
# LLM returns: {"voltage": "480Y/277V", "circuits": 42, "location": "warehouse"}

# Step 3: HRM validates and normalizes (HRM reasoning)
circuits = 42  # Must be even, 18-84 range ✓
voltage = "480Y/277V"  # Standard voltage ✓

# Step 4: HRM calculates phase balance (HRM reasoning)
optimal_assignment = hrm.optimize_phase_assignment(circuits)

# Step 5: HRM validates NEC compliance (HRM reasoning)
nec_valid = hrm.validate_nec_codes(optimal_assignment)

# Step 6: HRM generates panel IR (HRM reasoning)
panel_ir = hrm.create_panel_schedule(optimal_assignment)

# Step 7: HRM asks LLM to format report (LLM assist)
report = hrm.request_llm_assist(
    LLMRequest.GENERATE_REPORT,
    context=panel_ir
)

# Step 8: HRM delivers solution
return panel_ir, report
```

### Key Points:
- **HRM decides** it's a panel task (not LLM)
- **HRM plans** the execution steps (not LLM)
- **HRM performs** all calculations (not LLM)
- **HRM validates** engineering correctness (not LLM)
- **LLM only helps** with language I/O

## Benefits of HRM-First

### 1. **True Engineering Reasoning**
- Multi-step technical analysis
- Hierarchical problem decomposition
- Validated engineering decisions

### 2. **Speed**
- HRM inference: 100x faster than LLM CoT
- No waiting for language generation for technical tasks
- Single forward pass reasoning

### 3. **Cost**
- HRM: 27M parameters (tiny)
- Only call expensive LLM for language tasks
- Can run HRM locally (no API costs)

### 4. **Reliability**
- Deterministic engineering calculations
- No hallucinations in technical reasoning
- Validated against engineering rules

### 5. **Offline Capability**
- HRM runs without internet (edge deployment)
- LLM only needed for optional reporting
- Core engineering always available

## Implementation Priority

### Phase 1: Core HRM Orchestrator ✓
- [x] Create HRMOrchestrator class
- [x] Define engineering task types
- [x] Implement LLM request system
- [ ] Integrate with existing Elect_Engin1 routes

### Phase 2: Panel Optimization
- [ ] Train HRM on panel schedule datasets
- [ ] Replace GPT-based planning with HRM
- [ ] Benchmark phase balance improvements

### Phase 3: Circuit Routing
- [ ] Train HRM on floor plan datasets
- [ ] Integrate pathfinding with CAD generation
- [ ] Compare routes with manual designs

### Phase 4: NEC Validation
- [ ] Train HRM on code compliance datasets
- [ ] Replace GPT preflight with HRM validation
- [ ] Add multi-step rule chaining

## Code Integration Points

### Update FastAPI Routes
```python
# OLD: app/main.py
response = plan_from_prompt(user_text)  # LLM decides

# NEW: app/main.py
from app.ai.hrm_orchestrator import execute_engineering_task, TaskType

response = execute_engineering_task(
    task_type=TaskType.PANEL_OPTIMIZATION,  # HRM decides
    parameters={"user_text": user_text}
)
```

### Update AI Module
```python
# OLD: app/ai/llm.py calls OpenAI directly

# NEW: app/ai/hrm_orchestrator.py orchestrates
# app/ai/llm.py becomes support tool for HRM
```

## Next Steps

1. **Install HRM dependencies**
   ```bash
   pip install -r requirements.txt  # From HRM repo
   ```

2. **Create training datasets**
   - Panel schedules (1000 examples)
   - Floor plans (1000 examples)
   - NEC scenarios (1000 examples)

3. **Train initial HRM models**
   - Start with panel optimization (highest value)
   - Then circuit routing (proven capability)
   - Then NEC validation (most complex)

4. **Integrate with Elect_Engin1**
   - Replace LLM orchestration with HRM
   - Keep LLM for language tasks
   - Test on real projects

5. **Measure improvements**
   - Phase balance quality
   - Routing efficiency
   - Processing speed
   - Cost reduction
