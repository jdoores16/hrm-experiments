# HRM + Elect_Engin1 Integration Plan

## Overview
Integrate Sapient's HRM (Hierarchical Reasoning Model) to enhance Elect_Engin1's AI capabilities with advanced multi-step reasoning for electrical engineering tasks.

## Current Elect_Engin1 AI Capabilities
- **Intent Recognition**: OpenAI for understanding text/voice commands
- **Technical Review**: PE-level electrical analysis (NEC compliance, load balancing, conductor sizing)
- **OCR Enhancement**: Panel photo → Excel schedule extraction
- **CAD Generation**: One-line diagrams, power plans, lighting plans

## Where HRM Adds Value

### 1. **Advanced Panel Schedule Optimization** ⚡
**Current**: Basic load distribution and phase balance checking
**With HRM**: Multi-step reasoning for optimal breaker placement and circuit organization

HRM can solve complex optimization problems like:
- Finding optimal circuit-to-phase assignments for perfect balance
- Determining best breaker sizes considering upstream coordination
- Planning future expansion capacity allocation

**Integration Point**: `app/schemas/panel_ir.py` - enhance panel schedule generation

### 2. **Intelligent Circuit Routing** 🔌
**Current**: Manual specification of circuit paths
**With HRM**: Automated optimal routing for power/lighting plans

HRM excels at pathfinding (74.5% success on 30x30 mazes):
- Find shortest/cleanest conduit runs
- Avoid obstacles and maintain code-compliant spacing
- Minimize wire length while maintaining accessibility

**Integration Point**: `app/cad/power_plan.py`, `app/cad/lighting_plan.py`

### 3. **Deep NEC Code Compliance Reasoning** 📋
**Current**: GPT-based checklist review (single-pass)
**With HRM**: Multi-step hierarchical code analysis

HRM's hierarchical architecture can:
- Chain multiple NEC code rules together
- Reason about cascading requirements (e.g., voltage drop → conductor sizing → conduit sizing)
- Validate complex interdependent rules

**Integration Point**: `app/ai/gpt_preflight.py` - augment technical review

### 4. **Load Calculation Reasoning** 🔢
**Current**: Basic KVA calculations
**With HRM**: Complex load flow analysis and demand factor optimization

HRM can handle multi-variable optimization:
- Apply NEC demand factors correctly across multiple load types
- Calculate coincident loads for optimal panel sizing
- Reason about diversity and future load growth

**Integration Point**: New module `app/ai/hrm_load_calc.py`

### 5. **Design Error Detection** 🔍
**Current**: Rule-based validation
**With HRM**: Pattern recognition for subtle design issues

Train HRM to recognize:
- Common electrical design mistakes
- Unsafe configurations that pass basic rules
- Best practice violations

**Integration Point**: `app/ai/checklist.py` - enhance validation

## Implementation Phases

### Phase 1: Circuit Routing (Pathfinding)
- **Goal**: Use HRM for optimal circuit routing in power/lighting plans
- **Why First**: HRM's proven pathfinding capability (maze solving)
- **Effort**: Medium
- **Impact**: High - immediate visible improvement

### Phase 2: Panel Schedule Optimization
- **Goal**: Use HRM to optimize circuit-to-phase assignments
- **Why Second**: Builds on routing, demonstrates reasoning
- **Effort**: Medium-High
- **Impact**: High - measurably better phase balance

### Phase 3: NEC Code Reasoning
- **Goal**: Deep multi-step code compliance checking
- **Why Third**: Requires domain-specific training
- **Effort**: High
- **Impact**: Very High - professional engineering value

### Phase 4: Load Calculation Enhancement
- **Goal**: Advanced load flow analysis
- **Why Fourth**: Complex, needs extensive validation
- **Effort**: Very High
- **Impact**: Very High - core engineering capability

## Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Elect_Engin1 FastAPI                  │
│                                                         │
│  ┌──────────────┐          ┌──────────────┐           │
│  │   OpenAI     │          │     HRM      │           │
│  │   (GPT-4)    │          │   Reasoning  │           │
│  └──────┬───────┘          └──────┬───────┘           │
│         │                         │                    │
│         ├─────────────────────────┤                    │
│         │   AI Router/Orchestrator│                    │
│         │  (Choose best AI for    │                    │
│         │   each task)            │                    │
│         └─────────┬───────────────┘                    │
│                   │                                     │
│  ┌────────────────┴───────────────┐                   │
│  │  Intent: GPT-4 (language)      │                   │
│  │  Routing: HRM (pathfinding)    │                   │
│  │  Optimization: HRM (reasoning) │                   │
│  │  Review: GPT-4 + HRM (hybrid)  │                   │
│  └────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

## Data Format Conversion

### From Elect_Engin1 to HRM:
```python
# Power plan room layout
rooms = [
    {"name": "Office", "x": 0, "y": 0, "w": 20, "h": 15},
    {"name": "Corridor", "x": 20, "y": 0, "w": 5, "h": 15}
]

# Convert to HRM grid representation
grid = convert_floor_plan_to_grid(rooms)  # 2D maze-like tensor
start = device_location  # Source panel
end = outlet_location    # Destination receptacle

# HRM pathfinding
path = hrm_model.predict(grid, start, end)
```

### From Elect_Engin1 Panel IR to HRM:
```python
# Panel schedule data
panel_ir = PanelScheduleIR(
    header=PanelHeader(voltage="480Y/277V", phase=3),
    circuits=[
        Circuit(ckt=1, breaker_amps=20, load_amps=15, phA=True),
        Circuit(ckt=2, breaker_amps=20, load_amps=18, phB=True),
        # ...
    ]
)

# Convert to HRM optimization problem
problem = convert_panel_to_optimization_tensor(panel_ir)

# HRM optimization
optimal_assignment = hrm_model.predict(problem)
```

## Training Data Needed

### For Circuit Routing:
- **Dataset**: 1,000 examples of floor plans with optimal conduit routes
- **Source**: Generate synthetic + use historical projects
- **Format**: Grid-based (similar to maze dataset)

### For Panel Optimization:
- **Dataset**: 1,000 examples of panels with optimal phase assignments
- **Source**: PE-reviewed historical panel schedules
- **Format**: Structured electrical data (voltage, loads, phases)

### For NEC Reasoning:
- **Dataset**: 1,000 examples of designs with code violations
- **Source**: NEC examples, historical review findings
- **Format**: Panel IR + violation type + reasoning chain

## File Structure

```
elect_engin_app/
├── app/
│   ├── ai/
│   │   ├── llm.py              # Existing OpenAI
│   │   ├── hrm_reasoning.py    # NEW: HRM wrapper
│   │   ├── ai_router.py        # NEW: Route tasks to best AI
│   │   └── gpt_preflight.py    # Existing (enhance with HRM)
│   ├── cad/
│   │   ├── power_plan.py       # Enhance with HRM routing
│   │   └── lighting_plan.py    # Enhance with HRM routing
│   └── hrm_models/             # NEW: HRM-specific modules
│       ├── pathfinding.py      # Circuit routing
│       ├── optimization.py     # Panel optimization
│       └── code_reasoning.py   # NEC compliance
├── hrm_checkpoints/            # Pre-trained HRM models
│   ├── circuit_routing.pt
│   ├── panel_optimization.pt
│   └── nec_reasoning.pt
└── hrm_datasets/               # Training data
    ├── routing/
    ├── panels/
    └── nec/
```

## Benefits

### Immediate:
- **Better circuit routing**: Shorter conduit runs, cleaner layouts
- **Optimal panel schedules**: Perfect phase balance automatically

### Medium-term:
- **Faster design iterations**: HRM is 100x faster than LLM reasoning
- **Reduced costs**: Smaller model (27M vs billions) = lower API costs

### Long-term:
- **Professional-grade reasoning**: PE-level multi-step analysis
- **Edge deployment**: Run HRM locally (no API dependency)
- **Custom domain expertise**: Train on your specific design standards

## Next Steps

1. **Choose Phase 1 task**: Which to start with?
   - Circuit routing (easiest, proven capability)
   - Panel optimization (highest immediate value)

2. **Prepare training data**: 
   - Generate synthetic examples OR
   - Use existing project data

3. **Integrate HRM wrapper**:
   - Connect HRM to Elect_Engin1 FastAPI
   - Add AI router to choose GPT vs HRM per task

4. **Train and validate**:
   - Train HRM on electrical engineering tasks
   - Compare results with current GPT approach

Which phase would you like to start with?
