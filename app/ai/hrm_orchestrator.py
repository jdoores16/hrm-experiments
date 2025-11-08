"""
HRM Orchestrator - Core AI Decision Maker for Elect_Engin1

HRM is the central reasoning engine that:
1. Plans and executes multi-step electrical engineering tasks
2. Delegates language/data tasks to LLMs when needed
3. Makes all engineering decisions and validates designs
4. Orchestrates the entire design workflow

LLMs (GPT-4, etc.) serve as support tools for:
- Natural language understanding (user intent)
- Text generation (reports, summaries)
- Data extraction from unstructured text
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Engineering tasks that HRM can execute"""
    CIRCUIT_ROUTING = "circuit_routing"
    PANEL_OPTIMIZATION = "panel_optimization"
    LOAD_CALCULATION = "load_calculation"
    NEC_VALIDATION = "nec_validation"
    DESIGN_REVIEW = "design_review"
    CAD_GENERATION = "cad_generation"


class LLMRequest(Enum):
    """When HRM needs LLM assistance"""
    PARSE_USER_INTENT = "parse_user_intent"
    EXTRACT_PARAMETERS = "extract_parameters"
    GENERATE_REPORT = "generate_report"
    EXPLAIN_DECISION = "explain_decision"
    FORMAT_OUTPUT = "format_output"


@dataclass
class EngineeringTask:
    """Task specification for HRM reasoning"""
    task_type: TaskType
    parameters: Dict[str, Any]
    constraints: Dict[str, Any]
    session_id: str
    
    # Intermediate results during execution
    reasoning_steps: List[Dict] = None
    llm_assists: List[Dict] = None
    
    def __post_init__(self):
        if self.reasoning_steps is None:
            self.reasoning_steps = []
        if self.llm_assists is None:
            self.llm_assists = []


class HRMOrchestrator:
    """
    Core AI orchestrator using HRM as the reasoning engine.
    
    Architecture:
        User Request → HRM Reasoning → Engineering Solution
                              ↓
                          LLM Assist (when needed)
                          - Language understanding
                          - Data extraction
                          - Report generation
    """
    
    def __init__(self):
        """Initialize HRM orchestrator with reasoning models"""
        self.hrm_models = {}
        self.llm_available = False
        self._load_hrm_models()
        self._check_llm_availability()
    
    def _load_hrm_models(self):
        """Load HRM models for different engineering tasks"""
        # TODO: Load actual HRM checkpoints
        logger.info("Loading HRM reasoning models...")
        
        # Placeholder for HRM model loading
        # In production, load from hrm_checkpoints/
        self.hrm_models = {
            TaskType.CIRCUIT_ROUTING: None,  # Will load circuit_routing.pt
            TaskType.PANEL_OPTIMIZATION: None,  # Will load panel_optimization.pt
            TaskType.NEC_VALIDATION: None,  # Will load nec_reasoning.pt
        }
        
        logger.info(f"HRM models loaded: {list(self.hrm_models.keys())}")
    
    def _check_llm_availability(self):
        """Check if LLM (OpenAI) is available for language tasks"""
        try:
            from app.core.settings import settings
            self.llm_available = bool(settings.OPENAI_API_KEY)
            logger.info(f"LLM availability: {self.llm_available}")
        except Exception as e:
            logger.warning(f"LLM not available: {e}")
            self.llm_available = False
    
    def execute_task(self, task: EngineeringTask) -> Dict[str, Any]:
        """
        Main execution method - HRM reasons through the task
        and calls LLM only when needed.
        """
        logger.info(f"HRM executing task: {task.task_type.value}")
        
        # Step 1: HRM analyzes the task and creates execution plan
        execution_plan = self._hrm_plan_task(task)
        task.reasoning_steps.append({
            "step": "planning",
            "plan": execution_plan
        })
        
        # Step 2: HRM executes each step in the plan
        for step in execution_plan["steps"]:
            # HRM decides if it needs LLM assistance for this step
            if step.get("needs_llm"):
                llm_result = self._request_llm_assist(
                    step["llm_request_type"],
                    step["llm_context"]
                )
                task.llm_assists.append(llm_result)
                step["llm_result"] = llm_result
            
            # HRM performs the core reasoning
            step_result = self._hrm_execute_step(step, task)
            task.reasoning_steps.append(step_result)
        
        # Step 3: HRM synthesizes final solution
        solution = self._hrm_synthesize_solution(task)
        
        return {
            "success": True,
            "task_type": task.task_type.value,
            "solution": solution,
            "reasoning_trace": task.reasoning_steps,
            "llm_assists_used": len(task.llm_assists),
            "session_id": task.session_id
        }
    
    def _hrm_plan_task(self, task: EngineeringTask) -> Dict[str, Any]:
        """
        HRM's core planning capability - breaks down engineering task
        into executable steps.
        
        This is where HRM's hierarchical reasoning shines:
        - High-level module: strategic planning
        - Low-level module: detailed execution
        """
        # TODO: Use actual HRM model for planning
        # For now, use rule-based planning as placeholder
        
        if task.task_type == TaskType.CIRCUIT_ROUTING:
            return self._plan_circuit_routing(task)
        elif task.task_type == TaskType.PANEL_OPTIMIZATION:
            return self._plan_panel_optimization(task)
        elif task.task_type == TaskType.NEC_VALIDATION:
            return self._plan_nec_validation(task)
        elif task.task_type == TaskType.DESIGN_REVIEW:
            return self._plan_design_review(task)
        else:
            return {"steps": [], "error": "Unknown task type"}
    
    def _plan_circuit_routing(self, task: EngineeringTask) -> Dict[str, Any]:
        """HRM plans circuit routing strategy"""
        return {
            "task": "circuit_routing",
            "strategy": "optimal_pathfinding",
            "steps": [
                {
                    "name": "analyze_floor_plan",
                    "needs_llm": False,
                    "reasoning": "HRM analyzes spatial layout"
                },
                {
                    "name": "identify_obstacles",
                    "needs_llm": False,
                    "reasoning": "HRM detects walls, equipment, restricted areas"
                },
                {
                    "name": "calculate_optimal_paths",
                    "needs_llm": False,
                    "reasoning": "HRM uses pathfinding (maze solving capability)"
                },
                {
                    "name": "validate_nec_compliance",
                    "needs_llm": False,
                    "reasoning": "HRM checks code requirements"
                },
                {
                    "name": "generate_routing_report",
                    "needs_llm": True,
                    "llm_request_type": LLMRequest.GENERATE_REPORT,
                    "llm_context": "routing_results",
                    "reasoning": "LLM formats human-readable report"
                }
            ]
        }
    
    def _plan_panel_optimization(self, task: EngineeringTask) -> Dict[str, Any]:
        """HRM plans panel schedule optimization"""
        return {
            "task": "panel_optimization",
            "strategy": "multi_objective_optimization",
            "steps": [
                {
                    "name": "analyze_circuit_loads",
                    "needs_llm": False,
                    "reasoning": "HRM calculates load distribution"
                },
                {
                    "name": "optimize_phase_balance",
                    "needs_llm": False,
                    "reasoning": "HRM finds optimal circuit-to-phase assignment"
                },
                {
                    "name": "validate_breaker_coordination",
                    "needs_llm": False,
                    "reasoning": "HRM checks upstream/downstream coordination"
                },
                {
                    "name": "calculate_demand_factors",
                    "needs_llm": False,
                    "reasoning": "HRM applies NEC demand calculations"
                },
                {
                    "name": "explain_optimization",
                    "needs_llm": True,
                    "llm_request_type": LLMRequest.EXPLAIN_DECISION,
                    "llm_context": "optimization_rationale",
                    "reasoning": "LLM generates explanation in natural language"
                }
            ]
        }
    
    def _plan_nec_validation(self, task: EngineeringTask) -> Dict[str, Any]:
        """HRM plans multi-step NEC code validation"""
        return {
            "task": "nec_validation",
            "strategy": "hierarchical_rule_checking",
            "steps": [
                {
                    "name": "identify_applicable_codes",
                    "needs_llm": True,
                    "llm_request_type": LLMRequest.EXTRACT_PARAMETERS,
                    "llm_context": "nec_code_database",
                    "reasoning": "LLM extracts relevant NEC articles from text database"
                },
                {
                    "name": "validate_conductor_sizing",
                    "needs_llm": False,
                    "reasoning": "HRM applies NEC Table 310.15(B)(16)"
                },
                {
                    "name": "check_voltage_drop",
                    "needs_llm": False,
                    "reasoning": "HRM calculates and validates voltage drop"
                },
                {
                    "name": "verify_grounding",
                    "needs_llm": False,
                    "reasoning": "HRM validates per Article 250"
                },
                {
                    "name": "validate_protection_coordination",
                    "needs_llm": False,
                    "reasoning": "HRM checks breaker/fuse coordination"
                },
                {
                    "name": "generate_compliance_report",
                    "needs_llm": True,
                    "llm_request_type": LLMRequest.GENERATE_REPORT,
                    "llm_context": "nec_validation_results",
                    "reasoning": "LLM formats professional compliance report"
                }
            ]
        }
    
    def _plan_design_review(self, task: EngineeringTask) -> Dict[str, Any]:
        """HRM plans comprehensive design review"""
        return {
            "task": "design_review",
            "strategy": "multi_aspect_analysis",
            "steps": [
                {
                    "name": "parse_design_documents",
                    "needs_llm": True,
                    "llm_request_type": LLMRequest.EXTRACT_PARAMETERS,
                    "llm_context": "design_specifications",
                    "reasoning": "LLM extracts structured data from documents"
                },
                {
                    "name": "analyze_electrical_system",
                    "needs_llm": False,
                    "reasoning": "HRM performs deep technical analysis"
                },
                {
                    "name": "identify_design_issues",
                    "needs_llm": False,
                    "reasoning": "HRM detects potential problems"
                },
                {
                    "name": "suggest_improvements",
                    "needs_llm": False,
                    "reasoning": "HRM generates optimization recommendations"
                },
                {
                    "name": "write_review_report",
                    "needs_llm": True,
                    "llm_request_type": LLMRequest.GENERATE_REPORT,
                    "llm_context": "review_findings",
                    "reasoning": "LLM writes professional PE review report"
                }
            ]
        }
    
    def _hrm_execute_step(self, step: Dict, task: EngineeringTask) -> Dict[str, Any]:
        """
        HRM executes a single step of the plan.
        This is where the actual reasoning happens.
        """
        # TODO: Use actual HRM model inference
        logger.info(f"HRM executing: {step['name']}")
        
        # Placeholder for actual HRM execution
        return {
            "step": step["name"],
            "status": "completed",
            "reasoning": step.get("reasoning", "HRM reasoning performed"),
            "result": {}  # Actual HRM output would go here
        }
    
    def _hrm_synthesize_solution(self, task: EngineeringTask) -> Dict[str, Any]:
        """
        HRM synthesizes final solution from all reasoning steps.
        High-level module integrates all low-level results.
        """
        # TODO: Use actual HRM synthesis
        return {
            "status": "synthesized",
            "steps_completed": len(task.reasoning_steps),
            "llm_assists": len(task.llm_assists)
        }
    
    def _request_llm_assist(self, request_type: LLMRequest, context: Any) -> Dict[str, Any]:
        """
        HRM requests LLM assistance for language-specific tasks.
        LLM is a tool that HRM uses, not the orchestrator.
        """
        if not self.llm_available:
            logger.warning(f"LLM not available for {request_type.value}, using fallback")
            return self._llm_fallback(request_type, context)
        
        logger.info(f"HRM requesting LLM assist: {request_type.value}")
        
        try:
            from app.ai.llm import _chat_with_retries
            
            # HRM formulates the LLM request
            messages = self._format_llm_request(request_type, context)
            
            # Call LLM
            response = _chat_with_retries(
                messages=messages,
                temperature=0.2,
                max_tokens=1024
            )
            
            return {
                "request_type": request_type.value,
                "success": True,
                "content": response.choices[0].message.content
            }
        except Exception as e:
            logger.error(f"LLM assist failed: {e}")
            return self._llm_fallback(request_type, context)
    
    def _format_llm_request(self, request_type: LLMRequest, context: Any) -> List[Dict[str, str]]:
        """HRM formats the request to LLM based on what it needs"""
        
        if request_type == LLMRequest.PARSE_USER_INTENT:
            return [
                {"role": "system", "content": "Extract structured electrical engineering parameters from user text."},
                {"role": "user", "content": str(context)}
            ]
        elif request_type == LLMRequest.GENERATE_REPORT:
            return [
                {"role": "system", "content": "Generate a professional electrical engineering report."},
                {"role": "user", "content": f"Format this technical data into a report:\n{context}"}
            ]
        elif request_type == LLMRequest.EXPLAIN_DECISION:
            return [
                {"role": "system", "content": "Explain this engineering decision in clear language."},
                {"role": "user", "content": str(context)}
            ]
        else:
            return [
                {"role": "system", "content": "Process this request."},
                {"role": "user", "content": str(context)}
            ]
    
    def _llm_fallback(self, request_type: LLMRequest, context: Any) -> Dict[str, Any]:
        """Fallback when LLM is not available - HRM handles it alone"""
        logger.info(f"Using HRM fallback for {request_type.value}")
        return {
            "request_type": request_type.value,
            "success": True,
            "content": f"[HRM Fallback] {context}",
            "note": "LLM not available, HRM handled independently"
        }


# Global orchestrator instance
_orchestrator = None

def get_orchestrator() -> HRMOrchestrator:
    """Get or create the HRM orchestrator singleton"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = HRMOrchestrator()
    return _orchestrator


def execute_engineering_task(
    task_type: TaskType,
    parameters: Dict[str, Any],
    constraints: Dict[str, Any] = None,
    session_id: str = "default"
) -> Dict[str, Any]:
    """
    Main entry point: HRM executes engineering tasks.
    
    This replaces LLM-first orchestration with HRM-first reasoning.
    """
    orchestrator = get_orchestrator()
    
    task = EngineeringTask(
        task_type=task_type,
        parameters=parameters,
        constraints=constraints or {},
        session_id=session_id
    )
    
    return orchestrator.execute_task(task)
