# ERC3

## Architecture

### Generic

1. **Perception system.** Gathers data from environment.
2. **Reasoning engine.** Processes data and makes decisions.
3. **Memory store.** Retains information for future reference.
4. **Action module.** Executes decisions in the environment.

### Specific

```pseudo
SYSTEM ERC3_Architecture:

  COMPONENT PlanningExpert:
    INPUT: goals, current_state, memory, perception
    OUTPUT: execution_graph OR reevaluation_request
    
    FUNCTION generate_plan():
      current_state, memory = MemoryExpert.retrieve_relevant_context()
      plan = create_plan(goals, current_state, memory, perception)
      
      CALL ConstraintExpert.validate(plan)
      
      IF constraints_satisfied:
        RETURN execution_graph
      ELSE:
        RETURN reevaluation_request
    
    SUBCOMPONENT ConstraintExpert:
      FUNCTION validate(plan):
        IF plan violates constraints:
          trigger_reevaluation()
          RETURN false
        RETURN true

  COMPONENT ExecutionExpert:
    INPUT: execution_graph
    OUTPUT: actions
    
    FUNCTION execute_plan():
      FOR EACH step IN execution_graph:
        tool_result = ToolExpert.select_and_use(step)
        code_result = CodingExpert.write_and_debug(step)
        actions.append(tool_result, code_result)
        feedback = FeedbackExpert.monitor_and_adjust()
        IF feedback is good:
          continue
        ELSE:
          PlanningExpert.generate_plan()
      RETURN actions
    
    SUBCOMPONENT ToolExpert:
      FUNCTION select_and_use(step):
        tool = select_appropriate_tool(step)
        RETURN tool.execute()
    
    SUBCOMPONENT CodingExpert:
      FUNCTION write_and_debug(step):
        code = generate_code(step)
        RETURN debug_and_execute(code)

  COMPONENT FeedbackExpert:
    INPUT: execution_results
    OUTPUT: feedback_to_planning
    
    FUNCTION monitor_and_adjust():
      errors = ErrorHandlingExpert.identify_errors()
      context = MemoryExpert.retrieve_relevant_context()
      
      feedback = generate_feedback(errors, context)
      SEND feedback TO PlanningExpert
    
    SUBCOMPONENT MemoryExpert:
      FUNCTION store(information):
        memory.save(information)
      
      FUNCTION retrieve(query):
        RETURN memory.search(query)
    
    SUBCOMPONENT ErrorHandlingExpert:
      FUNCTION identify_errors():
        RETURN detected_errors
      
      FUNCTION resolve(error):
        RETURN resolution_strategy

  FLOW:
    PlanningExpert → ExecutionExpert → FeedbackExpert → PlanningExpert
```

1. **Planning expert.** Generates plans based on goals + current state (memory + perception).
   1. **Constraint expert.** Ensures plans adhere to specified constraints. Reevaluates plans if constraints are violated.
   2. Planning + Constraint = Execution graph if no reevaluation needed. ->
2. **Execution expert.** Executes the plan step-by-step.
   1. **Tool expert.** Selects and utilizes appropriate tools for each step.
   2. **Coding expert.** Writes and debugs code as needed during execution.
   3. Tool/Coding = Actions taken during execution. ->
3. **Feedback expert.** Monitors execution and provides feedback to the **Planning expert** for adjustments
   1. **Memory expert.** Manages storage and retrieval of information.
   2. **Error-handling expert.** Identifies and resolves errors during execution.


