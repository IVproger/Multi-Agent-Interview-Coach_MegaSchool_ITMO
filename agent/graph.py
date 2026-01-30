from langgraph.graph import StateGraph, END, START
from agent.state import InterviewState
from agent.nodes import mentor_node, interviewer_node, logger_node, reporting_node

def route_logger(state: InterviewState):
    if state.get("status") == "stop_requested":
        return "reporting_node"
    return END

def build_graph():
    builder = StateGraph(InterviewState)
    
    # Конструкция графа
    builder.add_node("mentor_node", mentor_node)
    builder.add_node("interviewer_node", interviewer_node)
    builder.add_node("logger_node", logger_node)
    builder.add_node("reporting_node", reporting_node)
        
    builder.add_edge(START, "mentor_node")
    
    # Mentor -> Interviewer (Always flow through Interviewer to acknowledge stop)
    builder.add_edge("mentor_node", "interviewer_node")
    
    # Interviewer -> Logger
    builder.add_edge("interviewer_node", "logger_node")
    
    # Logger -> Reporting (if stopping) OR End
    builder.add_conditional_edges(
        "logger_node",
        route_logger,
        {
            "reporting_node": "reporting_node",
            END: END
        }
    )
    
    # Reporting -> END
    builder.add_edge("reporting_node", END)
  
    return builder.compile()