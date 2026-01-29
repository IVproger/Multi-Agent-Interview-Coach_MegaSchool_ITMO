from langgraph.graph import StateGraph, END, START
from agent.state import InterviewState
from agent.nodes import mentor_node, interviewer_node, logger_node, reporting_node

def route_mentor(state: InterviewState):
    if state.get("status") == "stop_requested":
        return "reporting_node"
    return "interviewer_node"

def route_interviewer(state: InterviewState):
    if state.get("call_mentor", True):
        return "mentor_node" 
    return "logger_node" 

def build_graph():
    builder = StateGraph(InterviewState)
    
    # Конструкция графа
    builder.add_node("mentor_node", mentor_node)
    builder.add_node("interviewer_node", interviewer_node)
    builder.add_node("logger_node", logger_node)
    builder.add_node("reporting_node", reporting_node)
        
    builder.add_edge(START, "mentor_node")
    
    # Условный переход Mentor -> (Reporting или Interviewer)
    builder.add_conditional_edges(
        "mentor_node",
        route_mentor,
        {
            "reporting_node": "reporting_node",
            "interviewer_node": "interviewer_node"
        }
    )
    
     # Условный переход Interviewer -> (Mentor или Logger)
    builder.add_conditional_edges(
        "interviewer_node",
        lambda state: "mentor_node" if state.get("call_mentor") and state.get("last_interviewer_question") == "REQUEST_MENTOR_HELP" else "logger_node",
        {
            "mentor_node": "mentor_node",
            "logger_node": "logger_node"
        }
    )
    # Logger -> Interviewer    
    builder.add_edge("interviewer_node", "logger_node")
    
    # Logger -> END (Wait for next user input)
    builder.add_edge("logger_node", END)
    
    # Reporting -> END
    builder.add_edge("reporting_node", END)
  
    return builder.compile()