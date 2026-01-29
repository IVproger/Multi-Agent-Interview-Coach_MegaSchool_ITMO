from langgraph.graph import StateGraph, END, START
from agent.state import InterviewState
from agent.nodes import mentor_node, interviewer_node, logger_node, reporting_node

def route_mentor(state: InterviewState):
    if state.get("status") == "stop_requested":
        return "reporting_node"
    return "interviewer_node"

def build_graph():
    builder = StateGraph(InterviewState)
    
    # Add nodes
    builder.add_node("mentor_node", mentor_node)
    builder.add_node("interviewer_node", interviewer_node)
    builder.add_node("logger_node", logger_node)
    builder.add_node("reporting_node", reporting_node)
    
    # Define edges
    # START -> mentor_node (System analyzes input)
    # Wait, START usually implies receiving user input in the state buffer.
    
    builder.add_edge(START, "mentor_node")
    
    # Conditional edge from Mentor
    builder.add_conditional_edges(
        "mentor_node",
        route_mentor,
        {
            "reporting_node": "reporting_node",
            "interviewer_node": "interviewer_node"
        }
    )
    
    # Interviewer -> Logger
    builder.add_edge("interviewer_node", "logger_node")
    
    # Logger -> END (Wait for next user input)
    builder.add_edge("logger_node", END)
    
    # Reporting -> END
    builder.add_edge("reporting_node", END)
  
    return builder.compile()
