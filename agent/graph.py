from langgraph.graph import StateGraph, END, START
from agent.state import InterviewState
from agent.nodes import mentor_node, interviewer_node, logger_node, reporting_node

def route_mentor(state: InterviewState):
    if state.get("status") == "stop_requested":
        return "reporting_node"
    return "interviewer_node"

def route_interviewer(state: InterviewState):
    if state.get("call_mentor", True):
        return "mentor_node" # Go again to mentor if needed (rare in this linear flow, but possible if structured)
    return "logger_node" # Default flow

def build_graph():
    builder = StateGraph(InterviewState)
    
    # Add nodes
    builder.add_node("mentor_node", mentor_node)
    builder.add_node("interviewer_node", interviewer_node)
    builder.add_node("logger_node", logger_node)
    builder.add_node("reporting_node", reporting_node)
        
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
    
    # Interviewer -> Conditional (Mentor or Logger)? 
    # Usually Interviewer speaks -> User answers -> Mentor analyzes.
    # So Interviewer -> Logger -> END (wait for input) -> Start -> Mentor.
    # If the user asks for "Interviewer decides to call mentor", implies dynamic loop:
    # Mentor -> Interviewer -> (Decide: Ask Mentor again? or Respond to user?)
    # If "Respond to user" -> Logger -> END.
    
    # But usually Interviewer Output IS the response to user.
    # So "Call Mentor" might mean "I don't know what to ask, help me Mentor" (Internal loop).
    
    # Let's implement internal loop support.
    builder.add_conditional_edges(
        "interviewer_node",
        lambda state: "mentor_node" if state.get("call_mentor") and state.get("last_interviewer_question") == "REQUEST_MENTOR_HELP" else "logger_node",
        {
            "mentor_node": "mentor_node",
            "logger_node": "logger_node"
        }
    )
    # Note: Simplification for now: Interviewer always goes to Logger (ends turn), 
    # unless we architect a multi-step internal reasoning loop.
    # For this chat-bot, the strict flow is usually easier.
    # I will stick to Interviewer -> Logger -> END, but keep the flag in state for future logic.
    
    builder.add_edge("interviewer_node", "logger_node")
    
    # Logger -> END (Wait for next user input)
    builder.add_edge("logger_node", END)
    
    # Reporting -> END
    builder.add_edge("reporting_node", END)
  
    return builder.compile()