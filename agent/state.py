import operator
from typing import Annotated, List, Optional, TypedDict, Dict, Any, Union
from langchain_core.messages import BaseMessage

class SessionMeta(TypedDict):
    position: str
    grade_target: str
    experience: str

class TurnLog(TypedDict):
    turn_id: int
    agent_visible_message: str
    user_message: str
    internal_thoughts: str # Combined thoughts

def add_and_window(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """Append new messages and keep only the last 12."""
    formatted_list = left + right
    return formatted_list[-12:]

class InterviewState(TypedDict):
    # Chat history (Short-Term Memory: Last 12 messages)
    messages: Annotated[List[BaseMessage], add_and_window]
    
    # Metadata
    participant_name: str
    session_meta: Optional[SessionMeta] # Make optional or ignored in final output logic
    
    # Interview progression
    turns: Annotated[List[TurnLog], operator.add]
    current_turn_id: int
    
    # Working Memory / Summary
    summary: str 
    
    # Internal state for flow control
    last_candidate_answer: str
    last_interviewer_question: str
    mentor_directive: Optional[str]
    mentor_thoughts: Optional[str]
    interviewer_thoughts: Optional[str]
    mentor_confidence_score: float
    
    # Status
    status: str # "active", "stop_requested", "finished"
    call_mentor: bool # New flag: Interviewer decides to call mentor
    
    # Final results
    final_feedback: Optional[Dict[str, Any]]
