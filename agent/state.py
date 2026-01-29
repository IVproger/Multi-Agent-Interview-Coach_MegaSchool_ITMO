import operator
from typing import Annotated, List, Optional, TypedDict, Dict, Any, Union
from langchain_core.messages import BaseMessage

class SessionMeta(TypedDict):
    position: str
    grade_target: str
    experience: str

class TurnLog(TypedDict):
    turn_id: int
    timestamp: str
    interviewer_question: str
    candidate_answer: str
    internal_thoughts: List[str] # [Interviewer]: ..., [Mentor]: ...
    mentor_directive: str
    feedback: Optional[Dict[str, Any]] # For immediate feedback if any

class InterviewState(TypedDict):
    # Chat history
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Metadata
    participant_name: str
    session_meta: SessionMeta
    
    # Interview progression
    turns: Annotated[List[TurnLog], operator.add]
    current_turn_id: int
    
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
