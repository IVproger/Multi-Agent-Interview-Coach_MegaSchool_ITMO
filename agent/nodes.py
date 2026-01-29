import json
import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.tools import DuckDuckGoSearchResults
import os
from agent.state import InterviewState, TurnLog
from agent.models import MentorOutput, FinalFeedback, RoadmapItem
from agent.prompts import INTERVIEWER_SYSTEM_PROMPT, MENTOR_SYSTEM_PROMPT, FINAL_REPORT_SYSTEM_PROMPT

# Инициализация моделей
interview_model = ChatOpenAI(
    model="openai/gpt-4o-mini", 
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
)

mentor_model = ChatOpenAI(
    model="openai/gpt-4o-mini", 
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
)

# Инициализация инструмента поиска
search_tool = DuckDuckGoSearchResults()

def mentor_node(state: InterviewState):
    """
    Mentor agent analysis.
    """
    messages = state['messages']
    candidate_answer = messages[-1].content
    
    # Form separate history for mentor to avoid polluting the main chat context # or just pass the full history.
    
    meta = state['session_meta']
    participant = state['participant_name']
    
    system_prompt = MENTOR_SYSTEM_PROMPT.format(
        participant_name=participant,
        position=meta['position'],
        grade_target=meta['grade_target'],
        experience=meta['experience']
    )
    
    # We want to bind the structured output
    mentor_runnable = mentor_model.with_structured_output(MentorOutput)
    
    # Analyze the conversation so far
    # The last message is from the candidate
    # The message before that is from the interviewer
    
    response: MentorOutput = mentor_runnable.invoke([
        SystemMessage(content=system_prompt),
        *messages
    ])
    
    return {
        "mentor_directive": response.directive,
        "mentor_thoughts": response.internal_thoughts,
        "mentor_confidence_score": response.confidence_score,
        "status": "stop_requested" if response.stop_interview_flag else state.get("status", "active"),
        "last_candidate_answer": candidate_answer
    }

def interviewer_node(state: InterviewState):
    """
    Interviewer agent generation.
    """
    directive = state['mentor_directive']
    meta = state['session_meta']
    
    system_prompt = INTERVIEWER_SYSTEM_PROMPT.format(
        position=meta['position'],
        grade_target=meta['grade_target'],
        experience=meta['experience']
    )
    
    directive_message = SystemMessage(content=f"ДИРЕКТИВА МЕНТОРА: {directive}")
    
    messages = [SystemMessage(content=system_prompt)] + state['messages'] + [directive_message]
    
    response = interview_model.invoke(messages)
       
    return {
        "messages": [response],
        "last_interviewer_question": response.content,
        "interviewer_thoughts": f"Следую директиве: {directive}" 
    }

def logger_node(state: InterviewState):
    """
    Log the completed turn.
    """
    
    turn_id = state.get('current_turn_id', 0) + 1
        
    prev_q = "Начальное приветствие"
    if len(state['turns']) > 0:
        # Get the NEXT question from the previous log?
        # No, that's complicated.
        # Let's rely on message history if possible.
        pass
        
    # Let's grab the question that the user responded to.
    # Messages: [... AI(Q_old), Human(A), AI(Q_new)]
    # We want Q_old.
    
    messages = state['messages']
    if len(messages) >= 3:
        prev_q = messages[-3].content
    elif len(messages) == 2:
        # Human -> AI (This implies user started? Or context was pre-seeded?)
        # If user started, prev_q is empty or "N/A"
        prev_q = "Н/Д (Пользователь начал)"
        
    
    new_log: TurnLog = {
        "turn_id": turn_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "interviewer_question": prev_q,
        "candidate_answer": state['last_candidate_answer'],
        "internal_thoughts": [
            f"[Interviewer]: {state.get('interviewer_thoughts', '')}",
            f"[Mentor]: {state.get('mentor_thoughts', '')}"
        ],
        "mentor_directive": state.get('mentor_directive', ''),
        "feedback": None
    }
    
    return {
        "turns": [new_log], # Append handled by Annotated add in state? No, TypedDict behavior is replace or update manually
        # Wait, state definition uses `turns: List[TurnLog]`. 
        # In LangGraph `Annotated[List, operator.add]` handles appending. 
        # I need to check my state definition in `state.py`.
        # I did NOT use Annotated for turns, I used just List[TurnLog]. 
        # So I need to append manually.
        
        # Correction: I should update agent/state.py to use Annotated for turns if I want automatic appending,
        # otherwise I need to read current, append, and return new list.
        # Let's update state to make it easier, or just handle it here.
        "current_turn_id": turn_id
    }

def reporting_node(state: InterviewState):
    """
    Generate the final report.
    """
    meta = state['session_meta']
    participant = state['participant_name']
    
    turns_text = json.dumps(state['turns'], indent=2, ensure_ascii=False)
    
    system_prompt = FINAL_REPORT_SYSTEM_PROMPT.format(
        participant_name=participant,
        position=meta['position'],
        grade_target=meta['grade_target']
    )
    
    input_text = f"Interview Transcript:\n{turns_text}"
    
    feedback_runnable = mentor_model.with_structured_output(FinalFeedback)
    
    response: FinalFeedback = feedback_runnable.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=input_text)
    ])
    
    
    if response.personal_roadmap:
        print(f"Generating links for roadmap...")
        for item in response.personal_roadmap:
            if not isinstance(item, RoadmapItem):
                # Fallback if model parsing failed weirdly, but shouldn't happen with strict types
                continue
                
            topic = item.topic
            try:
                # Search for documentation or articles
                query = f"{topic} tutorial documentation {meta.get('position', 'developer')}"
                search_results_str = search_tool.invoke(query)
                
                link = ""
                try:
                    import re
                    # Look for http/https url in the stringified result
                    # Exclude common noise or internal tracking if possible, but basic regex is fine for now
                    urls = re.findall(r'(https?://[^\s,\]"\']+)', search_results_str)
                    if urls:
                        link = urls[0]
                except:
                    pass
                
                if link:
                    item.resource_link = link
                else:
                    item.resource_link = "Не удалось найти прямую ссылку, рекомендуется поиск по теме."
                    
            except Exception as e:
                item.resource_link = f"Ошибка поиска: {str(e)}"
     
    return {
        "final_feedback": response.model_dump(),
        "status": "finished"
    }


