import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.tools import DuckDuckGoSearchResults
import os
from agent.state import InterviewState, TurnLog
from agent.models import MentorOutput, FinalFeedback, RoadmapItem
from agent.prompts import INTERVIEWER_SYSTEM_PROMPT, MENTOR_SYSTEM_PROMPT, FINAL_REPORT_SYSTEM_PROMPT, DIRECTIVE_CONTEXT_PROMPT, DIRECTIVE_CONTEXT_PROMPT

# Инициализация моделей
try:
    interviewer_model = ChatOpenAI(
        model="openai/gpt-4o-mini", 
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )

    mentor_model = ChatOpenAI(
        model="openai/gpt-4o-mini", 
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )
except Exception as e:
    print("Ошибка инициализации моделей. Проверьте переменные окружения API_KEY и BASE_URL.")
    raise e

# Инициализация инструмента поиска
search_tool = DuckDuckGoSearchResults()

def mentor_node(state: InterviewState):
    """
    Mentor agent analysis.
    """
    messages = state['messages']
    candidate_answer = messages[-1].content
    
    meta = state['session_meta']
    participant = state['participant_name']
    
    system_prompt = MENTOR_SYSTEM_PROMPT.format(
        participant_name=participant,
        position=meta['position'],
        grade_target=meta['grade_target'],
        experience=meta['experience']
    )


    mentor_runnable = mentor_model.with_structured_output(MentorOutput)
   
    response: MentorOutput = mentor_runnable.invoke([
        SystemMessage(content=system_prompt),
        *messages
    ])
    
    final_directive = response.directive
    if response.correction_needed and response.correction_details:
        final_directive += f" [CORRECTION INFO FOR INTERVIEWER: {response.correction_details}]"
    
    return {
        "mentor_directive": final_directive,
        "mentor_thoughts": response.internal_thoughts,
        "mentor_confidence_score": response.confidence_score,
        "status": "stop_requested" if response.stop_interview_flag else state.get("status", "active"),
        "last_candidate_answer": candidate_answer
    }


   
def interviewer_node(state: InterviewState):
    """
    Interviewer agent generation.
    """
    directive = state.get('mentor_directive')
    meta = state['session_meta']
    
    system_prompt = INTERVIEWER_SYSTEM_PROMPT.format(
        position=meta['position'],
        grade_target=meta['grade_target'],
        experience=meta['experience']
    )
    
    messages = [SystemMessage(content=system_prompt)] + state['messages']
    
    if directive:
         from agent.prompts import DIRECTIVE_CONTEXT_PROMPT
         directive_context = DIRECTIVE_CONTEXT_PROMPT.format(directive=directive)
         messages.append(SystemMessage(content=directive_context))
    
    from pydantic import BaseModel, Field
    class InterviewerOutput(BaseModel):
        thought_process: str = Field(description="Internal ReAct process: Understand answer -> Check Directive -> Formulate Plan.")
        response_text: str = Field(description="The actual response/question to the candidate.")
        call_mentor: bool = Field(description="True if you need Mentor's help/analysis (e.g. user answered tough question). False if you continue efficiently on your own.", default=True)

    interviewer_runnable = interviewer_model.with_structured_output(InterviewerOutput)
    
    response: InterviewerOutput = interviewer_runnable.invoke(messages)
    
    return {
        "messages": [AIMessage(content=response.response_text)],
        "last_interviewer_question": response.response_text,
        "interviewer_thoughts": response.thought_process,
        "call_mentor": response.call_mentor
    }



def logger_node(state: InterviewState):
    """
    Log the completed turn.
    """
    
    messages = state['messages']
    
    if len(messages) < 3:
        return {}
    
    question_msg = ""
    answer_msg = ""
    
    if isinstance(messages[-3], AIMessage):
        question_msg = messages[-3].content
    if isinstance(messages[-2], HumanMessage):
        answer_msg = messages[-2].content

    # Construct internal thoughts string
    mentor_thought = state.get('mentor_thoughts', '')
    interviewer_thought = state.get('interviewer_thoughts', '')
    
    internal_thoughts = f"[Observer]: {mentor_thought}\n[Interviewer]: {interviewer_thought}\n"
    
    turn_id = state.get('current_turn_id', 0) + 1
    
    new_log: TurnLog = {
        "turn_id": turn_id,
        "agent_visible_message": question_msg, 
        "user_message": answer_msg,            
        "internal_thoughts": internal_thoughts
    }
    
    return {
        "turns": [new_log], 
        "current_turn_id": turn_id
    }

def memory_update_node(state: InterviewState):
    """
    Updates the working memory (summary) based on the latest turn.
    """
    from agent.prompts import SUMMARY_PROMPT
    
    # Get latest turn info
    if not state.get('turns'):
        return {} # No turns yet
        
    last_turn = state['turns'][-1]
    
    current_summary = state.get('summary') or "Начало интервью."
    
    prompt = SUMMARY_PROMPT.format(
        current_summary=current_summary,
        user_message=last_turn.get('user_message', ''),
        agent_message=last_turn.get('agent_visible_message', ''),
        internal_thoughts=last_turn.get('internal_thoughts', '')
    )
    
    response = mentor_model.invoke([HumanMessage(content=prompt)])
    new_summary = response.content
    
    return {
        "summary": new_summary
    }

def reporting_node(state: InterviewState):
    """
    Generate the final report.
    """
    meta = state['session_meta']
    participant = state['participant_name']
    
    # Use summary + turns for final report
    turns_text = json.dumps(state['turns'], indent=2, ensure_ascii=False)
    summary_text = state.get('summary', 'Нет саммари.')
    
    system_prompt = FINAL_REPORT_SYSTEM_PROMPT.format(
        participant_name=participant,
        position=meta['position'],
        grade_target=meta['grade_target'],
        summary=summary_text,
        transcript=turns_text
    )

    feedback_runnable = mentor_model.with_structured_output(FinalFeedback)
    response: FinalFeedback = feedback_runnable.invoke([
        SystemMessage(content=system_prompt)
    ])
    
    if response.personal_roadmap:
        for item in response.personal_roadmap:
            if not isinstance(item, RoadmapItem):
                continue
                
            topic = item.topic
            try:
                query = f"{topic} tutorial documentation {meta.get('position', 'developer')}"
                search_results_str = search_tool.invoke(query)
                
                link = ""
                try:
                    import re
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
    
    formatted_feedback_str = json.dumps(response.model_dump(), indent=2, ensure_ascii=False)
    
    return {
        "final_feedback": formatted_feedback_str, 
        "status": "finished"
    }


