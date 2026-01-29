import json
import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.tools import DuckDuckGoSearchResults
import os
from agent.state import InterviewState, TurnLog
from agent.models import MentorOutput, FinalFeedback, RoadmapItem
from agent.prompts import INTERVIEWER_SYSTEM_PROMPT, MENTOR_SYSTEM_PROMPT, FINAL_REPORT_SYSTEM_PROMPT, DIRECTIVE_CONTEXT_PROMPT, DIRECTIVE_CONTEXT_PROMPT

# Инициализация моделей
try:
    interviewer_model = ChatOpenAI(
        model="openai/gpt-4o-mini", 
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
    )

    mentor_model = ChatOpenAI(
        model="openai/gpt-4o-mini", 
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
    )
except Exception as e:
    print("Ошибка инициализации моделей. Проверьте переменные окружения OPENROUTER_API_KEY и OPENROUTER_BASE_URL.")
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
    
    turn_id = state.get('current_turn_id', 0) + 1
        
    if len(state['turns']) > 0:
        pass

    
    # messages = state['messages']
    # if len(messages) >= 3:
    #     prev_q = messages[-3].content
    # elif len(messages) == 2:
    #     prev_q = "Н/Д (Пользователь начал)"
    
    new_log: TurnLog = {
        "turn_id": turn_id,
        "internal_thoughts": [
            f"[Interviewer]: {state.get('interviewer_thoughts', '')}",
            f"[Mentor]: {state.get('mentor_thoughts', '')} (Directive: {state.get('mentor_directive', '')})"
        ]
    }
    
    return {
        "turns": [new_log], 
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
    
    input_text += "\n\nВАЖНО: Сгенерируй детальный 'personal_roadmap' (список RoadmapItem) для устранения всех выявленных пробелов. Если есть 'knowledge_gaps', roadmap НЕ должен быть пустым."

    response: FinalFeedback = feedback_runnable.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=input_text)
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


