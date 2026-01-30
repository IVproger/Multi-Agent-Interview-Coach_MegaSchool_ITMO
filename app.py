import streamlit as st
from dotenv import load_dotenv
load_dotenv(".env")

import uuid
import json
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import build_graph

# Page configuration
st.set_page_config(page_title="Multi-Agent Interview Coach", page_icon="👨‍💻")

def format_feedback_to_markdown(feedback_dict):
    """Formats feedback dictionary to Markdown for Streamlit."""
    if not isinstance(feedback_dict, dict):
        return str(feedback_dict)
    
    md = "## 📊 Результаты Интервью\n\n"
    
    md += "### 1. Вердикт (Decision)\n"
    md += f"- **Грейд**: {feedback_dict.get('grade', 'Не определен')}\n"
    md += f"- **Рекомендация**: {feedback_dict.get('hiring_recommendation', 'Не указана')}\n"
    md += f"- **Уверенность**: {feedback_dict.get('confidence_score', 0)}%\n\n"
    
    md += "### 2. Анализ Hard Skills\n"
    md += "**Подтвержденные навыки:**\n"
    skills = feedback_dict.get('confirmed_skills', [])
    if skills:
        for s in skills:
            md += f"- {s}\n"
    else:
        md += "- (Нет подтвержденных навыков)\n"
        
    md += "\n**Пробелы в знаниях:**\n"
    gaps = feedback_dict.get('knowledge_gaps', [])
    if gaps:
        for g in gaps:
            md += f"- {g}\n"
    else:
        md += "- (Явных пробелов не выявлено)\n"
        
    md += "\n### 3. Soft Skills & Communication\n"
    md += f"- **Ясность**: {feedback_dict.get('soft_skills_clarity', 'нет данных')}\n"
    md += f"- **Честность**: {feedback_dict.get('soft_skills_honesty', 'нет данных')}\n"
    md += f"- **Вовлеченность**: {feedback_dict.get('soft_skills_engagement', 'нет данных')}\n\n"
    
    md += "### 4. Персональный Roadmap\n"
    roadmap = feedback_dict.get('personal_roadmap', [])
    if roadmap:
        for i, task in enumerate(roadmap, 1):
            md += f"**{i}. {task.get('topic', 'Тема')}**\n"
            md += f"- Цель: {task.get('goal', '')}\n"
            md += f"- План: {task.get('plan', '')}\n"
            if task.get('resource_link'):
                md += f"- [Ресурс]({task.get('resource_link')})\n"
            md += "\n"
    else:
        md += "План не сформирован\n"
        
    return md

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "interview_active" not in st.session_state:
    st.session_state.interview_active = False
if "graph_state" not in st.session_state:
    st.session_state.graph_state = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "final_report" not in st.session_state:
    st.session_state.final_report = None

st.title("👨‍💻 Multi-Agent Interview Coach")

# Sidebar for Configuration
with st.sidebar:
    st.header("Настройки кандидата")
    if not st.session_state.interview_active and not st.session_state.final_report:
        name = st.text_input("Имя", value="Кандидат")
        position = st.text_input("Позиция", value="Python Developer")
        grade = st.selectbox("Целевой грейд", ["Junior", "Middle", "Senior"])
        experience = st.text_area("Опыт", value="Нет опыта")
        
        if st.button("Начать интервью"):
            # Initialize Graph State
            initial_state_config = {
                "participant_name": name,
                "session_meta": {
                    "position": position,
                    "grade_target": grade,
                    "experience": experience
                },
                "messages": [],
                "turns": [],
                "current_turn_id": 0,
                "status": "active",
                "summary": "Начало интервью.",
                "mentor_directive": "Начни интервью с представления себя и задай первый релевантный вопрос.",
                "mentor_thoughts": "Начальное состояние.",
                "mentor_confidence_score": 100.0,
                "last_candidate_answer": "",
                "last_interviewer_question": ""
            }
            
            # Add initial system message or greeting trigger
            # We trigger the first run to get the greeting
            app = build_graph()
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Initial run with a dummy start message or empty required structure
            # To kickstart the agent, we can simulate a "Ready" signal or just invoke with initial state
            # The original main.py asked for a greeting. Let's send a standard signal.
            initial_state = {**initial_state_config, "messages": [HumanMessage(content="Здравствуйте, я готов к интервью.")]}
            
            with st.spinner("Генерация первого вопроса..."):
                current_state = app.invoke(initial_state, config=config)
            
            st.session_state.graph_state = current_state
            
            # Extract first AI message
            if current_state["messages"]:
                last_msg = current_state["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    st.session_state.messages.append({"role": "assistant", "content": last_msg.content})
            
            st.session_state.interview_active = True
            st.rerun()
            
    elif st.session_state.interview_active:
        st.info("Интервью в процессе...")
        if st.button("Закончить интервью (Stop)"):
            # Flag to trigger stop logic in main flow
            st.session_state.stop_trigger = True
            st.rerun()

# Logic to handle stop trigger from sidebar or chat command
prompt_text = None
if st.session_state.get("stop_trigger"):
    prompt_text = "Stop interview"
    st.session_state.stop_trigger = False # Reset flag

# Main Chat Interface
if st.session_state.final_report:
    st.markdown(st.session_state.final_report)
    
    # Download JSON
    json_str = json.dumps(st.session_state.graph_state.get('final_feedback_raw', {}), indent=2, ensure_ascii=False)
    st.download_button(
        label="Скачать полный отчет (JSON)",
        data=json_str,
        file_name="interview_report.json",
        mime="application/json"
    )
    
    if st.button("Начать заново"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

elif st.session_state.interview_active:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle Chat Input OR Stop Trigger
    # We use := for chat_input, but if prompt_text is set via button, we use that.
    
    chat_input_val = st.chat_input("Ваш ответ...")
    
    # Priority: Button Stop -> Chat Input
    prompt = prompt_text if prompt_text else chat_input_val
    
    if prompt:
        # 1. Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Invoke Graph
        app = build_graph()
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # Prepare state update
        current_state = st.session_state.graph_state
        current_state["messages"].append(HumanMessage(content=prompt))
        
        with st.spinner("Интервьюер думает..."):
            new_state = app.invoke(current_state, config=config)
        
        st.session_state.graph_state = new_state
        
        # 3. Handle Response
        if new_state.get("status") in ["stop_requested", "finished"]:
             # Check if we have final report
            if new_state.get("final_feedback"):
                try:
                    feedback_dict = json.loads(new_state["final_feedback"])
                    # Save raw report for download
                    new_state['final_feedback_raw'] = feedback_dict
                    
                    report_md = format_feedback_to_markdown(feedback_dict)
                    st.session_state.final_report = report_md
                    
                    # --- AUTO SAVE LOG (Like main.py) ---
                    log_data = {
                        "participant_name": new_state.get("participant_name", "Unknown"),
                        "turns": new_state.get("turns", []),
                        "final_feedback": feedback_dict
                    }
                    try:
                        with open("interview_log.json", "w", encoding="utf-8") as f:
                            json.dump(log_data, f, indent=2, ensure_ascii=False)
                        st.success("Лог сохранен в 'interview_log.json'")
                    except Exception as e:
                        st.error(f"Ошибка сохранения лога: {e}")
                    # ------------------------------------
                    
                except Exception as e:
                    st.error(f"Ошибка чтения отчета: {e}")
                    st.session_state.final_report = "Ошибка генерации отчета."
            
            # Check for last goodbye message
            last_msg = new_state["messages"][-1]
            if isinstance(last_msg, AIMessage) and not st.session_state.final_report:
                 # If just stopping but not yet reporting (though graph should handle it)
                 st.session_state.messages.append({"role": "assistant", "content": last_msg.content})
                 with st.chat_message("assistant"):
                    st.markdown(last_msg.content)
            
            st.session_state.interview_active = False
            st.rerun()
            
        else:
            # Continue conversation
            last_msg = new_state["messages"][-1]
            if isinstance(last_msg, AIMessage):
                st.session_state.messages.append({"role": "assistant", "content": last_msg.content})
                with st.chat_message("assistant"):
                    st.markdown(last_msg.content)

else:
    st.info("👈 Пожалуйста, заполните данные слева и нажмите 'Начать интервью'")
