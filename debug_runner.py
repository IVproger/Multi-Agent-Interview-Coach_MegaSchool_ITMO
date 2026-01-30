from dotenv import load_dotenv
load_dotenv('.env')
import time
import os
import json
import glob
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import build_graph
import yaml

USER_INPUT_FILE = "user_input.txt"
SYSTEM_OUTPUT_FILE = "system_output.txt"
LOG_DIR = "./logs"  # <--- Specify your log folder here

def get_next_log_filename():
    """Finds the next available case number for logging in LOG_DIR."""
    os.makedirs(LOG_DIR, exist_ok=True)
    existing_logs = glob.glob(os.path.join(LOG_DIR, "interview_log_case*.json"))
    max_num = 0
    for log in existing_logs:
        try:
            part = os.path.basename(log).replace("interview_log_case", "").replace(".json", "")
            if part.isdigit():
                num = int(part)
                if num > max_num:
                    max_num = num
        except:
            pass
    return os.path.join(LOG_DIR, f"interview_log_case{max_num + 1}.json")

def read_and_clear_input():
    """Reads content from user_input.txt and clears it if not empty."""
    if not os.path.exists(USER_INPUT_FILE):
        return None
        
    with open(USER_INPUT_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()
        
    if content:
        # Clear the file
        with open(USER_INPUT_FILE, "w", encoding="utf-8") as f:
            f.write("")
        return content
    return None

def write_output(text):
    """Writes system response to system_output.txt."""
    print(f"[System Output]: {text}") # Also print to console for visibility
    with open(SYSTEM_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(text)

def format_feedback_to_text(feedback_dict):
    """Форматирует словарь фидбэка в читаемый текстовый отчет."""
    if not isinstance(feedback_dict, dict):
        return str(feedback_dict)
        
    lines = []
    lines.append("=== РЕЗУЛЬТАТЫ ИНТЕРВЬЮ ===")
    
    lines.append("\n1. Вердикт (Decision)")
    lines.append(f"Грейд: {feedback_dict.get('grade', 'Не определен')}")
    lines.append(f"Рекомендация: {feedback_dict.get('hiring_recommendation', 'Не указана')}")
    lines.append(f"Уверенность: {feedback_dict.get('confidence_score', 0)}%")
    
    lines.append("\n2. Анализ Hard Skills (Technical Review)")
    lines.append("• Подтвержденные навыки:")
    skills = feedback_dict.get('confirmed_skills', [])
    if skills:
        for s in skills:
            lines.append(f"  - {s}")
    else:
        lines.append("  (Нет подтвержденных навыков)")
        
    lines.append("\n• Пробелы в знаниях:")
    gaps = feedback_dict.get('knowledge_gaps', [])
    if gaps:
        for g in gaps:
            lines.append(f"  - {g}")
    else:
        lines.append("  (Явных пробелов не выявлено)")
        
    lines.append("\n3. Анализ Soft Skills & Communication")
    lines.append(f"Ясность: {feedback_dict.get('soft_skills_clarity', 'нет данных')}")
    lines.append(f"Честность: {feedback_dict.get('soft_skills_honesty', 'нет данных')}")
    lines.append(f"Вовлеченность: {feedback_dict.get('soft_skills_engagement', 'нет данных')}")
    
    lines.append("\n4. Персональный Roadmap (Next Steps)")
    roadmap = feedback_dict.get('personal_roadmap', [])
    if roadmap:
        for i, task in enumerate(roadmap, 1):
            lines.append(f"\n{i}. {task.get('topic', 'Тема')}")
            lines.append(f"   Цель: {task.get('goal', '')}")
            lines.append(f"   План: {task.get('plan', '')}")
            if task.get('resource_link'):
                lines.append(f"   Ресурс: {task.get('resource_link')}")
    else:
        lines.append("План не сформирован")
        
    return "\n".join(lines)

def main():
    print("=== Debug Runner Started ===")
    print(f"Monitoring {USER_INPUT_FILE} for input...")
    print(f"Responses will be written to {SYSTEM_OUTPUT_FILE}")
    
    # 1. Configuration (Hardcoded for debug speed, or could be read from first input)
    
    
    with open("user.yaml", "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f)
    
    name = user_config["user_info"]["name"]
    position = user_config["user_info"]["position"]
    grade = user_config["user_info"]["grade"]
    experience = user_config["user_info"]["experience"]
    print(f"Session Config: {name} | {position} | {grade}")

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

    app = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Initial Greeting
    initial_state = {**initial_state_config, "messages": [HumanMessage(content=user_config["user_info"]["first_message"])]}
    current_state = app.invoke(initial_state, config=config)
    
    # Write initial greeting
    if current_state["messages"]:
        last_msg = current_state["messages"][-1]
        if isinstance(last_msg, AIMessage):
            write_output(last_msg.content)
            
    # Main Loop
    while True:
        if current_state.get("status") in ["stop_requested", "finished"]:
            print("Status is finished. Generating report...")
            break

        user_input = read_and_clear_input()
        
        if user_input:
            print(f"\n[User Input Received]: {user_input}")
            current_state["messages"].append(HumanMessage(content=user_input))
            current_state = app.invoke(current_state, config=config)
            if current_state.get("status") in ["stop_requested", "finished"]:
                break
            last_msg = current_state["messages"][-1]
            if isinstance(last_msg, AIMessage):
                write_output(last_msg.content)
        
        time.sleep(0.5)
        
    print("\nInterview Finished.")
    last_msg = current_state["messages"][-1]
    if isinstance(last_msg, AIMessage):
        write_output(last_msg.content)
        
    if current_state.get("final_feedback"):
        log_filename = get_next_log_filename()
        try:
            feedback_dict = json.loads(current_state["final_feedback"])
            feedback_str = format_feedback_to_text(feedback_dict)
            log_data = {
                "participant_name": name,
                "turns": current_state.get("turns", []),
                "final_feedback": feedback_str  # formatted text
            }
            with open(log_filename, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            print(f"Log saved to: {log_filename}")
            write_output(f"INTERVIEW FINISHED. Log saved to {log_filename}")
        except Exception as e:
            print(f"Error saving log: {e}")
            write_output(f"Error saving log: {e}")
    else:
        print("No final feedback generated.")
        
if __name__ == "__main__":
    main()