from dotenv import load_dotenv
load_dotenv(".env")

import json
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import build_graph

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
    print("=== Мульти-Агентная Тренировка Интервью ===")
    
    # Сбор информации о кандидате
    print("\nПожалуйста, укажите ваши данные:")
    name = input("Имя: ") or "Кандидат"
    position = input("Позиция (например, Backend Developer): ") or "Кандидат не указал позицию"
    
    # Проверка корректности введенного грейда
    valid_grades = ["Junior", "Middle", "Senior"]
    while True:
        grade = input("Целевой грейд (Junior/Middle/Senior): ") or "Junior"
        if grade in valid_grades:
            break
        else:
            print("Пожалуйста, введите корректный грейд: Junior, Middle или Senior.")

    experience = input("Кратко об опыте: ") or "У меня нет опыта. И я уставил это поле "
    
    # Инициализация состояния системы
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
        "mentor_directive": "Начни интервью с представления себя и задай первый релевантный вопрос.",
        "mentor_thoughts": "Начальное состояние.",
        "mentor_confidence_score": 100.0,
        "last_candidate_answer": "",
        "last_interviewer_question": ""
    }
    
    app = build_graph()
    
    first_user_message = input("\nПриветсвие. Введите ваше первое сообщение (или нажмите Enter, чтобы пропустить): ")
    messages = [HumanMessage(content=first_user_message or "Здравствуйте, я готов к интервью.")]
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = {**initial_state_config, "messages": messages}
    current_state = app.invoke(initial_state, config=config)
    
    last_msg = current_state["messages"][-1]
    if isinstance(last_msg, AIMessage):
        print(f"\n[Interviewer]: {last_msg.content}")
    
    while True:

        # Проверка остановки
        if current_state.get("status") in ["stop_requested", "finished"]:
            break
            
        try:
            user_input = input(f"\n[{name}]: ")
        except EOFError:
            break
            
        if not user_input.strip():
            continue
            
        # Поддержка русских и английских команд остановки
        if user_input.lower() in ["exit", "quit", "stop interview", "стоп", "стоп интервью", "выход", "стоп игра. давай фидбэк."]:
            pass
        
        # Добавляем сообщение пользователя в список сообщений состояния
        current_state["messages"].append(HumanMessage(content=user_input))
        
        # Снова вызываем граф с обновленным состоянием
        current_state = app.invoke(current_state, config=config)
        
        # Проверяем статус
        if current_state.get("status") in ["stop_requested", "finished"]:
            # print the last interviewer message before
            last_msg = current_state["messages"][-1]
            if isinstance(last_msg, AIMessage):
                print(f"\n[Interviewer]: {last_msg.content}")
            else:
                print(f"\n[Interviewer]: Собеседование завершено. Спасибо за участие.")
            break
            
        last_msg = current_state["messages"][-1]
        print(f"\n[Interviewer]: {last_msg.content}")

    # Логика генерации отчета
    if current_state.get("final_feedback"):
        feedback_str = current_state["final_feedback"]   
        final_output = feedback_str

        try:
            feedback_dict = json.loads(feedback_str)
            print("\n=== РЕЗУЛЬТАТЫ ИНТЕРВЬЮ ===")
            print(f"Грейд: {feedback_dict.get('grade')}")
            print(f"Рекомендация: {feedback_dict.get('hiring_recommendation')}")
            print(f"Уверенность: {feedback_dict.get('confidence_score')}%")
            
            # Format to text
            final_output = format_feedback_to_text(feedback_dict)
        except:
             print("\n=== РЕЗУЛЬТАТЫ ИНТЕРВЬЮ ===")
             print("Отчет получен (сырой формат).")

        print("\nПолный отчет сохранен в 'interview_log.json'.")
        
        # Сохранение в JSON  
        log_data = {
            "participant_name": name,
            "turns": current_state["turns"],
            "final_feedback": final_output
        }
        
        with open("interview_log.json", "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
            print("Лог сохранен в interview_log.json")
    else:
        print("Отчет не сгенерирован.")

if __name__ == "__main__":
    main()
