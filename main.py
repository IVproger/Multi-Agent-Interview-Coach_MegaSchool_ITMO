from dotenv import load_dotenv
load_dotenv(".env")

import json
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import build_graph

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
            print("\n[System]: Интервью завершается...")
            break
            
        last_msg = current_state["messages"][-1]
        print(f"\n[Interviewer]: {last_msg.content}")

    # Логика генерации отчета
    if current_state.get("final_feedback"):
        feedback_str = current_state["final_feedback"]   
        try:
            feedback_dict = json.loads(feedback_str)
            print("\n=== РЕЗУЛЬТАТЫ ИНТЕРВЬЮ ===")
            print(f"Грейд: {feedback_dict.get('grade')}")
            print(f"Рекомендация: {feedback_dict.get('hiring_recommendation')}")
            print(f"Уверенность: {feedback_dict.get('confidence_score')}%")
        except:
             print("\n=== РЕЗУЛЬТАТЫ ИНТЕРВЬЮ ===")
             print("Отчет получен (сырой формат).")

        print("\nПолный отчет сохранен в 'interview_log.json'.")
        
        # Сохранение в JSON  
        log_data = {
            "participant_name": name,
            "turns": current_state["turns"],
            "final_feedback": feedback_dict
        }
        
        with open("interview_log.json", "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
            print("Лог сохранен в interview_log.json")
    else:
        print("Отчет не сгенерирован.")

if __name__ == "__main__":
    main()
