import os
import json
import uuid
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import build_graph
from agent.state import InterviewState

def main():
    print("=== Мульти-Агентная Тренировка Интервью ===")
    
    # 1. Сбор информации о кандидате
    print("\nПожалуйста, укажите ваши данные:")
    name = input("Имя: ") or "Кандидат"
    position = input("Позиция (например, Backend Developer): ") or "Software Engineer"
    grade = input("Целевой грейд (Junior/Middle/Senior): ") or "Junior"
    experience = input("Кратко об опыте: ") or "Нет"
    
    # 2. Инициализация состояния
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
    
    # Пользователь вводит первое сообщение
    first_user_message = input("\nВведите ваше первое сообщение (или нажмите Enter, чтобы пропустить): ")

    # Хак: Вставляем фиктивное сообщение, чтобы запустить цикл.
    messages = [HumanMessage(content=first_user_message or "Здравствуйте, я готов к интервью.")]
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Явно конструируем словарь начального состояния
    initial_state = {**initial_state_config, "messages": messages}
     
    current_state = app.invoke(initial_state, config=config)
    
    # Вывод первого ответа
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
            # Мы хотим, чтобы логика графа обработала это через Ментора.
            # Если пользователь пишет "стоп", Ментор должен распознать это и установить флаг.
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
        feedback = current_state["final_feedback"]
        print("\n=== РЕЗУЛЬТАТЫ ИНТЕРВЬЮ ===")
        print(f"Грейд: {feedback.get('grade')}")
        print(f"Рекомендация: {feedback.get('hiring_recommendation')}")
        print(f"Уверенность: {feedback.get('confidence_score')}%")
        print("\nПолный отчет сохранен в 'interview_log.json'.")
        
        # Сохранение в JSON
        log_data = {
            "participant_name": name,
            "session_meta": current_state["session_meta"],
            "turns": current_state["turns"],
            "final_feedback": feedback
        }
        
        with open("interview_log.json", "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
            print("Лог сохранен в interview_log.json")
    else:
        print("Отчет не сгенерирован.")

if __name__ == "__main__":
    main()
