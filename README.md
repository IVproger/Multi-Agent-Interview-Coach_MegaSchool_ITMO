# Multi-Agent Interview Coach

A system that simulates a technical interview using multiple AI agents (Interviewer and Mentor).

## Features
- **Interviewer Agent**: Conducts the dialogue, adapts questions.
- **Mentor Agent**: Hidden observer that analyzes answers, provides directives, and fact-checks.
- **Reporting**: Generates a structured JSON log and learning roadmap after the session.
- **LangGraph**: Orchestrates the agent workflow.

## Usage

1. **Install Dependencies**:
   ```bash
   pip install -U langchain langchain-openai langgraph
   ```

2. **Environment**:
   Set `OPENAI_API_KEY` in `.env`.

3. **Run**:
   ```bash
   python main.py
   ```

4. **Process**:
   - Enter details.
   - Chat.
   - Type 'Stop interview' to finish.
   - Check `interview_log.json`.
