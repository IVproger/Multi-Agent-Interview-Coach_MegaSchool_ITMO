from typing import List, Optional
from pydantic import BaseModel, Field

class MentorOutput(BaseModel):
    internal_thoughts: str = Field(description="Скрытые размышления об ответе кандидата и текущем состоянии. На РУССКОМ языке.")
    directive: str = Field(description="Указание для интервьюера, что спрашивать дальше или как реагировать. На РУССКОМ языке.")
    correction_needed: bool = Field(description="Требуется ли исправление ошибки кандидата.")
    correction_details: Optional[str] = Field(description="Детали исправления, если требуется (для внутренних нужд). На РУССКОМ языке.")
    confidence_score: float = Field(description="Уверенность в оценке ответа (0-100).", ge=0, le=100)
    stop_interview_flag: bool = Field(description="True, если интервью следует остановить (достаточно данных или запрос пользователя).", default=False)

class FinalFeedback(BaseModel):
    grade: str = Field(description="Уровень кандидата: Junior / Middle / Senior")
    hiring_recommendation: str = Field(description="Рекомендация: Hire / No Hire / Strong Hire. На английском или транслите, но лучше как термины.")
    confidence_score: float = Field(description="Общая уверенность 0-100%")
    confirmed_skills: List[str] = Field(description="Список тем, на которые кандидат ответил правильно.")
    knowledge_gaps: List[str] = Field(description="Список тем, где кандидат ошибся или не знал.")
    gap_solutions: List[str] = Field(description="Правильные ответы или объяснения для пробелов в знаниях. На РУССКОМ языке.")
    soft_skills_clarity: str = Field(description="Оценка ясности изложения. На РУССКОМ языке.")
    soft_skills_honesty: str = Field(description="Оценка честности / признания незнания. На РУССКОМ языке.")
    soft_skills_engagement: str = Field(description="Оценка вовлеченности. На РУССКОМ языке.")
    personal_roadmap: List[str] = Field(description="Список конкретных тем/технологий для изучения.")
