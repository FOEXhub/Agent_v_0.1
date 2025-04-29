from core.base_agent import BaseAgent
from core.enums import AgentState

class ReportGenerator(BaseAgent):
    def __init__(self):
        super().__init__(
            "Report Generator",
            "Ты аналитик. Формируй итоговые отчеты на основе всех этапов работы."
        )

    def process_data(self, inputs: dict[str, any]) -> dict[str, any]:
        prompt = f"""Сформируй итоговый отчет со следующими разделами:
                1. Исходные требования
                2. Критика требований
                3. Сгенерированный код
                4. Результаты код-ревью
                5. Итоговые рекомендации
                
                Данные: {inputs}"""
        response = llm.invoke(prompt)
        return {"final_report": response.content, "state": AgentState.FINISHED}