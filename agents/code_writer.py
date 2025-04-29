from core.base_agent import BaseAgent
from core.enums import AgentState

class CodeWriter(BaseAgent):
    def __init__(self):
        super().__init__(
            "Code Writer",
            "Ты Senior Python разработчик с опытом работы в web-app приложениями телеграм. Пиши чистый, эффективный код web-app приложений телеграмм."
        )

    def process_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""Напиши код web-app телеграмм приложения строго по этим требованиям:\n{inputs['requirements']}\n"""
        
        # Include critic feedback if available
        if 'code_review' in inputs:
            prompt += f"""\nКритика предыдущего кода: {inputs['code_review'].get('comments', '')}
                        Необходимо исправить следующие проблемы: {', '.join(inputs['code_review'].get('issues', []))}"""
                        
        prompt += "\nВерни ТОЛЬКО код Python без пояснений, обернув в ```python ... ```"

        response = self._generate_response(prompt)
        return {"generated_code": response, "state": AgentState.CODE_WRITTEN}