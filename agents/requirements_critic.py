from core.base_agent import BaseAgent
from core.enums import AgentState

class RequirementsCritic(BaseAgent):
    def __init__(self):  # Добавленный конструктор
        super().__init__(
            name="Requirements Critic",
            role="Ты эксперт по анализу требований. Проверяй полноту и выполнимость требований для web-app приложений телеграм."
        )

    def process_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        requirements = inputs.get('requirements', '')
        if not requirements.strip():
            return {
                "requirements_review": {
                    "approved": False,
                    "comments": "Пустые или недопустимые требования",
                    "score": 0
                },
                "state": AgentState.ERROR
            }

        prompt = f"""Проанализируй следующие требования:
                {requirements}
                ВАЖНО! Ответ должен быть строго в JSON-формате:
                {{
                    "approved": boolean,
                    "comments": string,
                    "score": integer 1-10
                }}
                Только JSON без других текстов!"""

        
        try:
            response = self._generate_response(prompt)
            
            # Проверка на пустой ответ
            if not response.strip():
                raise ValueError("Empty response received")
                
            # Поиск JSON в тексте
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end] if json_start != -1 else response
            
            result = json.loads(json_str)
            
        except json.JSONDecodeError:
            result = {
                "approved": False,
                "comments": "Ошибка формата ответа от LLM",
                "issues": ["Некорректный JSON"]
            }
            
        return {
            "requirements_review": result,
            "state": AgentState.REQUIREMENTS_APPROVED if result.get("approved") else AgentState.ERROR
        }