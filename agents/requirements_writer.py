from core.base_agent import BaseAgent
from core.enums import AgentState

class RequirementsWriter(BaseAgent):
    def __init__(self):
        super().__init__(
            "Requirements Writer",
            "Ты эксперт по составлению технических требований для web-app приложений телеграм."
        )

    def process_data(self, inputs):
        user_input = inputs.get('user_input', '')
        
        prompt = """
        На основе следующего запроса пользователя:
        "{user_input}"

        Сгенерируй технические требования для Telegram WebApp.
        Требования должны включать:
        - Основную функциональность
        - Структуру интерфейса
        - Используемые технологии
        - Пример взаимодействия

        Верни только чёткий и структурированный текст требований без пояснений.
        """.format(user_input=user_input)
        
        response = self._generate_response(prompt)
        return {"requirements": response.strip(), "state": AgentState.REQUIREMENTS_WRITTEN}