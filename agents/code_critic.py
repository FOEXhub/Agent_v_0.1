from core.base_agent import BaseAgent
from core.enums import AgentState

class CodeCritic(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Code Critic",
            role="Ты Senior разработчик. Проверяй качество кода web-app приложений телеграм, находи ошибки и оптимизируй."
        )

    def _extract_code(self, code_block: str) -> str:
        """Извлекает код из блока с ```python"""
        try:
            if '```python' in code_block:
                return code_block.split('```python')[1].split('```')[0].strip()
            elif '```' in code_block:
                return code_block.split('```')[1].split('```')[0].strip()
            return code_block
        except IndexError:
            return code_block

    def _validate_python(self, code: str) -> bool:
        """Проверяет синтаксис Python"""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            self._log_thought(f"Syntax error: {str(e)}", "VALIDATION_ERROR")
            return False

    def process_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Извлекаем код с обработкой ошибок
            raw_code = inputs.get('generated_code', '')
            code = self._extract_code(raw_code)
            
            if not code:
                return {
                    "code_review": {
                        "approved": False,
                        "comments": "No code found",
                        "issues": ["Missing code block"]
                    },
                    "state": AgentState.ERROR
                }

            # Валидация синтаксиса
            if not self._validate_python(code):
                return {
                    "code_review": {
                        "approved": False,
                        "comments": "Syntax error in code",
                        "issues": ["Invalid Python syntax"]
                    },
                    "state": AgentState.ERROR
                }

            # Генерация запроса с явным указанием формата
            prompt = textwrap.dedent(f"""
                Проверь следующий код Telegram web-app и **обязательно** верни **только** JSON-ответ без дополнительного текста!

                Формат ответа:
                {{
                    "approved": boolean,
                    "comments": string,
                    "issues": list[string]
                }}

                Требования:
                - approved: true только если нет синтаксических и логических ошибок
                - comments: общий комментарий по качеству кода
                - issues: список конкретных проблем

                Пример ответа:
                {{
                    "approved": false,
                    "comments": "Отсутствует обработка ошибок",
                    "issues": ["Нет retry для API запросов", "Нет валидации входных данных"]
                }}

                Код для проверки:
                {code}
            """)

            response = self._generate_response(prompt)
            
            # Поиск JSON в ответе
            json_str = response[response.find('{'):response.rfind('}')+1]
            
            # Удаляем экранированные символы
            json_str = json_str.replace('\\', '')
            
            # Парсинг JSON
            try:
                result = json.loads(json_str)
                if not isinstance(result.get('issues', []), list):
                    raise ValueError("Issues should be a list")
            except Exception as e:
                self._log_thought(f"JSON parsing error: {str(e)}", "ERROR")
                return {
                    "code_review": {
                        "approved": False,
                        "comments": f"Invalid review format: {str(e)}",
                        "issues": ["Failed to parse review"]
                    },
                    "state": AgentState.ERROR
                }

            return {
                "code_review": result,
                "state": AgentState.CODE_APPROVED if result.get("approved") else AgentState.ERROR
            }

        except Exception as e:
            self._log_thought(f"Critical error: {str(e)}", "ERROR")
            return {
                "code_review": {
                    "approved": False,
                    "comments": f"System error: {str(e)}",
                    "issues": ["Critical processing error"]
                },
                "state": AgentState.ERROR
            }