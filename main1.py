import ast
from pathlib import Path
import json

import textwrap

from typing import Dict, Any, List, Union
from langchain.agents import AgentExecutor, Tool, create_openai_tools_agent
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from enum import Enum, auto
from datetime import datetime

from dotenv import load_dotenv
import os

load_dotenv()

# Настройка OpenRouter
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="qwen/qwq-32b:free", # 
    # qwen/qwq-32b:free
    # deepseek/deepseek-chat-v3-0324:free
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7
)

# Состояния выполнения
class AgentState(Enum):
    INIT = auto()
    REQUIREMENTS_WRITTEN = auto()
    REQUIREMENTS_APPROVED = auto()
    CODE_WRITTEN = auto()
    CODE_APPROVED = auto()
    FINISHED = auto()
    ERROR = auto()

# Базовый класс агента
class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name  # ✅ Ensure this line exists
        self.role = role  # ✅ Ensure this line exists
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            input_key="input"
        )
        self.logs = []
        self.tools = self._define_tools()
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=role),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])


    def _log_thought(self, thought: Union[str, Dict], type: str):
        """Логирует мысли агента"""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "type": type,
            "content": thought
        })
    
    def save_logs(self, file_path: Union[str, Path]):
        """Сохраняет логи агента в файл"""
        try:
            with open(file_path, 'a', encoding='utf-8') as f:  # 'a' - append mode
                for entry in self.logs:
                    json.dump(entry, f, ensure_ascii=False, indent=2)
                    f.write('\n')
            self.logs.clear()
        except Exception as e:
            print(f"Error saving logs: {str(e)}")


    def _generate_response(self, prompt: str) -> str:
        try:
            self._log_thought(prompt, "PROMPT")
            response = llm.invoke(prompt)
            raw_response = response.content
            
            # Логируем сырой ответ
            self._log_thought({
                "raw_response": raw_response,
                "length": len(raw_response)
            }, "RAW_RESPONSE")
            
            return raw_response
        except Exception as e:
            self._log_thought(str(e), "ERROR")
            raise



    def _define_tools(self) -> List[Tool]:
        return [
            Tool(
                name="process_data",
                func=self.process_data,
                description="Основная функция обработки данных"
            )
        ]

    def process_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        agent = create_openai_tools_agent(llm, self.tools, self.prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, memory=self.memory)
        return agent_executor.invoke({"input": str(context)})

# Реализация конкретных агентов
class RequirementsWriter(BaseAgent):
    def __init__(self):
        super().__init__(
            "Requirements Writer",
            "Ты эксперт по составлению технических требований для web-app приложений телеграм (не ботов, а именно приложений). Генерируй четкие и проверяемые требования."
        )

    def process_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
            На основе следующего запроса пользователя:
            "{inputs['user_input']}"

            Сгенерируй технические требования для Telegram WebApp.
            Требования должны включать:
            - Основную функциональность
            - Структуру интерфейса
            - Используемые технологии
            - Пример взаимодействия

            Верни только чёткий и структурированный текст требований без пояснений.
            """
        response = self._generate_response(prompt)

        if not response or len(response.strip()) == 0:
            raise ValueError("RequirementsWriter не смог сгенерировать требования")

        return {"requirements": response.strip(), "state": AgentState.REQUIREMENTS_WRITTEN}

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



class ReportGenerator(BaseAgent):
    def __init__(self):
        super().__init__(
            "Report Generator",
            "Ты аналитик. Формируй итоговые отчеты на основе всех этапов работы."
        )

    def process_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""Сформируй итоговый отчет со следующими разделами:
                1. Исходные требования
                2. Критика требований
                3. Сгенерированный код
                4. Результаты код-ревью
                5. Итоговые рекомендации
                
                Данные: {inputs}"""
        response = llm.invoke(prompt)
        return {"final_report": response.content, "state": AgentState.FINISHED}

# Система управления агентами
class AgentOrchestrator:
    def __init__(self, log_file: str = "agent_logs.json"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.thought_log = []
        self.agents = {
            "requirements_writer": RequirementsWriter(),
            "requirements_critic": RequirementsCritic(),
            "code_writer": CodeWriter(),
            "code_critic": CodeCritic(),
            "reporter": ReportGenerator()
        }
        # Исправлено: заменяем [...] на правильный workflow
        self.workflow = [
            ("requirements_writer", lambda x: True),
            ("requirements_critic", lambda x: x.get("state") == AgentState.REQUIREMENTS_WRITTEN),
            ("code_writer", lambda x: x.get("state") == AgentState.REQUIREMENTS_APPROVED),
            ("code_critic", lambda x: x.get("state") == AgentState.CODE_WRITTEN),
            ("reporter", lambda x: x.get("state") == AgentState.CODE_APPROVED)
        ]
        self.execution_log = []


    def _log_thoughts(self, agent: BaseAgent):
        """Сохраняет мысли агента в общий лог и выводит на экран"""
        for entry in agent.logs:
            timestamp = entry["timestamp"]
            agent_name = entry["agent"]
            log_type = entry["type"]
            content = entry["content"]

            # Печатаем только определённые типы логов
            if log_type == "PROMPT":
                print(f"[{timestamp}] 🤖 {agent_name} получил запрос:")
                print(textwrap.fill(str(content), width=100))
            elif log_type == "RAW_RESPONSE":
                print(f"[{timestamp}] 💬 Модель ответила ({content['length']} символов):")
                print(textwrap.fill(str(content["raw_response"]), width=100))
            elif log_type == "ERROR":
                print(f"[{timestamp}] ❗ Ошибка в работе агента {agent_name}:")
                print(str(content))

            self.thought_log.append(entry)
        agent.logs.clear()

    def execute_workflow(self, user_input: str) -> Dict[str, Any]:
        context = {"user_input": user_input, "state": AgentState.INIT}
        step_counter = 1
        MAX_RETRIES = 3

        try:
            for agent_name, condition in self.workflow:
                if not condition(context):
                    context["state"] = AgentState.ERROR
                    break

                agent = self.agents[agent_name]
                print(f"\n{'='*40}\nШаг {step_counter}: {agent.name}...")

                retries = 0
                while retries <= MAX_RETRIES:
                    try:
                        result = agent.process_data(context)
                        self._log_thoughts(agent)
                        break  # Если успешно — выходим из цикла
                    except Exception as e:
                        retries += 1
                        print(f"🔄 Попытка {retries} из {MAX_RETRIES}: {str(e)}")
                        if retries > MAX_RETRIES:
                            raise

                # Обновляем контекст после успешного выполнения
                context.update(result)

                # Форматированный вывод результатов...
                # Ваш существующий код вывода и логирования

                step_counter += 1

        finally:
            self._save_final_logs()

        return context


    def _save_final_logs(self):
        """Сохраняет системные логи оркестратора"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                for entry in self.thought_log:
                    json.dump(entry, f, ensure_ascii=False, indent=2)
                    f.write('\n')
                self.thought_log.clear()
        except Exception as e:
            print(f"Final log save failed: {str(e)}")


# Пример использования
if __name__ == "__main__":
    
    orchestrator = AgentOrchestrator()
    user_request = "Создай интерактивную форму в Telegram WebApp с полями имя, email и кнопкой отправки"
    final_result = orchestrator.execute_workflow(user_request)
    
    # Вывод финального отчета
    if final_result["state"] == AgentState.FINISHED:
        print("\n\n✨ Итоговый отчет:")
        print("-"*40)
        print(final_result.get("final_report", ""))
        print("-"*40)
    else:
        print("\n\n⚠️ Workflow завершен с ошибкой на шаге:")
        print(json.dumps(final_result, indent=2, ensure_ascii=False, default=str))