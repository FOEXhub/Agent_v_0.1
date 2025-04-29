from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime
import json
import textwrap
from typing import Dict, Any, List, Union

from config.llm_setup import llm

class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
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
        """Логирует мысли агента в консоль в формате Markdown и сохраняет в историю"""
        # Создаем запись для хранения в памяти
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.name,
            "type": type,
            "content": thought
        }
        self.logs.append(entry)
        
        # Формируем цветовой код для типа сообщения
        color_codes = {
            "PROMPT": "\033[94m",      # Синий
            "RAW_RESPONSE": "\033[92m",# Зеленый
            "ERROR": "\033[91m",       # Красный
            "TOOL": "\033[93m",        # Желтый
            "WARNING": "\033[95m",     # Фиолетовый
            "INFO": "\033[0m"          # Сброс
        }
        color = color_codes.get(type, "\033[0m")

        # Формируем строку времени
        time_str = f"{color}[{datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}]{color_codes['INFO']}"
        
        # Определяем тип сообщения
        type_str = f"{color}{type}{color_codes['INFO']}"
        
        # Форматируем содержимое
        if isinstance(thought, dict):
            content_str = json.dumps(
                thought, 
                ensure_ascii=False, 
                indent=2,
                default=lambda o: str(o)  # Для несериализуемых объектов
            )
            content_str = f"{color}```json\n{content_str}\n```{color_codes['INFO']}"
        elif isinstance(thought, str):
            content_str = textwrap.indent(
                thought, 
                '    ', 
                lambda line: True  # Используем indent для всех строк
            )
        else:
            content_str = str(thought)

        # Выводим в консоль
        print(f"{time_str} - {self.name} ({type_str})\n{content_str}\n" + "-"*80)

    def save_logs(self, file_path):
        """Сохраняет логи в JSONL-файл"""
        with open(file_path, 'a', encoding='utf-8') as f:
            for entry in self.logs:
                json.dump(entry, f, ensure_ascii=False, indent=2)
                f.write('\n')
        self.logs.clear()

    def _generate_response(self, prompt: str) -> str:
        self._log_thought(prompt, "PROMPT")
        response = llm.invoke(prompt)
        raw_response = response.content
        self._log_thought({"raw_response": raw_response, "length": len(raw_response)}, "RAW_RESPONSE")
        return raw_response

    def _define_tools(self) -> List[Tool]:
        return [
            Tool(name="process_data", func=self.process_data, description="Основная функция обработки данных")
        ]

    def process_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError