# Telegram WebApp Agent System 🤖

Это система автоматической генерации технических требований и кода для Telegram WebApp приложений, основанная на LLM (Large Language Models).

## 🔧 Особенности
- Автоматическая генерация требований по пользовательскому запросу
- Генерация рабочего Python-кода для Telegram WebApp
- Многоуровневая проверка качества через цепочку критиков
- Полная защита от JSON-ошибок и спецсимволов
- Интеграция с Telegram WebApp API
- Поддержка Markdown-форматирования в ответах

## 📦 Структура проекта
```
telegram-webapp-agent/
├── agents/                  # Агенты системы
│   ├── requirements_writer.py  # Генератор требований
│   ├── requirements_critic.py  # Проверка требований
│   ├── code_writer.py          # Генератор кода
│   ├── code_critic.py          # Проверка кода
│   └── report_generator.py     # Генератор отчетов
├── core/                    # Ядро системы
│   ├── base_agent.py           # Базовый класс агента
│   ├── enums.py                # Перечисления состояний
│   └── orchestrator.py         # Оркестровщик агентов
├── config/
│   └── llm_setup.py            # Настройка LLM
├── logs/
│   └── agent_logs.jsonl        # Логи работы системы (JSON Lines)
└── README.md                   # Документация
```

## 🛠️ Технологии
- Python 3.10+
- Flask (для серверной части)
- Telegram WebApp API
- LangChain / LLM
- JSONL (формат логов)

## ⚙️ Установка

```bash
# Клонирование репозитория
git clone https://github.com/yourname/telegram-webapp-agent.git
cd telegram-webapp-agent

# Установка зависимостей
pip install -r requirements.txt

# Настройка переменных окружения
cp .env.example .env
# Откройте .env и введите ваш Telegram Bot Token и другие параметры
```

## ▶️ Запуск

```bash
python main.py "Создай форму с полем email и кнопкой"
```

## 🧪 Пример вывода

```json
{
  "requirements": "# Технические требования\n...\n",
  "generated_code": "```python\nfrom flask import Flask, request\n...\n```",
  "final_report": {
    "status": "success",
    "approved_requirements": true,
    "approved_code": true
  }
}
```

## 📌 Переменные окружения

| Переменная | Описание |
|-----------|----------|
| `TELEGRAM_BOT_TOKEN` | Токен вашего Telegram бота |
| `LLM_MODEL` | Модель LLM (например, gpt-3.5-turbo) |
| `LOG_FILE` | Путь к файлу логов (по умолчанию: agent_logs.jsonl) |

## 🧼 Защита от ошибок

Система полностью защищена от:
- Пустых ответов от LLM
- Некорректных JSON-структур
- Спецсимволов в ответах (`\n`, `\t`, `"`, `'`)
- Markdown/HTML-разметки в ответах
- Циклических ссылок и сложных объектов

## 📋 Лицензия

[MIT License](LICENSE)

---

> Создано для автоматизации разработки Telegram WebApp приложений с использованием современных LLM технологий.