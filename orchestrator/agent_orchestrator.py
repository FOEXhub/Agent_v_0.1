from agents.requirements_writer import RequirementsWriter
# Импорт остальных агентов ...
from core.enums import AgentState
from pathlib import Path
import json
import textwrap

class AgentOrchestrator:
    def __init__(self, log_file: str = "agent_logs.json"):
        self.log_file = Path(log_file)
        self.agents = {
            "requirements_writer": RequirementsWriter(),
            # Другие агенты
        }
        self.workflow = [
            ("requirements_writer", lambda x: True),
            # Остальной workflow
        ]
        self.thought_log = []

    def execute_workflow(self, user_input):
        context = {"user_input": user_input, "state": AgentState.INIT}
        for agent_name, condition in self.workflow:
            if not condition(context):
                context["state"] = AgentState.ERROR
                break

            agent = self.agents[agent_name]
            result = agent.process_data(context)
            self._log_thoughts(agent)
            context.update(result)

        self._save_final_logs()
        return context

    def _log_thoughts(self, agent):
        for entry in agent.logs:
            self.thought_log.append(entry)
        agent.logs.clear()

    def _save_final_logs(self):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            for entry in self.thought_log:
                json.dump(entry, f, ensure_ascii=False, indent=2)
                f.write('\n')
        self.thought_log.clear()