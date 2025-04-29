from pathlib import Path
import json
from datetime import datetime

class AgentOrchestrator:
    def __init__(self, log_file: str = "agent_logs.json"):
        self.log_file = Path(log_file)
        self.agents = None
        self.workflow = None
        self.thought_log = []
    
    def initialize_agents(self):
        """Lazy initialization of agents to avoid circular imports"""
        from agents.requirements_writer import RequirementsWriter
        from agents.requirements_critic import RequirementsCritic
        from agents.code_writer import CodeWriter
        from agents.code_critic import CodeCritic
        from agents.report_generator import ReportGenerator
        from core.enums import AgentState
        
        self.AgentState = AgentState
        
        self.agents = {
            "requirements_writer": RequirementsWriter(),
            "requirements_critic": RequirementsCritic(),
            "code_writer": CodeWriter(),
            "code_critic": CodeCritic(),
            "reporter": ReportGenerator()
        }
        
        self.workflow = [
            ("requirements_writer", lambda x: True),
            ("requirements_critic", lambda x: x.get("state") == self.AgentState.REQUIREMENTS_WRITTEN),
            ("code_writer", lambda x: x.get("state") == self.AgentState.REQUIREMENTS_APPROVED),
            ("code_critic", lambda x: x.get("state") == self.AgentState.CODE_WRITTEN),
            ("reporter", lambda x: x.get("state") == self.AgentState.CODE_APPROVED)
        ]

    def execute_workflow(self, user_input):
        self.initialize_agents()
        
        context = {"user_input": user_input, "state": self.AgentState.INIT}
        
        for agent_name, condition in self.workflow:
            if not condition(context):
                context["state"] = self.AgentState.ERROR
                break

            agent = self.agents[agent_name]
            try:
                result = self._invoke_with_retry(agent, context)
                self._log_thoughts(agent)
                context.update(result)
            except Exception as e:
                print(f"Failed at agent {agent_name}: {str(e)}")
                context["state"] = self.AgentState.ERROR
                context["error"] = str(e)
                break

        self._save_final_logs()
        return context

    @staticmethod
    def _invoke_with_retry(agent, context):
        return agent.process_data(context)

    def _log_thoughts(self, agent):
        for entry in agent.logs:
            self.thought_log.append(entry)
        agent.logs.clear()

    def _save_final_logs(self):
        print("Trying to save logs:")
        print(json.dumps(self.thought_log[-1], indent=2, ensure_ascii=False))
        print("-" * 80)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            for entry in self.thought_log:
                try:
                    # Попытка сохранить как обычный JSON
                    json.dump(entry, f, ensure_ascii=False, indent=2)
                    f.write('\n')
                except Exception as e:
                    safe_entry = self._make_json_safe(entry)
                    json.dump(safe_entry, f, ensure_ascii=False, indent=2)
                    f.write('\n')
                    print(f"[WARNING] Исправлен невалидный JSON: {str(e)}")

    def _make_json_safe(self, data):
        def default(o):
            try:
                return str(o)
            except Exception:
                return repr(o)

        try:
            cleaned = self._deep_clean(data)
            return json.loads(json.dumps(cleaned, default=default, ensure_ascii=False))
        except Exception:
            return {"error": "Необработанные данные", "raw": str(data)}

    def _deep_clean(self, data):
        if isinstance(data, dict):
            return {
                self._deep_clean(k): self._deep_clean(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._deep_clean(item) for item in data]
        elif isinstance(data, str):
            # Экранируем опасные символы
            return (
                data.replace('\\', '\\\\')
                    .replace('"', '\\"')
                    .replace('\n', '\\n')
                    .replace('\r', '\\r')
                    .replace('\t', '\\t')
            )
        elif isinstance(data, (int, float, bool, type(None))):
            return data
        else:
            return str(data)