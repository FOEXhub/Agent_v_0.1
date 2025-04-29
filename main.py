from orchestrator.agent_orchestrator import AgentOrchestrator

if __name__ == "__main__":
    orchestrator = AgentOrchestrator()
    result = orchestrator.execute_workflow("Создай форму с полем email и кнопкой")
    print(result.get("final_report", result))