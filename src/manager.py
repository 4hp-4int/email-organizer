from typing import Optional, Dict

from agent_protocol import Agent

from src.message import Message


# AgentManager class definition
class AgentManager:
    def __init__(self):
        # Dictionary to hold registered agents by their name.
        self.agents: Dict[str, Agent] = {}

    def register_agent(self, agent: Agent) -> None:
        """
        Register an agent with the manager.
        """
        if agent.name in self.agents:
            raise ValueError(f"Agent with name '{agent.name}' is already registered.")
        self.agents[agent.name] = agent
        print(f"Registered agent: {agent.name}")

    def unregister_agent(self, agent_name: str) -> None:
        """
        Unregister an agent from the manager.
        """
        if agent_name in self.agents:
            del self.agents[agent_name]
            print(f"Unregistered agent: {agent_name}")
        else:
            raise ValueError(f"No agent registered with name '{agent_name}'.")

    def send_message(self, recipient_name: str, message: Message) -> Optional[Message]:
        """
        Send a message to the specified agent by name and return the agent's response.
        """
        if recipient_name not in self.agents:
            raise ValueError(f"No agent registered with name '{recipient_name}'.")

        agent = self.agents[recipient_name]
        print(f"Sending message to {recipient_name}: {message}")
        response = agent.on_message(message)
        print(f"Received response from {recipient_name}: {response}")
        return response
