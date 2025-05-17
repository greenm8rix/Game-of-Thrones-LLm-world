from typing import Dict, Optional
# Forward declaration for AgentState to avoid circular import if AgentState needs Registry
# This is a common pattern when two classes reference each other.
# Alternatively, AgentState could be imported within methods if full type hint is needed there.
if False: # TYPE_CHECKING:
    from .agent import AgentState

# Placeholder for logger
class PrintLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")

log = PrintLogger()

class Registry:
    """
    Manages a central registry of all agents in the simulation.
    This allows for quick lookups and provides a single source of truth for agent existence.
    """
    def __init__(self):
        global log
        try:
            from ..utils.logger import log as main_log # Relative import
            log = main_log
        except ImportError:
            pass # Keep print logger if utils.logger is not yet available

        self.agents: Dict[int, 'AgentState'] = {}
        log.info("Registry initialized.")

    def add(self, agent: 'AgentState') -> bool:
        """
        Adds an agent to the registry.
        Args:
            agent: The AgentState object to add.
        Returns:
            True if the agent was added successfully, False if an ID conflict occurs.
        """
        if agent.id in self.agents:
            log.warning(f"Agent ID {agent.id} ({agent.name}) conflict in registry. Not adding.")
            return False
        self.agents[agent.id] = agent
        # log.debug(f"Agent {agent.name} [{agent.id}] added to registry.")
        return True

    def remove(self, agent_id: int) -> bool:
        """
        Removes an agent from the registry.
        Args:
            agent_id: The ID of the agent to remove.
        Returns:
            True if the agent was found and removed, False otherwise.
        """
        if agent_id in self.agents:
            # agent_name = self.agents[agent_id].name
            del self.agents[agent_id]
            # log.debug(f"Agent {agent_name} [{agent_id}] removed from registry.")
            return True
        # log.warning(f"Attempted to remove non-existent agent ID {agent_id} from registry.")
        return False

    def get_agent(self, agent_id: int) -> Optional['AgentState']:
        """
        Retrieves an agent by its ID.
        Args:
            agent_id: The ID of the agent to retrieve.
        Returns:
            The AgentState object if found, otherwise None.
        """
        return self.agents.get(agent_id)

    def get_agent_name(self, agent_id: int) -> str:
        """
        Safely retrieves an agent's name by ID.
        Args:
            agent_id: The ID of the agent.
        Returns:
            The agent's name if found, or a placeholder string if not.
        """
        agent = self.agents.get(agent_id)
        return agent.name if agent else f"UnknownAgent[{agent_id}]"

    def is_agent_alive(self, agent_id: int) -> bool:
        """
        Checks if an agent with the given ID is alive.
        Args:
            agent_id: The ID of the agent.
        Returns:
            True if the agent exists and is alive, False otherwise.
        """
        agent = self.agents.get(agent_id)
        return agent is not None and agent.is_alive # Assumes AgentState has an 'is_alive' property

    def get_all_agents(self) -> list['AgentState']:
        """Returns a list of all agents in the registry."""
        return list(self.agents.values())

    def get_living_agents(self) -> list['AgentState']:
        """Returns a list of all living agents in the registry."""
        return [agent for agent in self.agents.values() if agent.is_alive]

    def clear(self):
        """Clears all agents from the registry."""
        self.agents.clear()
        log.info("Registry cleared.")

# Global registry instance (optional, can be managed by the main application)
# For this refactoring, we'll let the main simulation loop instantiate and pass it around.
# registry_instance = Registry()

if __name__ == '__main__':
    # Example Usage
    # This requires a mock AgentState or the actual AgentState class for testing.
    class MockAgentState:
        _next_id_test = 1
        def __init__(self, name):
            self.id = MockAgentState._next_id_test
            MockAgentState._next_id_test +=1
            self.name = name
            self.is_alive = True
            log.info(f"MockAgent {self.name} created with ID {self.id}")

    # Test with PrintLogger if run directly
    log.info("Testing Registry...")
    test_registry = Registry()

    agent1 = MockAgentState("Jon Snow")
    agent2 = MockAgentState("Daenerys Targaryen")

    test_registry.add(agent1)
    test_registry.add(agent2)

    log.info(f"Agent 1 Name: {test_registry.get_agent_name(agent1.id)}")
    retrieved_agent = test_registry.get_agent(agent2.id)
    if retrieved_agent:
        log.info(f"Retrieved Agent 2: {retrieved_agent.name}, Alive: {test_registry.is_agent_alive(agent2.id)}")

    log.info(f"All agents: {[a.name for a in test_registry.get_all_agents()]}")
    log.info(f"Living agents: {[a.name for a in test_registry.get_living_agents()]}")

    agent2.is_alive = False
    log.info(f"Daenerys is_alive set to False. Is she alive in registry? {test_registry.is_agent_alive(agent2.id)}")
    log.info(f"Living agents now: {[a.name for a in test_registry.get_living_agents()]}")


    test_registry.remove(agent1.id)
    log.info(f"Jon Snow removed. Agent 1 in registry? {test_registry.get_agent(agent1.id)}")
    log.info(f"All agents after removal: {[a.name for a in test_registry.get_all_agents()]}")

    test_registry.clear()
    log.info(f"Registry cleared. All agents: {test_registry.get_all_agents()}")
