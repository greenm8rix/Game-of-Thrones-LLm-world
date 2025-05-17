import asyncio
import json
import gzip
import random
import ast
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

# Forward declarations for type hinting
if TYPE_CHECKING:
    from .agent import AgentState
    from .registry import Registry
    from .world_elements import LocationNode, Structure
    from ..config import Config
    from chromadb import PersistentClient

# Placeholder for logger and config
class PrintLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg, exc_info=False): print(f"ERROR: {msg}")
    def critical(self, msg, exc_info=False): print(f"CRITICAL: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")

log = PrintLogger()

class MockConfigForWorld: # Placeholder
    LOCATIONS = ["DefaultLocation1"]
    INITIAL_RESOURCES = {"DefaultLocation1": Counter()}
    MAX_CONCURRENT_LLM_CALLS = 5
    TICKS_PER_DAY = 4
    EVENT_CHECK_INTERVAL = 10
    EVENT_BASE_CHANCE = 0.1
    RANDOM_EVENTS = []
    MIN_POP = 2
    MAX_POP = 10 # Added for respawn logic
    CODE_MODIFICATION_WHITELIST = set()
    FORBIDDEN_CODE_PATTERNS = []
    MAX_CODE_LINES = 10
    BASE_ATTACK_DAMAGE_FACTOR = 10.0
    SKILLS = ["war"] # For resolve_attack
    TRAITS = ["wrath"] # For resolve_attack


class World:
    def __init__(self,
                 config: 'Config', # Pass the actual config object
                 registry: 'Registry', # Pass the registry instance
                 chroma_client: Optional['PersistentClient'] = None,
                 load_data: Optional[Dict] = None):
        global log
        try:
            from ..utils.logger import log as main_log
            log = main_log
        except ImportError:
            pass

        self.config = config
        self.registry = registry
        self.chroma_client = chroma_client # Used for creating new agents if they need it

        self.agents: Dict[int, 'AgentState'] = {} # This will mirror registry.agents for convenience
        self.structures: Dict[int, 'Structure'] = {} # Global lookup for structures by ID
        self.locations: Dict[str, 'LocationNode'] = {}

        self.current_tick: int = 0
        self.day: int = 0
        self.season: str = "Spring" # TODO: Implement season changes

        self.new_agent_buffer: List['AgentState'] = [] # For agents born/spawned mid-tick
        self.agents_to_remove: List[int] = [] # Buffer for agent IDs to remove at end of tick

        self.llm_semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_LLM_CALLS)
        
        # For code modification
        self.pending_code_updates: List[Tuple[str, str]] = [] # (target_name, code_string)
        self.last_event_trigger_tick: int = 0

        if load_data:
            self._load_from_dict(load_data)
            log.info(f"World loaded. Tick: {self.current_tick}, Pop: {len(self.agents)}")
        else:
            self._initialize_locations()
            log.info("New world initialized.")

    def _initialize_locations(self):
        from .world_elements import LocationNode # Local import
        for loc_name in self.config.LOCATIONS:
            base_res = self.config.INITIAL_RESOURCES.get(loc_name, Counter())
            # Pass self.config to LocationNode so it can access TICK_PER_DAY etc.
            self.locations[loc_name] = LocationNode(name=loc_name, base_resources=base_res, config_ref=self.config)
        log.info(f"Initialized {len(self.locations)} location nodes.")

    def _load_from_dict(self, data: Dict):
        from .agent import AgentState # Local imports for from_dict methods
        from .world_elements import LocationNode, Structure
        from ..core.code_utils import restore_all_original_code, store_original_code # For restoring code state

        # Restore original code definitions before loading a world that might have run with modified code
        log.info("Restoring original code definitions before loading save file...")
        # These need the actual class definitions, not just instances.
        # This implies that store_original_code and restore_all_original_code
        # should be called from a place where Config, AgentState class, World class are defined/imported.
        # For now, we assume they are available or this part needs adjustment.
        # One way is to pass the class types themselves to these functions.
        # Let's assume the main script will handle calling store_original_code with correct modules.
        # And restore_all_original_code will be called similarly.
        # For now, this call might be problematic if AgentState/World classes aren't fully "known" here.
        # This is a structural challenge with dynamic code modification.
        # A simpler approach: saved games always run with the *current* codebase's original functions.
        # The `restore_all_original_code` is more for resetting during a session if code was changed.
        # So, on load, we just load data. If code was modified in the *previous* session,
        # that modification is lost unless the save format itself stored modified code (which it doesn't).
        # The `store_original_code` should be called once at app startup.
        # `restore_all_original_code` can be called if a "reset code" feature is desired.

        AgentState._next_id = data.get("next_agent_id", 1)
        Structure._next_id = data.get("next_structure_id", 1)
        self.current_tick = data.get("current_tick", 0)
        self.day = data.get("day", 0)
        self.season = data.get("season", "Spring")
        
        self.locations = {
            name: LocationNode.from_dict(ldata, config_ref=self.config)
            for name, ldata in data.get("locations", {}).items()
        }
        # Ensure all configured locations exist
        for loc_name in self.config.LOCATIONS:
            if loc_name not in self.locations:
                base_res = self.config.INITIAL_RESOURCES.get(loc_name, Counter())
                self.locations[loc_name] = LocationNode(name=loc_name, base_resources=base_res, config_ref=self.config)
                log.warning(f"Location {loc_name} from config missing in save, re-initialized.")

        self.structures.clear()
        for loc_node in self.locations.values():
            for struct_id, struct_obj in loc_node.structures.items():
                self.structures[struct_id] = struct_obj
        
        self.agents.clear()
        self.registry.clear() # Clear registry before repopulating
        loaded_agents_data = data.get("agents", {})
        for aid_str, adata in loaded_agents_data.items():
            aid = int(aid_str)
            agent = AgentState.from_dict(adata, config=self.config, chroma_client=self.chroma_client)
            self.agents[aid] = agent
            self.registry.add(agent) # Add to central registry

        self.last_event_trigger_tick = data.get("last_event_trigger_tick", self.current_tick)
        self.pending_code_updates.clear() # Don't load pending changes from save
        self.new_agent_buffer.clear()
        self.agents_to_remove.clear()

    def get_agents_at_loc(self, loc_name: str) -> List['AgentState']:
        return [a for a in self.agents.values() if a.loc == loc_name and a.is_alive]

    def get_living_agents(self) -> List['AgentState']:
        return [a for a in self.agents.values() if a.is_alive]

    def add_structure(self, structure: 'Structure') -> bool:
        if structure.id in self.structures:
            log.warning(f"Structure ID {structure.id} ({structure.name}) conflict. Cannot add to world master list.")
            return False
        
        loc_node = self.locations.get(structure.loc)
        if not loc_node:
            log.error(f"Cannot add structure {structure.name} to non-existent location {structure.loc}")
            return False
            
        self.structures[structure.id] = structure
        loc_node.add_structure(structure) # Also add to location's list
        return True

    def remove_structure(self, structure_id: int) -> Optional['Structure']:
        struct_to_remove = self.structures.pop(structure_id, None)
        if struct_to_remove:
            loc_node = self.locations.get(struct_to_remove.loc)
            if loc_node:
                loc_node.remove_structure(structure_id)
            else:
                log.warning(f"Structure {structure_id} removed from world master list, but its location {struct_to_remove.loc} was not found.")
            return struct_to_remove
        return None

    def kill_agent(self, agent_to_kill: 'AgentState'): # Changed from agent_id to agent object
        if agent_to_kill.id not in self.agents_to_remove:
            self.agents_to_remove.append(agent_to_kill.id)
            agent_to_kill.health = 0 # Mark health as 0 immediately
            log.info(f"Agent {agent_to_kill.name} [{agent_to_kill.id}] marked for removal.")


    def _trigger_random_event(self) -> Optional[str]:
        if not self.config.RANDOM_EVENTS: return None
        # Logic from original World._trigger_random_event, adapted for self.config
        event_config = random.choice(self.config.RANDOM_EVENTS)
        # ... (Full implementation of event triggering) ...
        # This is a simplified version for brevity
        message_template = event_config.get("message", "An event occurred.")
        location_str = "the realm"
        if event_config.get("loc_specific") and self.config.LOCATIONS:
            location_str = random.choice(self.config.LOCATIONS)
            # Apply effect to location_str if applicable
        # Apply global effects
        log.info(f"Random event triggered: {event_config.get('name')}")
        return f"[EVENT] {message_template.format(location=location_str)}"


    def tick_update(self) -> List[str]:
        """Advance world time, process needs, regen, events, code changes, agent removal."""
        from .agent import AgentState # For respawn
        from ..utils.helpers import generate_themed_name # For respawn

        self.current_tick += 1
        tick_events: List[str] = []

        # Apply Pending Code Changes (if any)
        if self.pending_code_updates:
            log.info(f"Applying {len(self.pending_code_updates)} pending code updates...")
            for target_name, code_string in self.pending_code_updates:
                 success, msg = self._apply_code_change_unsafe(target_name, code_string) # Internal method
                 event_msg = f"Code Change Applied: Target='{target_name}', Result='{msg}'"
                 tick_events.append(f"[SYSTEM] {event_msg}")
                 if not success: log.error(f"Error applying code change for {target_name}: {msg}")
            self.pending_code_updates.clear()

        # Daily / Seasonal Updates
        if self.current_tick % self.config.TICKS_PER_DAY == 0:
            self.day += 1
            for loc_node in self.locations.values():
                loc_node.regenerate_resources()

        # Random Event Check
        if self.current_tick - self.last_event_trigger_tick >= self.config.EVENT_CHECK_INTERVAL:
            if random.random() < self.config.EVENT_BASE_CHANCE:
                event_message = self._trigger_random_event()
                if event_message: tick_events.append(event_message)
            self.last_event_trigger_tick = self.current_tick

        # Agent Needs Update
        newly_dead_this_tick: List[int] = []
        for agent_id, agent in list(self.agents.items()): # Iterate copy for safe modification
            if agent.is_alive:
                try:
                    agent.update_needs()
                except Exception as e:
                    log.error(f"Error during agent {agent.id} update_needs: {e}", exc_info=True)
                if not agent.is_alive:
                    newly_dead_this_tick.append(agent.id)
                    log.info(f"Agent {agent.id} ({agent.name}) died. HP: {agent.health:.1f}.")
                    tick_events.append(f"[DEATH] {agent.name} [{agent.id}] has succumbed.")
        
        for dead_id in newly_dead_this_tick:
            if dead_id not in self.agents_to_remove: # Ensure not already marked
                 # Call kill_agent on the agent object if available, or just add ID
                 agent_obj = self.agents.get(dead_id)
                 if agent_obj: self.kill_agent(agent_obj) # This adds ID to agents_to_remove
                 elif dead_id not in self.agents_to_remove : # Fallback if agent object somehow gone
                      self.agents_to_remove.append(dead_id)


        # Process Agent Removal Buffer
        if self.agents_to_remove:
            for agent_id_to_remove in self.agents_to_remove:
                if agent_id_to_remove in self.agents:
                    dead_agent_obj = self.agents.pop(agent_id_to_remove) # Remove from world's dict
                    self.registry.remove(agent_id_to_remove) # Remove from central registry
                    log.info(f"Agent {dead_agent_obj.name} [{agent_id_to_remove}] fully removed from world.")
                    # Handle item drops, structure ownership changes etc. as in original
            self.agents_to_remove.clear()

        # Process New Agents
        if self.new_agent_buffer:
            for new_agent in self.new_agent_buffer:
                if new_agent.id not in self.agents:
                    self.agents[new_agent.id] = new_agent
                    self.registry.add(new_agent)
                    tick_events.append(f"[BIRTH/SPAWN] {new_agent.name} [{new_agent.id}] has appeared at {new_agent.loc}.")
                else:
                    log.error(f"Failed to add new agent {new_agent.id}, ID already exists in world.agents.")
            self.new_agent_buffer.clear()

        # Respawn Logic
        current_living_agents = len(self.get_living_agents())
        if current_living_agents < self.config.MIN_POP:
            needed = self.config.MIN_POP - current_living_agents
            log.info(f"Population low ({current_living_agents}/{self.config.MIN_POP}). Spawning {needed} new agent(s).")
            for _ in range(needed):
                if len(self.get_living_agents()) >= self.config.MAX_POP: break # Check against max_pop
                name = generate_themed_name()
                # New agents need the config and chroma_client
                agent = AgentState(name, config=self.config, chroma_client=self.chroma_client)
                # Add to new_agent_buffer to be processed at start of next tick or end of this one
                self.new_agent_buffer.append(agent) # Will be added properly in next iteration or above
        return tick_events

    def resolve_attack(self, attacker: 'AgentState', defender: 'AgentState') -> Tuple[float, float]:
        """Resolves combat. Logic from original World.resolve_attack."""
        if not attacker.is_alive or not defender.is_alive or attacker.attack_cooldown > 0:
            return 0.0, 0.0
        try:
            base_damage = getattr(self.config, "BASE_ATTACK_DAMAGE_FACTOR", 10.0)
            
            # Simplified combat logic from original
            attacker_war = attacker.dna["skills"][self.config.SKILLS.index("war")]
            defender_war = defender.dna["skills"][self.config.SKILLS.index("war")]
            
            attack_roll = random.uniform(0.5, 1.0) + attacker_war * 1.5
            defend_roll = random.uniform(0.3, 0.8) + defender_war * 1.2
            
            damage_dealt = 0.0
            if attack_roll > defend_roll:
                damage_dealt = max(1.0, (attack_roll - defend_roll) * base_damage * random.uniform(0.7, 1.3))
            defender.health -= damage_dealt

            damage_received = 0.0 # Simplified: no counter-attack in this stub
            # ... (full counter-attack logic would go here) ...

            attacker.health = max(0.0, attacker.health)
            defender.health = max(0.0, defender.health)

            if attacker.health <= 0: self.kill_agent(attacker)
            if defender.health <= 0: self.kill_agent(defender)
            
            return damage_dealt, damage_received
        except Exception as e:
            log.error(f"Error in resolve_attack between {attacker.id} and {defender.id}: {e}", exc_info=True)
            return 0.0, 0.0

    # --- Code Modification Methods ---
    def safe_get_target_code(self, target_name: str) -> Optional[str]:
        """Retrieves original source code/value using code_utils."""
        from ..core.code_utils import ORIGINAL_SOURCE_CODE # Access the global dict
        if target_name not in self.config.CODE_MODIFICATION_WHITELIST:
            log.warning(f"Attempt to get code for non-whitelisted target: {target_name}")
            return None
        original_code = ORIGINAL_SOURCE_CODE.get(target_name)
        if original_code is None:
            log.error(f"Original source/value for whitelisted target '{target_name}' not found in code_utils.ORIGINAL_SOURCE_CODE.")
        return original_code

    def safe_execute_agent_code(self, agent_id: int, target_name: str, code_string: str) -> Tuple[bool, str]:
        """Validates and queues agent-proposed code."""
        if target_name not in self.config.CODE_MODIFICATION_WHITELIST:
            return False, "Target not allowed for modification."
        
        lines = code_string.strip().splitlines()
        if len(lines) > self.config.MAX_CODE_LINES:
            return False, f"Code exceeds max lines ({len(lines)} > {self.config.MAX_CODE_LINES})."
        if not lines: return False, "Empty code proposed."

        for pattern in self.config.FORBIDDEN_CODE_PATTERNS:
            if pattern in code_string.lower():
                return False, f"Code contains forbidden pattern ('{pattern}')."
        try:
            if target_name.startswith("Config."):
                # Basic eval check for Config values
                allowed_globals = {'__builtins__': {'True': True, 'False': False, 'None': None, 'int': int, 'float': float, 'str': str, 'list': list, 'dict': dict, 'Counter': Counter, 'lambda': lambda:None, 'max':max, 'min':min}}
                eval(code_string, allowed_globals, {})
            else: # For methods, just AST syntax
                ast.parse(code_string)
        except Exception as e:
            return False, f"Validation failed: {type(e).__name__}: {e}"

        self.pending_code_updates.append((target_name, code_string))
        log.info(f"Agent {agent_id} queued code change for '{target_name}'. Will apply next tick.")
        return True, "Code validated and queued for execution."

    def _apply_code_change_unsafe(self, target_name: str, code_string: str) -> Tuple[bool, str]:
        """Applies validated code. UNSAFE. Uses code_utils for restoration logic if needed."""
        # This method now acts as a high-level wrapper. The actual modification
        # and restoration logic is complex and better suited for code_utils.
        # However, the direct modification (setattr, exec) happens here based on target_name.
        # code_utils.restore_specific_original_code would be called if this application fails.
        
        log.warning(f"Applying code change for {target_name}. New code:\n{code_string}")
        from ..config import Config as AppConfig # Actual Config class
        from .agent import AgentState # Actual AgentState class
        # from ..core.code_utils import restore_specific_original_code # For error recovery

        try:
            if target_name.startswith("Config."):
                attr_name = target_name.split('.', 1)[1]
                allowed_globals = {'__builtins__': {'True': True, 'False': False, 'None': None, 'int': int, 'float': float, 'str': str, 'list': list, 'dict': dict, 'Counter': Counter, 'lambda': lambda:None, 'max':max, 'min':min}}
                # If lambda refers to Config, it needs Config in its scope
                lambda_globals = allowed_globals.copy()
                lambda_globals['Config'] = AppConfig # Provide the actual Config class for lambdas
                
                value = eval(code_string, lambda_globals, {})
                
                # Type check (optional, but good for safety)
                # original_value = getattr(AppConfig, attr_name, None)
                # if original_value is not None and not isinstance(value, type(original_value)):
                #    if not (isinstance(value, (int,float)) and isinstance(original_value, (int,float))): # Allow int/float interchange
                #        raise TypeError(f"Type mismatch for {target_name}. Expected {type(original_value)}, got {type(value)}")
                
                setattr(AppConfig, attr_name, value)
                return True, f"Config.{attr_name} updated."

            elif target_name.startswith("AgentState.") or target_name.startswith("World."):
                class_name, method_name = target_name.split('.', 1)
                target_class = None
                if class_name == "AgentState": target_class = AgentState
                elif class_name == "World": target_class = World # self.__class__ could also work if called on World instance
                else: return False, f"Unsupported class '{class_name}' for method update."

                local_exec_namespace = {}
                # Careful with globals for exec. Based on original World._apply_code_change_unsafe
                safe_globals = {
                     '__builtins__': { 'min': min, 'max': max, 'random': random, 'np': np, 'Counter': Counter, 'log': log, 'True':True, 'False':False, 'None':None, 'int':int, 'float':float, 'str':str, 'list':list, 'dict':dict, 'getattr': getattr, 'isinstance': isinstance, 'callable': callable, 'type': type, 'any':any},
                     'Config': AppConfig, 'AgentState': AgentState, 'World': self.__class__, # Pass World class
                     'Structure': None, 'LocationNode': None, # Need to import these if methods use them
                     'random': random, 'np': np, 'log': log,
                     # For HarmCategory etc. if modifying LLM parts of AgentState
                     # 'HarmCategory': HarmCategory, 'HarmBlockThreshold': HarmBlockThreshold 
                }
                from .world_elements import Structure, LocationNode # Make them available
                safe_globals['Structure'] = Structure
                safe_globals['LocationNode'] = LocationNode
                safe_globals[class_name] = target_class # For super()

                exec(code_string, safe_globals, local_exec_namespace)
                new_method = local_exec_namespace.get(method_name)
                if not new_method or not callable(new_method):
                    return False, f"Code did not define function '{method_name}' correctly."
                setattr(target_class, method_name, new_method)
                return True, f"{target_name} method updated."
            else:
                return False, "Unsupported target type for code change."
        except Exception as e:
            log.error(f"CRITICAL ERROR applying code change for {target_name}: {e}", exc_info=True)
            # Attempt to restore original code for this specific target
            # This requires access to the original code_utils functions and the module references
            # from ..core.code_utils import restore_specific_original_code, ORIGINAL_SOURCE_CODE
            # if restore_specific_original_code(target_name, AppConfig, AgentState, self.__class__): # Pass class types
            #     return False, f"Execution Error: {type(e).__name__}. ORIGINAL CODE RESTORED for {target_name}."
            # else:
            #     return False, f"Execution Error: {type(e).__name__}. FAILED TO RESTORE ORIGINAL CODE for {target_name}."
            return False, f"Execution Error: {type(e).__name__}. Restoration attempt would be here."


    def to_dict(self) -> Dict:
        # Active code modifications are not saved; simulation reverts to original code on load.
        return {
            "current_tick": self.current_tick, "day": self.day, "season": self.season,
            "next_agent_id": getattr(AgentState, '_next_id', 1), # Get from AgentState class
            "next_structure_id": getattr(Structure, '_next_id', 1), # Get from Structure class
            "agents": {aid: agent.to_dict() for aid, agent in self.agents.items()},
            # Structures are part of locations, but save a master list too for easier lookup if needed
            # "structures": {sid: struct.to_dict() for sid, struct in self.structures.items()},
            "locations": {lname: loc.to_dict() for lname, loc in self.locations.items()},
            "last_event_trigger_tick": self.last_event_trigger_tick,
        }

    @classmethod
    def from_file(cls, filepath: Path, config: 'Config', registry: 'Registry', chroma_client: Optional['PersistentClient']) -> Optional['World']:
        if not filepath.exists():
            log.error(f"Save file not found: {filepath}")
            return None
        try:
            with gzip.open(filepath, "rt", encoding='utf-8') as f:
                data = json.load(f)
            
            # On load, the simulation uses the current codebase's original functions.
            # Any runtime code modifications from the saved session are not preserved.
            # `store_original_code` should be called at app startup.
            # `restore_all_original_code` could be called here if we wanted to ensure a clean slate
            # from the *current* codebase, but it's generally not needed if saves don't store modified code.

            world_instance = cls(config=config, registry=registry, chroma_client=chroma_client, load_data=data)
            
            # Ensure next IDs from save file are respected and set on the classes
            from .agent import AgentState
            from .world_elements import Structure
            AgentState._next_id = data.get("next_agent_id", AgentState._next_id)
            Structure._next_id = data.get("next_structure_id", Structure._next_id)
            
            return world_instance
        except Exception as e:
            log.error(f"Failed to load world from {filepath}: {e}", exc_info=True)
            return None

    def save_to_file(self, filepath: Path):
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            world_dict = self.to_dict()
            # Ensure agents marked for removal are not saved
            world_dict["agents"] = {
                aid_str: adata for aid_str, adata in world_dict["agents"].items()
                if int(aid_str) not in self.agents_to_remove
            }
            with gzip.open(filepath, "wt", encoding='utf-8') as f:
                json.dump(world_dict, f, indent=2)
            log.info(f"World state saved successfully to {filepath}")
        except Exception as e:
            log.error(f"Failed to save world to {filepath}: {e}", exc_info=True)


if __name__ == '__main__':
    log.info("Testing World class...")
    # This would require substantial mocking of Config, Registry, AgentState, LocationNode, ChromaClient
    # For a simple test:
    mock_config_instance = MockConfigForWorld()
    
    class MockRegistryForWorld:
        def add(self, agent): pass
        def remove(self, agent_id): pass
        def clear(self): pass

    mock_registry_instance = MockRegistryForWorld()

    world = World(config=mock_config_instance, registry=mock_registry_instance)
    log.info(f"World initialized with {len(world.locations)} locations.")
    log.info(f"Current tick: {world.current_tick}")
    
    # Simulate a few ticks
    # for i in range(5):
    #     events = world.tick_update()
    #     log.info(f"Tick {world.current_tick} events: {events}")

    # Test save/load (very basic, needs file IO)
    # test_save_path = Path("./test_world_save.json.gz")
    # world.save_to_file(test_save_path)
    # if test_save_path.exists():
    #     log.info(f"World saved to {test_save_path}")
    #     loaded_world = World.from_file(test_save_path, mock_config_instance, mock_registry_instance)
    #     if loaded_world:
    #         log.info(f"World loaded, tick: {loaded_world.current_tick}")
    #         assert loaded_world.current_tick == world.current_tick
    #     test_save_path.unlink() # Clean up
