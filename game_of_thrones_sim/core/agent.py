import asyncio
import random
import string
from collections import deque, Counter
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

# Attempt to import google.generativeai types for LLM interaction parts
try:
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    SDK_AVAILABLE = True
except ModuleNotFoundError:
    SDK_AVAILABLE = False

# Forward declarations for type hinting to avoid circular imports
if TYPE_CHECKING:
    from .world import World
    from .registry import Registry
    from ..config import Config # Assuming config.py is one level up from core
    from chromadb import PersistentClient # For type hinting chroma_client

# Placeholder for logger and config
class PrintLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg, exc_info=False): print(f"ERROR: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")
    def exception(self, msg): print(f"EXCEPTION: {msg}")

log = PrintLogger()

class MockConfigForAgent: # Used until actual Config is properly importable by AgentState
    MAX_HEALTH = 100.0
    MAX_HUNGER = 20.0
    MIN_MORALE = 0.0
    MAX_MORALE = 100.0
    LOCATIONS = ["DefaultLocation1", "DefaultLocation2"]
    TRAITS = ["trait1", "trait2"]
    SKILLS = ["skill1", "skill2"]
    FORBIDDEN = {"forbidden_word"}
    RAG_TOPK = 3
    API_TIMEOUT = 20.0
    SYSTEM_PROMPT = "You are a test agent."
    MAX_AGE = 300
    HUNGER_PER_TICK = 0.5
    HEALTH_DAMAGE_FROM_HUNGER = 1.0
    MORALE_DECAY_PER_TICK = 0.1
    AGE_DEATH_CHANCE_PER_TICK = lambda age: 0.001 if age > 280 else 0
    COMBAT_MORALE_LOSS_ON_ATTACK = 5
    COMBAT_MORALE_LOSS_ON_KILL = 10
    COMBAT_MORALE_GAIN_ON_VICTORY = 5
    COMBAT_COOLDOWN = 2
    POST_KILL_ATTACK_COOLDOWN = 5
    CRAFTING_RECIPES = {"TestItem": {"Wood": 1}}
    STRUCTURE_BUILD_VERBS = {"TestShelter"}
    CONSUMABLE_ITEMS = {"TestFood": {"hunger_relief": 5}}
    ALLOW_CODE_MODIFICATION = False
    CODE_MODIFICATION_WHITELIST = set()
    MAX_CODE_LINES = 10
    BASE_HEAL_AMOUNT = 10.0
    SKILL_DECAY = 0.001 # Used as mutation std in original crossover
    TRAIT_MUT_STD = 0.04


class AgentState:
    _next_id = 1

    def __init__(self,
                 name: str,
                 config: 'Config', # Pass the config object
                 dna: Optional[Dict[str, np.ndarray]] = None,
                 loc: Optional[str] = None,
                 agent_id: Optional[int] = None,
                 age: int = 0,
                 health: float = -1.0, # Default to -1 to use config's MAX_HEALTH
                 hunger: float = 0.0,
                 morale: float = 70.0,
                 inventory: Optional[Counter] = None,
                 profession: str = "Peasant",
                 role: str = "Commoner",
                 relations: Optional[Dict[int, int]] = None,
                 chroma_client: Optional['PersistentClient'] = None,
                 attack_cooldown: int = 0):
        global log
        try:
            from ..utils.logger import log as main_log
            log = main_log
        except ImportError:
            pass # Keep print logger

        self.config = config # Store config instance

        if agent_id is not None:
            self.id = agent_id
            if agent_id >= AgentState._next_id:
                AgentState._next_id = agent_id + 1
        else:
            self.id = AgentState._next_id
            AgentState._next_id += 1

        self.name = name
        
        # Initialize DNA using helper function and config lists
        if dna:
            self.dna = dna
        else:
            from ..utils.helpers import random_dna as random_dna_helper
            self.dna = random_dna_helper(self.config.TRAITS, self.config.SKILLS)

        self.loc = loc or random.choice(list(self.config.LOCATIONS))

        # Core State
        self.age = age
        self.health = float(self.config.MAX_HEALTH if health < 0 else health)
        self.hunger = float(hunger)
        self.morale = float(np.clip(morale, self.config.MIN_MORALE, self.config.MAX_MORALE))
        self.inventory = inventory or Counter()
        self.profession = profession
        self.role = role
        self.relations = relations or {}  # agent_id -> relationship score (-100 to 100)

        # Internal / Transient State
        self.memory = deque(maxlen=25)
        self.chroma_client = chroma_client # This is the chromadb client instance
        self.attack_cooldown = attack_cooldown

        # Code Modification State (transient, not saved)
        self.requested_code_target: Optional[str] = None
        self.requested_code_content: Optional[str] = None # The actual code string received
        self.last_code_proposal_result: Optional[str] = None

    @property
    def is_alive(self) -> bool:
        return self.health > 0

    def update_needs(self):
        """Update hunger, health, morale, age per tick."""
        if not self.is_alive:
            return

        self.age += 1
        
        # Access config values through self.config
        hunger_per_tick = getattr(self.config, "HUNGER_PER_TICK", 0.7)
        max_hunger = getattr(self.config, "MAX_HUNGER", 20.0)
        dmg_from_hunger = getattr(self.config, "HEALTH_DAMAGE_FROM_HUNGER", 1.5)
        morale_decay = getattr(self.config, "MORALE_DECAY_PER_TICK", 0.1)
        max_age_val = getattr(self.config, "MAX_AGE", 320) # Renamed to avoid conflict
        
        # Get the age death chance function from config
        # It's a lambda, so it should be callable
        age_death_func = getattr(self.config, "AGE_DEATH_CHANCE_PER_TICK", lambda a: 0)


        self.hunger += hunger_per_tick
        if self.hunger > max_hunger:
            damage_taken = dmg_from_hunger * ((self.hunger - max_hunger) / max_hunger + 1)
            self.health -= damage_taken
            # log.debug(f"Agent {self.id} takes {damage_taken:.2f} damage from hunger (Hunger: {self.hunger:.1f}). HP: {self.health:.1f}")

        self.morale -= morale_decay
        self.morale = max(self.config.MIN_MORALE, self.morale)

        if self.age > max_age_val * 0.85:
            death_chance = age_death_func(self.age) # Call the lambda
            if random.random() < death_chance:
                log.info(f"Agent {self.id} ({self.name}) dies of old age at {self.age} ticks (chance: {death_chance:.5f}).")
                self.health = 0

        self.health = max(0.0, self.health)

        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

    def modify_relation(self, other_id: int, amount: int):
        if other_id == self.id: return
        current_score = self.relations.get(other_id, 0)
        new_score = np.clip(current_score + amount, -100, 100)
        if new_score != current_score:
            self.relations[other_id] = int(new_score)

    def get_relation_summary(self, registry: 'Registry') -> Tuple[str, str]:
        if not self.relations: return "none", "none"
        living_relations = {oid: score for oid, score in self.relations.items() if registry.is_agent_alive(oid)}
        if not living_relations: return "none", "none"

        sorted_relations = sorted(living_relations.items(), key=lambda item: item[1], reverse=True)
        friends = [f"{registry.get_agent_name(oid)}[{oid}]({score:+})" for oid, score in sorted_relations if score > 30][:3]
        enemies = [f"{registry.get_agent_name(oid)}[{oid}]({score:+})" for oid, score in sorted_relations if score < -30][:3]
        return (", ".join(friends) or "none"), (", ".join(enemies) or "none")

    def get_inventory_summary(self) -> str:
        if not self.inventory: return "empty"
        return ", ".join(f"{item}:{count}" for item, count in self.inventory.items())[:100]

    def oov(self, txt: str) -> bool:
        """Check for Out-Of-Vocabulary (forbidden) words using Config.FORBIDDEN."""
        from ..utils.helpers import oov_check # Use the helper
        return oov_check(txt, self.config.FORBIDDEN)

    def recent_verbs(self, n: int = 5) -> List[str]:
        verbs = []
        valid_actions = {
            "move", "propose_reproduction", "reflect", "speak", "gather", "craft",
            "store", "take", "attack", "heal", "assign_role", "consume",
            "request_code", "propose_code_change"
        } # From original AgentState.decide
        for entry in reversed(list(self.memory)):
            if isinstance(entry, str) and entry.startswith("action:"):
                parts = entry.split(':', 2)
                if len(parts) > 1:
                    verb_candidate = parts[1].strip().lower()
                    if verb_candidate in valid_actions:
                        verbs.append(verb_candidate)
                        if len(verbs) >= n: break
        return verbs
    
    def recent_built_items(self, n: int = 3) -> List[str]:
        built = []
        for entry in reversed(list(self.memory)):
             if isinstance(entry, str) and entry.startswith("event:built_"):
                 item_name = entry.split('_', 1)[1]
                 built.append(item_name)
                 if len(built) >= n: break
        return built

    async def rag_query(self, prompt: str) -> str:
        """Perform RAG query using the llm.rag module."""
        if not self.chroma_client: # Chroma client is passed during AgentState init
            # log.warning(f"Agent {self.id}: Chroma client not available for RAG.") # Can be noisy
            return ""
        try:
            from ..llm.rag import query_rag # Use the centralized RAG query function
            return await query_rag(prompt, self.config.RAG_TOPK)
        except ImportError:
            log.error("Failed to import query_rag from ..llm.rag module.")
            return ""
        except Exception as e:
            log.warning(f"Agent {self.id} RAG query failed: {e}", exc_info=False)
            return ""

    async def llm_call(self, prompt_content: str, lore_content: str) -> str:
        """Perform LLM call using the llm.client module."""
        from ..llm.client import make_llm_call # Use the centralized LLM call function

        # Construct prompt parts as expected by make_llm_call
        # This structure is based on the original AgentState.llm method
        combined_prompt_parts = [self.config.SYSTEM_PROMPT]
        if lore_content:
            combined_prompt_parts.append(f"\n--- Relevant Lore Snippets ---\n{lore_content}\n---")
        combined_prompt_parts.append("\n--- Your Task ---")
        combined_prompt_parts.append("Based on your persona, state, memory, and the provided context/lore, decide on exactly ONE action from the 'Available Action Formats' list.")
        combined_prompt_parts.append("Your response MUST start *immediately* with the chosen action line (e.g., `speak Hello there` or `gather Wood`).")
        combined_prompt_parts.append("Do NOT include explanations, prefixes like 'ACTION:', or any text *before* the action line.")
        combined_prompt_parts.append("\n--- Current Situation & Action Prompt ---")
        combined_prompt_parts.append(prompt_content) # Main prompt with state, memory, available actions etc.

        model_params = {
            "temperature": 0.75, # From original AgentState.llm
            "max_output_tokens": 200 # From original AgentState.llm (was 2000, reduced to 200)
        }
        
        # Safety settings from original AgentState.llm
        # Ensure SDK_AVAILABLE is checked before using HarmCategory etc.
        safety_params = None
        if SDK_AVAILABLE:
            safety_params = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }

        raw_text_output = await make_llm_call(
            prompt_parts=combined_prompt_parts,
            model_config=model_params,
            safety_settings_dict=safety_params,
            api_timeout=self.config.API_TIMEOUT
        )

        # Post-processing from original AgentState.llm
        if not raw_text_output or raw_text_output.startswith("["): # Handle error codes or empty response
            if not raw_text_output: raw_text_output="[error: empty]"
            log.warning(f"Agent {self.id} LLM call resulted in error/empty: {raw_text_output}")
            return raw_text_output

        action_line = raw_text_output.splitlines()[0].strip()
        if not action_line:
            log.warning(f"Agent {self.id} LLM output started with empty line. Raw: {raw_text_output[:100]}")
            return "[error: format]"

        verb_test = action_line.split(maxsplit=1)[0].lower()
        if verb_test in ["speak", "reflect"]:
            args_test = action_line.split(maxsplit=1)[1] if len(action_line.split(maxsplit=1)) > 1 else ""
            if self.oov(args_test):
                log.warning(f"Agent {self.id} generated OOV content in {verb_test}: {args_test}")
                return "[oov]"
        
        log.debug(f"Agent {self.id} proposed action: {action_line}")
        return action_line

    async def decide(self, world: 'World', registry: 'Registry') -> str:
        """LLM call for agent decision-making."""
        if not self.is_alive: return "[error: dead]"

        current_loc_node = world.locations.get(self.loc)
        if not current_loc_node:
             log.error(f"Agent {self.id} in invalid location {self.loc}. Cannot decide.")
             return "[error: invalid_location]"

        others_here = [a for a in world.get_agents_at_loc(self.loc) if a.id != self.id]
        structures_here = list(current_loc_node.structures.values())
        # my_structures_here = [s for s in structures_here if s.owner_id == self.id] # Not used in original prompt string
        
        population_room = max(0, self.config.MAX_POP - len(world.get_living_agents()))
        can_reproduce = population_room > 0 and len(others_here) > 0
        nearby_agent_ids = [o.id for o in others_here]
        friends_summary, enemies_summary = self.get_relation_summary(registry)
        lore_skill = self.dna["skills"][self.config.SKILLS.index("lore")]

        # Build Action Formats String (copied and adapted from original)
        location_list = "|".join(self.config.LOCATIONS)
        resource_list = "|".join(self.config.RESOURCE_REGEN_RATE.keys())
        craftable_items = "|".join(self.config.CRAFTING_RECIPES.keys())
        consumable_items_list = "|".join(self.config.CONSUMABLE_ITEMS.keys()) # Renamed to avoid conflict
        roles_list = "Guard|Merchant|Farmer|Mason|Healer|Commander|Leader"
        nearby_ids_str = "|".join(map(str, nearby_agent_ids)) if nearby_agent_ids else "AGENT_ID"
        structure_ids_str = "|".join(map(str, (s.id for s in structures_here))) if structures_here else "STRUCTURE_ID"

        action_formats_list = [
            f"`move <{location_list}>`",
            "`speak <message_to_people_here>`",
            "`reflect <internal_thought_or_plan>`",
            f"`gather <{resource_list}>`",
            f"`craft <{craftable_items}>`",
            f"`consume <{consumable_items_list}>`",
            f"`store <item_name> <{structure_ids_str}>`",
            f"`take <item_name> <{structure_ids_str}>`",
            f"`attack <{nearby_ids_str}>`",
            f"`heal <{nearby_ids_str}>`",
            f"`propose_reproduction <{nearby_ids_str}>`",
            f"`assign_role <{nearby_ids_str}> <{roles_list}>`"
        ]

        if self.config.ALLOW_CODE_MODIFICATION and lore_skill > 0.7:
            allowed_targets = "|".join(self.config.CODE_MODIFICATION_WHITELIST)
            action_formats_list.append(f"`request_code <{allowed_targets}>` # Ask to see simulation rules code")
            if self.requested_code_target and self.requested_code_content: # Check if content is present
                 action_formats_list.append(f"`propose_code_change {self.requested_code_target} <python_code_string>` # Suggest changes (Max {self.config.MAX_CODE_LINES} lines)")
            else:
                 action_formats_list.append(f"`# propose_code_change <target> <code>` (Requires using request_code first and receiving code)")
        else:
            action_formats_list.append("`# request_code <target>` (Requires high lore skill & feature enabled)")
            action_formats_list.append("`# propose_code_change <target> <code>` (Requires high lore skill & feature enabled)")

        final_action_formats = []
        all_verbs_available = set()

        for line in action_formats_list:
            verb = line.split(maxsplit=1)[0].lower().strip('`# ')
            is_available = True
            reason = ""
            if line.strip().startswith("`#"):
                 is_available = False
                 if "(" in line and ")" in line: reason = line[line.find("(")+1:line.find(")")]
                 final_action_formats.append(line)
                 continue

            if verb == "propose_reproduction":
                if not can_reproduce: is_available = False; reason = "Pop full or no others here"
                elif not nearby_agent_ids: is_available = False; reason = "No valid partner nearby"
            elif verb in ["attack", "heal", "assign_role"]:
                if not nearby_agent_ids: is_available = False; reason = "No living agents nearby"
                elif verb == "attack" and self.attack_cooldown > 0: is_available = False; reason = f"Cooldown: {self.attack_cooldown}"
                elif verb == "assign_role" and self.role not in ["Lord", "Leader", "Commander"]: is_available = False; reason = f"Role ({self.role}) lacks authority"
            elif verb in ["store", "take"]:
                if not structures_here: is_available = False; reason = "No structures nearby"
            elif verb == "craft":
                can_craft_anything = False
                for item, recipe in self.config.CRAFTING_RECIPES.items():
                    if all(self.inventory.get(ing, 0) >= recipe[ing] for ing in recipe):
                        if item == "Iron_Sword" and not any(s.name == "Forge" for s in structures_here): continue
                        can_craft_anything = True
                        if item in self.config.STRUCTURE_BUILD_VERBS:
                            recently_built = self.recent_built_items()
                            owns_same_type = any(s.name == item and s.owner_id == self.id for s in structures_here)
                            if item in recently_built or owns_same_type:
                                is_available = False; reason = f"Already built {item} recently or own one here"
                                break
                if not can_craft_anything and is_available:
                    is_available = False; reason = "Lack ingredients/tools for any recipe"
            elif verb == "consume":
                 if not any(item in self.inventory for item in self.config.CONSUMABLE_ITEMS):
                     is_available = False; reason = "No consumable items in inventory"
            
            if is_available:
                final_action_formats.append(line)
                all_verbs_available.add(verb)
            else:
                final_action_formats.append(f"`# {line.strip('`')} ({reason})`")
        action_formats_string = "\n".join(final_action_formats)

        # Constraints and Encouragement (simplified for brevity, can be expanded as original)
        constraint = ""
        if self.hunger >= self.config.MAX_HUNGER * 0.7:
            constraint += f"\nCONSTRAINT: Getting hungry ({self.hunger:.0f}). Should prioritize food."
        if self.health < self.config.MAX_HEALTH * 0.5:
            constraint += f"\nCONSTRAINT: Health low ({self.health:.0f}). Consider safety/healing."

        encouragement = "\n--- Encouragement & Context ---"
        if self.config.ALLOW_CODE_MODIFICATION and lore_skill > 0.8:
             encouragement += "\nYour high lore skill allows you to inspect and potentially modify the world's rules."
        if self.requested_code_target and self.requested_code_content: # Check if content is present
             encouragement += f"\nYou have received code for '{self.requested_code_target}'. You can analyze or propose changes."
        if self.last_code_proposal_result:
             encouragement += f"\nResult of last code proposal: {self.last_code_proposal_result}"
             self.last_code_proposal_result = None


        from ..utils.helpers import vec_str as vec_str_helper # Use helper
        code_section = ""
        if self.requested_code_target and self.requested_code_content:
            code_section = f"\n--- Code Content for {self.requested_code_target} ---\n```python\n{self.requested_code_content}\n```\n---"
            # self.requested_code_content = None # Clear after showing, or keep until proposal? Original cleared.

        prompt_str = (
            f"--- Your State & Surroundings ---\n"
            f"You are {self.name} (ID:{self.id}), a {self.role} ({self.profession}). Age: {self.age}, Health: {self.health:.0f}/{self.config.MAX_HEALTH}, Hunger: {self.hunger:.0f}, Morale: {self.morale:.0f}.\n"
            f"Location: {self.loc}.\n"
            f"Resources here: {current_loc_node.get_resource_summary()}.\n"
            f"Structures here: {current_loc_node.get_structure_summary()}.\n"
            f"People here: {', '.join(f'{o.name}[{o.id}]' for o in others_here) or 'none'}.\n"
            f"Inventory: {self.get_inventory_summary()}.\n"
            f"Skills: {vec_str_helper(self.dna['skills'], self.config.SKILLS)}.\n"
            f"Friends: {friends_summary} | Enemies: {enemies_summary}.\n"
            f"Memory (Recent Events/Thoughts): {'; '.join(self.memory) or 'none'}.\n"
            f"{code_section}"
            f"{constraint}"
            f"{encouragement}"
            f"\n--- Available Action Formats ---\n{action_formats_string}"
            f"\n--- Your ONE Action (MUST start on the next line, NO commentary before it) ---"
        )

        rag_query_text = f"Action decision context for {self.name} at {self.loc}. Needs: H={self.health:.0f}, G={self.hunger:.0f}, M={self.morale:.0f}. Role: {self.role}. Skills: {vec_str_helper(self.dna['skills'], self.config.SKILLS)}. Memory: {'; '.join(self.memory)}"
        lore = await self.rag_query(rag_query_text)
        return await self.llm_call(prompt_str, lore)

    async def act(self, world: 'World', registry: 'Registry') -> str:
        """Parse LLM decision and execute the corresponding action."""
        if not self.is_alive: return f"[{self.id}] {self.name} is dead."

        # Use a semaphore from the world or a global one if LLM calls are rate-limited
        # Assuming world.llm_semaphore exists as in original code
        async with world.llm_semaphore: # Ensure semaphore is passed or accessible
            raw_decision = await self.decide(world, registry)

        if raw_decision.startswith("[") and raw_decision.endswith("]"):
            verb, arg_str = raw_decision, ""
        else:
            parts = raw_decision.split(maxsplit=1)
            verb = parts[0].lower().strip("` ")
            arg_str = parts[1].strip() if len(parts) > 1 else ""
        
        ev = "" # Event description
        log.debug(f"Agent {self.id} ({self.name}) attempting action: {verb} {arg_str[:50]}")
        current_loc_node = world.locations.get(self.loc)

        # Action execution logic (simplified for brevity, needs full implementation from original)
        # This part is very long in the original script. I will summarize the structure.
        # For a full refactor, each action would ideally be its own method or handled by a dispatcher.
        try:
            if verb == "speak":
                msg = arg_str[:150].replace("\n", " ")
                ev = f"says \"{msg}\"" if msg else "mutters."
            elif verb == "move":
                if arg_str in self.config.LOCATIONS and arg_str != self.loc:
                    old_loc = self.loc; self.loc = arg_str; ev = f"travels from {old_loc} to {arg_str}."
                else: ev = f"tries to move to '{arg_str}' but cannot."
            elif verb == "reflect":
                ev = f"reflects internally ('{arg_str[:50]}...')."
            # ... (other verbs: gather, craft, consume, store, take, attack, heal, propose_reproduction, assign_role) ...
            # ... (code modification verbs: request_code, propose_code_change) ...
            elif verb == "gather":
                resource = arg_str.strip().capitalize()
                if resource in self.config.RESOURCE_REGEN_RATE and current_loc_node:
                    if current_loc_node.resources.get(resource, 0.0) >= 1.0:
                        current_loc_node.resources[resource] -= 1.0
                        self.inventory[resource] += 1
                        ev = f"gathers 1 {resource}."
                    else: ev = f"tries to gather {resource}, but finds none."
                else: ev = f"tries to gather unknown resource '{resource}'."
            # Add more verb handlers here based on the original script's logic
            # For example, for 'attack':
            elif verb == "attack":
                if self.attack_cooldown > 0:
                    ev = f"tries to attack, but is recovering (Cooldown: {self.attack_cooldown})."
                else:
                    try:
                        target_id = int(arg_str.strip())
                        target = registry.get_agent(target_id)
                        if target and target.is_alive and target.loc == self.loc and target.id != self.id:
                            dmg_done, dmg_received = world.resolve_attack(self, target) # World handles combat
                            ev = f"attacks {target.name}! Deals {dmg_done:.1f}, receives {dmg_received:.1f}."
                            # Update relations, morale, cooldowns as in original
                            self.attack_cooldown = self.config.COMBAT_COOLDOWN
                            if not target.is_alive:
                                ev += f" {target.name} has been slain!"
                                world.kill_agent(target) # Mark for removal
                                self.attack_cooldown = self.config.POST_KILL_ATTACK_COOLDOWN
                        else:
                            ev = f"tries to attack non-existent or invalid target ID {arg_str}."
                    except ValueError:
                        ev = f"uses attack with invalid target ID '{arg_str}'."
            
            elif verb == "request_code":
                target_name = arg_str.strip()
                if not self.config.ALLOW_CODE_MODIFICATION:
                    ev = "thinks about changing rules, but feature is disabled."
                elif self.dna["skills"][self.config.SKILLS.index("lore")] < 0.7:
                     ev = "lacks lore skill to inspect world code."
                elif target_name not in self.config.CODE_MODIFICATION_WHITELIST:
                    ev = f"tries to request code for forbidden target '{target_name}'."
                else:
                    code_content = world.safe_get_target_code(target_name) # world method
                    if code_content:
                        self.requested_code_target = target_name
                        self.requested_code_content = code_content
                        ev = f"requests and receives source code for '{target_name}'."
                    else:
                        ev = f"tries to request code for '{target_name}', but it could not be retrieved."
            
            elif verb == "propose_code_change":
                # Simplified - full logic in original
                if not self.config.ALLOW_CODE_MODIFICATION or self.dna["skills"][self.config.SKILLS.index("lore")] < 0.7:
                    ev = "cannot propose code changes (disabled or lacks skill)."
                else:
                    parts = arg_str.split(maxsplit=1)
                    if len(parts) >= 2:
                        target_name, code_str = parts[0].strip(), parts[1].strip().strip('`')
                        if target_name == self.requested_code_target and target_name in self.config.CODE_MODIFICATION_WHITELIST:
                            success, message = world.safe_execute_agent_code(self.id, target_name, code_str) # world method
                            self.last_code_proposal_result = message
                            ev = f"proposes code change for '{target_name}'. Result: {message}"
                        else:
                            ev = f"cannot propose code for '{target_name}' (mismatch or not allowed)."
                            self.last_code_proposal_result = "Target mismatch or not whitelisted"
                    else:
                        ev = "proposes code change with invalid format."
                        self.last_code_proposal_result = "Invalid format"


            elif verb.startswith("[") and verb.endswith("]"): # LLM error codes
                 ev = f"experiences a cognitive error ({verb})."
            else:
                ev = f"tries an unrecognized action '{verb}'."
        except Exception as e:
            log.error(f"Error during {self.name}.act ({verb} '{arg_str[:50]}'): {type(e).__name__}: {e}", exc_info=True)
            ev = f"encounters an internal error during action!"

        if not ev: ev = "lingers uncertainly."
        
        # Memory update
        all_valid_verbs = { "move", "propose_reproduction", "reflect", "speak", "gather", "craft", "store", "take", "attack", "heal", "assign_role", "consume", "request_code", "propose_code_change"}
        if verb in all_valid_verbs:
             self.memory.append(f"action:{verb}")
             if verb == "propose_code_change": self.requested_code_target = None # Clear after proposal
        elif verb.startswith("["):
             self.memory.append(f"action_fail:{verb}")
        self.memory.append(f"event:{ev[:70]}")

        return f"[{self.id}] {self.name} @ {self.loc} (H:{self.health:.1f}|G:{self.hunger:.1f}|M:{self.morale:.1f}): {ev}"

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "name": self.name, "loc": self.loc,
            "dna": {"traits": self.dna["traits"].tolist(), "skills": self.dna["skills"].tolist()},
            "age": self.age, "health": self.health, "hunger": self.hunger, "morale": self.morale,
            "inventory": dict(self.inventory), "profession": self.profession, "role": self.role,
            "relations": self.relations,
            "memory": list(self.memory),
            "attack_cooldown": self.attack_cooldown,
            # Transient states like requested_code_target/content are not saved
        }

    @classmethod
    def from_dict(cls, data: Dict, config: 'Config', chroma_client: Optional['PersistentClient']) -> 'AgentState':
        # Ensure _next_id is correctly managed
        current_max_id = data.get("id", 0)
        if current_max_id >= cls._next_id:
            cls._next_id = current_max_id + 1
        
        from ..utils.helpers import random_dna as random_dna_helper # For default DNA

        dna_data = data.get("dna", {})
        try:
            traits = np.array(dna_data.get("traits", [])).astype(np.float32)
            if traits.shape != (len(config.TRAITS),): traits = random_dna_helper(config.TRAITS, [])["traits"] # Partial random
            skills = np.array(dna_data.get("skills", [])).astype(np.float32)
            if skills.shape != (len(config.SKILLS),): skills = random_dna_helper([], config.SKILLS)["skills"] # Partial random
            dna = {"traits": traits, "skills": skills}
        except Exception:
            dna = random_dna_helper(config.TRAITS, config.SKILLS)


        agent = cls(
            agent_id=data["id"], name=data["name"], config=config, dna=dna, loc=data["loc"],
            age=data.get("age", 0),
            health=data.get("health", config.MAX_HEALTH), # Use config for default health
            hunger=data.get("hunger", 0.0),
            morale=data.get("morale", 70.0),
            inventory=Counter(data.get("inventory", {})),
            profession=data.get("profession", "Peasant"), role=data.get("role", "Commoner"),
            relations=data.get("relations", {}),
            chroma_client=chroma_client,
            attack_cooldown=data.get("attack_cooldown", 0)
        )
        agent.memory = deque(data.get("memory", []), maxlen=25)
        return agent


if __name__ == '__main__':
    # Example Usage (requires more setup for full test)
    log.info("Testing AgentState...")
    
    # Mock dependencies for testing
    mock_config_instance = MockConfigForAgent()

    class MockChromaClient: pass
    mock_chroma = MockChromaClient()

    class MockWorld:
        llm_semaphore = asyncio.Semaphore(1)
        locations = {"DefaultLocation1": object()} # Mock location
        def get_agents_at_loc(self, loc): return []
        def get_living_agents(self): return []
        def resolve_attack(self, attacker, defender): return 0,0
        def kill_agent(self, agent): pass
        def safe_get_target_code(self, target): return None
        def safe_execute_agent_code(self, aid, t, c): return False, "Not implemented"


    class MockRegistry:
        def is_agent_alive(self, aid): return True
        def get_agent_name(self, aid): return f"Agent{aid}"
        def get_agent(self, aid): return None

    mock_world_instance = MockWorld()
    mock_registry_instance = MockRegistry()

    agent = AgentState(name="TestAgent", config=mock_config_instance, chroma_client=mock_chroma)
    log.info(f"Created agent: {agent.name} ID: {agent.id} Loc: {agent.loc} HP: {agent.health}")
    agent.update_needs()
    log.info(f"After needs update: HP: {agent.health}, Hunger: {agent.hunger}")

    # To test async methods:
    async def run_agent_test():
        # decision = await agent.decide(mock_world_instance, mock_registry_instance)
        # log.info(f"Agent decision: {decision}") # This would make an LLM call
        
        # Simulate an action based on a predefined decision for testing act() structure
        # For a real test of act(), you'd need a mock LLM or a simple decision.
        # Here, we'll just call act with a simple, known verb.
        # To do this properly, the act() method needs to be filled out more.
        # For now, we can test its basic structure.
        
        # Let's assume a simple action like 'reflect'
        # The 'act' method calls 'decide', so we'd need to mock 'decide' or provide a simple path.
        # For now, this test is more about class structure than full async execution.
        log.info("Async test part would require more mocking or a running event loop.")

    # asyncio.run(run_agent_test()) # Uncomment to run async tests if event loop is available

    agent_dict_data = agent.to_dict()
    log.info(f"Agent to_dict: {agent_dict_data}")
    loaded_agent = AgentState.from_dict(agent_dict_data, mock_config_instance, mock_chroma)
    log.info(f"Loaded agent: {loaded_agent.name} ID: {loaded_agent.id}")
    assert agent.id == loaded_agent.id
