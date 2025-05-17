import os
import sys
import time
import json
import random
import secrets
import logging
import string
import gzip
import argparse
import asyncio
from collections import deque, Counter
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import queue
import threading
import inspect # For agent code modification
import ast     # For agent code validation

# --- GUI Imports ---
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import simpledialog, messagebox # Added messagebox for confirmations
# try:
#     from tkinterdnd2 import DND_FILES, TkinterDnD # Optional DND
#     HAS_DND = True
# except ImportError:
#     HAS_DND = False
#     print("tkinterdnd2 not found. Drag-and-drop features disabled.")

# --- Simulation Core Imports ---
import numpy as np
# import openai # Not used directly for Gemini
import chromadb

# ①  Detect which SDK is installed (Using google-genai)
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    from google.generativeai.types import BlockedPromptException
    from google.api_core import exceptions as gapi_exc # For specific API errors
    import google.api_core # **** ADD THIS LINE ****
    SDK = "new"
    log = logging.getLogger("got-sim-v3") # Update logger name
    log.info("Using google-genai SDK.")
except ModuleNotFoundError:
    print("ERROR: google-generativeai SDK not found. Please install it: pip install google-generativeai")
    sys.exit(1)


# ──────────────────────────── CONFIG ─────────────────────────────

class Config:
    # --- Core Simulation ---
    MAX_POP         = 100
    START_POP       = 15
    MIN_POP         = 5 # If population drops below this, new agents might spawn
    TICK_DELAY      = 0.1  # Min seconds between sim loop iterations
    DEFAULT_SAVE_INTERVAL = 50 # Ticks between auto-saves (0 to disable)
    SAVE_DIR        = "./saves_got_v3" # Updated save dir name
    LOCATIONS       = ["Winterfell", "KingsLanding", "TheWall", "Dragonstone", "Braavos"]
    # --- LLM ---
    MODEL           = "gemini-2.5-flash-preview-04-17" # Using Gemini Flash as updated in original code
    API_TIMEOUT     = 25.0 # Increased timeout slightly for potentially complex calls
    MAX_CONCURRENT_LLM_CALLS = 10 # Reduced slightly for stability
    FORBIDDEN = { # Words forbidden in *agent direct output* (speak, reflect, code comments)
        # Fewer forbidden words, rely more on prompt and safety settings
        "internet", "america", "nasa", "iphone", "quantum", "computer",
        "electricity", "database", "server", "api", "token", "language model"
    }
    # More specific forbidden patterns for code generation attempts
    FORBIDDEN_CODE_PATTERNS = ["import ", " open(", "eval(", "exec(", "__", "subprocess", "requests", "socket", "urllib"]

    # --- Agent State & Needs ---
    MAX_AGE         = 80 * 4 # Ticks per year arbitrary, e.g., 4 ticks/year -> 320 max ticks
    MAX_HEALTH      = 100
    MAX_HUNGER      = 20 # Agent starts taking damage above this
    HUNGER_PER_TICK = 0.7 # Increased hunger rate to add pressure
    HEALTH_DAMAGE_FROM_HUNGER = 1.5 # Increased damage from hunger (now float)
    MAX_MORALE      = 100
    MIN_MORALE      = 0
    MORALE_DECAY_PER_TICK = 0.1
    BASE_HEAL_AMOUNT = 15
    AGE_DEATH_CHANCE_PER_TICK = lambda age: max(0, (age - Config.MAX_AGE * 0.9) / (Config.MAX_AGE * 0.1) * 0.0005) # Slightly increased chance

    # --- Combat ---
    BASE_ATTACK_DAMAGE_FACTOR = 18 # Higher factor = more damage (Increased)
    COMBAT_MORALE_LOSS_ON_ATTACK = 5
    COMBAT_MORALE_LOSS_ON_KILL = 15 # Was gain before, now loss (killing is stressful)
    COMBAT_MORALE_GAIN_ON_VICTORY = 10 # Morale gain for surviving/winning an attack
    POST_KILL_ATTACK_COOLDOWN = 5 # Ticks before attacker can attack again
    COMBAT_COOLDOWN = 2 # Ticks before attacker can attack ANYONE again after attacking

    # --- Economy / Crafting ---
    INITIAL_RESOURCES = { # Base resources per location type (example)
        "Winterfell": Counter({"Wood": 50, "Food": 30, "Stone": 20}),
        "KingsLanding": Counter({"Food": 40, "Stone": 40, "Ore": 10}),
        "TheWall": Counter({"Stone": 60, "Wood": 10, "Food": 5}),
        "Dragonstone": Counter({"Stone": 50, "Ore": 30}),
        "Braavos": Counter({"Fish": 50, "Stone": 20}), # Using "Fish" as food example
    }
    RESOURCE_REGEN_RATE = { # Amount per resource per day
        "Wood": 2.0, "Food": 3.0, "Stone": 1.0, "Ore": 0.5, "Fish": 4.0
    }
    TICKS_PER_DAY = 4 # How many ticks constitute a "day" for regen
    CRAFTING_RECIPES = {
        "Wooden_Spear": {"Wood": 3},
        "Stone_Axe": {"Wood": 2, "Stone": 3},
        "Basic_Shelter": {"Wood": 10, "Stone": 5}, # Becomes a Structure
        "Forge": {"Stone": 15, "Wood": 5, "Ore": 5}, # Becomes a Structure
        "Cooked_Meal": {"Food": 2, "Wood": 1}, # Requires Campfire/Forge nearby? (Simplification)
        "Iron_Sword": {"Ore": 8, "Wood": 2}, # Requires Forge
    }
    STRUCTURE_BUILD_VERBS = {"Basic_Shelter", "Forge"} # Verbs that create Structures
    CONSUMABLE_ITEMS = {
        "Cooked_Meal": {"hunger_relief": 10, "health_gain": 0, "morale_gain": 5},
        "Food": {"hunger_relief": 3, "health_gain": 0, "morale_gain": 1}, # Can consume raw food
        "Fish": {"hunger_relief": 4, "health_gain": 0, "morale_gain": 1}, # Can consume raw fish
    }

    # --- DNA / Traits / Skills ---
    TRAITS          = ["cunning","honour","wrath","piety"]
    SKILLS          = ["smith","war","lore","diplomacy"]
    TRAIT_MUT_STD   = 0.04
    SKILL_DECAY     = 0.001 # Reduced skill decay

    # --- RAG ---
    CHROMA_DIR      = "./chroma_db_auto_gui_v3" # Updated dir name
    BOOK_CHUNKS     = "./data/got_chunks.json"
    RAG_TOPK        = 3

    # --- GUI ---
    GUI_UPDATE_INTERVAL = 150 # ms
    EVENT_LOG_DISPLAY_SIZE = 100 # Max lines in GUI Listbox

    # --- Agent Code Modification (EXPERIMENTAL) ---
    ALLOW_CODE_MODIFICATION = True # Default to disabled for safety
    MAX_CODE_LINES = 20
    # Whitelist of targets agents can request/modify. BE EXTREMELY CAREFUL ADDING TO THIS.
    CODE_MODIFICATION_WHITELIST = {
        "Config.RESOURCE_REGEN_RATE",
        "Config.BASE_ATTACK_DAMAGE_FACTOR",
        "Config.HUNGER_PER_TICK",
        "AgentState.update_needs",
        "World.resolve_attack",
    }

    # --- Simple Random Event System ---
    EVENT_CHECK_INTERVAL = 30 # Check roughly every N ticks
    EVENT_BASE_CHANCE = 0.15 # Base chance of *an* event happening when checked
    RANDOM_EVENTS = [
        {"name": "Good Harvest", "type": "resource_loc", "resource": "Food", "amount": 15.0, "loc_specific": True, "message": "Reports of an unusually good harvest arrive from {location}!"},
        {"name": "Resource Find", "type": "resource_loc", "resource": "Stone", "amount": 10.0, "loc_specific": True, "message": "A new vein of stone has been discovered near {location}."},
        {"name": "Resource Find", "type": "resource_loc", "resource": "Ore", "amount": 5.0, "loc_specific": True, "message": "Scouts report finding a deposit of ore near {location}."},
        {"name": "Mild Sickness", "type": "morale_loc", "amount": -5, "loc_specific": True, "message": "A bout of mild sickness spreads through {location}, lowering spirits."},
        {"name": "Bad Omen", "type": "morale_loc", "amount": -8, "loc_specific": True, "message": "Strange lights in the sky above {location} are seen as a bad omen."},
        {"name": "Inspiring News", "type": "morale_all", "amount": 3, "loc_specific": False, "message": "Good news travels across the land, slightly lifting everyone's spirits."},
        {"name": "Bandit Sighting", "type": "flavor", "loc_specific": True, "message": "Travelers report increased bandit activity near {location}."},
        # {"name": "Harsh Weather", "type": "hunger_all", "amount": 0.1, "loc_specific": False, "message": "Unseasonably harsh weather makes survival slightly harder for everyone."}, # Example affecting needs
    ]


# --- GUI Constants ---
# (Unchanged from previous version)
CANVAS_WIDTH = 600
CANVAS_HEIGHT = 450
LOCATION_COORDS = {
    "Winterfell": (150, 100), "KingsLanding": (400, 350),
    "TheWall": (300, 50), "Dragonstone": (500, 200), "Braavos": (550, 100),
}
AGENT_DOT_SIZE = 5
STRUCTURE_BOX_SIZE = 8
LOCATION_RADIUS = 20

# --- System Prompt (Enhanced) ---
SYSTEM_PROMPT = (
    "You are a character living *only* inside the world of the published ASOIAF novels (A Game of Thrones, A Clash of Kings, A Storm of Swords, A Feast for Crows, A Dance with Dragons). "
    "Your knowledge is limited to the events, people, places, technologies, and social structures within those books. Focus on authentic actions and speech. "
    "Do not mention anything from the TV show if it differs, or anything from the real world (like computers, AI, the internet, specific programming concepts unless prompted via `request_code`). "
    "Strive for survival, influence, power, or establishing a legacy. Consider your needs (hunger, health, morale), skills, relationships (friends/enemies), inventory, and surroundings. "
    "Think strategically. Vary your actions when appropriate. Form alliances, wage war, build structures, gather resources, reproduce, or even attempt to subtly influence the rules of this world if the opportunity arises via code modification actions (if enabled and you possess high 'lore' skill). Be creative but stay in character."
)

# ─────────────────────────── LOGGING ─────────────────────────────
log_file = Path("./got_sim_gui_v3.log") # Updated log filename
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(threadName)s | %(filename)s:%(lineno)d | %(message)s", # Added threadName
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=log_file,
    filemode='w'
)
log = logging.getLogger("got-sim-v3") # Consistent logger name

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("google.ai").setLevel(logging.INFO) # Allow some info from google-ai
logging.getLogger("google.api_core").setLevel(logging.INFO)

# --- Global Variables ---
_GEMINI_MODEL = None
_ACTION_QUEUE = None # To allow world to enqueue events related to code exec
_UPDATE_QUEUE = None # To allow world to enqueue events related to code exec


# --- Gemini Model Initialization ---
def initialize_llm_client():
    global _GEMINI_MODEL
    # KEEPING HARDCODED KEY AS REQUESTED BY USER
    api_key ="AIzaSyAAFpDNLv8EYWpqQpmd8ijRAekynL0N3HE" # Using this env var name
    if not api_key:
        log.critical("API key (OPENAI_API_KEY environment variable OR hardcoded) not found.")
        return False
    try:
        genai.configure(api_key=api_key)
        _GEMINI_MODEL = genai.GenerativeModel(model_name=Config.MODEL)
        # Test call (optional, but good for validation)
        # _GEMINI_MODEL.generate_content("test")
        log.info(f"Gemini model '{Config.MODEL}' initialized successfully.")
        return True
    except Exception as e:
        log.critical(f"Failed to initialize Gemini model '{Config.MODEL}': {e}", exc_info=True)
        _GEMINI_MODEL = None
        return False

# ─────────────────── UTILITY & DATA STRUCTURES ───────────────────
# (Mostly unchanged, added safe_get_target_code and safe_execute_code)

# KEEPING HARDCODED KEY AS REQUESTED BY USER - even though it seems redundant here
genai.configure(api_key="AIzaSyAAFpDNLv8EYWpqQpmd8ijRAekynL0N3HE")

def rand_vec(n: int) -> np.ndarray:
    return np.random.rand(n).astype(np.float32)

def random_dna() -> Dict[str, np.ndarray]:
    return {"traits": rand_vec(len(Config.TRAITS)), "skills": rand_vec(len(Config.SKILLS))}

def crossover(d1: Dict[str, np.ndarray], d2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    child = {}
    # Traits: Average + mutation
    t = (d1["traits"] + d2["traits"]) / 2
    child["traits"] = np.clip(t + np.random.normal(0, Config.TRAIT_MUT_STD, len(t)), 0, 1).astype(np.float32)
    # Skills: Max of parents + slight mutation/learning potential (instead of decay)
    s = np.maximum(d1["skills"], d2["skills"]) + np.random.normal(0, Config.SKILL_DECAY, len(Config.SKILLS)) # Treat SKILL_DECAY as std dev for skill change
    child["skills"] = np.clip(s, 0, 1).astype(np.float32)
    return child

def vec_str(v: np.ndarray, labels: List[str]) -> str:
    """Converts a numpy vector to a labeled string summary."""
    if not isinstance(v, np.ndarray) or v.ndim != 1 or len(v) != len(labels):
        log.error(f"Vector/label mismatch or invalid vector type: v={v}, labels={labels}")
        return "Invalid DNA/Labels"
    return " ".join(f"{lab[:1]}={val:.1f}" for lab, val in zip(labels, v))


def generate_themed_name(female_chance=0.5):
    """Generates a somewhat thematic ASOIAF name."""
    if random.random() < female_chance:
         p1 = random.choice(["Ary", "San", "Dan", "Cer", "Lyn", "Cat", "Mar", "Lys", "Ash", "Bri", "Eli", "Gil", "Shae", "Myr", "Tal", "Ygr", "Mel", "Iri", "Val", "Lei"])
         p2 = random.choice(["a", "ys", "ei", "elle", "ene", "enne", "ia", "wyn", "sa", "ya", "ara", "i", "a", "itte", "is", "na"])
    else:
        p1 = random.choice(["Jon", "Aeg", "Rob", "Edd", "Tyr", "Dav", "Sam", "Bra", "Ger", "Pod", "Ben", "Jeo", "Vis", "Har", "Wal", "Tom", "Ric", "Bal", "Lor", "Pat", "Leo", "Bry", "Tor", "Hos"])
        p2 = random.choice(["on", "ar", "os", "en", "ys", "yn", "ard", "ert", "ion", "as", "an", "ae", "or", "ic", "jen", "ell", "ald", "win", "mar", "ek", "rey", "nar", "ook", "mund"])

    surnames_list = [
        "Snow", "Stone", "Rivers", "Hill", "Pyke", "Flowers", "Sand", "Waters", # Bastard names (8)
        "Stark", "Lannister", "Targaryen", "Baratheon", "Greyjoy", "Martell", "Tyrell", "Arryn", # Great houses (8)
        "Bolton", "Frey", "Mormont", "Karstark", "Umber", "Glover", "Reed", # Northern (7)
        "Clegane", "Payne", "Marbrand", "Westerling", # Westerlands (4)
        "Tarly", "Florent", "Hightower", "Redwyne", # Reach (4)
        "Royce", "Waynwood", "Corbray", # Vale (3)
        "Dayne", "Yronwood", # Dorne (2)
        "Seaworth", "Velaryon", "Celtigar", # Crownlands/Dragonstone (3)
        "", "", "", "", "", "", "", "", "", "" # Commoner (10)
    ] # Total 49

    weights_list = [
        # Bastard (8)
        2, 2, 2, 1, 1, 1, 1, 1,
        # Great Houses (8)
        0.1, 0.1, 0.05, 0.1, 0.1, 0.05, 0.1, 0.1,
        # Northern (7)
        0.2, 0.2, 0.5, 0.3, 0.3, 0.3, 0.2,
        # Westerlands (4)
        0.5, 0.5, 0.3, 0.3,
        # Reach (4)
        0.3, 0.3, 0.3, 0.3,
        # Vale (3)
        0.3, 0.3, 0.2,
        # Dorne (2)
        0.2, 0.2,
        # Crownlands (3)
        0.3, 0.2, 0.2,
        # Commoner (10) - Distribute the total weight (e.g., 60) across the 10 slots
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6  # 10 * 6 = 60
    ] # Total 49 weights

    assert len(surnames_list) == len(weights_list), f"Length mismatch: {len(surnames_list)} surnames vs {len(weights_list)} weights"

    # Select the surname using the corrected lists
    selected_surname = random.choices(surnames_list, weights=weights_list, k=1)[0]

    name = f"{p1}{p2}"
    if selected_surname: # Only add surname if it's not an empty string
        name += f" {selected_surname}"

    return name[:20] # Limit length

# ─────────────────────── Embedding Setup ─────────────────────────
# (Unchanged)
def ensure_embeddings():
    """Load or create Chroma collection of ASOIAF lore snippets."""
    log.info("Checking for ChromaDB embeddings...")
    path = Path(Config.CHROMA_DIR)
    path.mkdir(exist_ok=True)
    client = None
    try:
        client = chromadb.PersistentClient(path=str(path))
        try:
            client.get_collection("got")
            log.info("Chroma collection 'got' already exists.")
            return client
        except Exception:
            log.info("Chroma collection 'got' not found. Creating...")

        coll = client.create_collection("got")
        fp = Path(Config.BOOK_CHUNKS)
        if not fp.exists():
            log.warning("Lore file not found at %s. RAG disabled.", Config.BOOK_CHUNKS)
            print(f"Warning: Lore file '{Config.BOOK_CHUNKS}' not found. RAG will be less effective.")
            try:
                 dummy_data = ["Winter is coming.", "The Targaryens ruled Westeros with dragons.", "The Lannisters always pay their debts.", "The Wall protects the realms of men.", "King's Landing is the capital.", "Braavos is a city of canals and assassins.", "The North remembers."]
                 with fp.open('w', encoding='utf-8') as f: json.dump(dummy_data, f, indent=2)
                 log.info("Created dummy lore file.")
                 chunks = dummy_data
                 print("Created dummy lore file for RAG.")
            except Exception as create_e:
                 log.error("Failed to create dummy lore file: %s. RAG will be empty.", create_e)
                 chunks = []
        else:
            try:
                with fp.open('r', encoding='utf-8') as f: chunks = json.load(f)
                log.info(f"Loaded {len(chunks)} lore chunks from {Config.BOOK_CHUNKS}")
            except json.JSONDecodeError:
                log.error("Failed to decode JSON from %s. RAG disabled.", Config.BOOK_CHUNKS)
                print(f"Error reading {Config.BOOK_CHUNKS}. Check JSON format. RAG disabled.")
                return client
            except Exception as e:
                log.error("Error reading lore file %s: %s. RAG disabled.", Config.BOOK_CHUNKS, e)
                print(f"Error reading {Config.BOOK_CHUNKS}: {e}. RAG disabled.")
                return client

        try:
            if chunks:
                ids = [f"doc_{i}" for i in range(len(chunks))]
                doc_chunks = [str(chunk) if not isinstance(chunk, str) else chunk for chunk in chunks]
                batch_size = 100
                for i in range(0, len(doc_chunks), batch_size):
                    batch_ids = ids[i:i+batch_size]
                    batch_docs = doc_chunks[i:i+batch_size]
                    if batch_ids and batch_docs:
                        coll.add(ids=batch_ids, documents=batch_docs)
                log.info("Embedded %d lore snippets.", len(doc_chunks))
            else:
                log.warning("Lore file %s was empty or failed to load. RAG will have no data.", Config.BOOK_CHUNKS)
                print(f"Lore file was empty. RAG will have no data.")
        except Exception as e:
            log.error("Error adding documents to Chroma: %s", e, exc_info=True)
            print(f"Error adding documents to ChromaDB: {e}")
        return client
    except Exception as e:
        log.error("Failed to initialize ChromaDB client at %s: %s", path, e, exc_info=True)
        print(f"CRITICAL ERROR: Failed to initialize ChromaDB at {path}. Check permissions/dependencies.")
        print(f"Details: {e}")
        return None

# ─────────────────── DOMAIN MODEL (Sprint 0, 2) ────────────────────
# (Structure and LocationNode are unchanged)
class Structure:
    _next_id = 1

    def __init__(self, name: str, loc: str, owner_id: int, durability: int = 100, capacity: int = 10, stored_items: Optional[Counter] = None, structure_id: Optional[int] = None):
        self.id = structure_id if structure_id is not None else Structure._next_id; Structure._next_id += 1
        self.name = name
        self.loc = loc
        self.owner_id = owner_id
        self.durability = durability
        self.capacity = capacity
        self.stored_items = stored_items or Counter()

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "name": self.name, "loc": self.loc, "owner_id": self.owner_id,
            "durability": self.durability, "capacity": self.capacity,
            "stored_items": dict(self.stored_items)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Structure':
        current_max_id = data.get("id", 0)
        if current_max_id >= Structure._next_id:
            Structure._next_id = current_max_id + 1
        return cls(
            structure_id=data["id"], name=data["name"], loc=data["loc"], owner_id=data.get("owner_id", 0),
            durability=data.get("durability", 100), capacity=data.get("capacity", 10),
            stored_items=Counter(data.get("stored_items", {}))
        )

    def get_info(self) -> str:
         items_str = ", ".join(f"{item}:{count}" for item, count in self.stored_items.items()) if self.stored_items else "empty"
         return f"{self.name} (ID:{self.id}, Owner:{self.owner_id}, Dur:{self.durability}, Cap:{self.capacity}, Items:[{items_str[:50]}])"

class LocationNode:
    def __init__(self, name: str, base_resources: Counter):
        self.name = name
        self.resources = Counter({k: float(v) for k, v in base_resources.items()}) # Store as floats
        self.base_resources = Counter({k: float(v) for k, v in base_resources.items()}) # Store as floats
        self.structures: Dict[int, Structure] = {}

    def add_structure(self, structure: Structure):
        self.structures[structure.id] = structure
        log.debug(f"Structure {structure.id} ({structure.name}) added to {self.name}.")

    def remove_structure(self, structure_id: int):
        if structure_id in self.structures:
            del self.structures[structure_id]
            log.debug(f"Structure {structure_id} removed from {self.name}.")

    def regenerate_resources(self):
        # Use float rates directly from Config
        for resource, rate in Config.RESOURCE_REGEN_RATE.items():
            if resource in self.base_resources:
                 current_amount = self.resources.get(resource, 0.0)
                 max_amount = self.base_resources.get(resource, 0.0) * 1.5 # Allow regen slightly above base
                 regen_per_tick = rate / Config.TICKS_PER_DAY
                 amount_to_add = min(regen_per_tick, max(0, max_amount - current_amount)) # Prevent negative regen
                 if amount_to_add > 1e-5: # Check for meaningful addition
                     self.resources[resource] = current_amount + amount_to_add
                     # log.debug(f"Regenerated {amount_to_add:.2f} {resource} at {self.name}. New total: {self.resources[resource]:.2f}")

    def get_structure_summary(self) -> str:
        if not self.structures: return "none"
        return ", ".join(f"{s.name}[{s.id}]" for s in self.structures.values())[:150]

    def get_resource_summary(self) -> str:
         # Display rounded integers, but keep internal floats
         return ", ".join(f"{r}:{int(round(c))}" for r,c in self.resources.items() if c >= 1.0) or "none"

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "resources": dict(self.resources), # Save floats
            "base_resources": dict(self.base_resources), # Save floats
            "structures": {sid: s.to_dict() for sid, s in self.structures.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'LocationNode':
        # Load floats correctly
        base_resources = Counter({k: float(v) for k, v in data.get("base_resources", {}).items()})
        node = cls(name=data["name"], base_resources=base_resources)
        node.resources = Counter({k: float(v) for k, v in data.get("resources", base_resources).items()})
        node.structures = {int(sid): Structure.from_dict(s_data) for sid, s_data in data.get("structures", {}).items()}
        return node

# Agent State - Now includes code modification attributes
class AgentState:
    _next_id = 1

    def __init__(self, name: str, dna: Optional[Dict[str, np.ndarray]] = None,
                 loc: Optional[str] = None, agent_id: Optional[int] = None,
                 age: int = 0, health: float = Config.MAX_HEALTH, hunger: float = 0.0, morale: float = 70.0,
                 inventory: Optional[Counter] = None, profession: str = "Peasant", role: str = "Commoner",
                 relations: Optional[Dict[int, int]] = None,
                 chroma_client=None,
                 attack_cooldown: int = 0):

        self.id = agent_id if agent_id is not None else AgentState._next_id; AgentState._next_id += 1
        self.name = name
        self.dna = dna or random_dna()
        self.loc = loc or random.choice(list(Config.LOCATIONS))

        # Core State
        self.age = age
        self.health = float(health)
        self.hunger = float(hunger)
        self.morale = float(np.clip(morale, Config.MIN_MORALE, Config.MAX_MORALE))
        self.inventory = inventory or Counter()
        self.profession = profession
        self.role = role
        self.relations = relations or {} # agent_id -> relationship score (-100 to 100)

        # Internal / Transient State
        self.memory = deque(maxlen=25) # Increased memory size slightly
        self.chroma_client = chroma_client
        self.attack_cooldown = attack_cooldown

        # Code Modification State
        self.requested_code_target: Optional[str] = None
        self.requested_code_content: Optional[str] = None
        self.last_code_proposal_result: Optional[str] = None # Info about last attempt

    @property
    def is_alive(self) -> bool:
        return self.health > 0

    def update_needs(self):
        """Update hunger, health, morale, age per tick. (Potentially modifiable by agents)"""
        if not self.is_alive: return

        self.age += 1
        # Use getattr to allow Config modification at runtime
        hunger_per_tick = getattr(Config, "HUNGER_PER_TICK", 0.7) # Using updated default
        max_hunger = getattr(Config, "MAX_HUNGER", 20)
        dmg_from_hunger = getattr(Config, "HEALTH_DAMAGE_FROM_HUNGER", 1.5) # Using updated default
        morale_decay = getattr(Config, "MORALE_DECAY_PER_TICK", 0.1)
        max_age = getattr(Config, "MAX_AGE", 320)
        age_death_func = getattr(Config, "AGE_DEATH_CHANCE_PER_TICK", lambda a: 0)

        self.hunger += hunger_per_tick
        if self.hunger > max_hunger:
            damage_taken = dmg_from_hunger * ((self.hunger - max_hunger) / max_hunger + 1) # Scale damage slightly by how hungry
            self.health -= damage_taken
            log.debug(f"Agent {self.id} takes {damage_taken:.2f} damage from hunger (Hunger: {self.hunger:.1f}). HP: {self.health:.1f}")

        self.morale -= morale_decay
        self.morale = max(Config.MIN_MORALE, self.morale)

        # Stochastic old age death check
        if self.age > max_age * 0.85: # Start checking a bit earlier
             death_chance = age_death_func(self.age)
             if random.random() < death_chance:
                 log.info(f"Agent {self.id} ({self.name}) dies of old age at {self.age} ticks (chance: {death_chance:.5f}).")
                 self.health = 0 # Mark for death

        self.health = max(0.0, self.health)

        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

    def modify_relation(self, other_id: int, amount: int):
        """ Safely modifies the relationship score towards another agent. """
        if other_id == self.id: return
        current_score = self.relations.get(other_id, 0)
        new_score = np.clip(current_score + amount, -100, 100)
        if new_score != current_score:
             self.relations[other_id] = int(new_score)
             # log.debug(f"Agent {self.id}'s relation towards {other_id} changed by {amount} to {new_score}")

    def get_relation_summary(self, registry: 'Registry') -> Tuple[str, str]:
        """ Gets summaries of top friends and enemies, filtering out dead agents. """
        if not self.relations: return "none", "none"
        living_relations = {oid: score for oid, score in self.relations.items() if registry.is_agent_alive(oid)}
        if not living_relations: return "none", "none"

        sorted_relations = sorted(living_relations.items(), key=lambda item: item[1], reverse=True)
        friends = [f"{registry.get_agent_name(oid)}[{oid}]({score:+})" for oid, score in sorted_relations if score > 30][:3]
        enemies = [f"{registry.get_agent_name(oid)}[{oid}]({score:+})" for oid, score in sorted_relations if score < -30][:3] # Get reversed for enemies? No, sort handles it.

        return (", ".join(friends) or "none"), (", ".join(enemies) or "none")

    def get_inventory_summary(self) -> str:
        if not self.inventory: return "empty"
        return ", ".join(f"{item}:{count}" for item, count in self.inventory.items())[:100]

    # --- LLM Interaction ---
    def oov(self, txt:str) -> bool:
        """Check for Out-Of-Vocabulary (forbidden) words in free text."""
        if not txt: return False
        lw = txt.lower()
        words = lw.translate(str.maketrans('', '', string.punctuation)).split()
        return any(word in Config.FORBIDDEN for word in words)

    def recent_verbs(self, n:int = 5) -> List[str]:
        """Return the valid verbs used in the last *n* actions."""
        verbs = []
        # Include new code modification verbs
        valid_actions = {"move", "propose_reproduction", "reflect", "code_build", "speak",
                         "gather", "craft", "store", "take", "attack", "heal", "assign_role", "consume",
                         "request_code", "propose_code_change"}
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
        """ Returns names of structures built recently """
        built = []
        for entry in reversed(list(self.memory)):
             if isinstance(entry, str) and entry.startswith("event:built_"):
                 item_name = entry.split('_', 1)[1]
                 built.append(item_name)
                 if len(built) >= n: break
        return built

    async def rag(self, prompt:str) -> str: # Async
        """Fetch top-K lore snippets relevant to prompt."""
        if not self.chroma_client:
            # log.warning("Agent %s: Chroma client not available for RAG.", self.id) # Too noisy
            return ""
        lore_content = ""
        try:
            def _query_chroma_sync(p: str, n: int):
                try:
                    # Ensure collection exists before querying
                    coll_list = self.chroma_client.list_collections()
                    if not any(c.name == "got" for c in coll_list):
                        log.warning("Chroma collection 'got' not found during query.")
                        return ""
                    coll = self.chroma_client.get_collection("got")
                    count = coll.count()
                    if count > 0:
                        n_results = min(n, count)
                        res = coll.query(query_texts=[p], n_results=n_results)
                        if res and res.get("documents") and res["documents"][0]:
                             # Filter out empty strings just in case
                             return "\n".join(filter(None, res["documents"][0]))
                    return ""
                except Exception as e:
                    log.warning(f"Agent {self.id} sync Chroma query failed: {e}", exc_info=False)
                    return ""

            lore_content = await asyncio.get_running_loop().run_in_executor(
                 None, _query_chroma_sync, prompt, Config.RAG_TOPK
            )
            # log.debug(f"Agent {self.id} RAG query returned {len(lore_content)} chars.") # Less verbose log

        except Exception as e:
            log.warning(f"Agent {self.id} RAG execution failed: {e}", exc_info=False)
        return lore_content

    async def llm(self, prompt: str, lore: str) -> str:
        """Perform LLM call with error handling and improved parsing."""
        if _GEMINI_MODEL is None:
            log.error("Agent %s: Gemini model not initialized for LLM call.", self.id)
            return "[error: gemini client missing]"

        combined_prompt_parts = [SYSTEM_PROMPT] # SYSTEM_PROMPT already encourages variety
        if lore:
            combined_prompt_parts.append(f"\n--- Relevant Lore Snippets ---\n{lore}\n---")
        combined_prompt_parts.append("\n--- Your Task ---")
        combined_prompt_parts.append("Based on your persona, state, memory, and the provided context/lore, decide on exactly ONE action from the 'Available Action Formats' list.")
        combined_prompt_parts.append("Your response MUST start *immediately* with the chosen action line (e.g., `speak Hello there` or `gather Wood`).")
        combined_prompt_parts.append("Do NOT include explanations, prefixes like 'ACTION:', or any text *before* the action line.")
        combined_prompt_parts.append("\n--- Current Situation & Action Prompt ---")
        combined_prompt_parts.append(prompt) # Main prompt with state, memory, available actions etc.

        full_prompt = "\n".join(combined_prompt_parts)

        # Gemini safety settings (adjust as needed) - Blocking harmful content
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        MAX_OUTPUT_TOKENS = 2000 # Reduced slightly, actions should be concise
        TEMPERATURE = 0.75 # Slightly increased variability

        def _call_sync() -> str:
            try:
                resp = _GEMINI_MODEL.generate_content(
                    contents=full_prompt, # Send as single string
                    generation_config=genai.types.GenerationConfig( # Use GenerationConfig object
                        candidate_count=1,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        temperature=TEMPERATURE,
                    ),
                    safety_settings=safety_settings
                )
                # Robust text extraction
                if resp.candidates and resp.candidates[0].content.parts:
                    return resp.candidates[0].content.parts[0].text.strip()
                elif hasattr(resp, 'text'): # Fallback for older SDK versions or different response structures
                    return resp.text.strip()
                else:
                     # Check for finish_reason=SAFETY
                     finish_reason = getattr(resp.candidates[0], 'finish_reason', None)
                     if finish_reason == genai.types.FinishReason.SAFETY:
                        log.warning(f"Agent {self.id}: LLM response blocked by safety settings (detected in response obj).")
                        return "[blocked: safety]"
                     log.warning(f"Agent {self.id}: Received unexpected LLM response format: {resp}")
                     return "[error: empty response or malformed]"

            # Handle specific Gemini exceptions
            except BlockedPromptException as bpe:
                 log.warning(f"Agent {self.id}: Prompt blocked by safety settings. Details: {bpe}")
                 return "[blocked: safety]" # Specific error code
            except google.api_core.exceptions.ResourceExhausted as ree: # Specific error for quota/rate limit
                 log.warning(f"Agent {self.id} hit ResourceExhausted error (rate limit/quota): {ree}")
                 return "[error: rate limited]"
            except google.api_core.exceptions.GoogleAPIError as api_e: # Catch other Google API errors
                 log.error(f"Agent {self.id}: GoogleAPIError during LLM call: {api_e}", exc_info=False)
                 # You could try to extract specific codes like api_e.code if available/needed
                 return f"[error: api_error_{getattr(api_e, 'code', 'unknown')}]"
            except Exception as sync_e:
                 log.error(f"Agent {self.id}: Unhandled error in _call_sync: {sync_e}", exc_info=True)
                 return "[error: sync_call_failed]"

        # Execute in thread
        raw_text_output = "[error: no response]"
        try:
            raw_text_output = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(None, _call_sync),
                timeout=Config.API_TIMEOUT
            )
            # log.debug("Agent %s raw Gemini output: %.100s", self.id, raw_text_output) # Log less output

            # --- Extract First Line ---
            # The prompt now instructs the LLM to start directly with the action.
            if not raw_text_output or raw_text_output.startswith("["): # Handle error codes or empty response
                if not raw_text_output: raw_text_output="[error: empty]" # Assign code if empty
                log.warning(f"Agent {self.id} LLM call resulted in error/empty: {raw_text_output}")
                return raw_text_output # Return the error code directly

            # Get the first line as the potential action
            action_line = raw_text_output.splitlines()[0].strip()

            if not action_line:
                 log.warning(f"Agent {self.id} LLM output started with empty line. Raw: {raw_text_output[:100]}")
                 return "[error: format]"

            # Basic verb check (optional, relying more on prompt structure)
            # parts = action_line.split(maxsplit=1)
            # if not parts or parts[0].lower() not in ALL_VERBS: # Need set of all possible verbs
            #     log.warning(f"Agent {self.id} first line not a recognized verb: '{action_line}'. Raw: {raw_text_output[:100]}")
            #     # Decide whether to return format error or try anyway
            #     # return "[error: format]" # Stricter
            #     pass # Less strict, try processing it

            # OOV check *only* on free text parts (speak, reflect arguments) - not on code proposals
            verb_test = action_line.split(maxsplit=1)[0].lower()
            if verb_test in ["speak", "reflect"]:
                args_test = action_line.split(maxsplit=1)[1] if len(action_line.split(maxsplit=1)) > 1 else ""
                if self.oov(args_test):
                    log.warning(f"Agent {self.id} generated OOV content in {verb_test}: {args_test}")
                    return "[oov]"
            # For code proposals, OOV check is less relevant than syntax/safety checks later

            log.debug(f"Agent {self.id} proposed action: {action_line}")
            return action_line

        except asyncio.TimeoutError:
            log.error(f"Agent {self.id}: LLM call timed out after {Config.API_TIMEOUT}s.")
            return "[error: timeout]"
        except Exception as e:
            log.exception(f"Unhandled Gemini error for agent {self.id}. Raw output potentially: {raw_text_output}")
            return f"[error: unhandled_llm_exception]"

    async def decide(self, world: 'World', registry: 'Registry') -> str: # Async
        """LLM call for agent decision-making, enhanced prompting."""
        if not self.is_alive: return "[error: dead]"

        # --- 1. Gather Context ---
        current_loc_node = world.locations.get(self.loc)
        if not current_loc_node:
             log.error(f"Agent {self.id} in invalid location {self.loc}. Cannot decide.")
             return "[error: invalid location]"

        others_here = [a for a in world.get_agents_at_loc(self.loc) if a.id != self.id]
        structures_here = list(current_loc_node.structures.values())
        my_structures_here = [s for s in structures_here if s.owner_id == self.id]
        population_room = max(0, Config.MAX_POP - len(world.get_living_agents()))
        can_reproduce = population_room > 0 and len(others_here) > 0
        nearby_agent_ids = [o.id for o in others_here]
        friends_summary, enemies_summary = self.get_relation_summary(registry)
        lore_skill = self.dna["skills"][Config.SKILLS.index("lore")] # Get lore skill for code mod check

        # --- 2. Build Action Formats String ---
        location_list = "|".join(Config.LOCATIONS)
        resource_list = "|".join(Config.RESOURCE_REGEN_RATE.keys())
        craftable_items = "|".join(Config.CRAFTING_RECIPES.keys())
        consumable_items = "|".join(Config.CONSUMABLE_ITEMS.keys())
        roles_list = "Guard|Merchant|Farmer|Mason|Healer|Commander|Leader" # Expanded roles
        nearby_ids_str = "|".join(map(str, nearby_agent_ids)) if nearby_agent_ids else "AGENT_ID"
        structure_ids_str = "|".join(map(str, (s.id for s in structures_here))) if structures_here else "STRUCTURE_ID" # Use IDs from structures_here

        # Base action formats
        action_formats_list = [
            f"`move <{location_list}>`",
            "`speak <message_to_people_here>`",
            "`reflect <internal_thought_or_plan>`",
            f"`gather <{resource_list}>`",
            f"`craft <{craftable_items}>`",
            f"`consume <{consumable_items}>`",
            # f"`code_build <item_name> <PROP1=VAL1;...>`", # Deprecated in favor of craft
            f"`store <item_name> <{structure_ids_str}>`",
            f"`take <item_name> <{structure_ids_str}>`",
            f"`attack <{nearby_ids_str}>`",
            f"`heal <{nearby_ids_str}>`",
            f"`propose_reproduction <{nearby_ids_str}>`",
            f"`assign_role <{nearby_ids_str}> <{roles_list}>`"
        ]

        # Add code modification actions if enabled and agent has high lore skill
        if Config.ALLOW_CODE_MODIFICATION and lore_skill > 0.7:
            allowed_targets = "|".join(Config.CODE_MODIFICATION_WHITELIST)
            action_formats_list.append(f"`request_code <{allowed_targets}>` # Ask to see simulation rules code")
            # Only allow proposal if code has been requested and received
            if self.requested_code_target and self.requested_code_content:
                 action_formats_list.append(f"`propose_code_change {self.requested_code_target} <python_code_string>` # Suggest changes (Max {Config.MAX_CODE_LINES} lines)")
            else:
                 action_formats_list.append(f"`# propose_code_change <target> <code>` (Requires using request_code first)")
        else:
            # Comment out if not allowed/skilled
            action_formats_list.append("`# request_code <target>` (Requires high lore skill & feature enabled)")
            action_formats_list.append("`# propose_code_change <target> <code>` (Requires high lore skill & feature enabled)")


        # --- 3. Apply Action Constraints/Availability ---
        final_action_formats = []
        all_verbs_available = set() # Track verbs *before* disabling

        for line in action_formats_list:
            verb = line.split(maxsplit=1)[0].lower().strip('`# ')
            is_available = True
            reason = ""

            # Basic check if it's already commented out
            if line.strip().startswith("`#"):
                 is_available = False
                 # Extract existing reason if possible
                 if "(" in line and ")" in line: reason = line[line.find("(")+1:line.find(")")]
                 # Keep the commented line
                 final_action_formats.append(line)
                 continue # Skip further checks for this line


            # Specific checks
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
                can_build_structure = False
                buildable_structure = ""
                for item, recipe in Config.CRAFTING_RECIPES.items():
                    if all(self.inventory.get(ing, 0) >= recipe[ing] for ing in recipe):
                        # Check structure reqs simply
                        if item == "Iron_Sword" and not any(s.name == "Forge" for s in structures_here): continue
                        can_craft_anything = True
                        if item in Config.STRUCTURE_BUILD_VERBS:
                            can_build_structure = True
                            buildable_structure = item
                            # Specific check for building repetition
                            recently_built = self.recent_built_items()
                            owns_same_type = any(s.name == item and s.owner_id == self.id for s in structures_here)
                            if item in recently_built or owns_same_type:
                                is_available = False # Disable crafting this *specific* structure
                                reason = f"Already built {item} recently or own one here"
                                break # Stop checking other recipes if we disable this build
                        # If not disabled by build check, we can craft something
                if not can_craft_anything and is_available: # is_available could be False from build check
                    is_available = False; reason = "Lack ingredients/tools for any recipe"

            elif verb == "consume":
                 if not any(item in self.inventory for item in Config.CONSUMABLE_ITEMS):
                     is_available = False; reason = "No consumable items in inventory"


            if is_available:
                final_action_formats.append(line)
                all_verbs_available.add(verb) # Add to set of actually available verbs
            else:
                # Comment out the action with a reason
                final_action_formats.append(f"`# {line.strip('`')} ({reason})`")

        action_formats_string = "\n".join(final_action_formats)

        # --- 4. Analyze State & Recent Actions for Constraints ---
        window = 5
        recent_verbs_list = self.recent_verbs(window)
        speak_reflect_count = sum(1 for v in recent_verbs_list if v in ["speak", "reflect"])
        move_count = recent_verbs_list.count("move")
        gather_count = recent_verbs_list.count("gather") # Check gather repetition too

        constraint = ""
        # Penalize repeating speak/reflect
        if len(recent_verbs_list) >= window and speak_reflect_count >= 3 and len(all_verbs_available) > 2:
             suggested_verbs = {v for v in all_verbs_available if v not in ["speak", "reflect"]}
             if suggested_verbs:
                 constraint += f"\nCONSTRAINT: Avoid 'speak' or 'reflect'. You did this {speak_reflect_count} times recently. Try: {', '.join(suggested_verbs)}."
        # Penalize excessive movement
        elif len(recent_verbs_list) >= window and move_count >= 3 and len(all_verbs_available) > 1:
             suggested_verbs = {v for v in all_verbs_available if v != "move"}
             if suggested_verbs:
                  constraint += f"\nCONSTRAINT: Avoid 'move'. You did this {move_count} times recently. Consider: {', '.join(suggested_verbs)}."
        # Penalize excessive gathering
        elif len(recent_verbs_list) >= window and gather_count >= 4 and len(all_verbs_available) > 1: # Higher threshold for gather
             suggested_verbs = {v for v in all_verbs_available if v != "gather"}
             if suggested_verbs:
                  constraint += f"\nCONSTRAINT: Avoid 'gather'. You did this {gather_count} times recently. Consider: {', '.join(suggested_verbs)}."


        # Needs constraints (Now more likely to trigger due to increased hunger/damage)
        if self.hunger >= Config.MAX_HUNGER * 0.7: # Trigger earlier
            options = []
            if "gather" in all_verbs_available and any(f in Config.RESOURCE_REGEN_RATE for f in ["Food", "Fish"]): options.append("gather Food/Fish")
            if "craft" in all_verbs_available and "Cooked_Meal" in Config.CRAFTING_RECIPES: options.append("craft Cooked_Meal")
            if "consume" in all_verbs_available and any(f in self.inventory for f in ["Food", "Fish", "Cooked_Meal"]): options.append("consume items")
            if options: constraint += f"\nCONSTRAINT: Getting hungry ({self.hunger:.0f}). Should prioritize getting food (e.g., {', '.join(options)})."
            else: constraint += f"\nCONSTRAINT: Getting hungry ({self.hunger:.0f}), but few options to get food right now."
        if self.health < Config.MAX_HEALTH * 0.5: # Trigger earlier
             options = []
             if "heal" in all_verbs_available and nearby_agent_ids: options.append("ask for healing (use heal action on someone)") # Hacky way to phrase
             if "consume" in all_verbs_available and "Cooked_Meal" in self.inventory: options.append("consume Cooked_Meal (might help morale)") # Assume meal might help slightly
             if "reflect" in all_verbs_available: options.append("reflect on finding safety/help")
             if options: constraint += f"\nCONSTRAINT: Health low ({self.health:.0f}). Consider: {', '.join(options)}."
             else: constraint += f"\nCONSTRAINT: Health low ({self.health:.0f}). Survival is becoming difficult."

        # --- 5. Encouragement & Context ---
        encouragement = "\n--- Encouragement & Context ---"
        if self.role in ["Lord", "Leader", "Commander"]:
            encouragement += "\nAs a leader, consider strategic actions: assign roles, organize defenses, manage resources, or influence others."
        if lore_skill > 0.8 and Config.ALLOW_CODE_MODIFICATION:
             encouragement += "\nYour high lore skill allows you to inspect and potentially modify the underlying rules of this world via `request_code` and `propose_code_change`."
        elif self.requested_code_content:
             encouragement += f"\nYou have received the code for '{self.requested_code_target}'. You can now analyze it (`reflect`) or attempt to modify it using `propose_code_change`."
        if self.last_code_proposal_result:
             encouragement += f"\nResult of last code proposal: {self.last_code_proposal_result}"
             self.last_code_proposal_result = None # Clear after showing once

        if enemies_summary != "none":
            encouragement += f"\nYou have enemies nearby ({enemies_summary}). Consider attacking, defending, or moving away."
        elif friends_summary != "none":
            encouragement += f"\nAllies are nearby ({friends_summary}). Consider cooperating, trading (if possible), or healing them."
        # Add encouragement based on inventory or surroundings
        if "Wood" in self.inventory and self.inventory["Wood"] >= 10 and "Stone" in self.inventory and self.inventory["Stone"] >= 5:
            if "craft" in all_verbs_available and "Basic_Shelter" not in self.recent_built_items(3): # Check recent builds
                 encouragement += "\nYou have enough materials to build a Basic_Shelter."


        # --- 6. Compose Full Prompt ---
        # Include requested code if available
        code_section = ""
        if self.requested_code_target and self.requested_code_content:
            code_section = f"\n--- Code Content for {self.requested_code_target} ---\n```python\n{self.requested_code_content}\n```\n---"
            # Clear the stored code after displaying it in the prompt
            # self.requested_code_target = None # Keep target until proposal/new request
            self.requested_code_content = None

        prompt = (
            f"--- Your State & Surroundings ---\n"
            f"You are {self.name} (ID:{self.id}), a {self.role} ({self.profession}). Age: {self.age}, Health: {self.health:.0f}/{Config.MAX_HEALTH}, Hunger: {self.hunger:.0f}, Morale: {self.morale:.0f}.\n"
            f"Location: {self.loc}.\n"
            f"Resources here: {current_loc_node.get_resource_summary()}.\n"
            f"Structures here: {current_loc_node.get_structure_summary()}.\n"
            f"People here: {', '.join(f'{o.name}[{o.id}]' for o in others_here) or 'none'}.\n"
            f"Inventory: {self.get_inventory_summary()}.\n"
            f"Skills: {vec_str(self.dna['skills'], Config.SKILLS)}.\n"
            f"Friends: {friends_summary} | Enemies: {enemies_summary}.\n"
            f"Memory (Recent Events/Thoughts): {'; '.join(self.memory) or 'none'}.\n"
            f"{code_section}" # Include code if requested
            f"{constraint}" # Include constraints
            f"{encouragement}" # Include encouragement
            f"\n--- Available Action Formats ---\n{action_formats_string}"
            f"\n--- Your ONE Action (MUST start on the next line, NO commentary before it) ---" # Final instruction
        )

        # --- 7. Get Lore & Call LLM ---
        rag_query = f"Action decision context for {self.name} at {self.loc}. Needs: H={self.health:.0f}, G={self.hunger:.0f}, M={self.morale:.0f}. Role: {self.role}. Skills: {vec_str(self.dna['skills'], Config.SKILLS)}. Relationships: Friends={friends_summary}, Enemies={enemies_summary}. Memory: {'; '.join(self.memory)}"
        lore = await self.rag(rag_query)
        return await self.llm(prompt, lore)

    async def act(self, world: 'World', registry: 'Registry') -> str: # Async
        """Parse LLM decision and execute the corresponding action async."""
        if not self.is_alive: return f"[{self.id}] {self.name} is dead."

        # Clear previous code request results before making a new decision
        # self.last_code_proposal_result = None # Moved clearing to decide

        # --- Get Decision ---
        async with world.llm_semaphore:
            raw_decision = await self.decide(world, registry)

        # --- Parse Decision ---
        if raw_decision.startswith("[") and raw_decision.endswith("]"): # Handle error codes or [oov], [blocked]
            verb = raw_decision
            arg_str = ""
            args = []
        else:
            parts = raw_decision.split(maxsplit=1)
            verb = parts[0].lower().strip("` ") # Clean verb
            arg_str = parts[1].strip() if len(parts) > 1 else ""

        ev = "" # Event description string
        log.debug(f"Agent {self.id} attempting action: {verb} {arg_str[:50]}")

        # --- Execute Action ---
        try:
            current_loc_node = world.locations.get(self.loc)
            if not current_loc_node and verb not in ["reflect"]: # Need location for most actions
                 ev = f"is in an invalid location ({self.loc}) and cannot act."
                 return f"[{self.id}] {self.name}: {ev}"

            # === Standard Actions ===
            if verb == "speak":
                 msg = arg_str[:150].replace("\n", " ")
                 if not msg: ev = "mutters incoherently."
                 # OOV check done in LLM wrapper now
                 # elif self.oov(msg): ev = f"tries to speak forbidden words."
                 else: ev = f"says \"{msg}\""
            elif verb == "move":
                target_loc = arg_str
                if target_loc in Config.LOCATIONS:
                    if target_loc != self.loc:
                        old_loc = self.loc
                        self.loc = target_loc
                        ev = f"travels from {old_loc} to {target_loc}."
                        self.morale += 1
                    else: ev = f"decides to stay at {self.loc}."
                else: ev = f"tries to move to an unknown location '{target_loc}'."
            elif verb == "propose_reproduction":
                # (Logic mostly unchanged, but uses registry.get_agent_name)
                try:
                     partner_id = int(arg_str)
                     partner = registry.agents.get(partner_id)
                     if partner and partner.is_alive and partner.loc == self.loc and partner.id != self.id:
                         if len(world.get_living_agents()) < Config.MAX_POP:
                             # Simplified success chance
                             success_chance = 0.5 + (self.relations.get(partner_id, 0)/200) # Base 50% + relation mod
                             if random.random() < success_chance:
                                 child_dna = crossover(self.dna, partner.dna)
                                 child_name = generate_themed_name()
                                 child = AgentState(child_name, child_dna, loc=self.loc, chroma_client=self.chroma_client)
                                 if registry.add(child):
                                     world.new_agent_buffer.append(child)
                                     self.modify_relation(partner_id, 15)
                                     partner.modify_relation(self.id, 15)
                                     self.morale += 10
                                     partner.morale += 10
                                     ev = f"and {partner.name}[{partner_id}] successfully reproduce offspring: {child.name}[{child.id}]!"
                                 else: ev = f"and {partner.name}[{partner_id}] attempt reproduction, but registry failed to add offspring." # Should not happen
                             else:
                                 ev = f"proposes reproduction to {partner.name}[{partner_id}], but is rejected this time."
                                 self.modify_relation(partner_id, -5)
                                 partner.modify_relation(self.id, -3) # Less penalty for rejecting
                         else: ev = f"tries to reproduce with {partner.name}[{partner_id}], but the world feels too crowded."
                     elif not partner or not partner.is_alive: ev = f"tries to reproduce with someone who isn't here or is dead (ID {arg_str})."
                     elif partner.loc != self.loc: ev = f"tries to reproduce with {partner.name}[{partner_id}] who is in {partner.loc}."
                     else: ev = f"has confusing thoughts about reproduction."
                except ValueError: ev = f"uses propose_reproduction with invalid target ID '{arg_str}'."
            elif verb == "reflect":
                thought = arg_str[:150].replace("\n", " ")
                if not thought: ev = "reflects silently."
                # OOV check done in LLM wrapper
                # elif self.oov(thought): ev = "has forbidden thoughts."
                else: ev = f"reflects internally ('{thought[:50]}...')."
            # --- REMOVED code_build as it's handled by craft ---
            # elif verb == "code_build": ...
            elif verb == "gather":
                resource = arg_str.strip().capitalize() # Canonicalize resource name
                if not resource: ev = "tries to gather nothing."
                elif resource in Config.RESOURCE_REGEN_RATE:
                     available = current_loc_node.resources.get(resource, 0.0)
                     if available >= 1.0: # Require at least 1.0 unit to gather
                         amount_gathered = 1.0
                         current_loc_node.resources[resource] -= amount_gathered
                         self.inventory[resource] += int(amount_gathered) # Add as int
                         self.hunger += 0.3 # Slightly more hunger increase from work
                         self.morale += 0.5
                         ev = f"gathers 1 {resource}. Inv: {self.get_inventory_summary()}"
                     else:
                         ev = f"tries to gather {resource}, but finds less than one unit left ({available:.1f})."
                else:
                    ev = f"tries to gather unknown resource '{resource}'."
            elif verb == "craft":
                item_to_craft = arg_str.strip().replace(" ", "_") # Allow spaces in LLM output but map to underscore
                if not item_to_craft: ev = "tries to craft nothing."
                elif item_to_craft in Config.CRAFTING_RECIPES:
                    recipe = Config.CRAFTING_RECIPES[item_to_craft]
                    missing_ingredients = [f"{amount} {ing}" for ing, amount in recipe.items() if self.inventory.get(ing, 0) < amount]

                    # Check for structure requirements
                    needs_forge = item_to_craft == "Iron_Sword"
                    has_forge = any(s.name == "Forge" for s in world.get_structures_at_loc(self.loc))

                    if missing_ingredients:
                        ev = f"tries to craft {item_to_craft}, but lacks: {', '.join(missing_ingredients)}."
                    elif needs_forge and not has_forge:
                        ev = f"tries to craft {item_to_craft}, but needs a Forge nearby."
                    else:
                        # Consume ingredients
                        for ingredient, amount in recipe.items():
                            self.inventory[ingredient] -= amount
                            if self.inventory[ingredient] <= 0: del self.inventory[ingredient]

                        # Add result
                        if item_to_craft in Config.STRUCTURE_BUILD_VERBS:
                            # Check if already owns one here to prevent spam (double check, prompt should prevent but good safeguard)
                            if any(s.name == item_to_craft and s.owner_id == self.id for s in world.get_structures_at_loc(self.loc)):
                                ev = f"considers building another {item_to_craft}, but already has one here. Crafting materials consumed."
                                # Restore ingredients? Or let them be consumed? Let's say consumed for now.
                            else:
                                new_struct = Structure(name=item_to_craft, loc=self.loc, owner_id=self.id)
                                if world.add_structure(new_struct): # Use world helper
                                    ev = f"crafts and builds a {item_to_craft} (ID:{new_struct.id})."
                                    self.memory.append(f"event:built_{item_to_craft}") # Add specific build event to memory
                                else:
                                    ev = f"tries to build {item_to_craft}, but failed to add structure to world." # Should not happen
                        else:
                            self.inventory[item_to_craft] += 1
                            ev = f"crafts 1 {item_to_craft}. Inv: {self.get_inventory_summary()}"
                        self.morale += 2
                        self.hunger += 0.2 # Crafting takes some energy
                else:
                    ev = f"tries to craft unknown item '{item_to_craft}'."
            elif verb == "consume":
                item_to_consume = arg_str.strip().replace(" ", "_")
                if not item_to_consume: ev = "tries to consume nothing."
                elif item_to_consume in Config.CONSUMABLE_ITEMS:
                     if self.inventory.get(item_to_consume, 0) > 0:
                         effects = Config.CONSUMABLE_ITEMS[item_to_consume]
                         self.inventory[item_to_consume] -= 1
                         if self.inventory[item_to_consume] <= 0: del self.inventory[item_to_consume]

                         hunger_relief = effects.get("hunger_relief", 0)
                         health_gain = effects.get("health_gain", 0)
                         morale_gain = effects.get("morale_gain", 0)

                         self.hunger = max(0.0, self.hunger - hunger_relief)
                         self.health = min(Config.MAX_HEALTH, self.health + health_gain)
                         self.morale = np.clip(self.morale + morale_gain, Config.MIN_MORALE, Config.MAX_MORALE)

                         ev = f"consumes 1 {item_to_consume}. Effects: Hunger -{hunger_relief}, HP +{health_gain}, Morale +{morale_gain}."
                     else:
                         ev = f"tries to consume {item_to_consume}, but has none."
                elif item_to_consume in self.inventory: # Allow consuming non-defined items (e.g. raw Wood - bad idea)
                    if self.inventory[item_to_consume] > 0:
                        self.inventory[item_to_consume] -= 1
                        if self.inventory[item_to_consume] <= 0: del self.inventory[item_to_consume]
                        # Default minimal effect for unknown consumables (usually bad)
                        hunger_relief = 0.5 # Minimal relief, maybe slight negative effect?
                        morale_change = -1 if item_to_consume not in ["Food", "Fish"] else 0 # Penalty for eating non-food
                        self.hunger = max(0.0, self.hunger - hunger_relief)
                        self.morale = np.clip(self.morale + morale_change, Config.MIN_MORALE, Config.MAX_MORALE)
                        ev = f"consumes 1 raw {item_to_consume}. Seems unpleasant."
                    else: # Should not happen if check above is correct
                         ev = f"tries to consume {item_to_consume}, but has none."
                else:
                    ev = f"tries to consume unknown or non-existent item '{item_to_consume}'."
            elif verb == "store":
                args = arg_str.split(maxsplit=1)
                if len(args) >= 2:
                    item_name = args[0].strip()
                    try:
                        structure_id = int(args[1].strip())
                        target_structure = world.structures.get(structure_id)
                        if target_structure and target_structure.loc == self.loc:
                            if self.inventory.get(item_name, 0) > 0:
                                 current_item_count = sum(target_structure.stored_items.values())
                                 if current_item_count < target_structure.capacity:
                                     amount_to_store = 1
                                     self.inventory[item_name] -= amount_to_store
                                     if self.inventory[item_name] <= 0: del self.inventory[item_name]
                                     target_structure.stored_items[item_name] += amount_to_store
                                     ev = f"stores 1 {item_name} in {target_structure.name}[{structure_id}]."
                                     self.morale += 0.5
                                 else: ev = f"tries to store {item_name}, but {target_structure.name}[{structure_id}] is full."
                            else: ev = f"tries to store {item_name}, but has none."
                        elif not target_structure: ev = f"tries to store in non-existent structure ID {structure_id}."
                        else: ev = f"tries to store in {target_structure.name}[{structure_id}], but it's in {target_structure.loc}."
                    except ValueError: ev = f"uses store with invalid structure ID '{args[1]}'."
                else: ev = f"uses store action incorrectly. Format: store <item_name> <structure_id>."
            elif verb == "take":
                 args = arg_str.split(maxsplit=1)
                 if len(args) >= 2:
                     item_name = args[0].strip()
                     try:
                        structure_id = int(args[1].strip())
                        target_structure = world.structures.get(structure_id)
                        if target_structure and target_structure.loc == self.loc:
                             if target_structure.stored_items.get(item_name, 0) > 0:
                                 amount_to_take = 1
                                 target_structure.stored_items[item_name] -= amount_to_take
                                 if target_structure.stored_items[item_name] <= 0: del target_structure.stored_items[item_name]
                                 self.inventory[item_name] += amount_to_take
                                 # Stealing check and relation impact
                                 if target_structure.owner_id != self.id and target_structure.owner_id != 0:
                                     owner = registry.agents.get(target_structure.owner_id)
                                     owner_name = owner.name if owner else f"Agent {target_structure.owner_id}"
                                     ev = f"takes 1 {item_name} from {target_structure.name}[{structure_id}] (owned by {owner_name})."
                                     self.modify_relation(target_structure.owner_id, -15)
                                     if owner and owner.is_alive: owner.modify_relation(self.id, -15)
                                     self.morale -= 2 # Morale penalty for stealing
                                 else:
                                     ev = f"takes 1 {item_name} from {target_structure.name}[{structure_id}]."
                             else: ev = f"tries to take {item_name} from {target_structure.name}[{structure_id}], but it has none."
                        elif not target_structure: ev = f"tries to take from non-existent structure ID {structure_id}."
                        else: ev = f"tries to take from {target_structure.name}[{structure_id}], but it's in {target_structure.loc}."
                     except ValueError: ev = f"uses take with invalid structure ID '{args[1]}'."
                 else: ev = f"uses take action incorrectly. Format: take <item_name> <structure_id>."
            elif verb == "attack":
                if self.attack_cooldown > 0:
                     ev = f"tries to attack, but is recovering (Cooldown: {self.attack_cooldown})."
                else:
                    try:
                        target_id = int(arg_str.strip())
                        target = registry.agents.get(target_id)
                        if target and target.is_alive and target.loc == self.loc and target.id != self.id:
                           dmg_done, dmg_received = world.resolve_attack(self, target) # World handles combat logic
                           ev = f"attacks {target.name}[{target_id}]! Deals {dmg_done:.1f} dmg (Target HP: {target.health:.1f}), receives {dmg_received:.1f} dmg (HP: {self.health:.1f})."
                           self.modify_relation(target_id, -30)
                           target.modify_relation(self.id, -30)
                           self.morale -= Config.COMBAT_MORALE_LOSS_ON_ATTACK
                           target.morale -= Config.COMBAT_MORALE_LOSS_ON_ATTACK / 2 # Target also loses morale
                           self.attack_cooldown = Config.COMBAT_COOLDOWN # Base cooldown
                           self.hunger += 0.5 # Combat takes energy

                           if not target.is_alive:
                               ev += f" {target.name} has been slain!"
                               world.kill_agent(target) # Mark for removal
                               self.morale -= Config.COMBAT_MORALE_LOSS_ON_KILL # Additional loss for killing
                               self.attack_cooldown = Config.POST_KILL_ATTACK_COOLDOWN # Longer cooldown after kill
                               # Transfer some inventory? Simplification: just drop items for now? Or remove? Let's just remove for now.
                               # target.inventory.clear()
                           elif dmg_done > 0 or dmg_received > 0: # If combat actually occurred
                               self.morale += Config.COMBAT_MORALE_GAIN_ON_VICTORY / 2 # Small gain for surviving
                               target.morale += Config.COMBAT_MORALE_GAIN_ON_VICTORY / 4 # Smaller gain for defender if survived

                        elif not target or not target.is_alive: ev = f"tries to attack non-existent/dead target ID {arg_str}."
                        elif target.loc != self.loc: ev = f"tries to attack {target.name}[{target_id}] who is in {target.loc}."
                        elif target.id == self.id: ev = f"considers attacking self, but thinks better of it."
                    except ValueError: ev = f"uses attack with invalid target ID '{arg_str}'."
            elif verb == "heal":
                 try:
                     target_id = int(arg_str.strip())
                     target = registry.agents.get(target_id)
                     if target and target.is_alive and target.loc == self.loc:
                         # Simplified healing - maybe require a "Medicine" item later?
                         if target.health < Config.MAX_HEALTH:
                             heal_amount = Config.BASE_HEAL_AMOUNT * (1 + self.dna["skills"][Config.SKILLS.index("lore")] * 0.5) # Lore skill improves healing? Or diplomacy? Let's use Lore for now.
                             actual_heal = min(heal_amount, Config.MAX_HEALTH - target.health) # Don't overheal
                             target.health += actual_heal
                             ev = f"attempts to heal {target.name}[{target_id}] for {actual_heal:.0f} HP. Target HP is now {target.health:.0f}."
                             self.modify_relation(target_id, 8)
                             target.modify_relation(self.id, 8)
                             self.morale += 3
                         else:
                             ev = f"tries to heal {target.name}[{target_id}], but they are already at full health."
                     elif not target or not target.is_alive: ev = f"tries to heal non-existent/dead target ID {arg_str}."
                     else: ev = f"tries to heal {target.name}[{target_id}] who is not here ({target.loc})."
                 except ValueError: ev = f"uses heal with invalid target ID '{arg_str}'."
            elif verb == "assign_role":
                 args = arg_str.split(maxsplit=1)
                 if len(args) >= 2:
                     try:
                         target_id = int(args[0].strip())
                         # Validate role against a known list? For now, just capitalize.
                         new_role = args[1].strip().capitalize().replace("_", " ")
                         target = registry.agents.get(target_id)

                         if self.role not in ["Lord", "Leader", "Commander"]:
                              ev = f"tries to assign role, but lacks authority (Role: {self.role})."
                         elif target and target.is_alive and target.loc == self.loc:
                             old_role = target.role
                             target.role = new_role
                             ev = f"assigns the role of '{new_role}' to {target.name}[{target_id}] (was '{old_role}')."
                             self.modify_relation(target_id, 3)
                             target.modify_relation(self.id, 5) # Target might appreciate new role more
                             self.morale += 2
                         elif not target or not target.is_alive: ev = f"tries to assign role to invalid target {target_id}."
                         else: ev = f"tries to assign role to {target.name}[{target_id}] who is not here."
                     except ValueError: ev = f"uses assign_role with invalid target ID '{args[0]}'."
                 else: ev = f"uses assign_role incorrectly. Format: assign_role <agent_id> <role_name>."

            # === Code Modification Actions (EXPERIMENTAL) ===
            elif verb == "request_code":
                target_name = arg_str.strip()
                if not Config.ALLOW_CODE_MODIFICATION:
                    ev = "thinks about changing the world's rules, but the mechanism is disabled."
                elif self.dna["skills"][Config.SKILLS.index("lore")] < 0.7:
                     ev = f"lacks the deep understanding (Lore skill) required to inspect the world's code."
                elif target_name not in Config.CODE_MODIFICATION_WHITELIST:
                    ev = f"tries to request code for forbidden target '{target_name}'. Allowed: {', '.join(Config.CODE_MODIFICATION_WHITELIST)}"
                    self.requested_code_target = None # Clear any previous request
                    self.requested_code_content = None
                else:
                    code_content = world.safe_get_target_code(target_name)
                    if code_content:
                        self.requested_code_target = target_name
                        self.requested_code_content = code_content # Will be added to next prompt
                        ev = f"requests and receives the source code for '{target_name}' for study."
                    else:
                        ev = f"tries to request code for '{target_name}', but it could not be retrieved."
                        self.requested_code_target = None
                        self.requested_code_content = None

            elif verb == "propose_code_change":
                if not Config.ALLOW_CODE_MODIFICATION:
                     ev = "tries to propose code changes, but the mechanism is disabled."
                elif self.dna["skills"][Config.SKILLS.index("lore")] < 0.7:
                     ev = f"lacks the deep understanding (Lore skill) required to modify the world's code."
                else:
                    parts = arg_str.split(maxsplit=1)
                    if len(parts) < 2:
                        ev = "tries to propose a code change, but provides incomplete arguments. Format: propose_code_change <target> <python_code_string>"
                        self.last_code_proposal_result = "Invalid format"
                    else:
                        target_name = parts[0].strip()
                        code_string = parts[1].strip().strip('`') # Remove backticks if LLM adds them

                        if target_name != self.requested_code_target:
                             ev = f"tries to propose code for '{target_name}', but must first `request_code {target_name}`."
                             self.last_code_proposal_result = f"Target mismatch (requested {self.requested_code_target})"
                        elif target_name not in Config.CODE_MODIFICATION_WHITELIST:
                             ev = f"tries to modify forbidden target '{target_name}'."
                             self.last_code_proposal_result = "Target not whitelisted"
                        else:
                            # Perform validation and execution via World method
                            success, message = world.safe_execute_agent_code(self.id, target_name, code_string)
                            self.last_code_proposal_result = message # Store result for next prompt
                            if success:
                                ev = f"successfully proposes and implements a code change for '{target_name}'! ({message})"
                                self.morale += 15 # Significant boost for successful modification
                            else:
                                ev = f"proposes a code change for '{target_name}', but it fails. Reason: {message}"
                                self.morale -= 5 # Penalty for failed/bad code


            # --- Fallback / Error Handling ---
            elif verb.startswith("[") and verb.endswith("]"): # Handle LLM error codes
                 if verb == "[oov]": ev = "tries to speak/think forbidden words."
                 elif verb == "[blocked: safety]": ev = "has thoughts blocked by world's safety filters."
                 elif verb == "[error: format]": ev = "fails to decide on a valid action format."
                 elif verb == "[error: rate limited]": ev = "thinks too fast and hits a limit, pausing thought."
                 elif verb == "[error: timeout]": ev = "takes too long to decide and gives up for now."
                 else: ev = f"experiences a cognitive error ({verb})."
            elif not verb: ev = f"gives an empty/invalid response ('{raw_decision[:50]}...')."
            else: ev = f"tries an unrecognized action '{verb}' ('{raw_decision[:50]}...')."

        except Exception as e:
            log.error(f"Unhandled error during {self.name}.act ({verb} '{arg_str[:50]}...'): {type(e).__name__}: {e}", exc_info=True)
            ev = f"encounters an unexpected internal error ({type(e).__name__}) during action!"

        # --- Final Processing ---
        if not ev: ev = "lingers uncertainly."

        # Add action verb to memory (if valid action attempted)
        all_valid_verbs = {"move", "propose_reproduction", "reflect", "speak",
                           "gather", "craft", "store", "take", "attack", "heal", "assign_role", "consume",
                           "request_code", "propose_code_change"} # Removed code_build
        if verb in all_valid_verbs:
             self.memory.append(f"action:{verb}")
             # If code was proposed, clear the request state
             if verb == "propose_code_change":
                  self.requested_code_target = None
        # Also record failed attempts / errors?
        elif verb.startswith("["):
             self.memory.append(f"action_fail:{verb}")


        # Add event to memory *only if it's not the build event added earlier*
        # to avoid duplication
        if not ev.startswith("crafts and builds") or not any(ev.endswith(f"(ID:{s.id}).") for s in Config.STRUCTURE_BUILD_VERBS):
             self.memory.append(f"event:{ev[:70]}") # Slightly longer event memory


        return f"[{self.id}] {self.name} @ {self.loc} (H:{self.health:.1f}|G:{self.hunger:.1f}|M:{self.morale:.1f}): {ev}"


    # --- Persistence Methods ---
    def to_dict(self) -> Dict:
        # Add code modification state to save data
        return {
            "id": self.id, "name": self.name, "loc": self.loc,
            "dna": {"traits": self.dna["traits"].tolist(), "skills": self.dna["skills"].tolist()},
            "age": self.age, "health": self.health, "hunger": self.hunger, "morale": self.morale,
            "inventory": dict(self.inventory), "profession": self.profession, "role": self.role,
            "relations": self.relations,
            "memory": list(self.memory),
            "attack_cooldown": self.attack_cooldown,
            # Code mod state is transient, not saved
        }

    @classmethod
    def from_dict(cls, data: Dict, chroma_client) -> 'AgentState':
        # (Load logic mostly unchanged, new attributes default to None)
        current_max_id = data.get("id", 0)
        if current_max_id >= AgentState._next_id:
            AgentState._next_id = current_max_id + 1

        dna_data = data.get("dna", {})
        # Handle potential loading errors for DNA gracefully
        try:
             traits = np.array(dna_data.get("traits", [])).astype(np.float32)
             if traits.shape != (len(Config.TRAITS),): traits = rand_vec(len(Config.TRAITS))
             skills = np.array(dna_data.get("skills", [])).astype(np.float32)
             if skills.shape != (len(Config.SKILLS),): skills = rand_vec(len(Config.SKILLS))
             dna = {"traits": traits, "skills": skills}
        except Exception as e:
             log.warning(f"Failed to load DNA for agent {data.get('id','?')}, using random. Error: {e}")
             dna = random_dna()

        agent = cls(
            agent_id=data["id"], name=data["name"], dna=dna, loc=data["loc"],
            age=data.get("age", 0),
            health=data.get("health", Config.MAX_HEALTH),
            hunger=data.get("hunger", 0.0),
            morale=data.get("morale", 70.0),
            inventory=Counter(data.get("inventory", {})),
            profession=data.get("profession", "Peasant"), role=data.get("role", "Commoner"),
            relations=data.get("relations", {}),
            chroma_client=chroma_client,
            attack_cooldown=data.get("attack_cooldown", 0)
        )
        agent.memory = deque(data.get("memory", []), maxlen=25) # Use new maxlen
        # Code mod state is transient and resets on load
        agent.requested_code_target = None
        agent.requested_code_content = None
        agent.last_code_proposal_result = None
        return agent


# ────────────────────────── World Class ─────────────────
# (Added methods for code modification and random events)

# Store the original source code of modifiable targets at startup
ORIGINAL_SOURCE_CODE = {}

def store_original_code():
    """Stores the source code of whitelisted targets."""
    global ORIGINAL_SOURCE_CODE
    ORIGINAL_SOURCE_CODE.clear() # Clear previous state if called again
    for target_name in Config.CODE_MODIFICATION_WHITELIST:
        try:
            target = None
            original_repr = None # To store the representation
            if '.' in target_name: # Class attribute or method
                class_name, attr_name = target_name.split('.', 1)
                target_class = None
                if class_name == "Config":
                    target_class = Config
                elif class_name == "AgentState":
                    target_class = AgentState
                elif class_name == "World":
                    target_class = World
                else:
                    log.warning(f"Unknown class '{class_name}' in whitelist target '{target_name}'"); continue

                target = getattr(target_class, attr_name, None)

                if target is None:
                     log.warning(f"Attribute/method '{attr_name}' not found in class '{class_name}'")
                     continue

                if callable(target):
                    # Check if it's a lambda or a regular function/method
                    if isinstance(target, type(lambda:0)) and target.__name__ == "<lambda>":
                         # Try to get source, might fail for complex lambdas
                         try: original_repr = inspect.getsource(target).strip()
                         except (TypeError, OSError): original_repr = repr(target) # Fallback to repr
                    else:
                         original_repr = inspect.getsource(target)
                else: # It's an attribute
                    original_repr = repr(target)

            else: # Module-level function? (Not currently whitelisted)
                log.warning(f"Direct module-level function modification not supported for '{target_name}'")
                continue

            if original_repr is not None:
                 ORIGINAL_SOURCE_CODE[target_name] = original_repr
                 # log.debug(f"Stored original source for {target_name}: {original_repr[:100]}...")
            else:
                 log.warning(f"Could not retrieve source/value for whitelisted target: {target_name}")

        except Exception as e:
            log.error(f"Error retrieving source for {target_name}: {e}", exc_info=False)
    log.info(f"Stored original source/value for {len(ORIGINAL_SOURCE_CODE)} whitelisted targets.")

class World:
    def __init__(self, load_data: Optional[Dict] = None, chroma_client=None):
        self.agents: Dict[int, AgentState] = {}
        self.structures: Dict[int, Structure] = {}
        self.locations: Dict[str, LocationNode] = {}
        self.current_tick: int = 0
        self.day: int = 0
        self.season: str = "Spring"
        self.new_agent_buffer: List[AgentState] = []
        self.agents_to_remove: List[int] = [] # Buffer for agent removal
        self.llm_semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_LLM_CALLS)
        self.chroma_client = chroma_client
        self.pending_code_updates: List[Tuple[str, str]] = [] # (target_name, code_string)
        self.last_event_trigger_tick: int = 0 # For random event timing

        if load_data:
            self._load_from_dict(load_data)
            log.info(f"World loaded. Tick: {self.current_tick}, Pop: {len(self.agents)}")
        else:
            self._initialize_locations()
            log.info("New world initialized.")

    def _initialize_locations(self):
        # (Unchanged)
        for loc_name in Config.LOCATIONS:
            base_res = Config.INITIAL_RESOURCES.get(loc_name, Counter())
            self.locations[loc_name] = LocationNode(name=loc_name, base_resources=base_res)
        log.info("Initialized location nodes.")

    def get_agents_at_loc(self, loc: str) -> List[AgentState]:
        """Get living agents at a specific location."""
        # More efficient list comprehension using agent's loc attribute
        return [a for a in self.agents.values() if a.loc == loc and a.is_alive]

    def _load_from_dict(self, data: Dict):
        # (Load logic mostly unchanged)
        AgentState._next_id = data.get("next_agent_id", 1)
        Structure._next_id = data.get("next_structure_id", 1)
        self.current_tick = data.get("current_tick", 0)
        self.day = data.get("day", 0)
        self.season = data.get("season", "Spring")
        self.locations = {name: LocationNode.from_dict(ldata) for name, ldata in data.get("locations", {}).items()}
        for loc_name in Config.LOCATIONS:
             if loc_name not in self.locations:
                 base_res = Config.INITIAL_RESOURCES.get(loc_name, Counter())
                 self.locations[loc_name] = LocationNode(name=loc_name, base_resources=base_res)
                 log.warning(f"Location {loc_name} from config missing in save, initialized.")
        self.structures = {}
        for loc_node in self.locations.values():
             for struct_id, struct in loc_node.structures.items():
                 self.structures[struct_id] = struct
        # Load agents - passes chroma_client
        self.agents = {int(aid): AgentState.from_dict(adata, self.chroma_client) for aid, adata in data.get("agents", {}).items()}
        self.last_event_trigger_tick = self.current_tick # Reset event timer on load


    def _trigger_random_event(self) -> Optional[str]:
        """Selects and applies a random world event."""
        if not Config.RANDOM_EVENTS: return None

        event_config = random.choice(Config.RANDOM_EVENTS)
        event_name = event_config["name"]
        event_type = event_config["type"]
        message_template = event_config["message"]
        location = "the realm"
        applied_effect = False

        try:
            target_loc_name = None
            if event_config.get("loc_specific", False):
                if not Config.LOCATIONS: return None # No locations to target
                target_loc_name = random.choice(Config.LOCATIONS)
                location = target_loc_name

            message = message_template.format(location=location)

            if event_type == "resource_loc" and target_loc_name:
                loc_node = self.locations.get(target_loc_name)
                if loc_node:
                    resource = event_config["resource"]
                    amount = event_config["amount"] * random.uniform(0.8, 1.2) # Add variability
                    current = loc_node.resources.get(resource, 0.0)
                    loc_node.resources[resource] = current + amount
                    applied_effect = True
                    log.info(f"Event '{event_name}' triggered: Added {amount:.1f} {resource} to {target_loc_name}.")

            elif event_type == "morale_loc" and target_loc_name:
                amount = event_config["amount"]
                agents_affected = 0
                for agent in self.get_agents_at_loc(target_loc_name):
                    agent.morale = np.clip(agent.morale + amount * random.uniform(0.8, 1.2), Config.MIN_MORALE, Config.MAX_MORALE)
                    agents_affected += 1
                if agents_affected > 0: applied_effect = True
                log.info(f"Event '{event_name}' triggered: Adjusted morale by {amount} for {agents_affected} agents at {target_loc_name}.")

            elif event_type == "morale_all":
                amount = event_config["amount"]
                agents_affected = 0
                for agent in self.agents.values():
                    if agent.is_alive:
                        agent.morale = np.clip(agent.morale + amount * random.uniform(0.8, 1.2), Config.MIN_MORALE, Config.MAX_MORALE)
                        agents_affected += 1
                if agents_affected > 0: applied_effect = True
                log.info(f"Event '{event_name}' triggered: Adjusted morale globally by {amount} for {agents_affected} agents.")

            elif event_type == "flavor":
                 log.info(f"Event '{event_name}' triggered: {message}")
                 applied_effect = True # Flavor counts as applied

            # Add more event types here (hunger_all, etc.) if defined in Config

            if applied_effect:
                 return f"[EVENT] {message}"
            else:
                 log.debug(f"Event '{event_name}' triggered but had no applicable target/effect.")
                 return None # Event happened but didn't do anything

        except Exception as e:
            log.error(f"Error processing random event '{event_name}': {e}", exc_info=True)
            return f"[EVENT] An unusual occurrence was noted, but its meaning is unclear." # Generic message on error


    def tick_update(self):
        """Advance world time, process needs, regen, events, code changes, agent removal."""
        self.current_tick += 1
        tick_events = []

        # --- Apply Pending Code Changes ---
        # Apply changes submitted in the *previous* tick's actions
        if self.pending_code_updates:
            log.info(f"Applying {len(self.pending_code_updates)} pending code updates...")
            for target_name, code_string in self.pending_code_updates:
                 success, msg = self._apply_code_change_unsafe(target_name, code_string)
                 event_msg = f"Code Change Applied: Target='{target_name}', Result='{msg}'"
                 tick_events.append(f"[SYSTEM] {event_msg}")
                 if not success: log.error(f"Error applying code change for {target_name}: {msg}")
            self.pending_code_updates.clear()


        # --- Daily / Seasonal Updates ---
        if self.current_tick % Config.TICKS_PER_DAY == 0:
            self.day += 1
            for loc_node in self.locations.values():
                loc_node.regenerate_resources()
            # log.debug(f"Day {self.day} started. Resources regenerated.") # Too noisy

        # --- Random Event Check ---
        if self.current_tick - self.last_event_trigger_tick >= Config.EVENT_CHECK_INTERVAL:
            if random.random() < Config.EVENT_BASE_CHANCE:
                event_message = self._trigger_random_event()
                if event_message:
                    tick_events.append(event_message)
            self.last_event_trigger_tick = self.current_tick # Reset timer regardless of trigger

        # --- Agent Needs Update ---
        newly_dead_agent_ids = []
        for agent in list(self.agents.values()): # Iterate over copy for safe removal later
            if agent.is_alive:
                try: # Protect against errors in agent update_needs (especially if modified)
                    agent.update_needs()
                except Exception as e:
                     log.error(f"Error during agent {agent.id} update_needs: {e}", exc_info=True)
                     # Potentially penalize agent or revert code? For now, just log.
                if not agent.is_alive:
                    newly_dead_agent_ids.append(agent.id)
                    log.info(f"Agent {agent.id} ({agent.name}) died. HP reached {agent.health:.1f}.")
                    event_msg = f"Agent {agent.name} [{agent.id}] has succumbed to their ailments."
                    if agent.hunger > Config.MAX_HUNGER * 1.5: event_msg += " (Likely starvation)"
                    elif agent.age >= Config.MAX_AGE: event_msg += " (Old age)"
                    tick_events.append(f"[DEATH] {event_msg}")

        # Add newly dead agents to the removal buffer
        for agent_id in newly_dead_agent_ids:
            if agent_id not in self.agents_to_remove:
                 self.agents_to_remove.append(agent_id)

        # --- Process Agent Removal Buffer ---
        if self.agents_to_remove:
            for agent_id in self.agents_to_remove:
                 if agent_id in self.agents:
                     # Basic item drop simulation: Drop some items on the ground (represented by adding to location resources)
                     dead_agent = self.agents[agent_id]
                     loc_node = self.locations.get(dead_agent.loc)
                     if loc_node:
                         items_dropped = []
                         for item, count in dead_agent.inventory.items():
                             if item in Config.RESOURCE_REGEN_RATE and random.random() < 0.5: # Chance to drop resources
                                 drop_count = random.randint(1, count)
                                 loc_node.resources[item] = loc_node.resources.get(item, 0.0) + drop_count
                                 items_dropped.append(f"{drop_count} {item}")
                         if items_dropped:
                              drop_msg = f"Items dropped by {dead_agent.name}: {', '.join(items_dropped)}"
                              log.info(drop_msg)
                              tick_events.append(f"[INFO] {drop_msg}")

                     log.info(f"Agent {dead_agent.name} [{agent_id}] removed from world (killed).")
                     del self.agents[agent_id]
                     # Clean up relationships pointing to the dead agent
                     for other_agent in self.agents.values():
                          if agent_id in other_agent.relations:
                              del other_agent.relations[agent_id]
                     # Clean up structures owned by the dead agent? Transfer ownership? Make owner 0?
                     for structure in self.structures.values():
                          if structure.owner_id == agent_id:
                              structure.owner_id = 0 # Mark as unowned
                              tick_events.append(f"[INFO] Structure {structure.name}[{structure.id}] is now unowned.")

                 if agent_id in registry.agents: # Update registry too if it somehow didn't get deleted above
                     registry.agents[agent_id].health = 0 # Ensure marked as dead in registry


            self.agents_to_remove.clear()

        # --- Process New Agents ---
        if self.new_agent_buffer:
            log.info(f"Adding {len(self.new_agent_buffer)} new agents to world.")
            for child in self.new_agent_buffer:
                if child.id not in self.agents:
                    self.agents[child.id] = child
                    log.info(f"Added new agent {child.name} [{child.id}] to world at {child.loc}.")
                    tick_events.append(f"[BIRTH] A new child, {child.name}, has been born at {child.loc}.")
                else: log.error(f"Failed to add agent {child.id}, ID already exists.")
            self.new_agent_buffer.clear()

        # --- Respawn Logic ---
        living_agents_count = len(self.get_living_agents())
        if living_agents_count < Config.MIN_POP:
            needed = Config.MIN_POP - living_agents_count
            log.info(f"Population low ({living_agents_count}/{Config.MIN_POP}). Spawning {needed} new agent(s).")
            for _ in range(needed):
                if len(self.get_living_agents()) >= Config.MIN_POP: break
                name = generate_themed_name()
                agent = AgentState(name, chroma_client=self.chroma_client)
                if registry.add(agent):
                    self.agents[agent.id] = agent
                    tick_events.append(f"[SPAWN] A stranger named {agent.name} arrives at {agent.loc}.")
                else: log.error(f"Failed to add spawned agent {name} to registry.")

        return tick_events # Return events generated during the world update

    def add_structure(self, structure: Structure):
         """Adds a structure to the world and the correct location."""
         if structure.id in self.structures:
             log.warning(f"Structure ID {structure.id} conflict. Cannot add.")
             return False
         self.structures[structure.id] = structure
         if structure.loc in self.locations:
             self.locations[structure.loc].add_structure(structure)
             return True
         else:
             log.error(f"Structure {structure.id} added to world but invalid location {structure.loc}")
             # Attempt recovery: remove from main dict if location invalid
             if structure.id in self.structures: del self.structures[structure.id]
             return False # Indicate issue

    def get_structures_at_loc(self, loc: str) -> List[Structure]:
        """Get structures at a specific location."""
        if loc in self.locations:
            return list(self.locations[loc].structures.values())
        return []

    def get_living_agents(self) -> List[AgentState]:
        # Ensure agents dict hasn't been corrupted
        return [a for a in self.agents.values() if isinstance(a, AgentState) and a.is_alive]

    def kill_agent(self, agent: AgentState):
        """Marks an agent for removal in the next tick update."""
        if agent.id not in self.agents_to_remove:
             self.agents_to_remove.append(agent.id)
             agent.health = 0 # Ensure health is 0 immediately


    def resolve_attack(self, attacker: AgentState, defender: AgentState) -> Tuple[float, float]:
        """Resolves combat between two agents. (Potentially modifiable)"""
        if not attacker.is_alive or not defender.is_alive: return 0.0, 0.0
        if attacker.attack_cooldown > 0: return 0.0, 0.0

        try:
            # Use getattr for potentially modified config values
            base_damage_factor = getattr(Config, "BASE_ATTACK_DAMAGE_FACTOR", 18) # Using updated default

            attacker_war = attacker.dna["skills"][Config.SKILLS.index("war")]
            defender_war = defender.dna["skills"][Config.SKILLS.index("war")]

            # Simple weapon bonus
            attacker_wpn = 0.3 if "Iron_Sword" in attacker.inventory else 0.15 if "Stone_Axe" in attacker.inventory else 0.08 if "Wooden_Spear" in attacker.inventory else 0
            defender_wpn = 0.3 if "Iron_Sword" in defender.inventory else 0.15 if "Stone_Axe" in defender.inventory else 0.08 if "Wooden_Spear" in defender.inventory else 0

            # Add slight randomness and skill influence
            attack_roll = random.uniform(0.5, 1.0) + attacker_war * 1.5 + attacker_wpn
            defend_roll = random.uniform(0.3, 0.8) + defender_war * 1.2 # Defense slightly less effective

            damage_dealt = 0.0
            if attack_roll > defend_roll:
                 damage_multiplier = (attack_roll - defend_roll) # Raw difference matters
                 damage_dealt = max(1.0, damage_multiplier * base_damage_factor * random.uniform(0.7, 1.3)) # Wider random range

            defender.health -= damage_dealt

            # Counter attack chance depends more on defender skill and aggression (wrath trait?)
            counter_chance = 0.2 + defender_war * 0.3 + defender.dna["traits"][Config.TRAITS.index("wrath")] * 0.2
            damage_received = 0.0
            if defender.is_alive and random.random() < counter_chance:
                 counter_attack = random.uniform(0.4, 0.9) + defender_war * 1.3 + defender_wpn
                 counter_defend = random.uniform(0.3, 0.7) + attacker_war * 1.0
                 if counter_attack > counter_defend:
                     counter_dmg_mult = (counter_attack - counter_defend)
                     counter_dmg = max(1.0, counter_dmg_mult * base_damage_factor * 0.7 * random.uniform(0.7, 1.2)) # Counter attacks slightly weaker base
                     damage_received = counter_dmg
                     attacker.health -= damage_received

            log.debug(f"Combat: {attacker.id}(A:{attack_roll:.2f}|W:{attacker_wpn:.1f}) vs {defender.id}(D:{defend_roll:.2f}|W:{defender_wpn:.1f}). Dmg: {damage_dealt:.1f}/{damage_received:.1f}.")

            attacker.health = max(0.0, attacker.health)
            defender.health = max(0.0, defender.health)

            # Mark for death immediately if health drops below zero
            if attacker.health <= 0: self.kill_agent(attacker)
            if defender.health <= 0: self.kill_agent(defender)


            return damage_dealt, damage_received
        except Exception as e:
             log.error(f"Error during resolve_attack between {attacker.id} and {defender.id}: {e}", exc_info=True)
             return 0.0, 0.0 # Return no damage if calculation fails


    # --- Code Modification Methods ---

    def safe_get_target_code(self, target_name: str) -> Optional[str]:
        """Safely retrieves the *original* source code/value representation of a whitelisted target."""
        if target_name not in Config.CODE_MODIFICATION_WHITELIST:
            log.warning(f"Attempt to get code for non-whitelisted target: {target_name}")
            return None
        # Return the stored original source code/value representation
        original_code = ORIGINAL_SOURCE_CODE.get(target_name)
        if original_code is None:
             log.error(f"Original source code/value for whitelisted target '{target_name}' was not found/stored.")
        return original_code

    def safe_execute_agent_code(self, agent_id: int, target_name: str, code_string: str) -> Tuple[bool, str]:
        """Validates and queues agent-proposed code for execution."""
        if target_name not in Config.CODE_MODIFICATION_WHITELIST:
            return False, "Target not allowed for modification."

        # 1. Line Count Check
        lines = code_string.strip().splitlines()
        if len(lines) > Config.MAX_CODE_LINES:
            return False, f"Code exceeds maximum lines ({len(lines)} > {Config.MAX_CODE_LINES})."
        if not lines: return False, "Empty code proposed."

        # 2. Basic Forbidden Pattern Check
        for pattern in Config.FORBIDDEN_CODE_PATTERNS:
            if pattern in code_string.lower():
                return False, f"Code contains forbidden pattern ('{pattern}')."

        # 3. Syntax Check (and basic validation for Config values)
        try:
            if target_name.startswith("Config."):
                # Try to evaluate it in a restricted way to check basic validity
                allowed_globals = {'__builtins__': {'True': True, 'False': False, 'None': None, 'int': int, 'float': float, 'str': str, 'list': list, 'dict': dict, 'Counter': Counter, 'lambda': lambda:None}} # Allow lambda keyword
                value = eval(code_string, allowed_globals, {})
                 # Add more sophisticated type/value range checks here if needed
            else: # For methods, just check AST syntax
                ast.parse(code_string)
        except SyntaxError as e:
            return False, f"Syntax Error: {e}"
        except NameError as e:
             return False, f"Name Error in Config value: {e}. Use only basic types/values or simple lambdas."
        except Exception as e:
            return False, f"Validation failed: {type(e).__name__}: {e}"

        # 4. Queue for execution at start of next tick
        # This avoids modifying code while agents are potentially using it mid-tick
        self.pending_code_updates.append((target_name, code_string))
        log.info(f"Agent {agent_id} queued code change for '{target_name}'. Will apply next tick.")

        return True, "Code validated and queued for execution next tick."

    def _apply_code_change_unsafe(self, target_name: str, code_string: str) -> Tuple[bool, str]:
         """Applies the validated code change. Called by tick_update. UNSAFE."""
         log.warning(f"Applying code change for {target_name}. Previous source/value:\n{ORIGINAL_SOURCE_CODE.get(target_name, 'N/A')}\nNew code:\n{code_string}")

         try:
            if target_name.startswith("Config."):
                 # Modify Config class attribute
                 attr_name = target_name.split('.', 1)[1]
                 # VERY basic sandbox for evaluating the value
                 allowed_globals = {'__builtins__': {'True': True, 'False': False, 'None': None, 'int': int, 'float': float, 'str': str, 'list': list, 'dict': dict, 'Counter': Counter, 'lambda': lambda:None}} # Allow lambda keyword but it won't capture scope correctly here generally.
                 value = eval(code_string, allowed_globals, {}) # DANGEROUS - Eval is risky! Only allows evaluating simple values or basic lambdas.

                 # Type check? Maybe ensure it matches original type?
                 original_value_repr = ORIGINAL_SOURCE_CODE.get(target_name)
                 original_type = type(getattr(Config, attr_name, None)) # Get type of current value

                 # Basic check: Don't allow changing type drastically for non-callables
                 if not callable(getattr(Config, attr_name, None)) and not isinstance(value, original_type):
                      # Allow int to float promotion or vice versa perhaps?
                      if not ((isinstance(value, (int, float))) and (original_type in [int, float])):
                          return False, f"Type mismatch for Config.{attr_name}. Expected ~{original_type}, got {type(value)}."

                 setattr(Config, attr_name, value)
                 return True, f"Config.{attr_name} updated."

            elif '.' in target_name:
                 # Modify a class method (e.g., AgentState.update_needs)
                 class_name, method_name = target_name.split('.', 1)
                 target_class = None
                 if class_name == "AgentState": target_class = AgentState
                 elif class_name == "World": target_class = World
                 else: return False, f"Unknown class '{class_name}'"

                 # Execute the code string to define the new function
                 # Use a restricted environment
                 local_exec_namespace = {}
                 # Provide access to necessary modules/classes VERY carefully
                 safe_globals = {
                     '__builtins__': { 'min': min, 'max': max, 'random': random, 'np': np, 'Counter': Counter, 'log': log, # Allow logging
                                        'True':True, 'False':False, 'None':None, 'int':int, 'float':float, 'str':str, 'list':list, 'dict':dict, 'getattr': getattr, # Safe builtins
                                        'isinstance': isinstance, 'callable': callable, 'type': type, 'any':any, # More utils
                                        'HarmCategory': HarmCategory, 'HarmBlockThreshold': HarmBlockThreshold # Needed for Gemini if modifying LLM call
                                        },
                     'Config': Config, # Make Config accessible directly
                     'AgentState': AgentState,
                     'World': World,
                     'Structure': Structure,
                     'LocationNode': LocationNode, # Make domain classes available
                     'random': random,
                     'np': np,
                     'log': log,
                 }
                 # Add the class itself to the globals if needed for super() etc.
                 safe_globals[class_name] = target_class

                 # Execute the function definition
                 exec(code_string, safe_globals, local_exec_namespace)

                 # Find the newly defined function in the local namespace
                 new_method = local_exec_namespace.get(method_name)
                 if not new_method or not callable(new_method):
                     # Check if it defined a class or something else unexpected
                     defined_items = list(local_exec_namespace.keys())
                     return False, f"Code did not define function '{method_name}' correctly. Defined: {defined_items}"


                 # Replace the method on the class
                 setattr(target_class, method_name, new_method)
                 return True, f"{target_name} method updated."

            else:
                 return False, "Unsupported target type (must be Config attribute or Class.method)."

         except Exception as e:
            log.error(f"CRITICAL ERROR applying code change for {target_name}: {e}", exc_info=True)
            # Attempt to restore original code? Difficult to do safely mid-error. Let's try.
            log.warning(f"Attempting to restore original code for {target_name} due to error.")
            if restore_specific_original_code(target_name):
                 return False, f"Execution Error: {type(e).__name__}. ORIGINAL CODE RESTORED."
            else:
                 return False, f"Execution Error: {type(e).__name__}. FAILED TO RESTORE ORIGINAL CODE."

    # --- Persistence Methods ---
    def to_dict(self) -> Dict:
        """Serialize the entire world state."""
        # Note: Modified code is NOT saved. Simulation reverts to original code on load.
        return {
            "current_tick": self.current_tick, "day": self.day, "season": self.season,
            "next_agent_id": AgentState._next_id, "next_structure_id": Structure._next_id,
            "agents": {aid: agent.to_dict() for aid, agent in self.agents.items()},
            "structures": {sid: struct.to_dict() for sid, struct in self.structures.items()},
            "locations": {lname: loc.to_dict() for lname, loc in self.locations.items()},
            "last_event_trigger_tick": self.last_event_trigger_tick,
        }

    @classmethod
    def from_file(cls, filepath: Path, chroma_client) -> Optional['World']:
        # (Unchanged load, but restore happens after)
        if not filepath.exists(): log.error(f"Save file not found: {filepath}"); return None
        try:
            with gzip.open(filepath, "rt", encoding='utf-8') as f: data = json.load(f)
            # Ensure simulation starts with original code definitions after loading
            log.info("Restoring original code definitions after loading save file...")
            restore_original_code() # Ensure we use original code definitions
            world = cls(load_data=data, chroma_client=chroma_client)
            # Ensure next IDs from save file are respected
            AgentState._next_id = data.get("next_agent_id", AgentState._next_id)
            Structure._next_id = data.get("next_structure_id", Structure._next_id)
            world.last_event_trigger_tick = data.get("last_event_trigger_tick", world.current_tick) # Load event timer
            return world
        except Exception as e:
            log.error(f"Failed to load world from {filepath}: {e}", exc_info=True)
            return None

    def save_to_file(self, filepath: Path):
        # (Unchanged)
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            # Make sure agent dict doesn't contain dead agents marked for removal
            world_dict = self.to_dict()
            # Filter out agents who might be pending removal but still in self.agents before save
            world_dict["agents"] = {aid: agent_data for aid, agent_data in world_dict["agents"].items() if int(aid) not in self.agents_to_remove}

            with gzip.open(filepath, "wt", encoding='utf-8') as f:
                json.dump(world_dict, f, indent=2)
            log.info(f"World state saved successfully to {filepath}")
        except Exception as e:
            log.error(f"Failed to save world to {filepath}: {e}", exc_info=True)

# ───────────────────── Registry Class ───────────────────
# (Unchanged)
class Registry:
    def __init__(self):
        self.agents: Dict[int, AgentState] = {}

    def add(self, agent: AgentState) -> bool:
        if agent.id in self.agents:
             log.warning(f"Agent ID {agent.id} conflict in registry.")
             return False
        self.agents[agent.id] = agent
        # log.debug(f"Agent {agent.name} [{agent.id}] added to registry.")
        return True

    def get_agent_name(self, agent_id: int) -> str:
         agent = self.agents.get(agent_id)
         # Handle case where agent might be dead but still queried briefly
         return agent.name if agent else f"Unknown[{agent_id}]"

    def is_agent_alive(self, agent_id: int) -> bool:
        agent = self.agents.get(agent_id)
        return agent is not None and agent.is_alive

# Global registry instance
registry = Registry()

# --- Utility to restore original code (e.g., after loading a save or error) ---
def restore_specific_original_code(target_name: str) -> bool:
    """Restores a single whitelisted code target to its original state."""
    if target_name not in ORIGINAL_SOURCE_CODE:
        log.error(f"Cannot restore '{target_name}': Original code not found.")
        return False

    original_code_repr = ORIGINAL_SOURCE_CODE[target_name]
    log.warning(f"Restoring original code for target: {target_name}")
    try:
        if target_name.startswith("Config."):
            attr_name = target_name.split('.', 1)[1]
            allowed_globals = {'__builtins__': {'True': True, 'False': False, 'None': None, 'int': int, 'float': float, 'str': str, 'list': list, 'dict': dict, 'Counter': Counter, 'lambda': lambda:None}}
            original_value = eval(original_code_repr, allowed_globals, {})
            setattr(Config, attr_name, original_value)
            log.info(f"Restored Config.{attr_name} successfully.")
            return True
        elif '.' in target_name:
            class_name, method_name = target_name.split('.', 1)
            target_class = None
            if class_name == "AgentState": target_class = AgentState
            elif class_name == "World": target_class = World
            else: return False # Should not happen if whitelisted

            local_exec_namespace = {}
            safe_globals = {
                 '__builtins__': { 'min': min, 'max': max, 'random': random, 'np': np, 'Counter': Counter, 'log': log,
                                    'True':True, 'False':False, 'None':None, 'int':int, 'float':float, 'str':str, 'list':list, 'dict':dict, 'getattr': getattr,
                                    'isinstance': isinstance, 'callable': callable, 'type': type, 'any':any },
                 'Config': Config, 'AgentState': AgentState, 'World': World, 'random': random, 'np': np, 'log': log,
                 'Structure': Structure, 'LocationNode': LocationNode,
            }
            safe_globals[class_name] = target_class
            exec(original_code_repr, safe_globals, local_exec_namespace) # Execute original source
            original_method = local_exec_namespace.get(method_name)
            if original_method and callable(original_method):
                 setattr(target_class, method_name, original_method)
                 log.info(f"Restored method {target_name} successfully.")
                 return True
            else:
                 log.error(f"Failed to restore method {method_name} from original source execution.")
                 return False
        else:
            log.error(f"Unsupported target type for restoration: {target_name}")
            return False
    except Exception as e:
        log.error(f"CRITICAL error restoring original code for {target_name}: {e}", exc_info=True)
        return False


def restore_original_code():
     """Restores all whitelisted code targets to their original state."""
     log.warning("Restoring original code for all whitelisted targets...")
     applied_count = 0
     failed_count = 0
     if not ORIGINAL_SOURCE_CODE: store_original_code() # Ensure originals are loaded

     for target_name in Config.CODE_MODIFICATION_WHITELIST: # Iterate through whitelist ensure all are attempted
         if target_name in ORIGINAL_SOURCE_CODE:
            if restore_specific_original_code(target_name):
                applied_count += 1
            else:
                failed_count += 1
         else:
             log.error(f"Cannot restore '{target_name}': Original code was not stored.")
             failed_count += 1
     log.info(f"Finished restoring code: {applied_count} successful, {failed_count} failed.")

# ───────────────────── GUI Class ───────────────────────
# (Mostly unchanged, added confirmation for code changes)
class AppRoot(tk.Tk): pass

class SimulationGUI:
    def __init__(self, root: tk.Tk, update_queue: queue.Queue, action_queue: queue.Queue):
        self.root = root
        self.update_queue = update_queue
        self.action_queue = action_queue

        self.root.title("ASOIAF Autonomous Simulation v3 (Code Mod Enabled: {})".format(Config.ALLOW_CODE_MODIFICATION))
        self.root.minsize(1200, 800)

        self._configure_styles()
        self._create_widgets()
        self._layout_widgets()
        self.process_queue()

    def _configure_styles(self):
        # (Unchanged)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'))
        style.configure("Treeview", rowheight=25, font=('Helvetica', 9))
        style.map('Treeview', background=[('selected', '#ADD8E6')])
        style.configure("Healthy.Treeview", background="#DFF0D8")
        style.configure("Injured.Treeview", background="#FCF8E3")
        style.configure("Critical.Treeview", background="#F2DEDE")
        style.configure("Dead.Treeview", background="#D3D3D3", foreground="#555555")
        style.configure("TLabel", padding=5)
        style.configure("TFrame", background="#ECECEC")
        style.configure("Header.TLabel", background="#4682B4", foreground="white", font=('Helvetica', 14, 'bold'), anchor=tk.CENTER)
        style.configure("Footer.TLabel", background="#D3D3D3", font=('Helvetica', 8), anchor=tk.W)
        style.configure("Events.TLabel", font=('Helvetica', 11, 'bold'), background="#ECECEC", foreground="#333")
        style.configure("Structures.TLabel", font=('Helvetica', 11, 'bold'), background="#ECECEC", foreground="#333")
        style.configure("Agents.TLabel", font=('Helvetica', 11, 'bold'), background="#ECECEC", foreground="#333")
        style.configure("Map.TLabel", font=('Helvetica', 11, 'bold'), background="#ECECEC", foreground="#333")

    def _create_widgets(self):
        # (Unchanged)
        self.header_frame = ttk.Frame(self.root, height=40, style="TFrame")
        self.main_frame = ttk.Frame(self.root, style="TFrame")
        self.footer_frame = ttk.Frame(self.root, height=30, style="TFrame")
        self.left_frame = ttk.Frame(self.main_frame, style="TFrame")
        self.right_frame = ttk.Frame(self.main_frame, style="TFrame")
        self.map_frame = ttk.Frame(self.left_frame, style="TFrame")
        self.agents_frame = ttk.Frame(self.left_frame, style="TFrame")
        self.events_frame = ttk.Frame(self.right_frame, style="TFrame")
        self.structures_frame = ttk.Frame(self.right_frame, style="TFrame")
        self.tick_label = ttk.Label(self.header_frame, text="Tick: 0 | Day: 0 | Season: ?", style="Header.TLabel")
        self.status_label = ttk.Label(self.footer_frame, text="Status: Initializing...", style="Footer.TLabel")
        self.map_label = ttk.Label(self.map_frame, text="World Map", style="Map.TLabel")
        self.agents_label = ttk.Label(self.agents_frame, text="Agents (0/0)", style="Agents.TLabel")
        self.events_label = ttk.Label(self.events_frame, text="Recent Events", style="Events.TLabel")
        self.structures_label = ttk.Label(self.structures_frame, text="World Structures (0)", style="Structures.TLabel")
        self.map_canvas = tk.Canvas(self.map_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="#DDEEFF", relief=tk.SUNKEN, bd=1)
        self._draw_location_markers()
        self.pop_tree = ttk.Treeview(
            self.agents_frame, columns=("ID", "Name", "Loc", "HP", "Hunger", "Morale", "Role", "Inventory"),
            show="headings", selectmode="browse"
        )
        self.pop_tree.heading("ID", text="ID", anchor=tk.W); self.pop_tree.column("ID", width=40, stretch=False, anchor=tk.E)
        self.pop_tree.heading("Name", text="Name", anchor=tk.W); self.pop_tree.column("Name", width=100, stretch=False)
        self.pop_tree.heading("Loc", text="Location", anchor=tk.W); self.pop_tree.column("Loc", width=90, stretch=False)
        self.pop_tree.heading("HP", text="HP", anchor=tk.E); self.pop_tree.column("HP", width=40, stretch=False, anchor=tk.E)
        self.pop_tree.heading("Hunger", text="Hunger", anchor=tk.E); self.pop_tree.column("Hunger", width=50, stretch=False, anchor=tk.E)
        self.pop_tree.heading("Morale", text="Morale", anchor=tk.E); self.pop_tree.column("Morale", width=50, stretch=False, anchor=tk.E)
        self.pop_tree.heading("Role", text="Role", anchor=tk.W); self.pop_tree.column("Role", width=80, stretch=False)
        self.pop_tree.heading("Inventory", text="Inventory", anchor=tk.W); self.pop_tree.column("Inventory", width=180, stretch=True)
        self.pop_scroll = ttk.Scrollbar(self.agents_frame, orient="vertical", command=self.pop_tree.yview)
        self.pop_tree.configure(yscrollcommand=self.pop_scroll.set)
        self.pop_tree.tag_configure('Healthy', background='#DFF0D8')
        self.pop_tree.tag_configure('Injured', background='#FCF8E3')
        self.pop_tree.tag_configure('Critical', background='#F2DEDE')
        self.pop_tree.tag_configure('Dead', background='#D3D3D3', foreground='#555555')
        self.event_text = scrolledtext.ScrolledText(self.events_frame, wrap=tk.WORD, font=('Helvetica', 9), relief=tk.SUNKEN, bd=1, state=tk.DISABLED)
        self.structure_text = scrolledtext.ScrolledText(self.structures_frame, wrap=tk.WORD, height=10, state=tk.DISABLED, font=('Helvetica', 9), relief=tk.SUNKEN, bd=1)

    def _draw_location_markers(self):
        # (Unchanged)
        self.map_canvas.delete("location_marker")
        for loc, (x, y) in LOCATION_COORDS.items():
            self.map_canvas.create_oval(x - LOCATION_RADIUS, y - LOCATION_RADIUS, x + LOCATION_RADIUS, y + LOCATION_RADIUS, fill="#AAAAAA", outline="black", width=1, tags=("location_marker", loc))
            self.map_canvas.create_text(x, y + LOCATION_RADIUS + 5, text=loc, anchor=tk.N, font=('Helvetica', 8, 'bold'), tags=("location_marker", loc))

    def _layout_widgets(self):
        # (Unchanged)
        self.header_frame.pack(fill=tk.X, side=tk.TOP, expand=False)
        self.footer_frame.pack(fill=tk.X, side=tk.BOTTOM, expand=False)
        self.main_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP, padx=5, pady=5)
        self.tick_label.pack(fill=tk.X, expand=True, padx=10)
        self.status_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.main_frame.columnconfigure(0, weight=3, minsize=650)
        self.main_frame.columnconfigure(1, weight=2, minsize=450)
        self.main_frame.rowconfigure(0, weight=1)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.left_frame.rowconfigure(0, weight=3); self.left_frame.rowconfigure(1, weight=2); self.left_frame.columnconfigure(0, weight=1)
        self.map_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        self.agents_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        self.map_frame.rowconfigure(1, weight=1); self.map_frame.columnconfigure(0, weight=1)
        self.map_label.grid(row=0, column=0, sticky="ew")
        self.map_canvas.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.agents_frame.rowconfigure(1, weight=1); self.agents_frame.columnconfigure(0, weight=1); self.agents_frame.columnconfigure(1, weight=0)
        self.agents_label.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0,5))
        self.pop_tree.grid(row=1, column=0, sticky="nsew")
        self.pop_scroll.grid(row=1, column=1, sticky="ns")
        self.right_frame.rowconfigure(0, weight=3); self.right_frame.rowconfigure(1, weight=2); self.right_frame.columnconfigure(0, weight=1)
        self.events_frame.grid(row=0, column=0, sticky="nsew", pady=(0,5))
        self.structures_frame.grid(row=1, column=0, sticky="nsew", pady=(5,0))
        self.events_frame.rowconfigure(1, weight=1); self.events_frame.columnconfigure(0, weight=1)
        self.events_label.grid(row=0, column=0, sticky="ew")
        self.event_text.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.structures_frame.rowconfigure(1, weight=1); self.structures_frame.columnconfigure(0, weight=1)
        self.structures_label.grid(row=0, column=0, sticky="ew")
        self.structure_text.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

    def process_queue(self):
        # (Unchanged from v2)
        try:
            while True:
                update_data = self.update_queue.get_nowait()#timeout=0.01) # Using get_nowait for potentially faster processing
                self.update_gui(update_data)
                self.update_queue.task_done()
        except queue.Empty:
            pass
        except Exception as e:
            log.error(f"Error processing update queue in GUI: {e}", exc_info=True)
            try: self.status_label.config(text=f"Status: GUI Error: {e}")
            except: pass # Avoid errors if GUI is closing

        self.root.after(Config.GUI_UPDATE_INTERVAL, self.process_queue)

    def get_health_tag(self, health, max_health):
         # Handle potential non-numeric health temporarily
         try:
             health_f = float(health)
             max_health_f = float(max_health)
             ratio = health_f / max_health_f if max_health_f > 0 else 0
             if health_f <= 0: return 'Dead'
             if ratio < 0.3: return 'Critical'
             if ratio < 0.7: return 'Injured'
             return 'Healthy'
         except (ValueError, TypeError):
              return 'Unknown' # Fallback tag


    def update_gui(self, data: Dict):
        # (Mostly unchanged, slight tweak for agent count label)
        try:
            tick = data['tick']
            day = data['day']
            season = data['season']
            self.tick_label.config(text=f"Tick: {tick} | Day: {day} | Season: {season}")
            self.status_label.config(text=f"Status: {data['status']}")

            active_agents_data = data['agents']
            structures_data = data['structures']
            new_events = data['new_events']

            # Update Agent Table
            living_agent_count = len([a for a in active_agents_data if a.get('health', 0) > 0])
            self.agents_label.config(text=f"Agents ({living_agent_count} / {Config.MAX_POP})") # Show current living count
            selected_item = self.pop_tree.focus()
            current_tree_ids = set(self.pop_tree.get_children())
            agent_tree_map = {f"agent_{a['id']}": a for a in active_agents_data}
            agent_tree_ids_to_display = set(agent_tree_map.keys())

            ids_to_remove = current_tree_ids - agent_tree_ids_to_display
            if ids_to_remove: self.pop_tree.delete(*list(ids_to_remove))

            for item_id, agent_dict in agent_tree_map.items():
                inventory_str = agent_dict.get('inventory_summary', '?') # Use summary if provided
                health = agent_dict.get('health', 0)
                health_tag = self.get_health_tag(health, Config.MAX_HEALTH)
                values = (
                    agent_dict['id'], agent_dict['name'], agent_dict['loc'],
                    f"{health:.1f}", # Display float HP
                    f"{agent_dict.get('hunger', 0):.1f}", # Display float Hunger
                    f"{agent_dict.get('morale', 0):.1f}", # Display float Morale
                    agent_dict.get('role', '?'), inventory_str[:60] # Limit display length
                )
                if self.pop_tree.exists(item_id):
                    self.pop_tree.item(item_id, values=values, tags=(health_tag,))
                else:
                    # Sort by ID when inserting new items
                    all_items = list(self.pop_tree.get_children(''))
                    insert_pos = tk.END
                    # Find where to insert based on ID (simple numeric sort assumption)
                    agent_id_int = agent_dict['id']
                    for existing_item_id_str in all_items:
                        try:
                            existing_agent_id = int(existing_item_id_str.split('_')[1])
                            if agent_id_int < existing_agent_id:
                                insert_pos = self.pop_tree.index(existing_item_id_str)
                                break
                        except (IndexError, ValueError):
                            continue # Ignore malformed item IDs

                    self.pop_tree.insert("", insert_pos, iid=item_id, values=values, tags=(health_tag,))


            if selected_item and self.pop_tree.exists(selected_item): self.pop_tree.focus(selected_item)

            # Update Event Log
            if new_events:
                self.event_text.config(state=tk.NORMAL)
                for event in reversed(new_events):
                     self.event_text.insert('1.0', event + "\n") # Add timestamp? No, done by sim.
                     # Simple coloring based on keywords
                     event_lower = event.lower()
                     if "[death]" in event_lower or "fatal error" in event_lower or "critical" in event_lower:
                         self.event_text.tag_add("error", "1.0", "1.end")
                     elif "[event]" in event_lower or "[info]" in event_lower or "[birth]" in event_lower or "[spawn]" in event_lower:
                          self.event_text.tag_add("info", "1.0", "1.end")
                     elif "error" in event_lower or "fail" in event_lower:
                          self.event_text.tag_add("warning", "1.0", "1.end")

                # Configure tags (add more as needed)
                self.event_text.tag_config("error", foreground="red")
                self.event_text.tag_config("warning", foreground="orange")
                self.event_text.tag_config("info", foreground="blue")

                # Trim
                self.event_text.delete(f"{Config.EVENT_LOG_DISPLAY_SIZE+1}.0", tk.END)
                self.event_text.config(state=tk.DISABLED)

            # Update Structures List
            self.structures_label.config(text=f"World Structures ({len(structures_data)})")
            self.structure_text.config(state=tk.NORMAL)
            self.structure_text.delete('1.0', tk.END)
            sorted_structures = sorted(structures_data, key=lambda s: (s['loc'], s['id']))
            for s_dict in sorted_structures:
                items_str = ", ".join(f"{k}:{v}" for k, v in s_dict.get('stored_items', {}).items())
                owner_id = s_dict.get('owner_id',0)
                owner_name = registry.get_agent_name(owner_id) if owner_id != 0 else "Unowned"
                line = (f"- {s_dict['name']} [ID:{s_dict['id']}] @{s_dict['loc']} "
                        f"(Owner:{owner_name}, Dur:{s_dict.get('durability', '?')})\n"
                        f"  └ Items: {items_str or 'empty'}\n")
                self.structure_text.insert(tk.END, line)
            self.structure_text.config(state=tk.DISABLED)

            # Update Map Canvas
            self._update_canvas(active_agents_data, structures_data)

        except tk.TclError as e:
            if "invalid command name" not in str(e): # Ignore errors during shutdown
                 log.warning(f"Tkinter TclError during GUI update: {e}")
        except Exception as e:
             log.error(f"Unhandled error during GUI update: {e}", exc_info=True)

    def _update_canvas(self, agents_data: List[Dict], structures_data: List[Dict]):
        # (Unchanged)
        self.map_canvas.delete("agent")
        self.map_canvas.delete("structure")
        location_occupancy: Dict[str, Dict[str, List[Tuple[int, int]]]] = {loc: {"agent": [], "structure": []} for loc in Config.LOCATIONS}
        def get_next_grid_pos(loc_name, item_type, grid_size, radius):
            base_x, base_y = LOCATION_COORDS[loc_name]
            occupied = location_occupancy[loc_name][item_type]
            max_tries=100; i=0
            while i < max_tries:
                angle = i * 0.8 + random.uniform(-0.1, 0.1); r = 5 + i * (grid_size / 5.0) + random.uniform(-2, 2)
                if r > radius * 0.95 : r = radius * 0.95 # Clamp radius slightly inside marker
                ox, oy = r * np.cos(angle), r * np.sin(angle)
                px, py = base_x + ox, base_y + oy
                # Check collision with *all* items at location for better spacing
                all_occupied = location_occupancy[loc_name]["agent"] + location_occupancy[loc_name]["structure"]
                if not any(((px - ox2)**2 + (py - oy2)**2) < (grid_size * 1.8)**2 for ox2, oy2 in all_occupied):
                    occupied.append((px, py)); return px, py
                i += 1
            # Fallback: random placement if grid fails
            px, py = base_x + random.uniform(-radius*0.8, radius*0.8), base_y + random.uniform(-radius*0.8, radius*0.8)
            occupied.append((px, py)); return px, py

        for s_dict in structures_data:
            loc = s_dict.get('loc')
            if loc in LOCATION_COORDS:
                draw_x, draw_y = get_next_grid_pos(loc, "structure", STRUCTURE_BOX_SIZE, LOCATION_RADIUS - STRUCTURE_BOX_SIZE / 2)
                self.map_canvas.create_rectangle(draw_x - 4, draw_y - 4, draw_x + 4, draw_y + 4, fill="#A0522D", outline="black", width=1, tags=("structure", loc, f"struct_{s_dict['id']}"))
        for agent_dict in agents_data:
             # Only draw living agents
             health = agent_dict.get('health', 0)
             if health <= 0: continue

             loc = agent_dict.get('loc')
             if loc in LOCATION_COORDS:
                ratio = health / Config.MAX_HEALTH if Config.MAX_HEALTH > 0 else 0
                # color = "#7F8C8D" if health <= 0 else "#E74C3C" if ratio < 0.3 else "#F1C40F" if ratio < 0.7 else "#2ECC71"
                # Use tags for health status coloring
                health_tag = self.get_health_tag(health, Config.MAX_HEALTH)
                color = "#E74C3C" if health_tag == 'Critical' else "#F1C40F" if health_tag == 'Injured' else "#2ECC71" # Default Healthy

                draw_x, draw_y = get_next_grid_pos(loc, "agent", AGENT_DOT_SIZE, LOCATION_RADIUS - AGENT_DOT_SIZE / 2)
                self.map_canvas.create_oval(draw_x - 3, draw_y - 3, draw_x + 3, draw_y + 3, fill=color, outline="black", width=1, tags=("agent", loc, f"agent_{agent_dict['id']}"))


# ───────────────────── Simulation Loop ──────────────────

# Helper function for snapshotting world state for GUI
def get_snapshot_data(current_world: World) -> Tuple[List[Dict], List[Dict]]:
    """Creates serializable snapshots of agents and structures."""
    agents_snap = []
    # Use get_living_agents to only snapshot those alive for GUI display
    # But still include all agents from world.agents for registry checks etc.
    # Let's send all from world.agents and let GUI filter/tag dead ones
    all_agents_in_world = list(current_world.agents.values())
    for a in all_agents_in_world:
         try:
             # Include inventory summary directly for GUI efficiency
             a_dict = a.to_dict()
             a_dict['inventory_summary'] = a.get_inventory_summary()
             # Ensure health is included even if to_dict doesn't (it should)
             a_dict['health'] = a.health
             agents_snap.append(a_dict)
         except Exception as e:
             log.error(f"Failed to serialize agent {a.id}: {e}")

    structs_snap = [s.to_dict() for s in current_world.structures.values()]
    return agents_snap, structs_snap

async def run_simulation_loop(world: World, registry: Registry, update_queue: queue.Queue, action_queue: queue.Queue, stop_event: threading.Event, args: argparse.Namespace):
    """The main async simulation loop."""
    global _ACTION_QUEUE, _UPDATE_QUEUE # Allow world access
    _ACTION_QUEUE = action_queue
    _UPDATE_QUEUE = update_queue

    log.info("Async simulation loop starting.")
    save_interval = args.save_interval
    paused = False

    try:
        while not stop_event.is_set():
            start_tick_time = time.time()

            # --- Handle GUI Actions ---
            while not action_queue.empty():
                 try:
                     gui_action = action_queue.get_nowait()
                     action_type = gui_action.get("action")
                     log.info(f"Processing GUI action: {action_type}")

                     if action_type == "toggle_pause": paused = not paused
                     elif action_type == "save_now":
                         save_filename = Path(Config.SAVE_DIR) / f"world_tick_{world.current_tick}_manual.json.gz"
                         world.save_to_file(save_filename)
                         # Queue event back? Simpler to just log for now.
                         status_update = {"tick": world.current_tick, "day": world.day, "season": world.season, "agents": [], "structures": [], "new_events": [f"[SYSTEM] Manual save complete: {save_filename.name}"],"status": f"Saved manually to {save_filename.name}"}
                         update_queue.put(status_update)

                     elif action_type == "load_file":
                         filepath = Path(gui_action.get("filepath", ""))
                         if filepath.exists() and filepath.is_file():
                             log.info(f"Pausing simulation for load: {filepath}")
                             paused = True
                             # Send status update
                             agents_snap, structs_snap = get_snapshot_data(world)
                             update_queue.put({"tick": world.current_tick, "day": world.day, "season": world.season, "agents": agents_snap, "structures": structs_snap, "new_events": [f"[SYSTEM] Load requested: {filepath.name}. Pausing."], "status": f"Loading from {filepath.name}..."})
                             await asyncio.sleep(0.1) # Allow GUI to update

                             new_world = World.from_file(filepath, world.chroma_client)
                             if new_world:
                                 # Replace world state carefully
                                 world.agents = new_world.agents
                                 world.structures = new_world.structures
                                 world.locations = new_world.locations
                                 world.current_tick = new_world.current_tick
                                 world.day = new_world.day
                                 world.season = new_world.season
                                 world.last_event_trigger_tick = new_world.last_event_trigger_tick
                                 world.pending_code_updates = [] # Clear pending changes on load
                                 world.new_agent_buffer = []
                                 world.agents_to_remove = []

                                 # Update registry
                                 registry.agents.clear()
                                 for agent in world.agents.values(): registry.add(agent)

                                 # Reset transient agent states? (e.g., code request) - done in AgentState.from_dict
                                 # Restore next IDs
                                 AgentState._next_id = new_world.to_dict().get("next_agent_id", AgentState._next_id)
                                 Structure._next_id = new_world.to_dict().get("next_structure_id", Structure._next_id)

                                 log.info("World state loaded successfully. Resuming.")
                                 paused = False
                                 # Send immediate update after load
                                 agents_final_snap, structs_final_snap = get_snapshot_data(world)
                                 final_update = {
                                     "tick": world.current_tick, "day": world.day, "season": world.season,
                                     "agents": agents_final_snap, "structures": structs_final_snap,
                                     "new_events": [f"[SYSTEM] World loaded from {filepath.name}."],
                                     "status": f"Loaded tick {world.current_tick}. Running."
                                 }
                                 update_queue.put(final_update)

                             else:
                                 log.error(f"Failed to load world from {filepath}. Simulation remains paused.")
                                 # Keep paused and inform user
                                 update_queue.put({"tick": world.current_tick, "day": world.day, "season": world.season, "agents": [], "structures": [], "new_events": [f"[ERROR] Failed to load {filepath.name}. Check logs."], "status": f"Load failed. Paused."})


                         else: log.warning(f"GUI requested load from invalid path: {filepath}")

                     action_queue.task_done()
                 except queue.Empty: break # Should not happen with while not empty
                 except Exception as e: log.error(f"Error processing GUI action: {e}", exc_info=True)


            # --- Simulation Tick ---
            if paused:
                 # Send periodic status update while paused
                 if world.current_tick % 20 == 0: # Less frequent update
                     agents_snap, structs_snap = get_snapshot_data(world)
                     update_queue.put({"tick": world.current_tick, "day": world.day, "season": world.season, "agents": agents_snap, "structures": structs_snap, "new_events": [], "status": f"Tick {world.current_tick}. Simulation PAUSED."})
                 await asyncio.sleep(0.2) # Longer sleep when paused
                 continue

            current_tick_events = []

            # 1. World Update (Time, Needs, Regen, Code Apply, Events, Agent Removal)
            try:
                 world_events = world.tick_update()
                 current_tick_events.extend(world_events)
            except Exception as e:
                 log.critical(f"CRITICAL ERROR during world.tick_update: {e}", exc_info=True)
                 current_tick_events.append(f"T{world.current_tick}: [!!!] CRITICAL World Update Error: {type(e).__name__}")
                 stop_event.set() # Stop simulation on critical world update failure
                 # Send final error state
                 agents_snap, structs_snap = get_snapshot_data(world)
                 update_queue.put({"tick": world.current_tick, "day": world.day, "season": world.season, "agents": agents_snap, "structures": structs_snap, "new_events": current_tick_events, "status": f"FATAL ERROR @ Tick {world.current_tick}. Check logs."})
                 raise # Re-raise to exit loop

            # 2. Agent Actions (Concurrent)
            living_agents = world.get_living_agents()
            random.shuffle(living_agents) # Randomize action order each tick

            tasks = [asyncio.create_task(agent.act(world, registry)) for agent in living_agents]
            action_results = []
            if tasks:
                try:
                    action_results = await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as e: # Catch errors during gather itself
                    log.error(f"Error during asyncio.gather for agent actions: {e}", exc_info=True)
                    current_tick_events.append(f"T{world.current_tick}: [!!!] System Error during action processing: {type(e).__name__}")


            # Process results
            agent_exceptions = 0
            for result in action_results:
                 if isinstance(result, Exception):
                     log.error(f"Error during agent action task: {result}", exc_info=result)
                     current_tick_events.append(f"T{world.current_tick}: [!] System Error during action: {type(result).__name__}")
                     agent_exceptions += 1
                 elif isinstance(result, str):
                      # Prepend Tick number to agent action results for clarity
                      current_tick_events.append(f"T{world.current_tick}: {result}")

            # 3. Autosave
            if save_interval > 0 and world.current_tick > 0 and world.current_tick % save_interval == 0:
                 save_filename = Path(Config.SAVE_DIR) / f"world_tick_{world.current_tick}.json.gz"
                 world.save_to_file(save_filename)
                 current_tick_events.append(f"T{world.current_tick}: [SYSTEM] Autosaved world state to {save_filename.name}")

            # --- Final Update for Tick ---
            end_tick_time = time.time()
            tick_duration = end_tick_time - start_tick_time
            living_agent_count = len(world.get_living_agents()) # Re-check count after actions/deaths/births
            status = f"Tick {world.current_tick} | Day {world.day} | Pop: {living_agent_count} | Dur: {tick_duration:.2f}s"
            if agent_exceptions > 0: status += f" | Action Errors: {agent_exceptions}"
            log.info(status)

            agents_final_snap, structs_final_snap = get_snapshot_data(world)
            final_update = {
                "tick": world.current_tick, "day": world.day, "season": world.season,
                "agents": agents_final_snap, "structures": structs_final_snap,
                "new_events": current_tick_events, # Pass all collected events
                "status": status
            }
            try:
                # Limit queue size? If GUI lags, queue could grow indefinitely.
                # For now, assume GUI keeps up or drops frames.
                update_queue.put(final_update)
            except Exception as qe:
                log.error(f"Failed to put final update on queue: {qe}")


            # --- Delay ---
            time_to_wait = Config.TICK_DELAY - tick_duration
            if time_to_wait > 0:
                 await asyncio.sleep(time_to_wait)

    except asyncio.CancelledError:
        log.info("Simulation loop cancelled.")
    except Exception as e:
         log.critical(f"Fatal error in simulation loop: {e}", exc_info=True)
         # Send final error message to GUI
         if _UPDATE_QUEUE:
             try:
                 _UPDATE_QUEUE.put({"tick": world.current_tick, "day": world.day, "season": world.season, "agents": [], "structures": [], "new_events": [f"FATAL SIM ERROR: {type(e).__name__}. Check logs."],"status": f"FATAL ERROR @ Tick {world.current_tick}. Check logs."})
             except Exception: pass # Ignore queue errors during crash
         stop_event.set()
    finally:
        log.info("Async simulation loop stopped.")
        # Ensure queues are cleared/signalled? No, handled by daemon thread exit mostly.

# ───────────────────────── Main Execution ───────────────────────────

def setup_arg_parser() -> argparse.ArgumentParser:
     # Added --allow_code_change flag
     parser = argparse.ArgumentParser(description="ASOIAF Autonomous Agent Simulation v3")
     parser.add_argument("--load", action="store_true", help="Load the latest save file from the save directory.")
     parser.add_argument("--save_interval", type=int, default=Config.DEFAULT_SAVE_INTERVAL, help=f"Ticks between automatic saves (0 to disable). Default: {Config.DEFAULT_SAVE_INTERVAL}")
     parser.add_argument("--model", type=str, default=Config.MODEL, help=f"LLM model to use. Default: {Config.MODEL}")
     parser.add_argument("--start_pop", type=int, default=Config.START_POP, help=f"Initial agent population if not loading. Default: {Config.START_POP}")
     parser.add_argument("--max_pop", type=int, default=Config.MAX_POP, help=f"Maximum agent population. Default: {Config.MAX_POP}")
     parser.add_argument("--allow_code_change", action="store_true", help="[EXPERIMENTAL] Enable agent code modification feature. Use with caution!")
     return parser

def find_latest_save(save_dir: Path) -> Optional[Path]:
     # (Unchanged)
     if not save_dir.is_dir(): return None
     saves = list(save_dir.glob("world_tick_*.json*"))
     if not saves: return None
     # Sort by modification time first as a fallback
     saves.sort(key=os.path.getmtime, reverse=True)
     # Then sort by tick number primarily
     def extract_tick(p: Path) -> int:
         try:
             stem = p.stem; stem = Path(stem).stem # Handle .json.gz
             return int(stem.split('_')[-1].split('_manual')[0]) # Handle manual saves too
         except: return -1 # Invalid filenames sort last
     saves.sort(key=extract_tick, reverse=True)
     # Ensure the top one is valid before returning
     if extract_tick(saves[0]) >= 0:
        return saves[0]
     else: # If the 'latest' had an invalid tick name, maybe fallback to mtime? Or return None.
         valid_saves = [s for s in saves if extract_tick(s) >= 0]
         return valid_saves[0] if valid_saves else None


async def main_async(args: argparse.Namespace, stop_event: threading.Event, update_queue: queue.Queue, action_queue: queue.Queue):
    """The main asynchronous entry point."""
    log.info(f"Async main started. PID: {os.getpid()}, Thread: {threading.current_thread().name}")
    world = None # Initialize world to None

    try:
        # --- Initialize LLM ---
        if not initialize_llm_client():
             print("[CRITICAL ERROR] Failed to initialize LLM Client. Check API key and model name.")
             stop_event.set(); return # Stop if LLM fails

        # --- Initialize ChromaDB ---
        print("Initializing ChromaDB...")
        chroma_client = ensure_embeddings()
        if chroma_client is None: print("[WARNING] ChromaDB failed. RAG disabled.")

        # --- Initialize World (Load or New) ---
        # (Load logic slightly refined)
        if args.load:
            latest_save = find_latest_save(Path(Config.SAVE_DIR))
            if latest_save:
                print(f"Loading world from latest save: {latest_save}...")
                world = World.from_file(latest_save, chroma_client)
                if world:
                     print("World loaded successfully.")
                     # Registry is cleared and repopulated inside run_simulation_loop now on load action
                else: print("Failed to load save file. Starting new world.")
            else: print("No save file found to load. Starting new world.")

        if world is None:
             world = World(chroma_client=chroma_client) # Create new world if load failed/not requested

        # --- Seed Initial Population ---
        # (Only if world has no agents after load/new)
        if not world.agents: # Check if world.agents is empty
            print(f"Seeding initial population ({args.start_pop} agents)...")
            initial_agents_added = 0
            for _ in range(args.start_pop):
                if len(world.agents) >= args.max_pop: break # Check against world dict directly
                name = generate_themed_name()
                agent = AgentState(name, chroma_client=chroma_client)
                # Add directly to world and registry
                if registry.add(agent):
                    world.agents[agent.id] = agent
                    initial_agents_added += 1
                else: log.error(f"Failed to add initial agent {name} to registry.")
            print(f"Initial population seeded: {initial_agents_added} agents.")
        else:
            # If loading, populate registry from loaded world agents
            registry.agents.clear()
            for agent in world.agents.values():
                registry.add(agent)
            print(f"Loaded world with {len(world.agents)} agents.")


        # --- Start Simulation Loop ---
        print("Starting simulation loop...")
        await run_simulation_loop(world, registry, update_queue, action_queue, stop_event, args)

    except Exception as e:
        log.critical(f"Critical error in main_async: {e}", exc_info=True)
        print(f"\n[CRITICAL ERROR] Async main failed: {e}")
        # Try to inform GUI
        err_tick = world.current_tick if world else 0
        if update_queue: update_queue.put({"tick": err_tick, "day": 0, "season": "?", "agents": [], "structures": [], "new_events": [f"FATAL ASYNC ERROR: {type(e).__name__}. Check logs."], "status": f"FATAL ERROR @ Tick {err_tick}. Check logs."})
    finally:
        log.info("Async main finished.")
        stop_event.set() # Ensure stop event is set on exit

def run_gui(update_queue: queue.Queue, action_queue: queue.Queue, stop_event: threading.Event, loop: Optional[asyncio.AbstractEventLoop]):
     """Runs the Tkinter GUI."""
     log.info(f"GUI starting. PID: {os.getpid()}, Thread: {threading.current_thread().name}")
     root = AppRoot()
     gui = SimulationGUI(root, update_queue, action_queue)

     # --- Add Menu Bar ---
     menubar = tk.Menu(root)
     root.config(menu=menubar)

     file_menu = tk.Menu(menubar, tearoff=0)
     menubar.add_cascade(label="File", menu=file_menu)
     file_menu.add_command(label="Save Now", command=lambda: action_queue.put({"action": "save_now"}))
     file_menu.add_command(label="Load...", command=lambda: ask_load_file(action_queue))
     file_menu.add_separator()
     file_menu.add_command(label="Exit", command=root.quit) # Use quit for cleaner shutdown signal

     sim_menu = tk.Menu(menubar, tearoff=0)
     menubar.add_cascade(label="Simulation", menu=sim_menu)
     sim_menu.add_command(label="Pause/Resume", command=lambda: action_queue.put({"action": "toggle_pause"}))

     def ask_load_file(aq):
          # Ensure initialdir exists
          save_path = Path(Config.SAVE_DIR)
          save_path.mkdir(exist_ok=True)
          filepath = tk.filedialog.askopenfilename(
               title="Load Simulation State",
               initialdir=str(save_path.resolve()), # Use resolved absolute path
               filetypes=[("Compressed JSON", "*.json.gz"), ("All Files", "*.*")]
          )
          if filepath:
               log.info(f"GUI: Queuing load action for {filepath}")
               aq.put({"action": "load_file", "filepath": filepath})


     def on_closing():
         log.info("GUI window closing signal received.")
         print("\nRequesting simulation shutdown...")
         if not stop_event.is_set():
             stop_event.set() # Signal the simulation thread/asyncio task to stop
             log.info("Stop event set for simulation loop.")

             # Gracefully attempt to cancel running asyncio tasks
             if loop and loop.is_running():
                  log.info("Attempting to cancel asyncio tasks via loop.")
                  # Schedule cancellation from the loop's thread
                  loop.call_soon_threadsafe(lambda: [task.cancel() for task in asyncio.all_tasks(loop=loop) if not task.done()])
                  log.info("Cancellation scheduled via call_soon_threadsafe.")

         # Give threads a moment to react before destroying the window
         root.after(500, root.destroy) # Destroy root after a short delay

     root.protocol("WM_DELETE_WINDOW", on_closing)
     print("Simulation running. Close the window or use File > Exit to quit.")
     try:
         root.mainloop()
     except KeyboardInterrupt:
         print("\nCtrl+C detected in GUI thread. Shutting down...")
         on_closing()
     finally:
          log.info("GUI mainloop exited.")
          if not stop_event.is_set():
              stop_event.set() # Ensure stop is set if mainloop exits unexpectedly
              log.info("Stop event set after GUI mainloop exit.")


if __name__ == "__main__":
    if sys.version_info < (3, 8):
        print("Python 3.8+ required.")
        sys.exit(1)

    parser = setup_arg_parser()
    args = parser.parse_args()

    # --- Update Config from Args ---
    Config.MODEL = args.model
    Config.START_POP = args.start_pop
    Config.MAX_POP = args.max_pop
    Config.ALLOW_CODE_MODIFICATION = True# Set based on flag

    # --- Store Original Code ---
    # Done early before any potential modifications
    store_original_code()

    print("Starting Autonomous ASOIAF Simulation v3...")
    print(f"Logging to: {log_file.resolve()}")
    print(f"Using Model: {Config.MODEL}")
    print(f"Population: Start={Config.START_POP}, Max={Config.MAX_POP}, Min={Config.MIN_POP}")
    print(f"Save Interval: {args.save_interval} ticks (0=disabled)")
    if args.load: print(f"Attempting load from: {Config.SAVE_DIR}")
    print(f"Agent Code Modification: {'ENABLED (EXPERIMENTAL!)' if Config.ALLOW_CODE_MODIFICATION else 'DISABLED'}")
    if Config.ALLOW_CODE_MODIFICATION:
        print("!!! WARNING: Running with agent code modification enabled is highly experimental and potentially unstable/unsafe. !!!")

    # --- Setup Threading/Async Communication ---
    update_queue = queue.Queue(maxsize=100) # Add maxsize to prevent unbounded growth if GUI lags
    action_queue = queue.Queue()
    stop_event = threading.Event()
    sim_thread = None
    loop = None

    try:
        # Create and start the asyncio event loop in a separate thread
        loop = asyncio.new_event_loop()
        def run_async_loop_thread(loop, args, stop_event, update_queue, action_queue):
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(main_async(args, stop_event, update_queue, action_queue))
            finally:
                log.info("Async loop run_until_complete finished. Closing loop.")
                try: # Attempt graceful shutdown of remaining tasks
                    tasks = asyncio.all_tasks(loop=loop)
                    if tasks:
                         log.info(f"Cancelling {len(tasks)} outstanding asyncio tasks...")
                         for task in tasks: task.cancel()
                         loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                         log.info("Outstanding tasks cancelled.")
                except Exception as e: log.error(f"Error cancelling tasks during loop close: {e}")
                finally:
                     if not loop.is_closed():
                         loop.close()
                         log.info("Async loop closed.")

        sim_thread = threading.Thread(
            target=run_async_loop_thread,
            args=(loop, args, stop_event, update_queue, action_queue),
            name="SimulationThread", # Give the thread a name
            daemon=True # Make daemon so it exits if main thread exits unexpectedly
        )
        sim_thread.start()
        log.info("Simulation thread started.")

        # Run GUI in the main thread
        # Need to import filedialog for the menu command
        from tkinter import filedialog
        run_gui(update_queue, action_queue, stop_event, loop)

    except Exception as e:
        log.critical(f"Fatal error during setup or GUI launch: {e}", exc_info=True)
        print(f"\nFATAL ERROR: {e}")
    finally:
        log.info("Main thread exiting.")
        if not stop_event.is_set():
            stop_event.set()
            log.info("Stop event set in final cleanup.")

        if sim_thread and sim_thread.is_alive():
            log.info("Waiting for simulation thread to join...")
            sim_thread.join(timeout=5.0) # Generous timeout
            if sim_thread.is_alive(): log.warning("Simulation thread did not exit cleanly after join timeout.")
            else: log.info("Simulation thread joined.")

        # Ensure log handlers are closed
        logging.shutdown()
        print("Simulation shutdown complete.")