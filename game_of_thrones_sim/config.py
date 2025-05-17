import numpy as np
from collections import Counter

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
    MODEL           = "gemini-2.5-flash-preview-04-17" # Using Gemini Flash as updated in original code (Corrected model name based on common Gemini models, original had a typo "")
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
    MAX_HEALTH      = 100.0 # Made float for consistency
    MAX_HUNGER      = 20.0  # Made float
    HUNGER_PER_TICK = 0.7
    HEALTH_DAMAGE_FROM_HUNGER = 1.5
    MAX_MORALE      = 100.0 # Made float
    MIN_MORALE      = 0.0   # Made float
    MORALE_DECAY_PER_TICK = 0.1
    BASE_HEAL_AMOUNT = 15.0 # Made float
    AGE_DEATH_CHANCE_PER_TICK = lambda age: max(0, (age - Config.MAX_AGE * 0.9) / (Config.MAX_AGE * 0.1) * 0.0005)

    # --- Combat ---
    BASE_ATTACK_DAMAGE_FACTOR = 18.0 # Made float
    COMBAT_MORALE_LOSS_ON_ATTACK = 5
    COMBAT_MORALE_LOSS_ON_KILL = 15
    COMBAT_MORALE_GAIN_ON_VICTORY = 10
    POST_KILL_ATTACK_COOLDOWN = 5
    COMBAT_COOLDOWN = 2

    # --- Economy / Crafting ---
    INITIAL_RESOURCES = {
        "Winterfell": Counter({"Wood": 50, "Food": 30, "Stone": 20}),
        "KingsLanding": Counter({"Food": 40, "Stone": 40, "Ore": 10}),
        "TheWall": Counter({"Stone": 60, "Wood": 10, "Food": 5}),
        "Dragonstone": Counter({"Stone": 50, "Ore": 30}),
        "Braavos": Counter({"Fish": 50, "Stone": 20}),
    }
    RESOURCE_REGEN_RATE = {
        "Wood": 2.0, "Food": 3.0, "Stone": 1.0, "Ore": 0.5, "Fish": 4.0
    }
    TICKS_PER_DAY = 4
    CRAFTING_RECIPES = {
        "Wooden_Spear": {"Wood": 3},
        "Stone_Axe": {"Wood": 2, "Stone": 3},
        "Basic_Shelter": {"Wood": 10, "Stone": 5},
        "Forge": {"Stone": 15, "Wood": 5, "Ore": 5},
        "Cooked_Meal": {"Food": 2, "Wood": 1},
        "Iron_Sword": {"Ore": 8, "Wood": 2},
    }
    STRUCTURE_BUILD_VERBS = {"Basic_Shelter", "Forge"}
    CONSUMABLE_ITEMS = {
        "Cooked_Meal": {"hunger_relief": 10, "health_gain": 0, "morale_gain": 5},
        "Food": {"hunger_relief": 3, "health_gain": 0, "morale_gain": 1},
        "Fish": {"hunger_relief": 4, "health_gain": 0, "morale_gain": 1},
    }

    # --- DNA / Traits / Skills ---
    TRAITS          = ["cunning","honour","wrath","piety"]
    SKILLS          = ["smith","war","lore","diplomacy"]
    TRAIT_MUT_STD   = 0.04
    SKILL_DECAY     = 0.001

    # --- RAG ---
    CHROMA_DIR      = "./chroma_db_auto_gui_v3"
    BOOK_CHUNKS     = "./data/got_chunks.json" # Path relative to project root
    RAG_TOPK        = 3

    # --- GUI ---
    GUI_UPDATE_INTERVAL = 150 # ms
    EVENT_LOG_DISPLAY_SIZE = 100

    # --- Agent Code Modification (EXPERIMENTAL) ---
    ALLOW_CODE_MODIFICATION = True # Default to disabled for safety (will be set by argsparse)
    MAX_CODE_LINES = 20
    CODE_MODIFICATION_WHITELIST = {
        "Config.RESOURCE_REGEN_RATE",
        "Config.BASE_ATTACK_DAMAGE_FACTOR",
        "Config.HUNGER_PER_TICK",
        "AgentState.update_needs", # Will need to ensure AgentState is importable here or handle differently
        "World.resolve_attack",    # Will need to ensure World is importable here or handle differently
    }

    # --- Simple Random Event System ---
    EVENT_CHECK_INTERVAL = 30
    EVENT_BASE_CHANCE = 0.15
    RANDOM_EVENTS = [
        {"name": "Good Harvest", "type": "resource_loc", "resource": "Food", "amount": 15.0, "loc_specific": True, "message": "Reports of an unusually good harvest arrive from {location}!"},
        {"name": "Resource Find", "type": "resource_loc", "resource": "Stone", "amount": 10.0, "loc_specific": True, "message": "A new vein of stone has been discovered near {location}."},
        {"name": "Resource Find", "type": "resource_loc", "resource": "Ore", "amount": 5.0, "loc_specific": True, "message": "Scouts report finding a deposit of ore near {location}."},
        {"name": "Mild Sickness", "type": "morale_loc", "amount": -5, "loc_specific": True, "message": "A bout of mild sickness spreads through {location}, lowering spirits."},
        {"name": "Bad Omen", "type": "morale_loc", "amount": -8, "loc_specific": True, "message": "Strange lights in the sky above {location} are seen as a bad omen."},
        {"name": "Inspiring News", "type": "morale_all", "amount": 3, "loc_specific": False, "message": "Good news travels across the land, slightly lifting everyone's spirits."},
        {"name": "Bandit Sighting", "type": "flavor", "loc_specific": True, "message": "Travelers report increased bandit activity near {location}."},
    ]

# --- GUI Constants ---
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

# Note: The lambda function in AGE_DEATH_CHANCE_PER_TICK and
# the CODE_MODIFICATION_WHITELIST referencing AgentState and World
# will require AgentState and World to be defined or imported when Config is used,
# or these specific Config attributes might need to be initialized/updated later
# after AgentState and World classes are defined.
# For now, I'll keep them here, and we'll address import cycles or deferred initialization if it becomes an issue.
# A common pattern is to have Config values that are simple data types, and then
# have functions or methods elsewhere use these Config values.
# For CODE_MODIFICATION_WHITELIST, it's fine as strings.
# For AGE_DEATH_CHANCE_PER_TICK, the lambda refers to `Config.MAX_AGE`, which is fine.
