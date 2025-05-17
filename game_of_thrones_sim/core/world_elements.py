from collections import Counter
from typing import Dict, Optional

# Placeholder for logger and Config
class PrintLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")

log = PrintLogger()

class MockConfig: # Used until actual Config is properly importable
    TICKS_PER_DAY = 4
    RESOURCE_REGEN_RATE = {"Wood": 1.0, "Food": 1.0} # Example

config_instance = MockConfig() # Placeholder

class Structure:
    """Represents a structure built by agents in a location."""
    _next_id = 1

    def __init__(self, name: str, loc: str, owner_id: int,
                 durability: int = 100, capacity: int = 10,
                 stored_items: Optional[Counter] = None,
                 structure_id: Optional[int] = None):
        global log
        try:
            from ..utils.logger import log as main_log
            log = main_log
        except ImportError:
            pass

        if structure_id is not None:
            self.id = structure_id
            if structure_id >= Structure._next_id:
                Structure._next_id = structure_id + 1
        else:
            self.id = Structure._next_id
            Structure._next_id += 1

        self.name = name
        self.loc = loc # Location name (string)
        self.owner_id = owner_id # Agent ID of the owner, 0 if unowned/world-owned
        self.durability = durability
        self.capacity = capacity # Max number of item stacks or total weight, depending on game rules
        self.stored_items: Counter = stored_items or Counter() # item_name -> count

    def to_dict(self) -> Dict:
        """Serializes the structure to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "loc": self.loc,
            "owner_id": self.owner_id,
            "durability": self.durability,
            "capacity": self.capacity,
            "stored_items": dict(self.stored_items) # Convert Counter to dict for JSON
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Structure':
        """Deserializes a structure from a dictionary."""
        # Ensure _next_id is updated correctly if loading from save
        current_max_id = data.get("id", 0)
        if current_max_id >= cls._next_id:
            cls._next_id = current_max_id + 1

        return cls(
            structure_id=data["id"], # Pass explicitly to constructor
            name=data["name"],
            loc=data["loc"],
            owner_id=data.get("owner_id", 0), # Default to 0 if missing
            durability=data.get("durability", 100),
            capacity=data.get("capacity", 10),
            stored_items=Counter(data.get("stored_items", {}))
        )

    def get_info(self) -> str:
        """Returns a string summary of the structure."""
        items_str = ", ".join(f"{item}:{count}" for item, count in self.stored_items.items()) if self.stored_items else "empty"
        return f"{self.name} (ID:{self.id}, Owner:{self.owner_id}, Dur:{self.durability}, Cap:{self.capacity}, Items:[{items_str[:50]}])"

    def __repr__(self) -> str:
        return f"<Structure id={self.id} name='{self.name}' loc='{self.loc}' owner={self.owner_id}>"


class LocationNode:
    """Represents a location in the game world, holding resources and structures."""
    def __init__(self, name: str, base_resources: Counter, config_ref: Optional[object] = None):
        global log, config_instance
        try:
            from ..utils.logger import log as main_log
            log = main_log
        except ImportError:
            pass
        
        # Use passed config_ref if available, otherwise fallback to MockConfig
        # This allows the main application to pass the actual Config object.
        self.config = config_ref if config_ref else config_instance


        self.name = name
        # Store resources as floats for precise regeneration calculations
        self.resources: Counter = Counter({k: float(v) for k, v in base_resources.items()})
        self.base_resources: Counter = Counter({k: float(v) for k, v in base_resources.items()}) # Max capacity reference
        self.structures: Dict[int, Structure] = {} # structure_id -> Structure object

    def add_structure(self, structure: Structure):
        """Adds a structure to this location."""
        if structure.id in self.structures:
            log.warning(f"Structure ID {structure.id} ({structure.name}) already exists at {self.name}. Overwriting.")
        self.structures[structure.id] = structure
        # log.debug(f"Structure {structure.id} ({structure.name}) added to {self.name}.")

    def remove_structure(self, structure_id: int) -> Optional[Structure]:
        """Removes a structure from this location."""
        removed_structure = self.structures.pop(structure_id, None)
        if removed_structure:
            # log.debug(f"Structure {structure_id} ({removed_structure.name}) removed from {self.name}.")
            return removed_structure
        # log.warning(f"Attempted to remove non-existent structure ID {structure_id} from {self.name}.")
        return None

    def regenerate_resources(self):
        """Regenerates resources based on configured rates."""
        # This method relies on Config.RESOURCE_REGEN_RATE and Config.TICKS_PER_DAY
        # These should be accessible via self.config
        for resource, rate_per_day in self.config.RESOURCE_REGEN_RATE.items():
            if resource in self.base_resources: # Only regenerate resources that are part of this location's base set
                current_amount = self.resources.get(resource, 0.0)
                # Allow regeneration up to a certain multiplier of base (e.g., 1.5x or 2x)
                # This prevents infinite accumulation if not gathered.
                max_amount_at_loc = self.base_resources.get(resource, 0.0) * 1.5 # Example: 150% of base
                
                regen_per_tick = rate_per_day / self.config.TICKS_PER_DAY
                
                # Only add if below max and positive regen
                amount_to_add = 0.0
                if current_amount < max_amount_at_loc and regen_per_tick > 0:
                    amount_to_add = min(regen_per_tick, max_amount_at_loc - current_amount)
                
                if amount_to_add > 1e-5: # Check for meaningful addition to avoid float precision issues
                    self.resources[resource] = current_amount + amount_to_add
                    # log.debug(f"Regenerated {amount_to_add:.2f} {resource} at {self.name}. New total: {self.resources[resource]:.2f}")

    def get_structure_summary(self) -> str:
        """Returns a string summary of structures at this location."""
        if not self.structures:
            return "none"
        return ", ".join(f"{s.name}[ID:{s.id}]" for s in self.structures.values())[:150] # Limit length

    def get_resource_summary(self) -> str:
        """Returns a string summary of resources, rounded for display."""
        # Display rounded integers for user-friendliness, but keep internal floats
        return ", ".join(f"{r}:{int(round(c))}" for r, c in self.resources.items() if c >= 1.0) or "none"

    def to_dict(self) -> Dict:
        """Serializes the location node to a dictionary."""
        return {
            "name": self.name,
            "resources": dict(self.resources), # Save floats
            "base_resources": dict(self.base_resources), # Save floats
            "structures": {sid: s.to_dict() for sid, s in self.structures.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict, config_ref: Optional[object] = None) -> 'LocationNode':
        """Deserializes a location node from a dictionary."""
        # Load base resources as floats
        base_resources_data = data.get("base_resources", {})
        base_resources = Counter({k: float(v) for k, v in base_resources_data.items()})
        
        node = cls(name=data["name"], base_resources=base_resources, config_ref=config_ref)
        
        # Load current resources, defaulting to base_resources if not present (e.g., new save format)
        current_resources_data = data.get("resources", base_resources_data)
        node.resources = Counter({k: float(v) for k, v in current_resources_data.items()})
        
        # Load structures
        node.structures = {
            int(sid): Structure.from_dict(s_data)
            for sid, s_data in data.get("structures", {}).items()
        }
        return node

    def __repr__(self) -> str:
        return f"<LocationNode name='{self.name}' resources={len(self.resources)} structures={len(self.structures)}>"


if __name__ == '__main__':
    # Example Usage
    log.info("Testing World Elements...")

    # Mock Config for testing
    class TestConfig:
        TICKS_PER_DAY = 4
        RESOURCE_REGEN_RATE = {
            "Wood": 2.0, "Food": 3.0, "Stone": 1.0, "Ore": 0.5, "Fish": 4.0
        }

    test_config_instance = TestConfig()

    # Test Structure
    shelf = Structure(name="Shelf", loc="Winterfell", owner_id=1)
    log.info(f"Created structure: {shelf.get_info()}")
    shelf_data = shelf.to_dict()
    log.info(f"Serialized shelf: {shelf_data}")
    loaded_shelf = Structure.from_dict(shelf_data)
    log.info(f"Loaded shelf: {loaded_shelf.get_info()}")
    assert shelf.id == loaded_shelf.id and shelf.name == loaded_shelf.name

    # Test LocationNode
    winterfell_resources = Counter({"Wood": 50, "Food": 30, "Stone": 20})
    winterfell = LocationNode(name="Winterfell", base_resources=winterfell_resources, config_ref=test_config_instance)
    log.info(f"Created location: {winterfell.name}, Resources: {winterfell.get_resource_summary()}")

    winterfell.add_structure(loaded_shelf)
    log.info(f"Winterfell structures: {winterfell.get_structure_summary()}")

    for _ in range(test_config_instance.TICKS_PER_DAY * 2): # Simulate 2 days
        winterfell.regenerate_resources()
    log.info(f"Winterfell resources after 2 days: {winterfell.get_resource_summary()}")

    loc_data = winterfell.to_dict()
    log.info(f"Serialized Winterfell: {loc_data}")
    loaded_winterfell = LocationNode.from_dict(loc_data, config_ref=test_config_instance)
    log.info(f"Loaded Winterfell: {loaded_winterfell.name}, Resources: {loaded_winterfell.get_resource_summary()}")
    log.info(f"Loaded Winterfell structures: {loaded_winterfell.get_structure_summary()}")
    assert loaded_winterfell.structures[shelf.id].name == shelf.name
