import numpy as np
import random
import string
from typing import Dict, List, Tuple
from collections import Counter # Needed for generate_themed_name if it uses it, but it doesn't directly.

# Assuming Config class will be accessible, e.g., from game_of_thrones_sim.config import Config
# For now, to avoid circular dependency during refactoring, we might pass Config.TRAITS, Config.SKILLS, etc.
# or define them as constants here if they are static.
# Let's assume for now that the calling code will provide necessary Config values if needed.

def rand_vec(n: int) -> np.ndarray:
    """Generates a random numpy vector of float32."""
    return np.random.rand(n).astype(np.float32)

def random_dna(traits_list: List[str], skills_list: List[str]) -> Dict[str, np.ndarray]:
    """Generates random DNA based on provided trait and skill lists."""
    return {
        "traits": rand_vec(len(traits_list)),
        "skills": rand_vec(len(skills_list))
    }

def crossover(d1: Dict[str, np.ndarray], d2: Dict[str, np.ndarray],
              trait_mut_std: float, skill_mut_std: float) -> Dict[str, np.ndarray]:
    """
    Performs crossover between two DNA dictionaries.
    Uses trait_mut_std for trait mutation standard deviation.
    Uses skill_mut_std for skill mutation/learning potential standard deviation.
    """
    child = {}
    # Traits: Average + mutation
    t = (d1["traits"] + d2["traits"]) / 2
    child["traits"] = np.clip(t + np.random.normal(0, trait_mut_std, len(t)), 0, 1).astype(np.float32)

    # Skills: Max of parents + slight mutation/learning potential
    # skill_mut_std was Config.SKILL_DECAY, but it's better named as a mutation std dev here.
    s = np.maximum(d1["skills"], d2["skills"]) + np.random.normal(0, skill_mut_std, len(d1["skills"]))
    child["skills"] = np.clip(s, 0, 1).astype(np.float32)
    return child

def vec_str(v: np.ndarray, labels: List[str]) -> str:
    """Converts a numpy vector to a labeled string summary."""
    if not isinstance(v, np.ndarray) or v.ndim != 1 or len(v) != len(labels):
        # Consider logging this error instead of returning a string that might be displayed.
        # from .logger import log # Example: log.error(...)
        return "Invalid DNA/Labels"
    return " ".join(f"{lab[:1]}={val:.1f}" for lab, val in zip(labels, v))

def generate_themed_name(female_chance=0.5) -> str:
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

    if len(surnames_list) != len(weights_list):
        # This should ideally not happen if lists are correctly defined.
        # from .logger import log # Example: log.error("Surname and weight lists mismatch!")
        # Fallback or raise error
        pass # For now, let it proceed, but this is an issue.

    selected_surname = random.choices(surnames_list, weights=weights_list, k=1)[0]

    name = f"{p1}{p2}"
    if selected_surname: # Only add surname if it's not an empty string
        name += f" {selected_surname}"

    return name[:20] # Limit length

def oov_check(text: str, forbidden_words: set) -> bool:
    """
    Checks for Out-Of-Vocabulary (forbidden) words in free text.
    Args:
        text: The text to check.
        forbidden_words: A set of forbidden words.
    Returns:
        True if forbidden words are found, False otherwise.
    """
    if not text:
        return False
    lw = text.lower()
    # Create a translation table that removes all punctuation
    translator = str.maketrans('', '', string.punctuation)
    words = lw.translate(translator).split()
    return any(word in forbidden_words for word in words)

if __name__ == '__main__':
    # Example usage
    # Assuming Config.TRAITS and Config.SKILLS are defined elsewhere or passed
    example_traits = ["cunning", "honour", "wrath", "piety"]
    example_skills = ["smith", "war", "lore", "diplomacy"]

    dna1 = random_dna(example_traits, example_skills)
    dna2 = random_dna(example_traits, example_skills)
    print(f"DNA1 Traits: {vec_str(dna1['traits'], example_traits)}")
    print(f"DNA1 Skills: {vec_str(dna1['skills'], example_skills)}")

    child_dna = crossover(dna1, dna2, trait_mut_std=0.04, skill_mut_std=0.001)
    print(f"Child Traits: {vec_str(child_dna['traits'], example_traits)}")
    print(f"Child Skills: {vec_str(child_dna['skills'], example_skills)}")

    for _ in range(5):
        print(f"Generated Name: {generate_themed_name()}")

    print(f"OOV Check ('hello world'): {oov_check('hello world', {'internet', 'computer'})}")
    print(f"OOV Check ('hello internet'): {oov_check('hello internet', {'internet', 'computer'})}")
