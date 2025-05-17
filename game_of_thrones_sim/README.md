# ASOIAF Autonomous Agent Simulation

This project is an autonomous agent simulation set in the universe of George R.R. Martin's "A Song of Ice and Fire" (ASOIAF). Agents in this simulation are powered by a Large Language Model (LLM), allowing them to make decisions, interact with their environment, and pursue goals within the rich and dangerous world of Westeros.

## Key Features

*   **Autonomous Agents**: Each agent possesses unique DNA (traits and skills), needs (hunger, health, morale), and makes decisions based on their state and surroundings using Google's Gemini LLM.
*   **Dynamic World**: The simulation features multiple iconic locations from ASOIAF, each with its own resources and potential for development.
*   **Resource Management & Crafting**: Agents can gather resources (wood, food, stone, ore) and craft items, including tools, weapons, and structures like shelters and forges.
*   **Interactions & Social Dynamics**: Agents can interact with each other through speech, combat, healing, and even propose reproduction to create new agents. Relationships between agents can evolve based on their interactions.
*   **Retrieval Augmented Generation (RAG)**: A RAG system using ChromaDB provides agents with access to lore and information from the ASOIAF books, enhancing their in-character behavior and knowledge.
*   **Graphical User Interface (GUI)**: A Tkinter-based GUI allows for real-time visualization of the simulation, including agent locations, states, recent events, and world structures. Users can also interact with the simulation through the GUI (e.g., save/load, pause/resume).
*   **Experimental Code Modification**: A highly experimental feature allows agents with high 'lore' skill to attempt to inspect and propose modifications to certain simulation parameters or agent behaviors, offering a unique meta-interaction layer.
*   **Persistence**: The simulation state can be saved and loaded, allowing for long-running scenarios.

## Project Structure

The project is organized into a modular structure within the `game_of_thrones_sim` directory:

```
game_of_thrones_sim/
├── main.py                 # Main application entry point
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── config.py               # Core configuration settings for the simulation
├── core/                   # Core simulation logic
│   ├── agent.py            # AgentState class definition
│   ├── world.py            # World class, manages simulation state and progression
│   ├── registry.py         # Agent registry
│   ├── world_elements.py   # LocationNode and Structure classes
│   └── code_utils.py       # Utilities for the experimental code modification feature
├── llm/                    # LLM and RAG functionalities
│   ├── client.py           # Gemini LLM client setup and interaction
│   └── rag.py              # RAG system setup and querying (ChromaDB)
├── gui/                    # Graphical User Interface
│   └── main_window.py      # Tkinter GUI class
├── utils/                  # Utility modules
│   ├── helpers.py          # Helper functions (e.g., name generation, DNA manipulation)
│   └── logger.py           # Logging setup
├── data/
│   └── got_chunks.json     # (Optional) Pre-processed lore data for RAG
└── saves_got_v3/           # Default directory for saved simulation states
```

## Getting Started

### Prerequisites

*   Python 3.8 or higher.
*   Access to Google's Gemini API and an API key.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>/game_of_thrones_sim
    ```

2.  **Set up a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **API Key Setup**:
    You need to provide your Google API key for the Gemini LLM. Set it as an environment variable:
    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```
    Alternatively, you can pass it as a command-line argument when running `main.py`.

5.  **RAG Data** (Optional but Recommended):
    For the RAG system to function effectively, place a `got_chunks.json` file in the `game_of_thrones_sim/data/` directory. This file should contain text snippets from the ASOIAF books. If not found, a dummy file will be created, but RAG effectiveness will be minimal. The path is configurable in `config.py`.

### Running the Simulation

Navigate to the `game_of_thrones_sim` directory (if not already there) and run the main script:

```bash
python main.py
```

You can see available command-line options by running:

```bash
python main.py --help
```

Common options include:
*   `--load`: Load the latest saved simulation.
*   `--save_interval <ticks>`: Set the auto-save frequency.
*   `--model <model_name>`: Specify the Gemini model to use.
*   `--start_pop <number>`: Set the initial agent population.
*   `--allow_code_change` / `--disallow_code_change`: Enable or disable the experimental agent code modification feature.
*   `--api_key <your_key>`: Provide the Google API key directly.

## Configuration

Key simulation parameters can be adjusted in `game_of_thrones_sim/config.py`. This includes:
*   LLM model and API settings.
*   Agent population limits, needs, and behavior parameters.
*   World details like locations, resources, and crafting recipes.
*   RAG system settings (ChromaDB path, lore file path).
*   GUI update intervals.
*   Code modification whitelist and limits.

## Dependencies

The main external dependencies are listed in `requirements.txt`:
*   `numpy`: For numerical operations, especially in agent DNA and calculations.
*   `google-generativeai`: The official Google client library for Gemini.
*   `chromadb`: For the vector database used by the RAG system.

## Logging

The simulation logs events and errors to a file named `got_sim_gui_v3.log` (by default) located in the project's root directory (alongside the `game_of_thrones_sim` folder). Log messages are also printed to the console.

## Notes on Experimental Features

*   **Agent Code Modification**: This feature (`ALLOW_CODE_MODIFICATION` in `config.py` or via CLI args) is highly experimental. It allows agents to attempt to change parts of the simulation's Python code at runtime. While sandboxed to a degree, enabling this feature carries inherent risks and can lead to instability or unexpected behavior. Use with caution.

## Future Enhancements (Ideas)

*   More complex economic systems (trade, currency).
*   Advanced social structures (houses, factions, alliances).
*   More sophisticated combat mechanics.
*   Expanded crafting tree and technology progression.
*   More diverse random events and quests.
*   Improved GUI with more detailed agent inspection and world interaction.

Contributions and suggestions are welcome!
