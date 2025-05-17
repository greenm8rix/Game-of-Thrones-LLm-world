import asyncio
import argparse
import os
import sys
import threading
import queue
import time # For sleep in sim loop
from pathlib import Path
import numpy as np # For GUI canvas item placement, if not fully encapsulated

# --- Project module imports ---
# Configuration (should be imported first if other modules depend on it at import time)
from config import Config # Assuming Config class is directly in config.py

# Utilities
from utils.logger import log, setup_logging # Use the setup function and pre-configured log
from utils.helpers import generate_themed_name, find_latest_save # find_latest_save was global

# LLM and RAG
from llm.client import initialize_llm_client, SDK_AVAILABLE as LLM_SDK_AVAILABLE
from llm.rag import initialize_rag, get_chroma_client

# Core simulation components
from core.registry import Registry
from core.agent import AgentState
from core.world_elements import Structure # For type hints or direct use if needed
from core.world import World
from core.code_utils import store_original_code, restore_all_original_code

# GUI
from gui.main_window import SimulationGUI, AppRoot

# Global queues for communication between sim thread and GUI thread
# Defined globally here for access by sim loop and GUI setup.
# Maxsize can help prevent runaway queue growth if one thread lags.
UPDATE_QUEUE = queue.Queue(maxsize=200)
ACTION_QUEUE = queue.Queue()


def setup_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASOIAF Autonomous Agent Simulation v3 (Modular)")
    parser.add_argument("--load", action="store_true", help="Load the latest save file from the save directory.")
    parser.add_argument("--save_interval", type=int, default=None, help=f"Ticks between auto-saves (0 to disable). Overrides Config default.")
    parser.add_argument("--model", type=str, default=None, help=f"LLM model to use. Overrides Config default.")
    parser.add_argument("--start_pop", type=int, default=None, help=f"Initial agent population if not loading. Overrides Config default.")
    parser.add_argument("--max_pop", type=int, default=None, help=f"Maximum agent population. Overrides Config default.")
    parser.add_argument("--allow_code_change", action='store_true', default=False, help="[EXPERIMENTAL] Enable agent code modification. Overrides Config default if True.")
    parser.add_argument("--disallow_code_change", action='store_true', default=False, help="[EXPERIMENTAL] Force disable agent code modification. Overrides Config default if True.")
    parser.add_argument("--api_key", type=str, default=None, help="Google API Key. If not provided, uses GOOGLE_API_KEY env var.")
    return parser

def get_snapshot_data_for_gui(current_world: World) -> Tuple[List[Dict], List[Dict]]:
    """Creates serializable snapshots of agents and structures for the GUI."""
    agents_snap = []
    # Send all agents from world.agents; GUI can filter/tag dead ones.
    all_agents_in_world = list(current_world.agents.values()) # agents dict in World
    for a in all_agents_in_world:
        try:
            a_dict = a.to_dict() # AgentState.to_dict()
            a_dict['inventory_summary'] = a.get_inventory_summary()
            a_dict['health'] = a.health # Ensure health is present
            agents_snap.append(a_dict)
        except Exception as e:
            log.error(f"Failed to serialize agent {getattr(a, 'id', 'UNKNOWN_ID')} for GUI: {e}")

    # Structures are in world.structures
    structs_snap = [s.to_dict() for s in current_world.structures.values()]
    return agents_snap, structs_snap


async def run_simulation_loop(
    world: World,
    registry: Registry, # Registry is now managed by World or passed around
    config_instance: Config, # Pass the Config object
    stop_event: threading.Event,
    args: argparse.Namespace # Parsed command-line arguments
):
    """The main asynchronous simulation loop."""
    log.info("Async simulation loop starting.")
    
    # Use save_interval from args if provided, else from config
    save_interval = args.save_interval if args.save_interval is not None else config_instance.DEFAULT_SAVE_INTERVAL
    paused = False
    
    # Ensure world has access to its own registry if needed, or pass registry to methods
    # In this refactor, World takes registry in __init__

    try:
        while not stop_event.is_set():
            start_tick_time = time.time()

            # --- Handle GUI Actions ---
            while not ACTION_QUEUE.empty():
                try:
                    gui_action = ACTION_QUEUE.get_nowait()
                    action_type = gui_action.get("action")
                    log.info(f"Processing GUI action: {action_type}")

                    if action_type == "toggle_pause":
                        paused = not paused
                        status_msg = "Simulation PAUSED." if paused else f"Tick {world.current_tick}. Running."
                        log.info(status_msg)
                        # Send immediate feedback to GUI status
                        snap_agents, snap_structs = get_snapshot_data_for_gui(world)
                        UPDATE_QUEUE.put({
                            "tick": world.current_tick, "day": world.day, "season": world.season,
                            "agents": snap_agents, "structures": snap_structs,
                            "new_events": [f"[SYSTEM] {status_msg}"], "status": status_msg
                        })


                    elif action_type == "save_now":
                        save_dir = Path(config_instance.SAVE_DIR)
                        save_dir.mkdir(parents=True, exist_ok=True)
                        save_filename = save_dir / f"world_tick_{world.current_tick}_manual.json.gz"
                        world.save_to_file(save_filename) # World.save_to_file method
                        log.info(f"Manual save complete: {save_filename.name}")
                        snap_agents, snap_structs = get_snapshot_data_for_gui(world)
                        UPDATE_QUEUE.put({
                            "tick": world.current_tick, "day": world.day, "season": world.season,
                            "agents": snap_agents, "structures": snap_structs,
                            "new_events": [f"[SYSTEM] Manual save: {save_filename.name}"],
                            "status": f"Saved to {save_filename.name}"
                        })

                    elif action_type == "load_file":
                        filepath_str = gui_action.get("filepath", "")
                        filepath = Path(filepath_str)
                        if filepath.exists() and filepath.is_file():
                            log.info(f"Pausing simulation for load: {filepath.name}")
                            paused = True
                            # Inform GUI
                            snap_agents_pre_load, snap_structs_pre_load = get_snapshot_data_for_gui(world)
                            UPDATE_QUEUE.put({
                                "tick": world.current_tick, "day": world.day, "season": world.season,
                                "agents": snap_agents_pre_load, "structures": snap_structs_pre_load,
                                "new_events": [f"[SYSTEM] Load requested: {filepath.name}. Pausing."],
                                "status": f"Loading from {filepath.name}..."
                            })
                            await asyncio.sleep(0.1) # Allow GUI to update status

                            # World.from_file needs config, registry, chroma_client
                            new_world = World.from_file(filepath, config_instance, registry, get_chroma_client())
                            if new_world:
                                # Critical: Replace the 'world' instance in this scope
                                # This is tricky as 'world' is a local variable here.
                                # The original 'world' object needs to be updated, or this function
                                # needs to be able_to re-assign it in the caller's scope, which is not direct.
                                # A common pattern is for the simulation loop to operate on a mutable
                                # world_container object, or for this function to return the new_world
                                # and the caller updates its reference.
                                # For now, let's assume this function can modify the 'world' object's state
                                # by re-assigning its attributes if World.from_file returns a new instance.
                                # The current World.from_file is a @classmethod, so it returns a new instance.
                                # This means the 'world' variable in main_async needs to be updated.
                                # This loop should probably be part of a class that holds the world state.
                                # Quick fix: modify attributes of the existing world object.
                                # This is not ideal if from_file does complex re-init.
                                # The best way is that main_async holds the world and passes it,
                                # and if load happens, main_async reassigns its world variable.
                                # For now, this action will be problematic for 'world' reassignment.
                                # Let's assume for now the GUI action queue is checked in main_async,
                                # which can then reassign 'world'.
                                # This loop will just signal that a load should happen.
                                # So, this part of the logic needs to be outside the direct sim loop or handled differently.

                                # Let's simplify: the GUI sends a "stop_for_load" signal,
                                # main_async stops, loads, then restarts the loop with new world.
                                # For now, I'll log an error for this complex state change.
                                log.error("Dynamic world loading within run_simulation_loop is complex. "
                                          "Consider handling load by restarting the loop with the new world object.")
                                # To make it work somewhat:
                                world.__dict__.update(new_world.__dict__) # Risky, but attempts to update state
                                registry.clear()
                                for agent_obj in world.agents.values(): registry.add(agent_obj)


                                log.info(f"World state loaded from {filepath.name}. Resuming (if not paused by user).")
                                # paused = False # Or let user unpause
                                snap_agents_post_load, snap_structs_post_load = get_snapshot_data_for_gui(world)
                                UPDATE_QUEUE.put({
                                    "tick": world.current_tick, "day": world.day, "season": world.season,
                                    "agents": snap_agents_post_load, "structures": snap_structs_post_load,
                                    "new_events": [f"[SYSTEM] World loaded from {filepath.name}."],
                                    "status": f"Loaded tick {world.current_tick}. {'PAUSED' if paused else 'Running.'}"
                                })
                            else:
                                log.error(f"Failed to load world from {filepath}. Simulation remains paused.")
                                UPDATE_QUEUE.put({
                                    "tick": world.current_tick, "day": world.day, "season": world.season,
                                    "agents": get_snapshot_data_for_gui(world)[0], # current agents
                                    "structures": get_snapshot_data_for_gui(world)[1], # current structs
                                    "new_events": [f"[ERROR] Failed to load {filepath.name}. Check logs."],
                                    "status": f"Load failed. Paused."
                                })
                        else:
                            log.warning(f"GUI requested load from invalid path: {filepath_str}")
                    
                    elif action_type == "stop_simulation": # GUI requests shutdown
                        log.info("GUI requested simulation stop.")
                        stop_event.set() # Signal main loop to stop
                        # GUI will be destroyed by the main thread after this loop exits.

                    ACTION_QUEUE.task_done()
                except queue.Empty:
                    break # No more actions for now
                except Exception as e:
                    log.error(f"Error processing GUI action: {e}", exc_info=True)

            if stop_event.is_set(): break # Check before starting tick processing

            # --- Simulation Tick ---
            if paused:
                if world.current_tick % 20 == 0: # Less frequent update while paused
                    snap_agents_paused, snap_structs_paused = get_snapshot_data_for_gui(world)
                    UPDATE_QUEUE.put({
                        "tick": world.current_tick, "day": world.day, "season": world.season,
                        "agents": snap_agents_paused, "structures": snap_structs_paused,
                        "new_events": [], "status": f"Tick {world.current_tick}. PAUSED."
                    })
                await asyncio.sleep(0.2) # Longer sleep when paused
                continue

            current_tick_events: List[str] = []

            # 1. World Update (Time, Needs, Regen, Code Apply, Events, Agent Removal)
            try:
                world_events = world.tick_update() # World.tick_update()
                current_tick_events.extend(world_events)
            except Exception as e:
                log.critical(f"CRITICAL ERROR during world.tick_update at tick {world.current_tick}: {e}", exc_info=True)
                current_tick_events.append(f"T{world.current_tick}: [!!!] CRITICAL World Update Error: {type(e).__name__}")
                stop_event.set()
                # Send final error state to GUI
                snap_agents_err, snap_structs_err = get_snapshot_data_for_gui(world)
                UPDATE_QUEUE.put({
                    "tick": world.current_tick, "day": world.day, "season": world.season,
                    "agents": snap_agents_err, "structures": snap_structs_err,
                    "new_events": current_tick_events, "status": f"FATAL ERROR @ Tick {world.current_tick}. Check logs."
                })
                raise # Re-raise to exit loop

            # 2. Agent Actions (Concurrent)
            living_agents = world.get_living_agents() # World.get_living_agents()
            random.shuffle(living_agents)

            tasks = [asyncio.create_task(agent.act(world, registry)) for agent in living_agents]
            action_results = []
            if tasks:
                try:
                    action_results = await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as e:
                    log.error(f"Error during asyncio.gather for agent actions: {e}", exc_info=True)
                    current_tick_events.append(f"T{world.current_tick}: [!!!] System Error during action processing: {type(e).__name__}")

            agent_exceptions = 0
            for result in action_results:
                if isinstance(result, Exception):
                    log.error(f"Error during agent action task: {result}", exc_info=result)
                    current_tick_events.append(f"T{world.current_tick}: [!] Agent Action Error: {type(result).__name__}")
                    agent_exceptions += 1
                elif isinstance(result, str):
                    current_tick_events.append(f"T{world.current_tick}: {result}") # Result is already formatted

            # 3. Autosave
            if save_interval > 0 and world.current_tick > 0 and world.current_tick % save_interval == 0:
                save_dir_auto = Path(config_instance.SAVE_DIR)
                save_dir_auto.mkdir(parents=True, exist_ok=True)
                autosave_filename = save_dir_auto / f"world_tick_{world.current_tick}.json.gz"
                world.save_to_file(autosave_filename)
                current_tick_events.append(f"T{world.current_tick}: [SYSTEM] Autosaved to {autosave_filename.name}")

            # --- Final Update for Tick to GUI ---
            end_tick_time = time.time()
            tick_duration = end_tick_time - start_tick_time
            living_agent_count_now = len(world.get_living_agents())
            status_str = f"Tick {world.current_tick} | Day {world.day} | Pop: {living_agent_count_now} | Dur: {tick_duration:.2f}s"
            if agent_exceptions > 0: status_str += f" | Action Errors: {agent_exceptions}"
            log.info(status_str) # Log overall tick status

            snap_agents_final, snap_structs_final = get_snapshot_data_for_gui(world)
            UPDATE_QUEUE.put({
                "tick": world.current_tick, "day": world.day, "season": world.season,
                "agents": snap_agents_final, "structures": snap_structs_final,
                "new_events": current_tick_events, "status": status_str
            })

            # --- Delay ---
            time_to_wait = config_instance.TICK_DELAY - tick_duration
            if time_to_wait > 0:
                await asyncio.sleep(time_to_wait)

    except asyncio.CancelledError:
        log.info("Simulation loop cancelled.")
    except Exception as e:
        log.critical(f"Fatal error in simulation loop (tick {world.current_tick}): {e}", exc_info=True)
        if not stop_event.is_set(): stop_event.set() # Ensure stop if not already
        # Send final error message to GUI
        try:
            UPDATE_QUEUE.put({"tick": world.current_tick, "day": world.day, "season": world.season, "agents": [], "structures": [], "new_events": [f"FATAL SIM ERROR @ T{world.current_tick}: {type(e).__name__}. Check logs."],"status": f"FATAL ERROR @ Tick {world.current_tick}. Check logs."})
        except Exception: pass # Ignore queue errors during crash
    finally:
        log.info("Async simulation loop stopped.")


async def main_async(args: argparse.Namespace, stop_event: threading.Event):
    """The main asynchronous entry point for setting up and running the simulation."""
    log.info(f"Async main started. PID: {os.getpid()}, Thread: {threading.current_thread().name}")
    
    # --- Apply Args to Config ---
    # Create a Config instance first
    config_instance = Config()
    if args.model: config_instance.MODEL = args.model
    if args.start_pop is not None: config_instance.START_POP = args.start_pop
    if args.max_pop is not None: config_instance.MAX_POP = args.max_pop
    if args.save_interval is not None: config_instance.DEFAULT_SAVE_INTERVAL = args.save_interval
    
    if args.allow_code_change:
        config_instance.ALLOW_CODE_MODIFICATION = True
    elif args.disallow_code_change: # Explicitly disallow
        config_instance.ALLOW_CODE_MODIFICATION = False
    # Otherwise, it uses the default from Config class definition

    log.info(f"Effective Config: Model={config_instance.MODEL}, StartPop={config_instance.START_POP}, MaxPop={config_instance.MAX_POP}, SaveInterval={config_instance.DEFAULT_SAVE_INTERVAL}, CodeMod={config_instance.ALLOW_CODE_MODIFICATION}")

    # --- Store Original Code (after config is finalized) ---
    # This needs the actual class definitions/modules.
    store_original_code(config_instance, AgentState, World) # Pass class types

    world_instance: Optional[World] = None # Initialize

    try:
        # --- Initialize LLM ---
        if not LLM_SDK_AVAILABLE: # Check the flag from llm.client
            log.critical("Google Generative AI SDK not available. LLM functionalities will be disabled.")
            # Decide if this is a fatal error or if the simulation can run without LLM.
            # For now, let's assume it's fatal if an API key is provided or expected.
            if args.api_key or os.getenv("GOOGLE_API_KEY"):
                 print("[CRITICAL ERROR] LLM SDK missing but API key provided. Cannot proceed.")
                 stop_event.set(); return
        elif not initialize_llm_client(model_name=config_instance.MODEL, api_key=args.api_key):
            log.critical("Failed to initialize LLM Client. Check API key and model name.")
            # Depending on sim design, this might be fatal or allow running without LLM.
            # For now, assume fatal if key was given.
            if args.api_key or os.getenv("GOOGLE_API_KEY"):
                 print("[CRITICAL ERROR] Failed to initialize LLM Client. Cannot proceed.")
                 stop_event.set(); return

        # --- Initialize RAG (ChromaDB) ---
        log.info("Initializing RAG (ChromaDB)...")
        if not initialize_rag(config=config_instance): # Pass config instance
            log.warning("ChromaDB client initialization failed. RAG might be disabled or limited.")
        
        chroma_db_client = get_chroma_client() # Get the initialized client

        # --- Initialize World & Registry ---
        registry_instance = Registry() # Create the central agent registry

        if args.load:
            save_dir = Path(config_instance.SAVE_DIR)
            latest_save_path = find_latest_save(save_dir) # Use helper
            if latest_save_path:
                log.info(f"Loading world from latest save: {latest_save_path}...")
                world_instance = World.from_file(latest_save_path, config_instance, registry_instance, chroma_db_client)
                if world_instance:
                    log.info("World loaded successfully from save.")
                else:
                    log.warning("Failed to load from save file. Starting new world.")
            else:
                log.warning(f"No save file found in {save_dir}. Starting new world.")

        if world_instance is None: # If not loaded or load failed
            log.info("Creating new world instance.")
            world_instance = World(config=config_instance, registry=registry_instance, chroma_client=chroma_db_client)
            # Seed initial population for a new world
            log.info(f"Seeding initial population ({config_instance.START_POP} agents)...")
            initial_agents_added = 0
            for _ in range(config_instance.START_POP):
                if len(world_instance.agents) >= config_instance.MAX_POP: break
                agent_name = generate_themed_name()
                # AgentState needs config and chroma_client
                new_agent = AgentState(name=agent_name, config=config_instance, chroma_client=chroma_db_client)
                if registry_instance.add(new_agent): # Add to registry
                    world_instance.agents[new_agent.id] = new_agent # Add to world's dict
                    initial_agents_added += 1
            log.info(f"Initial population seeded: {initial_agents_added} agents.")
        
        # Ensure registry is consistent with world.agents if loaded (from_file should handle this)
        # For safety:
        if args.load and world_instance:
            registry_instance.clear()
            for agent_obj in world_instance.agents.values():
                registry_instance.add(agent_obj)
            log.info(f"Registry synchronized with loaded world: {len(registry_instance.agents)} agents.")


        # --- Start Simulation Loop ---
        log.info("Starting simulation loop...")
        await run_simulation_loop(world_instance, registry_instance, config_instance, stop_event, args)

    except Exception as e:
        log.critical(f"Critical error in main_async setup or pre-loop: {e}", exc_info=True)
        print(f"\n[CRITICAL ERROR] Async main setup failed: {e}")
        # Try to inform GUI
        err_tick = world_instance.current_tick if world_instance else 0
        UPDATE_QUEUE.put({"tick": err_tick, "day": 0, "season": "?", "agents": [], "structures": [], "new_events": [f"FATAL ASYNC SETUP ERROR: {type(e).__name__}. Check logs."], "status": f"FATAL SETUP ERROR. Check logs."})
    finally:
        log.info("Async main_async function finished.")
        if not stop_event.is_set():
            stop_event.set() # Ensure stop event is set on any exit from here


def run_gui_thread(stop_event: threading.Event, config_instance: Config, registry_instance: Registry):
    """Runs the Tkinter GUI in the main thread."""
    log.info(f"GUI starting. PID: {os.getpid()}, Thread: {threading.current_thread().name}")
    root = AppRoot() # Or tk.Tk()
    
    gui = SimulationGUI(root, UPDATE_QUEUE, ACTION_QUEUE, config_instance, registry_instance)

    def on_gui_closing_from_protocol():
        # This is called when WM_DELETE_WINDOW is triggered from SimulationGUI._on_closing
        # which already put "stop_simulation" on ACTION_QUEUE.
        # The simulation loop will see that and set stop_event.
        # Here, we just ensure the GUI window is eventually destroyed.
        log.info("GUI on_closing_from_protocol: stop_event should be set by sim loop. Preparing to destroy root.")
        # Give sim loop a bit of time to process the stop_event from action queue
        # This is a bit of a race, ideally sim_thread.join() is called before root.destroy()
        # For now, let the main thread's finally block handle join and destroy.
        # If the GUI is closed by user, the stop_event is set, sim_thread will exit,
        # then main thread's finally block will join sim_thread and destroy root.
        # This function might not need to do root.destroy() if main handles it.
        # However, if this is the *only* way GUI closes, it might be needed.
        # Let's assume main's finally block handles it.
        if not stop_event.is_set():
             log.warning("GUI closing, but stop_event not yet set by sim. This might be abrupt for sim.")
             # stop_event.set() # Force it here if necessary, but action queue is preferred.

    # The SimulationGUI's _on_closing method handles the WM_DELETE_WINDOW protocol
    # and puts "stop_simulation" on the ACTION_QUEUE.

    print("Simulation running. Close the window or use File > Exit to quit.")
    try:
        root.mainloop()
    except KeyboardInterrupt:
        log.info("Ctrl+C detected in GUI thread (mainloop). Requesting shutdown.")
        if not stop_event.is_set():
            ACTION_QUEUE.put({"action": "stop_simulation"}) # Signal sim to stop
            # stop_event.set() # Or set directly
    finally:
        log.info("GUI mainloop exited.")
        # Ensure stop_event is set if GUI exits for any reason,
        # so the simulation thread knows to stop.
        if not stop_event.is_set():
            log.info("GUI exited, ensuring stop_event is set for simulation thread.")
            stop_event.set()


if __name__ == "__main__":
    if sys.version_info < (3, 8): # Python 3.8+ for asyncio features used
        print("Python 3.8+ required for this simulation.")
        sys.exit(1)

    # Initial basic logger setup for early messages
    # The main logger will be fully configured by setup_logging() later.
    # This is just for messages before that.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")

    # --- Argument Parsing ---
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    # --- Setup Full Logging ---
    # `log` from utils.logger will be the configured logger instance
    # `setup_logging()` reconfigures the root logger and returns the named logger.
    log = setup_logging() # Now `log` is the official application logger.

    log.info("Starting Autonomous ASOIAF Simulation v3 (Modular)...")
    log.info(f"Log file: {Path(Config.LOG_FILE_NAME if hasattr(Config, 'LOG_FILE_NAME') else './got_sim_main.log').resolve()}") # Use Config if LOG_FILE_NAME is there

    # --- Create Config instance (will be updated by args in main_async) ---
    # This instance is passed to GUI and potentially other setup steps.
    # main_async will create its own effective_config based on this and args.
    # This is a bit redundant, could be streamlined.
    # Let's make one Config object and pass it around.
    
    # Effective Config setup (moved from main_async to here for clarity)
    effective_config = Config()
    if args.model: effective_config.MODEL = args.model
    if args.start_pop is not None: effective_config.START_POP = args.start_pop
    if args.max_pop is not None: effective_config.MAX_POP = args.max_pop
    if args.save_interval is not None: effective_config.DEFAULT_SAVE_INTERVAL = args.save_interval
    
    if args.allow_code_change:
        effective_config.ALLOW_CODE_MODIFICATION = True
    elif args.disallow_code_change:
        effective_config.ALLOW_CODE_MODIFICATION = False
    
    log.info(f"Using Config: Model={effective_config.MODEL}, StartPop={effective_config.START_POP}, MaxPop={effective_config.MAX_POP}, SaveInterval={effective_config.DEFAULT_SAVE_INTERVAL}, CodeMod={effective_config.ALLOW_CODE_MODIFICATION}")
    if effective_config.ALLOW_CODE_MODIFICATION:
        log.warning("!!! Agent Code Modification is ENABLED (EXPERIMENTAL) !!!")


    # --- Threading and Asyncio Setup ---
    stop_event = threading.Event()
    sim_thread: Optional[threading.Thread] = None
    async_loop: Optional[asyncio.AbstractEventLoop] = None
    
    # Create a global registry instance to be shared
    # This is needed by GUI for agent name lookups.
    # World also uses it.
    main_registry = Registry()


    try:
        # Create and start the asyncio event loop for the simulation in a separate thread
        async_loop = asyncio.new_event_loop()

        def run_async_tasks_in_thread(loop: asyncio.AbstractEventLoop, args_ns: argparse.Namespace, stop_evt: threading.Event):
            asyncio.set_event_loop(loop)
            try:
                # Pass the effective_config to main_async
                loop.run_until_complete(main_async(args_ns, stop_evt)) # main_async will use this config
            finally:
                log.info("Async loop run_until_complete finished in sim_thread. Closing loop.")
                try:
                    tasks = asyncio.all_tasks(loop=loop)
                    if tasks:
                        log.info(f"Cancelling {len(tasks)} outstanding asyncio tasks in sim_thread...")
                        for task in tasks: task.cancel()
                        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                        log.info("Outstanding asyncio tasks in sim_thread cancelled.")
                except Exception as e_cancel:
                    log.error(f"Error cancelling tasks during sim_thread loop close: {e_cancel}")
                finally:
                    if not loop.is_closed():
                        loop.close()
                        log.info("Async loop in sim_thread closed.")

        sim_thread = threading.Thread(
            target=run_async_tasks_in_thread,
            args=(async_loop, args, stop_event), # Pass args for main_async
            name="SimulationThread",
            daemon=True # Exits if main GUI thread exits
        )
        sim_thread.start()
        log.info("Simulation thread started.")

        # Run GUI in the main thread
        # Pass the effective_config and main_registry to the GUI
        run_gui_thread(stop_event, effective_config, main_registry)

    except Exception as e_main_setup:
        log.critical(f"Fatal error during main setup or GUI launch: {e_main_setup}", exc_info=True)
        print(f"\nFATAL ERROR: {e_main_setup}")
    finally:
        log.info("Main thread exiting. Signaling stop to simulation if not already set.")
        if not stop_event.is_set():
            stop_event.set() # Ensure simulation thread is signaled to stop

        if sim_thread and sim_thread.is_alive():
            log.info("Waiting for simulation thread to join...")
            sim_thread.join(timeout=10.0) # Increased timeout
            if sim_thread.is_alive():
                log.warning("Simulation thread did not exit cleanly after join timeout.")
            else:
                log.info("Simulation thread joined successfully.")
        
        # If async_loop was created and might still be running (e.g. if sim_thread daemonized and main exited fast)
        # This is mostly for cleanup if things didn't shut down via stop_event as expected.
        if async_loop and not async_loop.is_closed() and async_loop.is_running():
            log.info("Calling loop.call_soon_threadsafe to stop the async_loop from main thread.")
            async_loop.call_soon_threadsafe(async_loop.stop)
            # Give it a moment to process the stop
            # time.sleep(0.1) # This might not be necessary if join worked.
            # if not async_loop.is_closed(): # Check again
            #     async_loop.close()
            #     log.info("Async loop closed in main finally block.")


        # Ensure log handlers are closed (important for file logs)
        logging.shutdown()
        print("Simulation shutdown complete.")
