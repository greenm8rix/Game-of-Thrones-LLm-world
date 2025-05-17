import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import queue
from typing import Dict, List, Optional, TYPE_CHECKING
from pathlib import Path # For ask_load_file initialdir

if TYPE_CHECKING:
    from ..config import Config # To access GUI constants and Config.SAVE_DIR
    from ..core.registry import Registry # For registry.get_agent_name

# Placeholder for logger and config
class PrintLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg, exc_info=False): print(f"ERROR: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")

log = PrintLogger()

class MockConfigForGUI: # Placeholder
    GUI_UPDATE_INTERVAL = 150
    EVENT_LOG_DISPLAY_SIZE = 100
    MAX_POP = 100 # For agent label
    MAX_HEALTH = 100.0 # For health tag
    LOCATIONS = ["Winterfell", "KingsLanding", "TheWall", "Dragonstone", "Braavos"] # For map
    SAVE_DIR = "./saves_gui_test"
    ALLOW_CODE_MODIFICATION = False # For window title

# GUI Constants (originally global, now can be part of Config or passed)
# These are used by _draw_location_markers and _update_canvas
CANVAS_WIDTH = 600
CANVAS_HEIGHT = 450
LOCATION_COORDS = { # Should come from Config
    "Winterfell": (150, 100), "KingsLanding": (400, 350),
    "TheWall": (300, 50), "Dragonstone": (500, 200), "Braavos": (550, 100),
}
AGENT_DOT_SIZE = 5
STRUCTURE_BOX_SIZE = 8
LOCATION_RADIUS = 20


class AppRoot(tk.Tk):
    """Custom root window if needed for specific app-wide configurations."""
    pass

class SimulationGUI:
    def __init__(self, root: tk.Tk,
                 update_queue: queue.Queue,
                 action_queue: queue.Queue,
                 config: 'Config', # Pass the config object
                 registry_ref: 'Registry' # Pass a reference to the registry
                 ):
        global log
        try:
            # Try to import the actual logger
            from ..utils.logger import log as main_log
            log = main_log
        except ImportError:
            pass # Keep print logger

        self.root = root
        self.update_queue = update_queue
        self.action_queue = action_queue
        self.config = config
        self.registry = registry_ref # Store the registry reference

        # Use config values for GUI constants
        self.canvas_width = getattr(self.config, "CANVAS_WIDTH", CANVAS_WIDTH)
        self.canvas_height = getattr(self.config, "CANVAS_HEIGHT", CANVAS_HEIGHT)
        self.location_coords = getattr(self.config, "LOCATION_COORDS", LOCATION_COORDS)
        self.agent_dot_size = getattr(self.config, "AGENT_DOT_SIZE", AGENT_DOT_SIZE)
        self.structure_box_size = getattr(self.config, "STRUCTURE_BOX_SIZE", STRUCTURE_BOX_SIZE)
        self.location_radius = getattr(self.config, "LOCATION_RADIUS", LOCATION_RADIUS)


        self.root.title(f"ASOIAF Autonomous Simulation (Code Mod: {'ENABLED' if self.config.ALLOW_CODE_MODIFICATION else 'DISABLED'})")
        self.root.minsize(1200, 800)

        self._configure_styles()
        self._create_widgets()
        self._layout_widgets()
        self._create_menu() # Added menu creation call
        
        self.process_queue() # Start processing updates

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use('clam') # Or 'default', 'alt', 'vista' etc.
        style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'))
        style.configure("Treeview", rowheight=25, font=('Helvetica', 9))
        style.map('Treeview', background=[('selected', '#ADD8E6')]) # Light blue selection
        # Health status tags for Treeview
        style.configure("Healthy.Treeview", background="#DFF0D8") # Light green
        style.configure("Injured.Treeview", background="#FCF8E3") # Light yellow
        style.configure("Critical.Treeview", background="#F2DEDE") # Light red
        style.configure("Dead.Treeview", background="#D3D3D3", foreground="#555555") # Grey
        style.configure("Unknown.Treeview", background="#FFFFFF") # Default for unknown status

        style.configure("TLabel", padding=5)
        style.configure("TFrame", background="#ECECEC") # Light grey background for frames
        style.configure("Header.TLabel", background="#4682B4", foreground="white", font=('Helvetica', 14, 'bold'), anchor=tk.CENTER) # Steel blue
        style.configure("Footer.TLabel", background="#D3D3D3", font=('Helvetica', 8), anchor=tk.W) # Light grey footer
        # Section header labels
        style.configure("SectionHeader.TLabel", font=('Helvetica', 11, 'bold'), background="#ECECEC", foreground="#333333")


    def _create_widgets(self):
        # Main layout frames
        self.header_frame = ttk.Frame(self.root, height=40, style="TFrame")
        self.main_frame = ttk.Frame(self.root, style="TFrame")
        self.footer_frame = ttk.Frame(self.root, height=30, style="TFrame")

        # PanedWindow for resizable sections (optional, but good for usability)
        # self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)

        # Left and Right frames (will go into PanedWindow or directly into main_frame)
        self.left_frame = ttk.Frame(self.main_frame, style="TFrame") # Or self.paned_window
        self.right_frame = ttk.Frame(self.main_frame, style="TFrame")# Or self.paned_window
        # self.paned_window.add(self.left_frame, weight=3)
        # self.paned_window.add(self.right_frame, weight=2)


        # Sub-frames for content
        self.map_frame = ttk.Frame(self.left_frame, style="TFrame")
        self.agents_frame = ttk.Frame(self.left_frame, style="TFrame")
        self.events_frame = ttk.Frame(self.right_frame, style="TFrame")
        self.structures_frame = ttk.Frame(self.right_frame, style="TFrame")

        # Header and Footer labels
        self.tick_label = ttk.Label(self.header_frame, text="Tick: 0 | Day: 0 | Season: ?", style="Header.TLabel")
        self.status_label = ttk.Label(self.footer_frame, text="Status: Initializing...", style="Footer.TLabel")

        # Section Labels
        self.map_label = ttk.Label(self.map_frame, text="World Map", style="SectionHeader.TLabel")
        self.agents_label = ttk.Label(self.agents_frame, text=f"Agents (0/{self.config.MAX_POP})", style="SectionHeader.TLabel")
        self.events_label = ttk.Label(self.events_frame, text="Recent Events", style="SectionHeader.TLabel")
        self.structures_label = ttk.Label(self.structures_frame, text="World Structures (0)", style="SectionHeader.TLabel")

        # Map Canvas
        self.map_canvas = tk.Canvas(self.map_frame, width=self.canvas_width, height=self.canvas_height, bg="#DDEEFF", relief=tk.SUNKEN, bd=1)
        self._draw_location_markers() # Initial draw

        # Agent List (Treeview)
        self.pop_tree = ttk.Treeview(
            self.agents_frame,
            columns=("ID", "Name", "Loc", "HP", "Hunger", "Morale", "Role", "Inventory"),
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
        # Tags are configured in _configure_styles and applied in update_gui

        # Event Log (ScrolledText)
        self.event_text = scrolledtext.ScrolledText(self.events_frame, wrap=tk.WORD, font=('Helvetica', 9), relief=tk.SUNKEN, bd=1, state=tk.DISABLED)

        # Structures List (ScrolledText)
        self.structure_text = scrolledtext.ScrolledText(self.structures_frame, wrap=tk.WORD, height=10, state=tk.DISABLED, font=('Helvetica', 9), relief=tk.SUNKEN, bd=1)

    def _draw_location_markers(self):
        self.map_canvas.delete("location_marker") # Clear old markers
        for loc, (x, y) in self.location_coords.items():
            self.map_canvas.create_oval(
                x - self.location_radius, y - self.location_radius,
                x + self.location_radius, y + self.location_radius,
                fill="#AAAAAA", outline="black", width=1, tags=("location_marker", loc)
            )
            self.map_canvas.create_text(
                x, y + self.location_radius + 5, text=loc, anchor=tk.N,
                font=('Helvetica', 8, 'bold'), tags=("location_marker", loc)
            )

    def _layout_widgets(self):
        self.header_frame.pack(fill=tk.X, side=tk.TOP, expand=False)
        self.footer_frame.pack(fill=tk.X, side=tk.BOTTOM, expand=False)
        self.main_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP, padx=5, pady=5)
        # self.paned_window.pack(fill=tk.BOTH, expand=True) # If using PanedWindow

        self.tick_label.pack(fill=tk.X, expand=True, padx=10)
        self.status_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Grid layout for left and right frames within main_frame
        self.main_frame.columnconfigure(0, weight=3, minsize=650) # Left side (map, agents)
        self.main_frame.columnconfigure(1, weight=2, minsize=450) # Right side (events, structures)
        self.main_frame.rowconfigure(0, weight=1)

        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # Layout within left_frame
        self.left_frame.rowconfigure(0, weight=3) # Map gets more space
        self.left_frame.rowconfigure(1, weight=2) # Agents list
        self.left_frame.columnconfigure(0, weight=1)
        self.map_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        self.agents_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))

        # Layout within map_frame
        self.map_frame.rowconfigure(1, weight=1)
        self.map_frame.columnconfigure(0, weight=1)
        self.map_label.grid(row=0, column=0, sticky="ew")
        self.map_canvas.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Layout within agents_frame
        self.agents_frame.rowconfigure(1, weight=1)
        self.agents_frame.columnconfigure(0, weight=1)
        self.agents_frame.columnconfigure(1, weight=0) # Scrollbar column
        self.agents_label.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0,5))
        self.pop_tree.grid(row=1, column=0, sticky="nsew")
        self.pop_scroll.grid(row=1, column=1, sticky="ns")

        # Layout within right_frame
        self.right_frame.rowconfigure(0, weight=3) # Events log
        self.right_frame.rowconfigure(1, weight=2) # Structures list
        self.right_frame.columnconfigure(0, weight=1)
        self.events_frame.grid(row=0, column=0, sticky="nsew", pady=(0,5))
        self.structures_frame.grid(row=1, column=0, sticky="nsew", pady=(5,0))

        # Layout within events_frame
        self.events_frame.rowconfigure(1, weight=1)
        self.events_frame.columnconfigure(0, weight=1)
        self.events_label.grid(row=0, column=0, sticky="ew")
        self.event_text.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        # Layout within structures_frame
        self.structures_frame.rowconfigure(1, weight=1)
        self.structures_frame.columnconfigure(0, weight=1)
        self.structures_label.grid(row=0, column=0, sticky="ew")
        self.structure_text.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Now", command=lambda: self.action_queue.put({"action": "save_now"}))
        file_menu.add_command(label="Load...", command=self._ask_load_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing) # Use internal method for proper shutdown

        sim_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Simulation", menu=sim_menu)
        sim_menu.add_command(label="Pause/Resume", command=lambda: self.action_queue.put({"action": "toggle_pause"}))
        
        # Hook WM_DELETE_WINDOW to our closing method
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)


    def _ask_load_file(self):
        save_dir_path = Path(self.config.SAVE_DIR)
        save_dir_path.mkdir(parents=True, exist_ok=True) # Ensure it exists
        
        filepath = filedialog.askopenfilename(
            title="Load Simulation State",
            initialdir=str(save_dir_path.resolve()),
            filetypes=[("Compressed JSON", "*.json.gz"), ("All Files", "*.*")]
        )
        if filepath:
            log.info(f"GUI: Queuing load action for {filepath}")
            self.action_queue.put({"action": "load_file", "filepath": filepath})

    def _on_closing(self):
        log.info("GUI window closing signal received.")
        # Ask for confirmation
        if messagebox.askokcancel("Quit", "Do you want to quit the simulation?"):
            self.action_queue.put({"action": "stop_simulation"}) # Signal sim loop to stop
            # The main simulation loop or GUI runner should handle root.destroy()
            # For now, we just signal. If this GUI class is responsible for destroying root:
            # self.root.destroy()
            # However, it's better if the main thread that started the GUI handles this
            # after the simulation thread has had a chance to shut down.
            # The original code had root.after(500, root.destroy) in the main GUI runner.
            # This function now just signals the intent to stop.
            # The main loop will see the stop_event and then can destroy the GUI.
            # If this is the main thread, then it can call root.destroy after a delay.
            # For now, let's assume the main thread handles the destroy.
            pass # Let the main loop handle the actual destruction.

    def process_queue(self):
        try:
            while True: # Process all available updates
                update_data = self.update_queue.get_nowait()
                self.update_gui(update_data)
                self.update_queue.task_done() # Signal task completion
        except queue.Empty:
            pass # No more updates for now
        except Exception as e:
            log.error(f"Error processing update queue in GUI: {e}", exc_info=True)
            try: self.status_label.config(text=f"Status: GUI Error: {e}")
            except: pass # Avoid errors if GUI is closing

        # Schedule next check
        self.root.after(self.config.GUI_UPDATE_INTERVAL, self.process_queue)

    def get_health_tag(self, health_val: float, max_health_val: float) -> str:
        try:
            health = float(health_val)
            max_health = float(max_health_val)
            if max_health <= 0: return 'Unknown' # Avoid division by zero
            
            ratio = health / max_health
            if health <= 0: return 'Dead'
            if ratio < 0.3: return 'Critical'
            if ratio < 0.7: return 'Injured'
            return 'Healthy'
        except (ValueError, TypeError):
            return 'Unknown' # Fallback for non-numeric or unexpected health values

    def update_gui(self, data: Dict):
        try:
            tick = data.get('tick', self.tick_label.cget("text").split(" ")[1]) # Keep old if not present
            day = data.get('day', self.tick_label.cget("text").split(" ")[4])
            season = data.get('season', self.tick_label.cget("text").split(" ")[7])
            self.tick_label.config(text=f"Tick: {tick} | Day: {day} | Season: {season}")
            self.status_label.config(text=data.get('status', "Status: Running..."))

            active_agents_data = data.get('agents', [])
            structures_data = data.get('structures', [])
            new_events = data.get('new_events', [])

            # Update Agent Table
            living_agent_count = len([a for a in active_agents_data if a.get('health', 0) > 0])
            self.agents_label.config(text=f"Agents ({living_agent_count} / {self.config.MAX_POP})")
            
            selected_item_iid = self.pop_tree.focus() # Get IID of focused item
            current_tree_ids = set(self.pop_tree.get_children())
            agent_tree_map = {f"agent_{a['id']}": a for a in active_agents_data}
            agent_tree_ids_to_display = set(agent_tree_map.keys())

            ids_to_remove = current_tree_ids - agent_tree_ids_to_display
            if ids_to_remove: self.pop_tree.delete(*list(ids_to_remove))

            for item_id_str, agent_dict in agent_tree_map.items():
                inventory_str = agent_dict.get('inventory_summary', '?')
                health = agent_dict.get('health', 0.0)
                health_tag = self.get_health_tag(health, self.config.MAX_HEALTH)
                values = (
                    agent_dict['id'], agent_dict['name'], agent_dict['loc'],
                    f"{health:.1f}",
                    f"{agent_dict.get('hunger', 0.0):.1f}",
                    f"{agent_dict.get('morale', 0.0):.1f}",
                    agent_dict.get('role', '?'), inventory_str[:60]
                )
                if self.pop_tree.exists(item_id_str):
                    self.pop_tree.item(item_id_str, values=values, tags=(health_tag,))
                else:
                    # Simplified insert at end, sorting can be complex with continuous updates
                    self.pop_tree.insert("", tk.END, iid=item_id_str, values=values, tags=(health_tag,))
            
            if selected_item_iid and self.pop_tree.exists(selected_item_iid):
                 self.pop_tree.focus(selected_item_iid) # Re-focus if still exists
                 self.pop_tree.selection_set(selected_item_iid) # Also re-select


            # Update Event Log
            if new_events:
                self.event_text.config(state=tk.NORMAL)
                for event_str in reversed(new_events): # Add new events at the top
                     self.event_text.insert('1.0', event_str + "\n")
                     # Basic coloring (can be expanded)
                     event_lower = event_str.lower()
                     tag_to_apply = None
                     if "[death]" in event_lower or "fatal" in event_lower or "critical" in event_lower or "[error]" in event_lower:
                         tag_to_apply = "error"
                     elif "[event]" in event_lower or "[info]" in event_lower or "[birth]" in event_lower or "[spawn]" in event_lower or "[system]" in event_lower:
                          tag_to_apply = "info"
                     elif "warn" in event_lower or "fail" in event_lower: # Generic warning
                          tag_to_apply = "warning"
                     
                     if tag_to_apply:
                         self.event_text.tag_add(tag_to_apply, "1.0", "1.end") # Tag the newly inserted line

                # Configure tags if not already done (could be in _configure_styles)
                self.event_text.tag_config("error", foreground="red")
                self.event_text.tag_config("warning", foreground="orange")
                self.event_text.tag_config("info", foreground="blue")
                
                # Trim old events
                self.event_text.delete(f"{self.config.EVENT_LOG_DISPLAY_SIZE + 1}.0", tk.END)
                self.event_text.config(state=tk.DISABLED)

            # Update Structures List
            self.structures_label.config(text=f"World Structures ({len(structures_data)})")
            self.structure_text.config(state=tk.NORMAL)
            self.structure_text.delete('1.0', tk.END)
            sorted_structures = sorted(structures_data, key=lambda s_item: (s_item.get('loc', ''), s_item.get('id', 0)))
            for s_dict in sorted_structures:
                items_str = ", ".join(f"{k}:{v}" for k, v in s_dict.get('stored_items', {}).items())
                owner_id = s_dict.get('owner_id', 0)
                # Use registry to get owner name
                owner_name = self.registry.get_agent_name(owner_id) if owner_id != 0 else "Unowned"
                
                line = (f"- {s_dict.get('name','?')} [ID:{s_dict.get('id','?')}] @{s_dict.get('loc','?')}\n"
                        f"  (Owner:{owner_name}, Dur:{s_dict.get('durability', '?')})\n"
                        f"  └ Items: {items_str or 'empty'}\n")
                self.structure_text.insert(tk.END, line)
            self.structure_text.config(state=tk.DISABLED)

            # Update Map Canvas
            self._update_canvas(active_agents_data, structures_data)

        except tk.TclError as e:
            if "invalid command name" not in str(e).lower(): # Ignore errors during shutdown
                 log.warning(f"Tkinter TclError during GUI update: {e}")
        except Exception as e:
             log.error(f"Unhandled error during GUI update: {e}", exc_info=True)


    def _update_canvas(self, agents_data: List[Dict], structures_data: List[Dict]):
        self.map_canvas.delete("agent") # Clear old agent dots
        self.map_canvas.delete("structure_item") # Clear old structure markers (use a different tag)

        # Helper for positioning items within a location marker to avoid overlap
        location_occupancy: Dict[str, Dict[str, List[Tuple[int, int]]]] = {
            loc: {"agent": [], "structure": []} for loc in self.location_coords.keys()
        }

        def get_next_grid_pos(loc_name_key: str, item_type: str, item_size: int, loc_radius: int):
            base_x, base_y = self.location_coords[loc_name_key]
            occupied_slots = location_occupancy[loc_name_key][item_type]
            
            # Simple spiral placement logic (can be improved)
            num_items = len(occupied_slots)
            if num_items == 0:
                px, py = base_x, base_y
            else:
                angle_step = 360 / (num_items +1) # Distribute around center
                angle = num_items * angle_step * (3.14159 / 180) # Convert to radians
                # Spiral out slightly for more items
                current_radius = min(loc_radius * 0.6, 5 + num_items * (item_size / 2.5)) 
                
                offset_x = current_radius * np.cos(angle)
                offset_y = current_radius * np.sin(angle)
                px, py = base_x + offset_x, base_y + offset_y
            
            occupied_slots.append((int(px), int(py)))
            return int(px), int(py)

        # Draw structures first (usually fewer and larger)
        for s_dict in structures_data:
            loc = s_dict.get('loc')
            if loc in self.location_coords:
                draw_x, draw_y = get_next_grid_pos(loc, "structure", self.structure_box_size, self.location_radius - self.structure_box_size // 2)
                s_half = self.structure_box_size // 2
                self.map_canvas.create_rectangle(
                    draw_x - s_half, draw_y - s_half, draw_x + s_half, draw_y + s_half,
                    fill="#A0522D", outline="black", width=1, tags=("structure_item", loc, f"struct_{s_dict['id']}")
                )
        
        # Draw agents
        for agent_dict in agents_data:
            health = agent_dict.get('health', 0.0)
            if health <= 0: continue # Don't draw dead agents on map

            loc = agent_dict.get('loc')
            if loc in self.location_coords:
                health_tag = self.get_health_tag(health, self.config.MAX_HEALTH)
                color = "#E74C3C" if health_tag == 'Critical' else \
                        "#F1C40F" if health_tag == 'Injured' else \
                        "#2ECC71" # Healthy or Unknown (default to healthy color)

                draw_x, draw_y = get_next_grid_pos(loc, "agent", self.agent_dot_size, self.location_radius - self.agent_dot_size // 2)
                a_half = self.agent_dot_size // 2
                self.map_canvas.create_oval(
                    draw_x - a_half, draw_y - a_half, draw_x + a_half, draw_y + a_half,
                    fill=color, outline="black", width=1, tags=("agent", loc, f"agent_{agent_dict['id']}")
                )

if __name__ == '__main__':
    # Example of how to run the GUI standalone for testing
    # This requires a running event loop if async operations are involved in GUI actions,
    # or careful separation of GUI thread and simulation thread.

    # For simple visual testing:
    root = AppRoot() # Or tk.Tk()
    
    # Mock queues and config
    test_update_queue = queue.Queue()
    test_action_queue = queue.Queue()
    
    class MockRegistryForGUITest:
        def get_agent_name(self, agent_id): return f"AgentName[{agent_id}]"
    
    mock_registry = MockRegistryForGUITest()
    mock_config = MockConfigForGUI() # Use the GUI-specific mock

    # Ensure SAVE_DIR exists for the file dialog
    Path(mock_config.SAVE_DIR).mkdir(parents=True, exist_ok=True)

    gui = SimulationGUI(root, test_update_queue, test_action_queue, mock_config, mock_registry)

    # Example: Put some dummy data onto the queue to test updates
    def add_dummy_data():
        dummy_agent1 = {"id": 1, "name": "Jon Snow", "loc": "Winterfell", "health": 80.5, "hunger": 10.2, "morale": 75.0, "role": "Warrior", "inventory_summary": "Sword, Shield"}
        dummy_agent2 = {"id": 2, "name": "Daenerys", "loc": "Dragonstone", "health": 30.0, "hunger": 5.0, "morale": 90.5, "role": "Queen", "inventory_summary": "Crown"}
        dummy_agent3 = {"id": 3, "name": "Tyrion", "loc": "KingsLanding", "health": 0, "hunger": 0, "morale": 0, "role": "Hand", "inventory_summary": "Wine"}

        dummy_struct1 = {"id": 101, "name": "Barracks", "loc": "Winterfell", "owner_id": 1, "durability": 90, "stored_items": {"Wood": 5}}
        
        test_update_queue.put({
            "tick": 1, "day": 1, "season": "Spring",
            "status": "Simulating tick 1...",
            "agents": [dummy_agent1, dummy_agent2, dummy_agent3],
            "structures": [dummy_struct1],
            "new_events": ["[DEATH] An old raven died.", "[EVENT] A festival begins in KingsLanding!"]
        })
        root.after(2000, add_dummy_data_part2) # Schedule next update

    def add_dummy_data_part2():
        dummy_agent1_updated = {"id": 1, "name": "Jon Snow", "loc": "TheWall", "health": 70.0, "hunger": 15.0, "morale": 60.0, "role": "Commander", "inventory_summary": "Longclaw"}
        dummy_agent_new = {"id": 4, "name": "Arya Stark", "loc": "Braavos", "health": 95.0, "hunger": 2.0, "morale": 80.0, "role": "Assassin", "inventory_summary": "Needle"}
        test_update_queue.put({
            "tick": 2, "day": 1, "season": "Spring",
            "status": "Simulating tick 2...",
            "agents": [dummy_agent1_updated, dummy_agent_new], # Agent 2 and 3 are gone
            "structures": [], # Structure gone
            "new_events": ["[SYSTEM] Agent Daenerys vanished.", "[INFO] Arya arrives in Braavos."]
        })


    root.after(1000, add_dummy_data) # Add data after 1 second
    root.mainloop()
