import inspect
from typing import Dict, TYPE_CHECKING, Any

# Forward declarations for type hinting
if TYPE_CHECKING:
    from ..config import Config # Main config
    from .agent import AgentState # For type checking targets
    from .world import World      # For type checking targets

# Placeholder for logger
class PrintLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg, exc_info=False): print(f"ERROR: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")

log = PrintLogger()

# Stores the original source code or representation of whitelisted modifiable targets.
# Key: target_name (e.g., "Config.HUNGER_PER_TICK", "AgentState.update_needs")
# Value: string representation of the original code/value.
ORIGINAL_SOURCE_CODE: Dict[str, str] = {}

def store_original_code(config_module: 'Config', agent_module: Any, world_module: Any):
    """
    Stores the source code or value representation of whitelisted targets from the Config.
    Args:
        config_module: The application's main Config class/object.
        agent_module: The module or class for AgentState.
        world_module: The module or class for World.
    """
    global ORIGINAL_SOURCE_CODE, log
    try:
        from ..utils.logger import log as main_log
        log = main_log
    except ImportError:
        pass

    ORIGINAL_SOURCE_CODE.clear()
    log.info(f"Storing original code for whitelisted targets: {config_module.CODE_MODIFICATION_WHITELIST}")

    for target_name in config_module.CODE_MODIFICATION_WHITELIST:
        try:
            target_object: Any = None
            original_repr: str = ""

            if '.' not in target_name:
                log.warning(f"Unsupported whitelisted target format (must be Class.Attribute or Class.Method): {target_name}")
                continue

            class_name, attr_name = target_name.split('.', 1)
            target_class_definition: Any = None

            if class_name == "Config":
                target_class_definition = config_module
            elif class_name == "AgentState":
                target_class_definition = agent_module # This should be the AgentState class itself
            elif class_name == "World":
                target_class_definition = world_module # This should be the World class itself
            else:
                log.warning(f"Unknown class '{class_name}' in whitelist target '{target_name}'. Skipping.")
                continue

            if not hasattr(target_class_definition, attr_name):
                log.warning(f"Attribute/method '{attr_name}' not found in class '{class_name}' for target '{target_name}'. Skipping.")
                continue

            target_object = getattr(target_class_definition, attr_name)

            if callable(target_object):
                # Handle lambdas and regular functions/methods
                if isinstance(target_object, type(lambda: 0)) and target_object.__name__ == "<lambda>":
                    try:
                        original_repr = inspect.getsource(target_object).strip()
                    except (TypeError, OSError):
                        original_repr = repr(target_object) # Fallback for lambdas that can't get source
                        log.debug(f"Using repr for lambda target {target_name}: {original_repr}")
                else:
                    original_repr = inspect.getsource(target_object)
            else: # It's an attribute (e.g., Config.HUNGER_PER_TICK)
                original_repr = repr(target_object)

            ORIGINAL_SOURCE_CODE[target_name] = original_repr
            # log.debug(f"Stored original source/value for {target_name}: {original_repr[:100].replace(chr(10), ' ')}...")

        except Exception as e:
            log.error(f"Error retrieving source/value for whitelisted target '{target_name}': {e}", exc_info=False)
    
    log.info(f"Stored original source/value for {len(ORIGINAL_SOURCE_CODE)} whitelisted targets.")


def restore_specific_original_code(target_name: str, config_module: 'Config', agent_module: Any, world_module: Any) -> bool:
    """
    Restores a single whitelisted code target to its original state.
    Args:
        target_name: The name of the target to restore (e.g., "Config.HUNGER_PER_TICK").
        config_module: The application's main Config class/object.
        agent_module: The module or class for AgentState.
        world_module: The module or class for World.
    Returns:
        True if restoration was successful or target not found in originals, False on error.
    """
    global log
    if target_name not in ORIGINAL_SOURCE_CODE:
        log.error(f"Cannot restore '{target_name}': Original code/value not found in stored originals.")
        return False # Indicate original was not found, so no restoration attempted from here

    original_code_repr = ORIGINAL_SOURCE_CODE[target_name]
    log.warning(f"Attempting to restore original code/value for target: {target_name}")

    try:
        class_name, attr_name = target_name.split('.', 1)
        target_class_definition: Any = None

        if class_name == "Config":
            target_class_definition = config_module
        elif class_name == "AgentState":
            target_class_definition = agent_module
        elif class_name == "World":
            target_class_definition = world_module
        else:
            log.error(f"Unknown class '{class_name}' for restoration target '{target_name}'.")
            return False

        current_value = getattr(target_class_definition, attr_name, None)

        if callable(current_value) or (isinstance(current_value, type(lambda:0)) and current_value.__name__ == "<lambda>"):
            # For methods or lambdas, we need to exec the original source code
            # This requires a careful execution context.
            local_exec_namespace = {}
            # Provide access to necessary modules/classes VERY carefully
            # This list should be expanded based on what the original methods might need.
            # For now, it's minimal.
            safe_globals = {
                '__builtins__': { 'min': min, 'max': max, 'random': random, 'int': int, 'float': float, 'str': str, 'list': list, 'dict': dict, 'getattr': getattr, 'isinstance': isinstance, 'callable': callable, 'type': type, 'any':any, 'True':True, 'False':False, 'None':None, 'repr':repr, 'super':super, 'property':property, 'classmethod':classmethod, 'staticmethod':staticmethod},
                'Config': config_module, # Make Config accessible
                'AgentState': agent_module, # Make AgentState accessible
                'World': world_module, # Make World accessible
                # Add other necessary imports like numpy (np), Counter, log, etc. if methods use them.
                # Example:
                # 'np': __import__('numpy'),
                # 'Counter': __import__('collections').Counter,
                # 'log': log, # The logger instance
            }
            # Add the class itself to the globals if needed for super() etc.
            safe_globals[class_name] = target_class_definition

            exec(original_code_repr, safe_globals, local_exec_namespace)
            
            restored_method = local_exec_namespace.get(attr_name)
            if not restored_method or not callable(restored_method):
                # Check if it was a lambda defined directly on the class
                if original_code_repr.startswith("lambda"): # Very basic check
                    try:
                        restored_method = eval(original_code_repr, safe_globals, local_exec_namespace)
                    except: pass # eval might fail for complex lambdas not meant for direct eval

                if not restored_method or not callable(restored_method):
                    log.error(f"Failed to restore callable '{attr_name}' for {class_name}. Original code did not define it as expected in exec namespace. Defined: {list(local_exec_namespace.keys())}")
                    return False
            
            setattr(target_class_definition, attr_name, restored_method)
            log.info(f"Successfully restored method/callable: {target_name}")

        else: # For simple attributes (like Config values)
            # Evaluate the original representation to get the value
            # This eval is for simple values like numbers, strings, lists, dicts, or simple lambdas.
            allowed_globals_for_eval = {'__builtins__': {'True': True, 'False': False, 'None': None, 'int': int, 'float': float, 'str': str, 'list': list, 'dict': dict, 'Counter': __import__('collections').Counter, 'lambda': lambda:None, 'max':max, 'min':min}} # Added Counter, max, min
            # If the original repr was for a lambda assigned to a Config attribute:
            if original_code_repr.strip().startswith("lambda"):
                 # Need to ensure the lambda has access to Config if it refers to e.g. Config.MAX_AGE
                 # This is tricky. The lambda stored in ORIGINAL_SOURCE_CODE is a string.
                 # When eval'd, it needs its original context or have Config passed.
                 # For Config.AGE_DEATH_CHANCE_PER_TICK = lambda age: max(0, (age - Config.MAX_AGE * 0.9) / (Config.MAX_AGE * 0.1) * 0.0005)
                 # The `Config` in the lambda string refers to the Config class.
                 # We need to provide `Config` in the globals for eval.
                 lambda_globals = allowed_globals_for_eval.copy()
                 lambda_globals['Config'] = config_module # Provide the actual Config class
                 original_value = eval(original_code_repr, lambda_globals, {})

            else:
                 original_value = eval(original_code_repr, allowed_globals_for_eval, {})

            setattr(target_class_definition, attr_name, original_value)
            log.info(f"Successfully restored attribute: {target_name} to value: {str(original_value)[:100]}")

        return True

    except Exception as e:
        log.error(f"CRITICAL error during restoration of '{target_name}': {e}", exc_info=True)
        return False


def restore_all_original_code(config_module: 'Config', agent_module: Any, world_module: Any):
    """
    Restores all whitelisted code targets to their original states.
    Args:
        config_module: The application's main Config class/object.
        agent_module: The module or class for AgentState.
        world_module: The module or class for World.
    """
    global log
    log.warning("Restoring original code/values for all whitelisted targets...")
    applied_count = 0
    failed_count = 0

    if not ORIGINAL_SOURCE_CODE:
        log.info("ORIGINAL_SOURCE_CODE is empty. Attempting to store originals first.")
        store_original_code(config_module, agent_module, world_module)
        if not ORIGINAL_SOURCE_CODE:
            log.error("Failed to store original codes. Cannot proceed with restoration.")
            return

    # Iterate through the whitelist from Config to ensure all defined targets are attempted
    for target_name in config_module.CODE_MODIFICATION_WHITELIST:
        if target_name in ORIGINAL_SOURCE_CODE:
            if restore_specific_original_code(target_name, config_module, agent_module, world_module):
                applied_count += 1
            else:
                failed_count += 1
        else:
            log.error(f"Cannot restore '{target_name}': It was in whitelist but not found in stored ORIGINAL_SOURCE_CODE.")
            failed_count += 1
            
    log.info(f"Finished restoring code: {applied_count} successful, {failed_count} failed.")


if __name__ == '__main__':
    # Example Usage (conceptual)
    # This requires mock or actual Config, AgentState, World classes.

    # Setup logger for test
    try:
        from ..utils.logger import log as main_log_test
        log = main_log_test
    except ImportError:
        print("INFO: Using PrintLogger for code_utils test.")

    log.info("Testing Code Utils...")

    # --- Mock Config, AgentState, World for testing ---
    class MockConfigForCodeUtils:
        HUNGER_PER_TICK = 0.7
        MAX_AGE = 320
        # Example lambda that refers to Config.MAX_AGE
        AGE_DEATH_CHANCE_PER_TICK = lambda age: max(0, (age - MockConfigForCodeUtils.MAX_AGE * 0.9) / (MockConfigForCodeUtils.MAX_AGE * 0.1) * 0.0005)
        
        CODE_MODIFICATION_WHITELIST = {
            "MockConfigForCodeUtils.HUNGER_PER_TICK",
            "MockConfigForCodeUtils.AGE_DEATH_CHANCE_PER_TICK",
            "MockAgentForCodeUtils.test_method",
        }

    class MockAgentForCodeUtils:
        def test_method(self, value: int):
            log.info(f"Original MockAgent.test_method called with {value}")
            return value * 2

    class MockWorldForCodeUtils:
        pass # No modifiable parts in this simple mock

    mock_config_instance = MockConfigForCodeUtils()
    mock_agent_class = MockAgentForCodeUtils
    mock_world_class = MockWorldForCodeUtils

    # 1. Store originals
    store_original_code(mock_config_instance, mock_agent_class, mock_world_class)
    log.info(f"Stored originals: {ORIGINAL_SOURCE_CODE}")

    # 2. Simulate modification
    log.info(f"Original HUNGER_PER_TICK: {mock_config_instance.HUNGER_PER_TICK}")
    mock_config_instance.HUNGER_PER_TICK = 99.9
    log.info(f"Modified HUNGER_PER_TICK: {mock_config_instance.HUNGER_PER_TICK}")
    
    original_age_death_func = mock_config_instance.AGE_DEATH_CHANCE_PER_TICK
    log.info(f"Original AGE_DEATH_CHANCE_PER_TICK(300): {original_age_death_func(300)}")
    mock_config_instance.AGE_DEATH_CHANCE_PER_TICK = lambda age: 0.5 # New lambda
    log.info(f"Modified AGE_DEATH_CHANCE_PER_TICK(300): {mock_config_instance.AGE_DEATH_CHANCE_PER_TICK(300)}")


    agent_instance = mock_agent_class()
    log.info(f"Original agent_instance.test_method(5): {agent_instance.test_method(5)}")
    # Simulate modifying the method on the class
    def new_test_method(self, value: int):
        log.info(f"NEW MockAgent.test_method called with {value}")
        return value * 10
    mock_agent_class.test_method = new_test_method
    log.info(f"Modified agent_instance.test_method(5): {agent_instance.test_method(5)}")


    # 3. Restore all
    restore_all_original_code(mock_config_instance, mock_agent_class, mock_world_class)
    log.info(f"Restored HUNGER_PER_TICK: {mock_config_instance.HUNGER_PER_TICK}")
    assert mock_config_instance.HUNGER_PER_TICK == 0.7 
    
    # Check restored lambda
    log.info(f"Restored AGE_DEATH_CHANCE_PER_TICK(300): {mock_config_instance.AGE_DEATH_CHANCE_PER_TICK(300)}")
    # Compare output with original lambda's output for a test value
    expected_original_lambda_output = max(0, (300 - MockConfigForCodeUtils.MAX_AGE * 0.9) / (MockConfigForCodeUtils.MAX_AGE * 0.1) * 0.0005)
    assert abs(mock_config_instance.AGE_DEATH_CHANCE_PER_TICK(300) - expected_original_lambda_output) < 1e-9


    log.info(f"Restored agent_instance.test_method(5): {agent_instance.test_method(5)}") # Still uses the instance if method was changed on class
    # To test class method restoration, create a new instance or check the class's method
    restored_agent_instance = mock_agent_class()
    log.info(f"New instance agent.test_method(5) after class method restoration: {restored_agent_instance.test_method(5)}")
    assert restored_agent_instance.test_method(5) == 10 # Expected: 5 * 2 = 10

    log.info("Code utils test completed.")
