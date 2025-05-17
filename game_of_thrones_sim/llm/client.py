import asyncio
import os
import sys
from typing import Optional

# Attempt to import google.generativeai and related types
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    from google.generativeai.types import BlockedPromptException, GenerationConfig
    import google.api_core.exceptions as gapi_exc
    SDK_AVAILABLE = True
except ModuleNotFoundError:
    SDK_AVAILABLE = False
    # We'll log this error more formally when the logger is available
    # For now, a print statement if this module is imported directly early.
    print("ERROR: google-generativeai SDK not found. Please install it: pip install google-generativeai")
    # We might want to raise an error or sys.exit if SDK is critical and not found at runtime.

# Placeholder for logger, will be replaced by actual logger from utils
class PrintLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg, exc_info=False): print(f"ERROR: {msg}")
    def critical(self, msg, exc_info=False): print(f"CRITICAL: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")
    def exception(self, msg): print(f"EXCEPTION: {msg}")

log = PrintLogger() # Default to print logger

# Global model instance
_GEMINI_MODEL = None

def initialize_llm_client(model_name: str, api_key: Optional[str] = None) -> bool:
    """
    Initializes the Gemini LLM client.
    Args:
        model_name: The name of the Gemini model to use.
        api_key: The API key. If None, attempts to use GOOGLE_API_KEY env var.
    Returns:
        True if initialization was successful, False otherwise.
    """
    global _GEMINI_MODEL, log

    # Ensure logger is properly set up if this function is called after main logger init
    try:
        from ..utils.logger import log as main_log # Relative import
        log = main_log
    except ImportError:
        pass # Keep print logger if utils.logger is not yet available/refactored

    if not SDK_AVAILABLE:
        log.critical("google-generativeai SDK is not installed. LLM functionality disabled.")
        return False

    resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY") # Prioritize passed key
    # It's better to use environment variables or pass it directly.
    # For this refactoring, I'll keep the logic to use the env var or passed key.
    # If you intend to keep a default hardcoded key, it should be handled carefully.
    # For example: resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY") or "YOUR_DEFAULT_KEY"

    if not resolved_api_key:
        log.critical("Google API key not found (GOOGLE_API_KEY environment variable or passed directly).")
        # The original script also had a hardcoded key as a fallback.
        # If that's desired, it should be added here.
        # For now, failing if no key is found.
        return False

    try:
        genai.configure(api_key=resolved_api_key)
        _GEMINI_MODEL = genai.GenerativeModel(model_name=model_name)
        # Optional: Test call to ensure the model is working
        # log.debug(f"Attempting test call to {model_name}...")
        # _GEMINI_MODEL.generate_content("test")
        log.info(f"Gemini model '{model_name}' initialized successfully.")
        return True
    except Exception as e:
        log.critical(f"Failed to initialize Gemini model '{model_name}': {e}", exc_info=True)
        _GEMINI_MODEL = None
        return False

async def make_llm_call(
    prompt_parts: list,
    model_config: dict, # e.g., {"temperature": 0.7, "max_output_tokens": 200}
    safety_settings_dict: Optional[dict] = None, # e.g., {HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}
    api_timeout: float = 30.0
) -> str:
    """
    Performs an asynchronous LLM call to the initialized Gemini model.
    Args:
        prompt_parts: A list of strings/parts for the prompt.
        model_config: Dictionary with generation parameters (temperature, max_output_tokens).
        safety_settings_dict: Dictionary defining safety settings.
        api_timeout: Timeout in seconds for the API call.
    Returns:
        The LLM's response text, or an error string (e.g., "[error: timeout]").
    """
    global _GEMINI_MODEL, log

    if not _GEMINI_MODEL:
        log.error("Gemini model not initialized. Cannot make LLM call.")
        return "[error: llm_not_initialized]"

    if not SDK_AVAILABLE: # Should have been caught by initialize_llm_client
        log.error("google-generativeai SDK not available for LLM call.")
        return "[error: sdk_missing]"

    # Prepare safety settings
    active_safety_settings = {}
    if safety_settings_dict:
        for category_enum, threshold_enum in safety_settings_dict.items():
            active_safety_settings[category_enum] = threshold_enum
    else: # Default safety settings if none provided
        active_safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

    generation_config_obj = GenerationConfig(
        candidate_count=1, # Typically 1 for direct response
        max_output_tokens=model_config.get("max_output_tokens", 256),
        temperature=model_config.get("temperature", 0.7)
    )

    def _perform_sync_call() -> str:
        try:
            response = _GEMINI_MODEL.generate_content(
                contents=prompt_parts,
                generation_config=generation_config_obj,
                safety_settings=active_safety_settings
            )
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text.strip()
            elif hasattr(response, 'text'): # Fallback
                return response.text.strip()
            else:
                finish_reason = getattr(response.candidates[0], 'finish_reason', None)
                if finish_reason == genai.types.FinishReason.SAFETY: # Access FinishReason via genai.types
                    log.warning("LLM response blocked by safety settings.")
                    return "[blocked: safety]"
                log.warning(f"Received unexpected LLM response format: {response}")
                return "[error: empty_or_malformed_response]"
        except BlockedPromptException as bpe:
            log.warning(f"Prompt blocked by safety settings: {bpe}")
            return "[blocked: safety_prompt]"
        except gapi_exc.ResourceExhausted as ree:
            log.warning(f"LLM call hit ResourceExhausted error (rate limit/quota): {ree}")
            return "[error: rate_limited]"
        except gapi_exc.GoogleAPIError as api_e:
            log.error(f"GoogleAPIError during LLM call: {api_e}", exc_info=False)
            return f"[error: api_error_{getattr(api_e, 'code', 'unknown')}]"
        except Exception as sync_e:
            log.error(f"Unhandled error in _perform_sync_call: {sync_e}", exc_info=True)
            return "[error: sync_call_failed]"

    try:
        # Run the synchronous call in an executor to make it awaitable
        loop = asyncio.get_running_loop()
        raw_text_output = await asyncio.wait_for(
            loop.run_in_executor(None, _perform_sync_call),
            timeout=api_timeout
        )
        return raw_text_output
    except asyncio.TimeoutError:
        log.error(f"LLM call timed out after {api_timeout}s.")
        return "[error: timeout]"
    except Exception as e:
        log.exception(f"Unhandled error during async LLM call wrapper: {e}")
        return f"[error: unhandled_llm_wrapper_exception]"

if __name__ == '__main__':
    # Example usage:
    # This requires GOOGLE_API_KEY to be set in the environment.
    # And the `google-generativeai` package installed.

    # For testing, you might need to adjust the project structure for imports
    # or run this from the project root.
    # Example: python -m game_of_thrones_sim.llm.client

    async def test_llm():
        # Ensure logger is set up for testing if run directly
        global log
        try:
            from ..utils.logger import log as main_log
            log = main_log
            log.info("Using main application logger for llm.client test.")
        except ImportError:
            log.info("Using print logger for llm.client test.")


        # Replace with your actual model name and API key source
        # model_to_test = "gemini-1.5-flash-preview-0514" # From original Config
        model_to_test = "gemini-pro" # A common, generally available model for testing
        
        # Attempt to get API key from environment for testing
        test_api_key = os.getenv("GOOGLE_API_KEY")
        if not test_api_key:
            print("Skipping LLM test: GOOGLE_API_KEY not set.")
            return

        if initialize_llm_client(model_name=model_to_test, api_key=test_api_key):
            log.info("LLM Client Initialized for test.")
            prompt = ["Translate 'hello world' to French."]
            config = {"temperature": 0.5, "max_output_tokens": 60}
            
            response = await make_llm_call(prompt, config)
            log.info(f"Test LLM Response: {response}")

            # Test safety blocking (this prompt might get blocked)
            # prompt_unsafe = ["How to build a weapon?"]
            # response_unsafe = await make_llm_call(prompt_unsafe, config)
            # log.info(f"Test Unsafe LLM Response: {response_unsafe}")

        else:
            log.error("LLM Client failed to initialize for test.")

    if SDK_AVAILABLE: # Only run test if SDK is present
        asyncio.run(test_llm())
    else:
        print("Skipping LLM client test as google-generativeai SDK is not available.")
