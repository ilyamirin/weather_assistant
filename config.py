# config.py
# API key for weather service
OPENWEATHER_API_KEY = "67efc842d95cf8267a2763bb213308ef"

# Hugging Face Transformers model to use for the assistant
# You can change this to any local or cached chat/instruct model.
# Examples:
# - "meta-llama/Llama-3.1-8B-Instruct"
# - "Qwen/Qwen2.5-7B-Instruct"
# - "mistralai/Mistral-7B-Instruct-v0.3"
TRANSFORMERS_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# Whether the selected model supports tool calling reliably.
# If False, the app will not attempt to request tool invocations from the model
# and will answer as a plain chat assistant.
MODEL_SUPPORTS_TOOLS = True
