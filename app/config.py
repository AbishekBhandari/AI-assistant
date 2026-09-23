import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
VLLM_MODEL = os.getenv(
    "VLLM_MODEL",
    "Qwen/Qwen2.5-0.5B-Instruct"
)

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set")