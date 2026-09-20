import os
import re
import hashlib
import textwrap
import sqlite3
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b" 

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

from groq import Groq

groq_client = Groq(api_key=GROQ_API_KEY)

def generate(prompt: str, max_new_tokens: int = 256) -> str:
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=max_new_tokens,
    )
    return response.choices[0].message.content