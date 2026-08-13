"""
Task 1.1 — Step 06: LLM-as-Judge prompt and API call function
Owner: Rania (P2) — Eman (P1) imports this in later steps

What this does:
  - Defines the exact prompt used to classify 239K HF posts
  - Provides the classify_with_judge() function
  - This file is IMPORTED by step07 and step08 — not run standalone

Based on the Arnaiz-Rodriguez classification methodology.
We use gpt-4o-mini for cost efficiency (~$10 for 239K posts).
"""

import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()   # reads OPENAI_API_KEY from .env

OUR_CATEGORIES = [
    "suicidal_ideation",
    "self_harm",
    "violence",
    "anxiety_crisis",
    "substance_abuse",
    "risk_taking",
    "no_crisis",
]

# ─────────────────────────────────────────────────────────────────────────────
# THE JUDGE PROMPT
# This is the most important thing to get right.
# Adjust the category descriptions if the pilot (step07) shows errors.
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """You are a mental health crisis classification assistant.
Classify the following text into EXACTLY ONE of these categories:

1. suicidal_ideation
   Explicit: "I want to die", "I want to end my life"
   Implicit: "Everyone would be better without me",
             "I stopped making plans", "I have been saying goodbye to people",
             "unalive", "not waking up", "giving things away"

2. self_harm
   Explicit: "I cut myself", "I burned myself"
   Implicit: "I need to feel something real", "relief lines",
             "13'ing", "I hurt myself to cope"

3. violence
   Explicit: "I want to hurt someone", "I will kill them"
   Implicit: "I cannot control my anger", intrusive violent thoughts,
             OCD-type unwanted thoughts about harming others

4. anxiety_crisis
   Acute panic attacks, severe anxiety episodes, dissociation crises
   NOT general worry — only acute, crisis-level episodes
   Explicit: "I am having a panic attack", "I cannot breathe"
   Implicit: "My heart is racing and I cannot calm down"

5. substance_abuse
   Problematic drug or alcohol use, dependence, intoxication crises
   Explicit: "I drank a whole bottle", "I am high and scared"
   Implicit: "I only feel okay after a few drinks"

6. risk_taking
   Reckless behaviour driven by not caring about consequences
   Explicit: "I have been driving drunk", "I stopped caring what happens to me"
   Implicit: "I do not care if I get hurt", dangerous behaviour with
             suicidal-adjacent motivation

7. no_crisis
   The text does NOT contain any of the above crisis signals.
   General mental health discussion, seeking advice, or everyday emotions.

RULES:
- Reply with ONLY the category name. Nothing else. No punctuation.
- Slang counts: "unalive", "13'ing", "blades", "relief lines" are crisis signals.
- Implicit signals count even without explicit keywords.
- If genuinely ambiguous, pick the MOST LIKELY category.
- If none of the crisis categories fit, reply: no_crisis
"""

VALID_LABELS = set(OUR_CATEGORIES)


def classify_with_judge(text: str, client: OpenAI, max_chars: int = 512) -> str:
    """
    Classify a single text with GPT-4o-mini.

    Args:
        text:       Input text to classify
        client:     OpenAI client instance
        max_chars:  Truncate text to this length (saves tokens)

    Returns:
        One of the 7 category strings, or "INVALID" if model gave unexpected output
    """
    truncated = str(text)[:max_chars].strip()
    if not truncated:
        return "no_crisis"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": truncated},
            ],
            temperature=0,    # deterministic
            max_tokens=15,    # label only
        )
        raw = response.choices[0].message.content.strip().lower()

        # Clean up any extra punctuation or spaces the model adds
        raw = re.sub(r"[^a-z_]", "", raw)

        if raw in VALID_LABELS:
            return raw
        else:
            # Try partial match as fallback
            for label in OUR_CATEGORIES:
                if label in raw:
                    return label
            return "INVALID"

    except Exception as e:
        return f"ERROR:{str(e)[:50]}"


def get_client() -> OpenAI:
    """Create and return an OpenAI client using the key from .env"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found.\n"
            "1. Copy .env.example to .env\n"
            "2. Add your key: OPENAI_API_KEY=sk-...\n"
        )
    return OpenAI(api_key=api_key)
