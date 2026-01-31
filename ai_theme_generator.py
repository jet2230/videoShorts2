#!/usr/bin/env python3
"""
AI-powered theme generation using local LLM models.
Supports Ollama (Llama 3) and llama-cpp.
"""

import json
import requests
from typing import Optional, List, Dict


class AIThemeGenerator:
    """Generate theme titles using local AI models."""

    def __init__(self, model: str = 'llama3', api_base: str = 'http://localhost:11434'):
        """
        Initialize AI theme generator.

        Args:
            model: Model name (default: llama3)
            api_base: API base URL (default: Ollama)
        """
        self.model = model
        self.api_base = api_base

    def is_available(self) -> bool:
        """Check if the AI service is available."""
        try:
            response = requests.get(f'{self.api_base}/api/tags', timeout=2)
            return response.status_code == 200
        except:
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            response = requests.get(f'{self.api_base}/api/tags', timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
        except:
            pass
        return []

    def generate_title(self, transcript: str, duration: str = "") -> str:
        """
        Generate a theme title using AI.

        Args:
            transcript: The transcript text
            duration: Duration string (for context)

        Returns:
            Generated title
        """
        # Truncate transcript if too long
        transcript_text = transcript[:800]

        prompt = f"""You are a YouTube Shorts content expert. Generate a SHORT, CATCHY title for a short video clip.

Transcript (max {duration}):
{transcript_text}

Requirements:
- 2-6 words maximum
- Catchy and click-worthy
- No filler words like "The", "A", "An" at the start
- Must capture the MAIN topic
- Make viewers want to watch

Return ONLY the title, nothing else."""

        try:
            response = requests.post(
                f'{self.api_base}/api/generate',
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.8,
                        'num_predict': 40,
                        'top_p': 0.9,
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                title = response.json().get('response', '').strip()
                # Clean up
                title = title.strip('"\'').strip()
                # Remove any common prefixes
                for prefix in ['Title:', 'The title is', 'The video is about']:
                    if title.lower().startswith(prefix.lower()):
                        title = title[len(prefix):].strip()

                # Capitalize first letter
                if title:
                    title = title[0].upper() + title[1:]

                return title if len(title) > 3 else "Islamic Teaching"
            else:
                return None

        except Exception as e:
            print(f"  AI generation error: {e}")
            return None

    def generate_reason(self, transcript: str, title: str) -> str:
        """
        Generate a reason why this theme works as a short.

        Args:
            transcript: The transcript text
            title: The theme title

        Returns:
            Generated reason
        """
        prompt = f"""Explain in 10 words or less why this clip makes a good YouTube Short:

Title: {title}
Transcript: {transcript[:300]}

Return the reason only, nothing else."""

        try:
            response = requests.post(
                f'{self.api_base}/api/generate',
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'num_predict': 30,
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                reason = response.json().get('response', '').strip()
                reason = reason.strip('"\'').strip()
                return reason if len(reason) > 5 else "Engaging content for viewers"
            else:
                return None

        except Exception:
            return None


def test_ai_generator():
    """Test the AI theme generator."""
    print("Testing AI Theme Generator...")

    ai = AIThemeGenerator()

    if not ai.is_available():
        print("✗ AI service not available")
        return False

    models = ai.get_available_models()
    print(f"✓ Available models: {models}")

    # Test transcript
    test_transcript = """
    which is sometimes you ask yourself but it seems like my dua is not answering
    it do you get that feeling sometimes you say like my dua but it seems like my dua
    is not being answered this is something that also happened to so many people
    """

    title = ai.generate_title(test_transcript, "0m 30s")
    print(f"✓ Generated title: \"{title}\"")

    reason = ai.generate_reason(test_transcript, title)
    print(f"✓ Generated reason: \"{reason}\"")

    return True


if __name__ == '__main__':
    test_ai_generator()
