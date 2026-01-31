#!/usr/bin/env python3
"""
Test script for using Llama 3 to generate theme titles.
Tests various connection methods to local Llama 3 installation.
"""

import json
import subprocess
import sys
from pathlib import Path

# Test transcript sample (from video 002, theme 1)
SAMPLE_TRANSCRIPT = """
which is sometimes you ask yourself أمك دعاء but it seems like my duaاتوة الله
is not answering it do you get that feeling sometimes you say like my duaاتوة but
it seems like my duaاتو not being answered this is something that also happened to so
many people and it happens to people all the time they feel that way and I'm going to
share with you this morning إشاء الله تعالى the story of the great tabiri
"""

def test_ollama():
    """Test connection to Ollama (most common Llama 3 interface)."""
    print("Testing Ollama connection...")

    try:
        import requests

        # Check if Ollama is running
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✓ Ollama is running!")
            print(f"  Available models: {[m['name'] for m in models]}")

            # Find Llama 3 model
            llama_model = None
            for m in models:
                if 'llama' in m['name'].lower() and '3' in m['name']:
                    llama_model = m['name']
                    break

            if not llama_model and models:
                llama_model = models[0]['name']

            if llama_model:
                print(f"  Using model: {llama_model}")
                return test_ollama_generation(llama_model)
        else:
            print("✗ Ollama responded but not expected status")
    except requests.exceptions.ConnectionError:
        print("✗ Ollama not running on http://localhost:11434")
        print("  Start it with: ollama serve")
    except ImportError:
        print("✗ Requests library not installed")
        print("  Install it with: pip install requests")
    except Exception as e:
        print(f"✗ Ollama error: {e}")

    return False

def test_ollama_generation(model):
    """Test generating a title with Ollama."""
    try:
        import requests

        prompt = f"""You are a YouTube content analyst. Generate a SHORT, CATCHY title (2-6 words) for a video clip based on this transcript:

Transcript:
{SAMPLE_TRANSCRIPT[:500]}

Requirements:
- 2-6 words maximum
- Catchy and descriptive
- No filler words at the start
- Focus on the main topic

Return ONLY the title, nothing else."""

        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'num_predict': 50
                }
            },
            timeout=60
        )

        if response.status_code == 200:
            title = response.json().get('response', '').strip()
            # Clean up any quotes
            title = title.strip('"\'').strip()

            print(f"\n✓ Generated title: \"{title}\"")
            return True
        else:
            print(f"✗ Generation failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"✗ Generation error: {e}")
        return False

def test_llama_cpp():
    """Test connection to llama-cpp server."""
    print("\nTesting llama-cpp connection...")

    try:
        import requests
        response = requests.get('http://localhost:8080/health', timeout=2)
        if response.status_code == 200:
            print("✓ llama-cpp server is running!")
            return test_llama_cpp_generation()
        else:
            print("✗ llama-cpp responded but not healthy")
    except requests.exceptions.ConnectionError:
        print("✗ llama-cpp not running on http://localhost:8080")
    except Exception as e:
        print(f"✗ llama-cpp error: {e}")

    return False

def test_llama_cpp_generation():
    """Test generating a title with llama-cpp."""
    try:
        import requests

        prompt = f"""Generate a short, catchy title (2-6 words) for this clip:

{SAMPLE_TRANSCRIPT[:300]}

Title:"""

        response = requests.post(
            'http://localhost:8080/completion',
            json={
                'prompt': prompt,
                'n_predict': 30,
                'temperature': 0.7,
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            title = result.get('content', '').strip()
            print(f"\n✓ Generated title: \"{title}\"")
            return True
        else:
            print(f"✗ Generation failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"✗ Generation error: {e}")
        return False

def test_llamaf_cli():
    """Test using llama-cli directly."""
    print("\nTesting llama-cli...")

    # Check if llama-cli exists
    try:
        result = subprocess.run(['which', 'llama-cli'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ llama-cli found")
            # You would need to have a model file, this is just a check
            return True
        else:
            print("✗ llama-cli not found")
    except Exception as e:
        print(f"✗ llama-cli check error: {e}")

    return False

def main():
    """Run all connection tests."""
    print("=" * 60)
    print("Llama 3 Connection Test for Theme Generation")
    print("=" * 60)

    print("\nSample transcript for testing:")
    print(f"  {SAMPLE_TRANSCRIPT[:100]}...")

    success = False

    # Try Ollama first (most common)
    if test_ollama():
        success = True

    # Try llama-cpp server
    elif test_llama_cpp():
        success = True

    # Try llama-cli
    elif test_llamaf_cli():
        print("\nNote: llama-cli found but requires manual model path")
        success = True

    print("\n" + "=" * 60)
    if success:
        print("✓ SUCCESS: Llama 3 is accessible for theme generation!")
        print("\nNext steps:")
        print("  1. Create AI theme generation module")
        print("  2. Test on actual transcripts")
        print("  3. Compare AI vs pattern-based titles")
    else:
        print("✗ FAILED: Could not connect to Llama 3")
        print("\nTo get started:")
        print("  1. Install Ollama: https://ollama.ai/download")
        print("  2. Run: ollama run llama3")
        print("  3. Then run this test again")
    print("=" * 60)

    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
