import requests
import json
import sys


def chat(prompt, model="gpt-4"):
    """Send a chat request to the API."""
    url = "http://localhost:8000/chat"
    payload = {"prompt": prompt, "model": model}

    response = requests.post(url, json=payload)
    return response.json()


if __name__ == "__main__":
    # Get prompt from command line arguments or use default
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is LiteLLM?"

    print(f"Sending prompt: {prompt}")
    result = chat(prompt)

    print("\nResponse:")
    print(result.get("response", "Error: No response"))
    print(f"\nTrace ID: {result.get('trace_id', 'Not available')}")
    print(f"Model: {result.get('model', 'unknown')}")
