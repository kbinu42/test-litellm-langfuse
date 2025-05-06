import requests
import sys
import os
import json


def check_expiry(image_path, vision_model="gpt-4o-vision-poc", reasoning_model="gpt-4"):
    """Check product expiry date from an image."""
    url = "http://localhost:8000/check-expiry"

    # Verify image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Prepare the multipart form data
    files = {
        "file": (os.path.basename(image_path), open(image_path, "rb"), "image/jpeg")
    }

    data = {"vision_model": vision_model, "reasoning_model": reasoning_model}

    # Send the request
    print(f"Uploading image {image_path} for expiry check...")
    response = requests.post(url, files=files, data=data)

    # Close the file
    files["file"][1].close()

    # Process response
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python expiry_client.py <image_path> [vision_model] [reasoning_model]"
        )
        sys.exit(1)

    image_path = sys.argv[1]

    # Optional model parameters with defaults
    vision_model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4o-vision-poc"
    reasoning_model = sys.argv[3] if len(sys.argv) > 3 else "gpt-4"

    print(f"Using vision model: {vision_model}")
    print(f"Using reasoning model: {reasoning_model}")

    result = check_expiry(image_path, vision_model, reasoning_model)

    if result:
        print("\n=== Results ===")
        print(f"Extracted Date: {result.get('extracted_date', 'Not found')}")
        print(f"\nAnalysis: {result.get('analysis', 'No analysis available')}")
        print(f"\nTrace ID: {result.get('trace_id', 'Not available')}")

        # Display model information
        if "models_used" in result:
            models = result["models_used"]
            print(f"\n=== Models Used ===")
            print(f"Vision: {models.get('vision')}")
            print(f"Reasoning: {models.get('reasoning')}")

        print("\nYou can view the trace details in your Langfuse dashboard.")
