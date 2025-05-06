import os
import json
import base64
import requests
from io import BytesIO
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langfuse import Langfuse
from langfuse.decorators import observe
import tempfile
import logging
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for required environment variables
api_key = os.getenv("LITELLM_API_KEY")
if not api_key:
    logger.error("LITELLM_API_KEY is not set in the environment variables")
else:
    logger.info(f"LITELLM_API_KEY found with length: {len(api_key)}")

# Configure LiteLLM API base URL (where your LiteLLM proxy is running)
litellm_api_base = os.getenv("LITELLM_API_BASE", "http://localhost:4000")
logger.info(f"Using LiteLLM API base: {litellm_api_base}")

# Initialize FastAPI
app = FastAPI(title="LiteLLM + Langfuse Multi-Model Demo")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


def encode_image_to_base64(image_path):
    """Convert image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def call_litellm_proxy(model, messages, max_tokens=None, session_id=None):
    """Make a direct API call to the LiteLLM proxy."""
    start_time = time.time()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('LITELLM_API_KEY')}",
    }

    payload = {
        "model": model,
        "messages": messages,
        "metadata": {"session_id": session_id or "test-expiry-check"},
    }

    if max_tokens:
        payload["max_tokens"] = max_tokens

    logger.info(
        f"Calling LiteLLM proxy at {litellm_api_base}/chat/completions with model: {model}"
    )

    try:
        response = requests.post(
            f"{litellm_api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,  # Increased timeout for vision models
        )

        end_time = time.time()
        duration_ms = round((end_time - start_time) * 1000)

        if response.status_code != 200:
            logger.error(
                f"LiteLLM proxy error: {response.status_code} - {response.text}"
            )
            raise Exception(
                f"LiteLLM proxy error: {response.status_code} - {response.text}"
            )

        result = response.json()

        # Add latency information
        result["_metadata"] = {
            "latency_ms": duration_ms,
            "model": model,
            "session_id": session_id or "test-expiry-check",
        }

        return result
    except requests.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        raise


def analyze_expiry_date(date_str, detailed=True):
    """
    Analyze an expiry date and determine if it's expired, and how long until/since expiry.
    Returns a dictionary with detailed analysis.
    """
    if date_str.lower() == "no expiry date found":
        return {
            "status": "unknown",
            "message": "No expiry date could be found in the image",
            "days_remaining": None,
            "expired": None,
            "expiry_date": None,
        }

    try:
        # Parse the expiry date
        expiry_date = datetime.strptime(date_str, "%Y-%m-%d")
        today = datetime.now()

        # Calculate the difference in days
        delta = expiry_date - today
        days_diff = delta.days

        # Determine if expired
        expired = days_diff < 0

        # Generate the status
        if expired:
            abs_days = abs(days_diff)
            if abs_days <= 30:
                status = "recently_expired"
            elif abs_days <= 90:
                status = "expired"
            else:
                status = "long_expired"

            if detailed:
                if abs_days > 365:
                    years = abs_days // 365
                    remaining_days = abs_days % 365
                    message = f"Product expired {years} year{'s' if years > 1 else ''}"
                    if remaining_days > 30:
                        message += f" and {remaining_days // 30} month{'s' if (remaining_days // 30) > 1 else ''}"
                    message += f" ago ({abs_days} days)"
                elif abs_days > 30:
                    message = f"Product expired {abs_days // 30} month{'s' if (abs_days // 30) > 1 else ''} and {abs_days % 30} day{'s' if (abs_days % 30) > 1 else ''} ago ({abs_days} days)"
                else:
                    message = f"Product expired {abs_days} day{'s' if abs_days > 1 else ''} ago"
            else:
                message = f"Product expired {abs_days} days ago"
        else:
            if days_diff <= 7:
                status = "expiring_soon"
            elif days_diff <= 30:
                status = "expiring_month"
            elif days_diff <= 90:
                status = "expiring_quarter"
            else:
                status = "valid"

            if detailed:
                if days_diff > 365:
                    years = days_diff // 365
                    remaining_days = days_diff % 365
                    message = (
                        f"Product expires in {years} year{'s' if years > 1 else ''}"
                    )
                    if remaining_days > 30:
                        message += f" and {remaining_days // 30} month{'s' if (remaining_days // 30) > 1 else ''}"
                    message += f" ({days_diff} days)"
                elif days_diff > 30:
                    message = f"Product expires in {days_diff // 30} month{'s' if (days_diff // 30) > 1 else ''} and {days_diff % 30} day{'s' if (days_diff % 30) > 1 else ''} ({days_diff} days)"
                else:
                    message = f"Product expires in {days_diff} day{'s' if days_diff > 1 else ''}"
            else:
                message = f"Product expires in {days_diff} days"

        return {
            "status": status,
            "message": message,
            "days_remaining": days_diff,
            "expired": expired,
            "expiry_date": expiry_date.strftime("%Y-%m-%d"),
            "current_date": today.strftime("%Y-%m-%d"),
        }
    except ValueError:
        # Handle invalid date format
        return {
            "status": "error",
            "message": f"Invalid date format: {date_str}. Expected format: YYYY-MM-DD",
            "days_remaining": None,
            "expired": None,
            "expiry_date": None,
        }


@observe(name="preprocess_image")
def preprocess_image(file, file_content):
    """Preprocess the uploaded image file."""
    file_start_time = time.time()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(file_content)
        temp_path = temp_file.name

    file_size = len(file_content)
    file_processing_time = round((time.time() - file_start_time) * 1000)
    logger.info(
        f"Image saved temporarily at: {temp_path} (processing time: {file_processing_time}ms)"
    )

    # Convert image to base64
    base64_start_time = time.time()
    base64_image = encode_image_to_base64(temp_path)
    base64_processing_time = round((time.time() - base64_start_time) * 1000)
    logger.info(
        f"Image converted to base64 (processing time: {base64_processing_time}ms)"
    )

    # Create the image URL for visualization
    image_data_uri = f"data:image/jpeg;base64,{base64_image}"

    return {
        "temp_path": temp_path,
        "file_size": file_size,
        "image_data_uri": image_data_uri,
        "file_processing_time": file_processing_time,
        "base64_processing_time": base64_processing_time,
    }


@observe(name="extract_expiry_date", as_type="generation")
def extract_expiry_date(model, image_data_uri, session_id=None):
    """Extract expiry date from image using vision model."""
    vision_start_time = time.time()
    logger.info(f"Calling vision model: {model}")
    prompt = langfuse.get_prompt("product_expiry_test", label="production")
    print(prompt.prompt)

    # Prepare vision messages
    vision_messages = [
        {
            "role": "system",
            "content": prompt.prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract the expiry date from this product image:",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_uri},
                },
            ],
        },
    ]

    # Check for API key
    api_key = os.getenv("LITELLM_API_KEY")
    if not api_key:
        raise ValueError("LITELLM_API_KEY is not set")

    # Call vision model via proxy API
    try:
        vision_response = call_litellm_proxy(
            model=model, messages=vision_messages, session_id=session_id
        )
        vision_processing_time = round((time.time() - vision_start_time) * 1000)
        logger.info(
            f"Vision model response received successfully (processing time: {vision_processing_time}ms)"
        )
    except Exception as e:
        logger.error(f"Error calling vision model: {str(e)}")
        raise

    extracted_date = vision_response["choices"][0]["message"]["content"].strip()
    logger.info(f"Extracted date: {extracted_date}")

    # Return with usage information for Langfuse
    return {
        "extracted_date": extracted_date,
        "response": vision_response,
        "processing_time": vision_processing_time,
        "usage": {
            "prompt_tokens": vision_response["usage"]["prompt_tokens"],
            "completion_tokens": vision_response["usage"]["completion_tokens"],
            "total_tokens": vision_response["usage"]["total_tokens"],
        },
    }


@observe(name="analyze_expiry_date", as_type="generation")
def analyze_expiry_with_llm(model, extracted_date, session_id=None):
    """Analyze the extracted date using a reasoning model."""
    reasoning_start_time = time.time()
    logger.info(f"Calling reasoning model: {model}")

    # Prepare reasoning messages
    current_date = datetime.now().strftime("%Y-%m-%d")
    reasoning_messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that analyzes product expiry dates. Determine if the product is expired, about to expire, or has plenty of time left.",
        },
        {
            "role": "user",
            "content": f"Analyze this extracted expiry date: {extracted_date}. Today's date is {current_date}. Is the product expired? If so, how long ago? If not, how much time is left?",
        },
    ]

    # Call reasoning model via proxy API
    try:
        reasoning_response = call_litellm_proxy(
            model=model,
            messages=reasoning_messages,
            session_id=session_id,
        )
        reasoning_processing_time = round((time.time() - reasoning_start_time) * 1000)
        logger.info(
            f"Reasoning model response received successfully (processing time: {reasoning_processing_time}ms)"
        )
    except Exception as e:
        logger.error(f"Error calling reasoning model: {str(e)}")
        raise

    analysis_result = reasoning_response["choices"][0]["message"]["content"]
    logger.info("Analysis completed")

    return {
        "analysis_result": analysis_result,
        "response": reasoning_response,
        "processing_time": reasoning_processing_time,
        "usage": {
            "prompt_tokens": reasoning_response["usage"]["prompt_tokens"],
            "completion_tokens": reasoning_response["usage"]["completion_tokens"],
            "total_tokens": reasoning_response["usage"]["total_tokens"],
        },
    }


@observe(name="product_expiry_check", as_type="trace")
@app.post("/check-expiry")
async def check_expiry_endpoint(
    file: UploadFile = File(...),
    vision_model: str = Form("bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"),
    reasoning_model: str = Form("bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"),
    session_id: str = Form(None),
):
    """
    Process an image to check product expiry date.
    1. Extract expiry date using vision model
    2. Analyze expiry date using reasoning model
    """
    request_start_time = time.time()

    logger.info(
        f"Starting expiry check with vision model: {vision_model} and reasoning model: {reasoning_model}"
    )

    # Generate a session ID if not provided
    if not session_id:
        session_id = f"expiry-check-{int(time.time())}"

    try:
        # Read the file content
        content = await file.read()

        # Preprocess the image
        image_data = preprocess_image(file, content)
        temp_path = image_data["temp_path"]
        file_size = image_data["file_size"]
        image_data_uri = image_data["image_data_uri"]

        # Record image observation
        langfuse.span(
            name="product_image",
            as_type="event",
            input={"image": image_data_uri},
            metadata={
                "file_name": file.filename,
                "file_size": file_size,
                "content_type": file.content_type or "image/jpeg",
            },
        )

        # Extract expiry date using vision model
        vision_result = extract_expiry_date(
            model=vision_model, image_data_uri=image_data_uri, session_id=session_id
        )
        extracted_date = vision_result["extracted_date"]

        # Perform local analysis of the extracted date
        date_analysis = analyze_expiry_date(extracted_date)

        # Analyze expiry date using reasoning model
        reasoning_result = analyze_expiry_with_llm(
            model=reasoning_model, extracted_date=extracted_date, session_id=session_id
        )
        analysis_result = reasoning_result["analysis_result"]

        # Clean up the temporary file
        os.unlink(temp_path)
        logger.info("Temporary file cleaned up")

        # Calculate total request time
        total_time_ms = round((time.time() - request_start_time) * 1000)

        # Add scores
        if date_analysis["status"] != "error" and date_analysis["status"] != "unknown":
            langfuse.score(
                name="date_extraction_score",
                value=1.0,
                comment="Successfully extracted a valid date",
            )
        else:
            langfuse.score(
                name="date_extraction_score",
                value=0.0,
                comment=f"Failed to extract a valid date: {extracted_date}",
            )

        # Add a score for overall experience based on responsiveness
        response_score = 1.0
        if total_time_ms > 10000:  # More than 10 seconds
            response_score = 0.5
        if total_time_ms > 20000:  # More than 20 seconds
            response_score = 0.2

        langfuse.score(
            name="responsiveness",
            value=response_score,
            comment=f"Response time: {total_time_ms}ms",
        )

        # Return the complete result with enhanced analysis
        return {
            "extracted_date": extracted_date,
            "analysis_llm": analysis_result,
            "analysis_local": date_analysis,
            "models_used": {
                "vision": vision_model,
                "reasoning": reasoning_model,
            },
            "usage": {
                "total_time_ms": total_time_ms,
                "vision_tokens": vision_result["usage"]["total_tokens"],
                "reasoning_tokens": reasoning_result["usage"]["total_tokens"],
                "total_tokens": vision_result["usage"]["total_tokens"]
                + reasoning_result["usage"]["total_tokens"],
                "session_id": session_id,
            },
        }

    except Exception as e:
        # Log the error
        logger.error(f"Error in expiry check: {str(e)}")

        # Clean up if needed
        if "temp_path" in locals():
            try:
                os.unlink(temp_path)
            except:
                pass

        return JSONResponse(status_code=500, content={"error": str(e)})


@observe(name="chat_completion", as_type="trace")
@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()

    # Extract parameters from the request
    prompt = body.get("prompt", "Hello, how can I help you?")
    model = body.get(
        "model", "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
    )  # Changed default to azure/gpt-4

    logger.info(f"Chat request received - model: {model}")

    logger.info(f"Prompt: {prompt}")
    try:
        # Check for API key
        api_key = os.getenv("LITELLM_API_KEY")
        if not api_key:
            raise ValueError("LITELLM_API_KEY is not set")

        # Create message for the model
        messages = [{"role": "user", "content": prompt}]

        # Create a generation span using the langfuse instance
        generation = langfuse.generation(
            name="llm_call",
            model=model,
            input={"messages": messages},
        )

        logger.info(f"Calling model: {model} with API key length: {len(api_key)}")

        # Call the LiteLLM proxy directly
        try:
            response = call_litellm_proxy(model=model, messages=messages)
            logger.info("Model response received successfully")
        except Exception as e:
            logger.error(f"LiteLLM proxy error: {str(e)}")
            raise

        # Extract the response content
        content = response["choices"][0]["message"]["content"]
        logger.info("Response extracted successfully")

        # Update the generation with the result
        generation.end(
            output=content,
            usage={
                "prompt_tokens": response["usage"]["prompt_tokens"],
                "completion_tokens": response["usage"]["completion_tokens"],
                "total_tokens": response["usage"]["total_tokens"],
            },
        )

        return {
            "response": content,
            "model": model,
        }

    except Exception as e:
        # Log the error to Langfuse using the langfuse instance
        logger.error(f"Error in chat endpoint: {str(e)}")
        langfuse.event(name="error", level="ERROR", message=str(e))
        return {"error": str(e)}


# Add a simple test endpoint to verify LiteLLM configuration
@app.get("/test-litellm")
async def test_litellm():
    """Test endpoint to verify LiteLLM configuration"""
    try:
        # Check if LITELLM_API_KEY is set
        api_key = os.getenv("LITELLM_API_KEY")
        if not api_key:
            return {"status": "error", "message": "LITELLM_API_KEY is not set"}

        # Get the API base
        api_base = litellm_api_base

        # Try to make a direct API call to the proxy
        try:
            test_messages = [
                {
                    "role": "user",
                    "content": "This is a test. Reply with 'OK' if you receive this.",
                }
            ]

            response = call_litellm_proxy(
                model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",  # Use the Azure model format
                messages=test_messages,
                max_tokens=10,
            )
            connection_working = True
            test_response = response["choices"][0]["message"]["content"]
        except Exception as e:
            connection_working = False
            connection_error = str(e)

        # Try to get available models via direct API call
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            models_response = requests.get(f"{api_base}/models", headers=headers)
            if models_response.status_code == 200:
                available_models = models_response.json()
            else:
                available_models = (
                    f"Error fetching models: Status code {models_response.status_code}"
                )
        except Exception as e:
            available_models = f"Error fetching models: {str(e)}"

        result = {
            "status": "success" if connection_working else "error",
            "api_key_set": True,
            "api_key_length": len(api_key),
            "api_base": api_base,
            "connection_working": connection_working,
        }

        if connection_working:
            result["test_response"] = test_response
            result["available_models"] = available_models
        else:
            result["connection_error"] = connection_error

        return result
    except Exception as e:
        logger.error(f"Error testing LiteLLM configuration: {str(e)}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
