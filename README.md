# LiteLLM + Langfuse Multi-Model Demo

A demonstration app showcasing multi-model architecture with proper tracing using LiteLLM and Langfuse. This app includes both a simple chat endpoint and a more complex multi-model flow for product expiry date checking from images.

## Setup

1. Clone this repository
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Create a `.env` file from the example:
```
cp .env.example .env
```
4. Edit the `.env` file with your API keys:
   - Configure LiteLLM virtual key:
     - `LITELLM_API_KEY`: Your LiteLLM virtual API key (set up through LiteLLM proxy)
   - Configure Langfuse:
     - `LANGFUSE_PUBLIC_KEY`: Your Langfuse public key
     - `LANGFUSE_SECRET_KEY`: Your Langfuse secret key
     - `LANGFUSE_HOST`: Langfuse host URL

## Running the Application

Start the server:
```
python app.py
```

The API will be available at http://localhost:8000.

# IGNORE Below

## Testing with the Clients

### Basic Chat Client

Use the included chat client script to test the simple chat API:
```
python client.py "Your prompt goes here"
```

If no prompt is provided, a default prompt will be used.

### Product Expiry Checker

Use the expiry client to test the multi-model product expiry checking pipeline:
```
python expiry_client.py path/to/your/product_image.jpg
```

By default, this will use:
- `gpt-4o-vision-poc` for vision analysis
- `gpt-4` for reasoning

You can optionally specify which models to use:
```
python expiry_client.py path/to/your/product_image.jpg gpt-4o-vision-poc gpt-4
```

## API Endpoints

### 1. Chat Endpoint

- **POST /chat**: Send a chat message to an LLM
  - Request body:
    ```json
    {
      "prompt": "Your message here",
      "model": "gpt-4"  // Optional, defaults to gpt-4
    }
    ```
  - Response:
    ```json
    {
      "response": "The LLM's response",
      "trace_id": "langfuse-trace-id",
      "model": "model-used"
    }
    ```

### 2. Product Expiry Checker

- **POST /check-expiry**: Multi-model endpoint to analyze product expiry dates from images
  - Request format: `multipart/form-data`
  - Fields:
    - `file`: Image file (required)
    - `vision_model`: Model to use for image analysis (default: gpt-4o-vision-poc)
    - `reasoning_model`: Model to use for date analysis (default: gpt-4)
  - Response:
    ```json
    {
      "extracted_date": "2024-12-31",
      "analysis": "The product expires on December 31, 2024, which is X days from now...",
      "trace_id": "langfuse-trace-id",
      "models_used": {
        "vision": "gpt-4o-vision-poc",
        "reasoning": "gpt-4"
      }
    }
    ```

## Multi-Model Architecture

The product expiry checker demonstrates a multi-model architecture with proper tracing:

1. **Vision Model (First Stage)**:
   - Uses GPT-4o Vision model (via LiteLLM virtual key)
   - Processes the uploaded product image
   - Extracts the expiry date in a standardized format
   - Traced in Langfuse as a span with all relevant metadata

2. **Reasoning Model (Second Stage)**:
   - Uses GPT-4 model (via LiteLLM virtual key)
   - Receives the extracted expiry date from the first model
   - Analyzes whether the product is expired, about to expire, or has plenty of time left
   - Provides human-readable analysis with time calculations
   - Traced in Langfuse as a subsequent span in the same trace

3. **Complete Flow Tracing**:
   - The entire process is captured as a single trace in Langfuse
   - Each model call is recorded as a span within that trace
   - All inputs, outputs, token usage, and metadata are captured
   - Errors are properly tracked and reported

## Benefits of Using LiteLLM Virtual Keys

Using LiteLLM's virtual key approach offers several advantages:

1. **Simplified Code**: No need to handle model routing or different provider credentials in the application code
2. **Centralized Configuration**: All model routing and provider configuration is managed in one place
3. **Easy Provider Switching**: Change from Azure to other providers without modifying application code
4. **Enhanced Security**: API keys for providers are never exposed to the application
5. **Access Control**: LiteLLM proxy provides key management, rate limiting, and usage tracking

## Features

- Uses LiteLLM virtual keys for simplified model routing
- Demonstrates multi-model chains with proper data passing between models
- Comprehensive tracing with Langfuse for monitoring and analysis
- Error handling and proper cleanup of temporary files
- Simple clients for testing both endpoints 