from fastapi import FastAPI, HTTPException, status
import uvicorn
from helpers import generate
from pydantic import BaseModel
import asyncio

class GenerateRequest(BaseModel):
    prompt: str
    context: str

app = FastAPI(title="AI Content Generation API", version="1.0.0", description="This API provides a way to generate text content based on a given prompt and context.")

@app.post("/v1/generate", summary="Generate Text Content")
def generate_response(request: GenerateRequest):
    try:
        context = request.context
        response = asyncio.run(generate(request.prompt, context))
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "GENERATION_FAILED",
                "error_message": "Failed to generate content. Please try again later."
            }
        )

@app.get("/v1/", include_in_schema=False)
def documentation():
    """
    Detailed API Documentation

    This is the root endpoint of the API, which provides detailed documentation for all available endpoints and their usage.
    """
    return {
        "description": "Welcome to the AI Content Generation API!",
        "endpoints": [
            {
                "method": "POST",
                "path": "/v1/generate",
                "summary": "Generate Text Content",
                "request_body": {
                    "prompt": "The prompt for the content generation",
                    "context": "The context for the content generation"
                },
                "response": "The generated content",
                "errors": [
                    {
                        "error_code": "GENERATION_FAILED",
                        "error_message": "Failed to generate content. Please try again later.",
                        "status_code": 500
                    }
                ]
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)