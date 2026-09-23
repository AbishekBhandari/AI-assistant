from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.agent import run_agent


app = FastAPI(
    title="AI Assistant",
    description="Agentic AI Assistant",
    version="2.0.0"
)


limiter = Limiter(
    key_func=get_remote_address
)

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    topic: str
    confidence: float


@app.get("/")
def root():
    return {
        "message": "Agentic AI Assistant API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post(
    "/chat",
    response_model=ChatResponse
)
@limiter.limit("10/minute")
async def chat(
    request: Request,
    chat_request: ChatRequest
):

    if not chat_request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    try:

        result = await run_agent(
            chat_request.message
        )

        return ChatResponse(**result)

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        print(
            f"CHAT ERROR: {e}",
            flush=True
        )

        raise HTTPException(
            status_code=503,
            detail="AI service is temporarily unavailable."
        )