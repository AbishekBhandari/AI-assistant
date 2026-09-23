# AI Assistant — Applied AI & Engineering AI Systems

A production-oriented AI Assistant developed in two stages:

- **Task 1: Build an AI Assistant (Applied AI)**
- **Task 2: Productionize the AI Assistant (Engineering AI Systems)**

The project combines LLM integration, prompt engineering, structured output, tool calling, RAG, vector search, local LLM serving with vLLM, a web UI, reliability mechanisms, performance engineering, and Docker-based deployment.

---

# 1. Project Overview

The application can:

- Answer user questions using an LLM
- Answer questions from uploaded documents using RAG
- Perform arithmetic using an LLM-selected calculator tool
- Return structured JSON responses
- Use Groq as a hosted LLM provider
- Use an open-source model served through vLLM as a fallback/local deployment
- Provide a Streamlit chat interface
- Handle asynchronous API requests
- Retry temporary failures
- Rate-limit requests
- Fall back between LLM providers
- Handle errors gracefully
- Cache repeated responses
- Measure RAG and LLM latency
- Run using Docker and Docker Compose

---

# 2. Task 1 — Build an AI Assistant

## 2.1 Task 1 Architecture

The Task 1 architecture is:

```text
                         ┌─────────────────────┐
                         │        User         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      FastAPI        │
                         │      Backend        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    AI Pipeline      │
                         └──────────┬──────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
                ▼                   ▼                   ▼
        ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
        │ Prompt       │    │ Tool Calling │    │     RAG      │
        │ Engineering  │    │ Calculator   │    │   Pipeline   │
        └──────┬───────┘    └──────────────┘    └──────┬───────┘
               │                                       │
               │                              ┌────────┴────────┐
               │                              │                 │
               │                              ▼                 ▼
               │                         Embeddings          Qdrant
               │                              │                 │
               │                              └────────┬────────┘
               │                                       │
               └───────────────────┬───────────────────┘
                                   ▼
                           ┌────────────────┐
                           │      LLM       │
                           │ Groq / vLLM    │
                           └───────┬────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ Structured JSON    │
                         │ answer/topic/etc.  │
                         └────────────────────┘
```

---

# 3. Task 1 Components

## 3.1 LLM Integration

The application uses an OpenAI-compatible client.

Groq is used as the primary hosted provider.

Example:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)
```

Development model:

```text
openai/gpt-oss-20b
```

---

## 3.2 Prompt Engineering

A system prompt controls the behavior of the assistant.

The application also supports configurable:

```text
temperature
top_p
```

These values are stored in `.env`.

Example:

```env
TEMPERATURE=0.7
TOP_P=0.9
```

---

## 3.3 Structured JSON Output

The assistant returns a predictable JSON structure:

```json
{
  "answer": "Response from the AI",
  "topic": "machine learning",
  "confidence": 0.95
}
```

Pydantic validates the API response.

```python
class ChatResponse(BaseModel):
    answer: str
    topic: str
    confidence: float
```

---

# 4. Tool Calling

The assistant includes a calculator tool.

Supported operations:

```text
add
subtract
multiply
divide
```

Example:

```python
calculator(10, 5, "multiply")
```

Result:

```text
50
```

The LLM can request the calculator when arithmetic is required.

---

# 5. RAG — Retrieval-Augmented Generation

The application uses RAG to answer questions from documents.

## RAG Architecture

```text
             DOCUMENT INGESTION
                    │
                    ▼
              PDF / TXT file
                    │
                    ▼
              Text extraction
                    │
                    ▼
                 Chunking
                    │
                    ▼
               Embeddings
                    │
                    ▼
                 Qdrant
                    │
                    │
                    │
USER QUESTION ──────┘
       │
       ▼
Question embedding
       │
       ▼
Similarity search
       │
       ▼
Relevant document chunks
       │
       ▼
       LLM
       │
       ▼
     Answer
```

---

## 5.1 Document Ingestion

PDF documents are processed using PyPDF.

Example document:

```text
documents/BE Computer Syllabus.pdf
```

The document is extracted and divided into chunks.

---

## 5.2 Embeddings

The application uses:

```text
all-MiniLM-L6-v2
```

from Sentence Transformers.

Each document chunk is converted into a vector representation.

---

## 5.3 Vector Database

Qdrant stores:

- Document embeddings
- Corresponding text chunks

Similarity search retrieves the most relevant chunks for a user query.

---

# 6. Local LLM Deployment with vLLM

Task 1 requires local deployment of an open-source model using vLLM.

The local model was tested using:

```text
Google Colab
Tesla T4 GPU
15 GB VRAM
```

Example model:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Architecture:

```text
FastAPI
   │
   ▼
vLLM OpenAI-compatible API
   │
   ▼
Qwen open-source model
   │
   ▼
Response
```

The development Windows machine did not have an NVIDIA GPU, so vLLM was tested on a GPU-enabled Google Colab environment.

---

# 7. Task 2 — Productionize the AI Assistant

Task 2 transforms the Task 1 prototype into a production-oriented application.

## 7.1 Task 2 Architecture

```text
                              ┌─────────────────┐
                              │      User       │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   Streamlit UI  │
                              └────────┬────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │        FastAPI           │
                         │        Backend           │
                         └────────────┬─────────────┘
                                      │
              ┌───────────────────────┼──────────────────────┐
              │                       │                      │
              ▼                       ▼                      ▼
       Rate Limiting              Async API             Error Handling
              │                       │                      │
              └───────────────────────┼──────────────────────┘
                                      │
                                      ▼
                             ┌─────────────────┐
                             │   AI Pipeline   │
                             └────────┬────────┘
                                      │
                   ┌──────────────────┼──────────────────┐
                   │                  │                  │
                   ▼                  ▼                  ▼
                RAG/Qdrant         Cache              LLM Layer
                   │                  │                  │
                   │                  │           ┌──────┴──────┐
                   │                  │           │             │
                   │                  │           ▼             ▼
                   │                  │         Groq           vLLM
                   │                  │        Primary       Fallback
                   │                  │           │             │
                   └──────────────────┴───────────┴─────────────┘
                                      │
                                      ▼
                                  Response
```

---

# 8. Task 2 Production Features

## 8.1 Web UI

Streamlit provides a simple chat interface.

Architecture:

```text
Browser
   ↓
Streamlit
   ↓ HTTP POST
FastAPI /chat
   ↓
AI Assistant
```

Run:

```bash
python -m streamlit run app/ui.py
```

UI:

```text
http://localhost:8501
```

---

# 9. Asynchronous Request Handling

FastAPI uses asynchronous endpoints.

Example:

```python
@app.post("/chat")
async def chat(request: ChatRequest):
    response = await generate_response(request.message)
    return response
```

The LLM client uses `AsyncOpenAI`.

This allows I/O-bound LLM requests to be handled asynchronously.

---

# 10. Retry Mechanism

Tenacity is used to retry temporary failures.

Configuration:

```text
Maximum attempts: 3
Exponential backoff
```

Example:

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(
        multiplier=1,
        min=2,
        max=8
    )
)
```

Flow:

```text
Request
   ↓
LLM
   ↓ failure
Retry
   ↓ failure
Retry
   ↓ failure
Fallback / Error
```

---

# 11. Rate Limiting

SlowAPI is used for request limiting.

Current configuration:

```text
10 requests per minute per IP
```

Excess requests return:

```text
HTTP 429 Too Many Requests
```

---

# 12. Fallback Provider

The application supports a fallback provider.

```text
Primary provider:  Groq
Fallback provider: vLLM
```

Flow:

```text
              ┌─── Groq ───→ Response
FastAPI ──────┤
              └─── vLLM ───→ Response
                  ↑
              if Groq fails
```

---

# 13. Error Handling

The application handles common failures.

| Situation | HTTP Response |
|---|---:|
| Empty message | 400 |
| Invalid request | 400 |
| Rate limit exceeded | 429 |
| AI service unavailable | 503 |
| Unexpected server error | 500 |

Example:

```python
if not request.message.strip():
    raise HTTPException(
        status_code=400,
        detail="Message cannot be empty."
    )
```

RAG retrieval failures can also be handled so that a temporary retrieval failure does not necessarily crash the entire application.

---

# 14. Response Caching

A TTL cache is implemented using `cachetools`.

Configuration:

```text
Maximum entries: 100
TTL: 5 minutes
```

Flow:

```text
User Question
      │
      ▼
 Check Cache
   /       \
 HIT       MISS
  │          │
  ▼          ▼
Response    RAG + LLM
               │
               ▼
             Cache
```

Caching reduces repeated LLM calls for identical questions.

---

# 15. Performance Engineering

The application measures:

- RAG latency
- LLM latency
- Total request latency

Example:

```text
RAG time: 0.120s
LLM time: 1.850s
Total time: 1.970s
```

Performance improvements include:

- Asynchronous request handling
- Response caching
- Reduced prompt size
- Limited RAG context
- vLLM for local inference
- Appropriate model selection

---

# 16. ONNX Optimization

ONNX conversion was not applied to the main LLM.

The application uses externally served models through:

```text
Groq
vLLM
```

The application does not directly manage the main LLM model weights.

Therefore, inference optimization was instead addressed through:

- vLLM
- Response caching
- Reduced prompt/context size
- Asynchronous request handling
- Model selection

---

# 17. Complete System Architecture

The complete Task 1 + Task 2 architecture is:

```text
                                  ┌──────────────┐
                                  │     User     │
                                  └──────┬───────┘
                                         │
                                         ▼
                                  ┌──────────────┐
                                  │  Streamlit   │
                                  │      UI      │
                                  └──────┬───────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │       FastAPI       │
                              │       Backend       │
                              └──────────┬──────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
             Rate Limiting          Async API             Cache
                    │                    │                    │
                    └────────────────────┼────────────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │    AI Pipeline      │
                              └──────────┬──────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
                 RAG Layer          Tool Calling          Prompt
                    │                    │                Engineering
                    ▼                    ▼                    │
                 Qdrant             Calculator               │
                    │                    │                    │
                    └────────────────────┼────────────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │     LLM Layer       │
                              └──────────┬──────────┘
                                         │
                           ┌─────────────┴─────────────┐
                           │                           │
                           ▼                           ▼
                         Groq                         vLLM
                        Primary                     Fallback
                           │                           │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                              Structured JSON Response
                                         │
                                         ▼
                                    Streamlit UI
```

---

# 18. Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Backend | FastAPI |
| Web UI | Streamlit |
| Primary LLM | Groq |
| Local LLM Serving | vLLM |
| Open-source Model | Qwen |
| Embeddings | Sentence Transformers |
| Vector Database | Qdrant |
| PDF Processing | PyPDF |
| Structured Output | Pydantic + JSON |
| Tool Calling | OpenAI-compatible function calling |
| Retry | Tenacity |
| Rate Limiting | SlowAPI |
| Cache | Cachetools |
| Containerization | Docker |
| Multi-container Deployment | Docker Compose |

---

# 19. Project Structure

```text
ai-assistant/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── ui.py
│   ├── config.py
│   ├── llm.py
│   ├── rag.py
│   ├── tools.py
│   └── cache.py
│
├── documents/
│   └── BE Computer Syllabus.pdf
│
├── qdrant_data/
│
├── .env
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 20. Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key

TEMPERATURE=0.7
TOP_P=0.9

VLLM_BASE_URL=https://your-vllm-url/v1
VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
```

Never commit `.env` to GitHub.

---

# 21. Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Windows activation:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

# 22. Running the Application

## Start FastAPI

```bash
python -m uvicorn app.main:app --reload
```

API:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

## Start Streamlit

Open another terminal:

```bash
python -m streamlit run app/ui.py
```

UI:

```text
http://localhost:8501
```

---

# 23. RAG Document Ingestion

Place documents inside:

```text
documents/
```

Example:

```text
documents/
└── BE Computer Syllabus.pdf
```

Run ingestion:

```bash
python -c "from app.rag import ingest_document; print(ingest_document('documents/BE Computer Syllabus.pdf'))"
```

The process performs:

```text
PDF
 ↓
Text extraction
 ↓
Chunking
 ↓
Embedding generation
 ↓
Qdrant storage
```

---

# 24. Test RAG Retrieval

```bash
python -c "from app.rag import search_documents; print(search_documents('What is this syllabus about?'))"
```

---

# 25. API Example

### POST `/chat`

Request:

```json
{
  "message": "What topics are covered in Advanced Java Programming?"
}
```

Response:

```json
{
  "answer": "The course covers ...",
  "topic": "Advanced Java Programming",
  "confidence": 0.95
}
```

---

# 26. Docker

Build the application:

```bash
docker build -t ai-assistant .
```

Run:

```bash
docker run --rm -p 8000:8000 --env-file .env ai-assistant
```

API:

```text
http://localhost:8000
```

---

# 27. Docker Compose

The complete production-oriented deployment uses Docker Compose.

Planned services:

```text
Streamlit
FastAPI
Qdrant
Redis
```

Architecture:

```text
┌──────────────────┐
│    Streamlit     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│     FastAPI      │
└─────┬───────┬────┘
      │       │
      ▼       ▼
  ┌───────┐ ┌───────┐
  │Qdrant │ │ Redis │
  └───────┘ └───────┘
      │
      ▼
┌──────────────────┐
│   Groq / vLLM    │
└──────────────────┘
```

Start:

```bash
docker compose up --build
```

Stop:

```bash
docker compose down
```

---

# 28. Task 1 Milestones

```text
M1  Basic LLM Integration
 ↓
M2  Prompt Engineering
 ↓
M3  Structured JSON Output
 ↓
M4  Tool Calling
 ↓
M5  RAG + Qdrant
 ↓
M6  Local LLM + vLLM
 ↓
M7  Dockerization
```

---

# 29. Task 2 Milestones

```text
M1  Web UI
 ↓
M2  Async Request Handling
 ↓
M3  Retry Mechanism
 ↓
M4  Rate Limiting
 ↓
M5  Fallback Provider
 ↓
M6  Error Handling
 ↓
M7  Response Caching
 ↓
M8  Performance Engineering
 ↓
M9  ONNX Justification
 ↓
M10 Docker Compose
 ↓
M11 Testing
 ↓
M12 Documentation
```

---

# 30. Reliability Features

```text
✓ Asynchronous request handling
✓ Retry mechanism
✓ Rate limiting
✓ Fallback provider
✓ Error handling
✓ Graceful degradation
✓ Response caching
```

---

# 31. Security Considerations

- API keys are stored in environment variables.
- `.env` is excluded from Git.
- User input is validated with Pydantic.
- Rate limiting protects the API from excessive requests.
- API keys are not exposed to the Streamlit frontend.

---

# 32. Testing Checklist

The final application should be tested for:

```text
✓ Normal question
✓ RAG question
✓ Calculator/tool question
✓ Empty question
✓ Invalid request
✓ LLM timeout
✓ LLM failure
✓ Fallback provider
✓ Rate limiting
✓ Concurrent requests
✓ Repeated/cached question
✓ Docker deployment
✓ Docker Compose deployment
```

---

# 33. Limitations

- Practical vLLM inference requires suitable GPU resources.
- The development Windows machine does not have an NVIDIA GPU.
- vLLM was therefore tested using a Tesla T4 GPU in Google Colab.
- The current RAG implementation uses simple word-based chunking.
- Hosted LLM operation depends on provider availability.
- Colab-based vLLM is suitable for demonstration/testing rather than persistent production hosting.

---

# 34. Future Improvements

Possible improvements include:

- Persistent Redis caching
- Better semantic chunking
- Hybrid keyword + vector search
- Reranking retrieved documents
- Authentication and authorization
- HTTPS
- Centralized logging
- Prometheus/Grafana monitoring
- Automated unit and integration tests
- CI/CD
- Cloud deployment
- Persistent GPU-based vLLM deployment

---

# 35. Conclusion

This project demonstrates the progression from an AI prototype to a production-oriented AI application.

```text
                 TASK 1
                   │
                   ▼
          AI Assistant Prototype
                   │
      ┌────────────┼────────────┐
      ▼            ▼            ▼
     LLM          Tools         RAG
      │            │             │
      └────────────┼─────────────┘
                   ▼
              vLLM / Groq
                   │
                   ▼
                 TASK 2
                   │
                   ▼
          Productionization
                   │
      ┌────────────┼────────────┐
      ▼            ▼            ▼
     UI          Reliability  Performance
      │            │             │
      │       ┌────┼────┐        │
      │       ▼    ▼    ▼        │
      │     Retry Rate Fallback  │
      │       │    │    │        │
      └───────┴────┴────┴────────┘
                   │
                   ▼
             Docker / Compose
                   │
                   ▼
          Production-oriented
             AI Assistant
```

The project covers the complete AI engineering lifecycle:

**LLM integration → prompt engineering → structured output → tool calling → RAG → vector database → local model serving → web application → reliability → performance → caching → containerization → deployment.**
