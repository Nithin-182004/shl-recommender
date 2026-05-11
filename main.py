"""SHL Assessment Recommender — FastAPI entry point."""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import ChatRequest, ChatResponse, Recommendation
from agent import SHLAgent

load_dotenv()

app = FastAPI(
    title="SHL Assessment Recommender",
    description="A conversational agent that recommends SHL assessments",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = None


@app.on_event("startup")
def startup():
    """Initialize the agent on startup."""
    global agent
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("WARNING: GROQ_API_KEY not set!")
        return
    agent = SHLAgent(api_key=api_key)
    print("Agent initialized and ready!")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Stateless chat endpoint — takes full conversation history, returns next reply."""
    global agent

    if agent is None:
        return ChatResponse(
            reply="Service is starting up. Please try again in a moment.",
            recommendations=[],
            end_of_conversation=False
        )

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    result = agent.chat(messages)

    recs = [
        Recommendation(
            name=r["name"],
            url=r["url"],
            test_type=r["test_type"]
        )
        for r in result["recommendations"]
    ]

    return ChatResponse(
        reply=result["reply"],
        recommendations=recs,
        end_of_conversation=result["end_of_conversation"]
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
