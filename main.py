
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Invoice Agent is running!"}


@app.get("/.well-known/agent-card.json")
def agent_card():
    return {
        "name": "Invoice Action Agent",
        "description": "Reviews invoice claims and proposes safe invoice actions.",
        "version": "1.0.0",
        "capabilities": {},
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice Action Agent",
                "description": "Evaluates invoice packages and proposes actions for review.",
                "tags": ["invoice", "finance", "approval"]
            }
        ],
        "supportedInterfaces": [
            {
                "url": "https://tds-invoice-agent-apl4.onrender.com",
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0"
            }
        ],
        "defaultInputModes": [
            "application/vnd.ga5.invoice-claim-batch+json"
        ],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json"
        ]
    }
