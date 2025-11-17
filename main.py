import os
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db, create_document, get_documents
from schemas import Account, Transaction, Institution

app = FastAPI(title="AI Financial Companion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "AI Financial Companion Backend Running"}


# --------- Health & Schema Endpoints ---------
class SchemaField(BaseModel):
    name: str
    type: str
    required: bool = False


class SchemaModel(BaseModel):
    name: str
    fields: List[SchemaField]


@app.get("/schema", response_model=List[SchemaModel])
def get_schema():
    return [
        SchemaModel(
            name="institution",
            fields=[
                SchemaField(name="name", type="string", required=True),
                SchemaField(name="type", type="enum: bank|credit|upi|wallet|other", required=True),
                SchemaField(name="logo_url", type="string", required=False),
            ],
        ),
        SchemaModel(
            name="account",
            fields=[
                SchemaField(name="user_id", type="string"),
                SchemaField(name="name", type="string", required=True),
                SchemaField(name="institution_id", type="string"),
                SchemaField(name="type", type="enum: checking|savings|credit|upi|wallet|loan|other", required=True),
                SchemaField(name="currency", type="string"),
                SchemaField(name="balance", type="number"),
                SchemaField(name="last_synced_at", type="datetime"),
            ],
        ),
        SchemaModel(
            name="transaction",
            fields=[
                SchemaField(name="user_id", type="string"),
                SchemaField(name="account_id", type="string"),
                SchemaField(name="amount", type="number", required=True),
                SchemaField(name="currency", type="string"),
                SchemaField(name="date", type="datetime", required=True),
                SchemaField(name="description", type="string", required=True),
                SchemaField(name="category", type="string"),
                SchemaField(name="subcategory", type="string"),
                SchemaField(name="channel", type="string"),
                SchemaField(name="notes", type="string"),
                SchemaField(name="reference_id", type="string"),
            ],
        ),
    ]


# --------- Ingestion Endpoints ---------
class IngestAccount(BaseModel):
    data: Account


class IngestTransaction(BaseModel):
    data: Transaction


class IngestInstitution(BaseModel):
    data: Institution


@app.post("/ingest/account")
def ingest_account(payload: IngestAccount):
    try:
        inserted_id = create_document("account", payload.data)
        return {"status": "ok", "id": inserted_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/transaction")
def ingest_transaction(payload: IngestTransaction):
    try:
        # Ensure idempotency if reference_id exists
        if payload.data.reference_id:
            existing = db["transaction"].find_one({"reference_id": payload.data.reference_id})
            if existing:
                return {"status": "exists", "id": str(existing.get("_id"))}
        inserted_id = create_document("transaction", payload.data)
        return {"status": "ok", "id": inserted_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/institution")
def ingest_institution(payload: IngestInstitution):
    try:
        inserted_id = create_document("institution", payload.data)
        return {"status": "ok", "id": inserted_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------- Dashboard Queries ---------
class DashboardSummary(BaseModel):
    total_balance: float
    inflow_30d: float
    outflow_30d: float
    by_channel: Dict[str, float]
    by_category: Dict[str, float]


@app.get("/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary(user_id: Optional[str] = Query(None)):
    try:
        # Balances
        acct_filter = {"user_id": user_id} if user_id else {}
        accounts = get_documents("account", acct_filter)
        total_balance = sum(float(a.get("balance", 0) or 0) for a in accounts)

        # Transactions last 30 days
        from datetime import datetime, timedelta

        now = datetime.utcnow()
        start = now - timedelta(days=30)
        tx_filter: Dict[str, Any] = {"date": {"$gte": start}}
        if user_id:
            tx_filter["user_id"] = user_id
        txs = list(db["transaction"].find(tx_filter))

        inflow = sum(float(t.get("amount", 0)) for t in txs if float(t.get("amount", 0)) > 0)
        outflow = sum(abs(float(t.get("amount", 0))) for t in txs if float(t.get("amount", 0)) < 0)

        # Groupings
        by_channel: Dict[str, float] = {}
        by_category: Dict[str, float] = {}
        for t in txs:
            ch = t.get("channel") or "other"
            by_channel[ch] = by_channel.get(ch, 0.0) + abs(float(t.get("amount", 0)))
            cat = t.get("category") or "Uncategorized"
            by_category[cat] = by_category.get(cat, 0.0) + abs(float(t.get("amount", 0)))

        return DashboardSummary(
            total_balance=round(total_balance, 2),
            inflow_30d=round(inflow, 2),
            outflow_30d=round(outflow, 2),
            by_channel={k: round(v, 2) for k, v in by_channel.items()},
            by_category={k: round(v, 2) for k, v in by_category.items()},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------- AI Chat: Transaction-aware Q&A ---------
class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str


from textwrap import dedent


def synthesize_answer(question: str, summary: DashboardSummary) -> str:
    # Very simple, deterministic answer synthesizer based on available data.
    # In a real app, replace with an LLM call.
    q = question.lower()
    if "spent" in q or "outflow" in q or "expenses" in q:
        return f"You've spent approximately ₹{summary.outflow_30d:.2f} in the last 30 days. Top categories: " + \
            ", ".join(sorted(summary.by_category, key=summary.by_category.get, reverse=True)[:3])
    if "saved" in q or "surplus" in q or "left" in q or "balance" in q:
        net = summary.inflow_30d - summary.outflow_30d
        return f"Your current combined balance is ~₹{summary.total_balance:.2f}. Net over 30 days: ₹{net:.2f}."
    if "upi" in q:
        upi = summary.by_channel.get("upi", 0.0)
        return f"UPI volume over the last month is about ₹{upi:.2f}."
    if "credit" in q or "card" in q:
        card = summary.by_channel.get("card", 0.0)
        return f"Card transactions total roughly ₹{card:.2f} in the last 30 days."
    # Default
    return dedent(
        f"""
        Here's a quick snapshot for the last 30 days:
        - Combined balance: ₹{summary.total_balance:.2f}
        - Inflow: ₹{summary.inflow_30d:.2f}
        - Outflow: ₹{summary.outflow_30d:.2f}
        Top categories: {', '.join(sorted(summary.by_category, key=summary.by_category.get, reverse=True)[:3]) or 'N/A'}
        """
    ).strip()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        summary = dashboard_summary(user_id=req.user_id)  # reuse logic
        # If FastAPI returns model, ensure we have object
        if isinstance(summary, dict):
            summary = DashboardSummary(**summary)
        answer = synthesize_answer(req.question, summary)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Utility endpoint to seed sample data for demo
@app.post("/demo/seed")
def seed_demo():
    try:
        # Create sample institution & accounts
        bank_id = create_document("institution", {"name": "Demo Bank", "type": "bank"})
        upi_id = create_document("institution", {"name": "Demo UPI", "type": "upi"})

        create_document("account", {"name": "Savings", "type": "savings", "currency": "INR", "balance": 52340.25, "institution_id": bank_id})
        create_document("account", {"name": "Credit Card", "type": "credit", "currency": "INR", "balance": -12450.0, "institution_id": bank_id})
        create_document("account", {"name": "UPI Wallet", "type": "upi", "currency": "INR", "balance": 950.0, "institution_id": upi_id})

        from datetime import timedelta
        now = datetime.utcnow()
        sample = [
            {"amount": -450.0, "currency": "INR", "date": now - timedelta(days=2), "description": "Groceries - BigBazaar", "category": "Groceries", "channel": "card"},
            {"amount": -120.0, "currency": "INR", "date": now - timedelta(days=1), "description": "Tea & Snacks", "category": "Food", "channel": "upi"},
            {"amount": 35000.0, "currency": "INR", "date": now - timedelta(days=5), "description": "Salary", "category": "Income", "channel": "ach"},
            {"amount": -999.0, "currency": "INR", "date": now - timedelta(days=7), "description": "Electricity Bill", "category": "Utilities", "channel": "netbanking"},
            {"amount": -2300.0, "currency": "INR", "date": now - timedelta(days=12), "description": "Online Shopping", "category": "Shopping", "channel": "card"},
            {"amount": -180.0, "currency": "INR", "date": now - timedelta(days=15), "description": "Auto Ride", "category": "Transport", "channel": "upi"},
        ]
        for t in sample:
            create_document("transaction", t)

        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
