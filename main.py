import os
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
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
    inflow_lookback: float
    outflow_lookback: float
    by_channel: Dict[str, float]
    by_category: Dict[str, float]


@app.get("/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary(user_id: Optional[str] = Query(None), days: int = Query(30, ge=0)):
    try:
        # Balances (sum of account balances if present)
        acct_filter = {"user_id": user_id} if user_id else {}
        accounts = get_documents("account", acct_filter)
        total_balance = sum(float(a.get("balance", 0) or 0) for a in accounts)

        # Transactions lookback window
        tx_filter: Dict[str, Any] = {}
        if days and days > 0:
            from datetime import timedelta
            now = datetime.utcnow()
            start = now - timedelta(days=days)
            tx_filter["date"] = {"$gte": start}
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
            inflow_lookback=round(inflow, 2),
            outflow_lookback=round(outflow, 2),
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
        return f"You've spent approximately ₹{summary.outflow_lookback:.2f} in the selected period. Top categories: " + \
            ", ".join(sorted(summary.by_category, key=summary.by_category.get, reverse=True)[:3])
    if "saved" in q or "surplus" in q or "left" in q or "balance" in q:
        net = summary.inflow_lookback - summary.outflow_lookback
        return f"Your current combined balance is ~₹{summary.total_balance:.2f}. Net over period: ₹{net:.2f}."
    if "upi" in q:
        upi = summary.by_channel.get("upi", 0.0)
        return f"UPI volume over the selected period is about ₹{upi:.2f}."
    if "credit" in q or "card" in q:
        card = summary.by_channel.get("card", 0.0)
        return f"Card transactions total roughly ₹{card:.2f} in the selected period."
    # Default
    return dedent(
        f"""
        Here's a quick snapshot for the selected period:
        - Combined balance: ₹{summary.total_balance:.2f}
        - Inflow: ₹{summary.inflow_lookback:.2f}
        - Outflow: ₹{summary.outflow_lookback:.2f}
        Top categories: {', '.join(sorted(summary.by_category, key=summary.by_category.get, reverse=True)[:3]) or 'N/A'}
        """
    ).strip()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        # Use a wider default window for chat insights
        summary = dashboard_summary(user_id=req.user_id, days=365)  # reuse logic
        if isinstance(summary, dict):
            summary = DashboardSummary(**summary)
        answer = synthesize_answer(req.question, summary)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------- PDF Statement Upload & Parse ---------

def _parse_date(text: str) -> Optional[datetime]:
    from datetime import datetime
    import re
    # Supports: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, '01 Jan 2024'
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except Exception:
            pass
    # Try detect like 01/02/23
    m = __import__('re').match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2})$", text.strip())
    if m:
        d, mth, y = m.groups()
        y = int(y)
        y = 2000 + y if y < 70 else 1900 + y
        try:
            return datetime(int(y), int(mth), int(d))
        except Exception:
            return None
    return None


def _infer_channel(desc: str) -> Optional[str]:
    # For exactness, do not infer; return None
    return None


def _infer_category(desc: str) -> Optional[str]:
    # For exactness, do not infer; return None
    return None


@app.post("/ingest/statement/pdf")
async def ingest_statement_pdf(
    file: UploadFile = File(...),
    account_name: str = Form("Uploaded Statement"),
    account_type: str = Form("checking"),
    currency: str = Form("INR"),
    user_id: Optional[str] = Form(None),
):
    try:
        import io
        import re
        from decimal import Decimal, InvalidOperation
        import pdfplumber
        content = await file.read()
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            lines: List[str] = []
            for page in pdf.pages:
                txt = page.extract_text() or ""
                for ln in txt.splitlines():
                    lines.append(" ".join(ln.split()))  # normalize spaces only
        # Strict parse: require explicit Dr/Cr markers OR separate debit/credit columns
        txns: List[Dict[str, Any]] = []
        date_pat = re.compile(r"^(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2} [A-Za-z]{3,} \d{4})")
        # amount token matcher (no sign inference here)
        amt_token = re.compile(r"([\d,]+(?:\.\d{1,2})?)")
        crdr_token = re.compile(r"\b(CR|Cr|cr|DR|Dr|dr)\b")
        inserted = 0
        parsed = 0
        skipped_no_marker = 0

        for raw in lines:
            ln = raw.strip()
            if not ln:
                continue
            dm = date_pat.match(ln)
            if not dm:
                continue
            date_str = dm.group(1)
            rest = ln[dm.end():].strip()
            # Find all amount-like tokens at end
            tokens = amt_token.findall(rest)
            crdr_match = list(crdr_token.finditer(rest))

            amount: Optional[Decimal] = None
            sign: Optional[int] = None  # +1 credit, -1 debit

            # Case 1: explicit CR/DR marker near end with one amount
            if crdr_match:
                marker = crdr_match[-1].group(0).lower()
                sign = 1 if marker.startswith('cr') else -1
                # amount should be the last numeric token before/around the marker
                if tokens:
                    try:
                        amount = Decimal(tokens[-1].replace(',', ''))
                    except InvalidOperation:
                        amount = None
            else:
                # Without explicit markers we skip to keep exactness
                amount = None
                sign = None

            if amount is None or sign is None:
                skipped_no_marker += 1
                continue

            try:
                dt = _parse_date(date_str)
            except Exception:
                dt = None
            if not dt:
                continue

            # Description is rest with marker and trailing amount removed
            desc_end_idx = crdr_match[-1].start() if crdr_match else len(rest)
            description = rest[:desc_end_idx].strip()
            if not description:
                description = "Transaction"

            amt_signed = (amount * (1 if sign == 1 else -1))
            parsed += 1

            txns.append({
                "user_id": user_id,
                "account_id": None,
                "amount": float(amt_signed),  # stored as float in DB; parsed exactly via Decimal first
                "currency": currency,
                "date": dt,
                "description": description,
                "category": _infer_category(description),  # None for exactness
                "channel": _infer_channel(description),    # None for exactness
            })

        # Create/find institution & account for this user
        inst = db["institution"].find_one({"name": "Uploaded PDF"})
        if not inst:
            inst_id = create_document("institution", {"name": "Uploaded PDF", "type": "other"})
        else:
            inst_id = str(inst["_id"]) if "_id" in inst else inst.get("id")
        acct = db["account"].find_one({"name": account_name, "user_id": user_id}) if user_id else db["account"].find_one({"name": account_name})
        if not acct:
            acct_id = create_document("account", {
                "user_id": user_id,
                "name": account_name,
                "type": account_type,
                "currency": currency,
                "balance": 0.0,
                "institution_id": inst_id,
                "last_synced_at": datetime.utcnow(),
            })
        else:
            acct_id = str(acct["_id"]) if "_id" in acct else acct.get("id")
        # Insert transactions
        for t in txns:
            t["account_id"] = acct_id
            try:
                create_document("transaction", t)
                inserted += 1
            except Exception:
                continue
        return {"status": "ok", "inserted": inserted, "parsed": parsed, "skipped_no_marker": skipped_no_marker}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {e}")


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
