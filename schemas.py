"""
Database Schemas for AI Financial Companion

Each Pydantic model corresponds to a MongoDB collection (lowercased class name).
- Account -> "account"
- Transaction -> "transaction"
- Institution -> "institution"

"""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class Institution(BaseModel):
    name: str = Field(..., description="Institution display name")
    type: Literal["bank", "credit", "upi", "wallet", "other"] = Field(
        "bank", description="Type of financial institution"
    )
    logo_url: Optional[str] = Field(None, description="Logo URL for display")


class Account(BaseModel):
    user_id: Optional[str] = Field(None, description="User identifier (optional demo)")
    name: str = Field(..., description="Account nickname, e.g., HDFC Savings")
    institution_id: Optional[str] = Field(None, description="Related institution id")
    type: Literal["checking", "savings", "credit", "upi", "wallet", "loan", "other"] = Field(
        ..., description="Account type"
    )
    currency: str = Field("INR", description="ISO currency code")
    balance: float = Field(0.0, description="Current balance")
    last_synced_at: Optional[datetime] = Field(None, description="Last sync time")


class Transaction(BaseModel):
    user_id: Optional[str] = Field(None, description="User identifier (optional demo)")
    account_id: Optional[str] = Field(None, description="Account this transaction belongs to")
    amount: float = Field(..., description="Positive for inflow, negative for outflow")
    currency: str = Field("INR", description="ISO currency code")
    date: datetime = Field(..., description="Transaction date/time")
    description: str = Field(..., description="Narration or merchant name")
    category: Optional[str] = Field(None, description="High-level category, e.g., Groceries, Food, Travel")
    subcategory: Optional[str] = Field(None, description="Subcategory if any")
    channel: Optional[Literal["upi", "card", "netbanking", "cash", "imps", "neft", "ach", "other"]] = None
    notes: Optional[str] = None
    reference_id: Optional[str] = Field(None, description="External txn id for idempotency")
