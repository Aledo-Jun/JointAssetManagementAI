"""Pydantic data models for the Joint Asset Management prototype."""
from __future__ import annotations

from datetime import datetime, date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, validator


class IncomeBracket(str, Enum):
    """Discrete income ranges for user profiling."""

    LOW = "low"
    MIDDLE = "middle"
    HIGH = "high"


class AssetType(str, Enum):
    """Supported asset categories in the graph."""

    ACCOUNT = "account"
    REALESTATE = "real_estate"
    STOCK = "stock"


class MerchantCategory(str, Enum):
    """Merchant categories used for risk checks."""

    FOOD = "food"
    TRAVEL = "travel"
    RETAIL = "retail"
    HIGH_RISK = "high_risk"


class User(BaseModel):
    """Individual user participating in a household."""

    id: str = Field(..., description="Unique user identifier")
    name: str = Field(..., description="User name")
    age: int = Field(..., ge=0, description="User age")
    income_bracket: IncomeBracket = Field(..., description="Income bracket classification")


class Household(BaseModel):
    """Household node representing a joint asset pool."""

    id: str = Field(..., description="Household identifier")
    formation_date: date = Field(default_factory=date.today, description="Household creation date")
    total_asset: float = Field(0.0, ge=0.0, description="Aggregate asset value")


class Asset(BaseModel):
    """Asset node owned by one or more users."""

    id: str = Field(..., description="Asset identifier")
    type: AssetType = Field(..., description="Type of the asset")
    balance: float = Field(0.0, description="Asset balance or valuation")


class Merchant(BaseModel):
    """Merchant node participating in transactions."""

    id: str = Field(..., description="Merchant identifier")
    category: MerchantCategory = Field(..., description="Merchant industry category")
    risk_score: Optional[float] = Field(None, description="Optional precomputed risk score")


class MemberEdge(BaseModel):
    """Edge connecting a user to a household."""

    user_id: str = Field(..., description="User node id")
    household_id: str = Field(..., description="Household node id")
    role: str = Field(..., description="Role of the user within the household")


class OwnsEdge(BaseModel):
    """Edge connecting a user to an owned asset."""

    user_id: str = Field(..., description="User node id")
    asset_id: str = Field(..., description="Asset node id")
    ownership_ratio: float = Field(..., ge=0.0, le=1.0, description="Ownership share between 0 and 1")

    @validator("ownership_ratio")
    def validate_ratio(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Ownership ratio must be between 0 and 1")
        return value


class TransferEdge(BaseModel):
    """Transaction edge for transfers between nodes."""

    source_id: str = Field(..., description="Source node id (User or Asset)")
    target_id: str = Field(..., description="Target node id (Merchant or Asset)")
    amount: float = Field(..., gt=0.0, description="Transfer amount")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Transfer timestamp")
    merchant_category: Optional[MerchantCategory] = Field(
        None, description="Category of the merchant involved in the transaction"
    )

