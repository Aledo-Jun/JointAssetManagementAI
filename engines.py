"""Mock engines for policy retrieval and fraud detection."""
from __future__ import annotations

from typing import List

from models import Merchant, TransferEdge


class PolicyRAG:
    """Mock RAG engine returning policy recommendations based on embeddings."""

    def recommend_policies(self, household_embedding: List[float]) -> List[str]:
        """Return dummy policies informed by embedding magnitude."""

        member_count, total_assets, high_income_count = household_embedding
        recommendations = [
            "Consider joint tax filing benefits.",
            "Review housing subsidies for families.",
        ]
        if total_assets > 100000:
            recommendations.append("Evaluate wealth management options for high net worth households.")
        if high_income_count > 0:
            recommendations.append("Explore investment diversification for high earners.")
        if member_count > 2:
            recommendations.append("Assess family insurance packages.")
        return recommendations


class FraudDetector:
    """Simple fraud detector using merchant risk categories."""

    def is_suspicious(self, transaction: TransferEdge, merchant: Merchant | None = None) -> bool:
        """Flag a transaction if the merchant category is marked as high risk."""

        category = transaction.merchant_category
        if merchant is not None:
            category = merchant.category
        return category is not None and category.lower() == "high_risk"

