"""Simulation entrypoint demonstrating core graph logic and engines."""
from __future__ import annotations

from datetime import date

from engines import FraudDetector, PolicyRAG
from graph_manager import JointGraph
from models import (
    Asset,
    AssetType,
    Household,
    IncomeBracket,
    MemberEdge,
    Merchant,
    MerchantCategory,
    OwnsEdge,
    TransferEdge,
    User,
)


def run_simulation() -> None:
    """Build a sample graph and run mock policy and fraud checks."""

    graph = JointGraph()

    # Create users and household
    user_a = User(id="user_a", name="Alice", age=32, income_bracket=IncomeBracket.MIDDLE)
    user_b = User(id="user_b", name="Bob", age=35, income_bracket=IncomeBracket.HIGH)
    household = Household(id="household_ab", formation_date=date.today(), total_asset=50000)

    graph.add_user(user_a)
    graph.add_user(user_b)
    graph.add_household(household)
    graph.merge_users_into_household(
        household.id,
        [
            MemberEdge(user_id=user_a.id, household_id=household.id, role="partner"),
            MemberEdge(user_id=user_b.id, household_id=household.id, role="partner"),
        ],
    )

    # Create assets and link ownership
    savings = Asset(id="asset_savings", type=AssetType.ACCOUNT, balance=75000)
    car = Asset(id="asset_car", type=AssetType.REALESTATE, balance=20000)
    graph.add_asset(savings)
    graph.add_asset(car)
    graph.connect_ownership(
        [
            OwnsEdge(user_id=user_a.id, asset_id=savings.id, ownership_ratio=0.6),
            OwnsEdge(user_id=user_b.id, asset_id=savings.id, ownership_ratio=0.4),
            OwnsEdge(user_id=user_b.id, asset_id=car.id, ownership_ratio=1.0),
        ]
    )

    # Merchants and transactions
    grocery = Merchant(id="merchant_grocery", category=MerchantCategory.FOOD)
    risky_merchant = Merchant(id="merchant_risky", category=MerchantCategory.HIGH_RISK)
    graph.add_merchant(grocery)
    graph.add_merchant(risky_merchant)

    grocery_tx = TransferEdge(
        source_id=user_a.id,
        target_id=grocery.id,
        amount=120.5,
        merchant_category=grocery.category,
    )
    suspicious_tx = TransferEdge(
        source_id=user_b.id,
        target_id=risky_merchant.id,
        amount=5000,
        merchant_category=risky_merchant.category,
    )
    graph.record_transaction(grocery_tx)
    graph.record_transaction(suspicious_tx)

    # Engines
    embedding = graph.get_household_embedding(household.id)
    policy_rag = PolicyRAG()
    recommendations = policy_rag.recommend_policies(embedding)

    fraud_detector = FraudDetector()
    is_flagged = fraud_detector.is_suspicious(suspicious_tx, merchant=risky_merchant)

    # Output results
    print("Household embedding:", embedding)
    print("Policy recommendations:")
    for policy in recommendations:
        print(" -", policy)
    print("Suspicious transaction flagged:", is_flagged)


if __name__ == "__main__":
    run_simulation()

