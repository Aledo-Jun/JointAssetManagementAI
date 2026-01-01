"""Simulation entrypoint demonstrating graph-aware RAG and fraud detection."""
from __future__ import annotations

import random
from datetime import date, datetime, timedelta, UTC

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


def build_sample_graph() -> JointGraph:
    """Build a sample household graph with assets, merchants, and history."""

    graph = JointGraph()

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

    grocery = Merchant(id="merchant_grocery", category=MerchantCategory.FOOD)
    risky_merchant = Merchant(id="merchant_risky", category=MerchantCategory.HIGH_RISK)
    graph.add_merchant(grocery)
    graph.add_merchant(risky_merchant)

    base_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    for i in range(100):
        tx = TransferEdge(
            source_id=user_a.id if i % 2 == 0 else user_b.id,
            target_id=grocery.id,
            amount=round(max(random.normalvariate(80, 15), 10), 2),
            merchant_category=grocery.category,
            timestamp=base_time + timedelta(hours=i % 24),
        )
        graph.record_transaction(tx)

    return graph


def run_demo() -> None:
    """Demonstrate policy retrieval with text and household graph embeddings."""

    graph = build_sample_graph()
    household_id = "household_ab"

    policy_rag = PolicyRAG()  # 
    policy_rag.ingest_policies(  # TODO: replace with real policies by crawling
        [
            "First-time homebuyer support applies to households with assets under 100,000.",
            "Couples tax relief is available when combined assets stay below 1,000,000.",
            "Rent assistance applies when monthly rent is under 700 and deposit is under 10,000.",
            "High-risk merchants trigger manual review for transfers above 5,000.",
        ]
    )

    text_query = "joint household tax relief"
    text_results = policy_rag.search_policies(text_query, k=2)
    graph_results = policy_rag.search_policies_for_household(graph, household_id, k=2)

    print("Text query:", text_query)
    for policy in text_results:
        print(" -", policy)

    print("\nHousehold graph query:", household_id)
    for policy in graph_results:
        print(" -", policy)

    fraud_detector = FraudDetector(graph)
    normal_transactions = [
        TransferEdge(
            source_id="user_a",
            target_id="merchant_grocery",
            amount=round(max(random.normalvariate(80, 15), 10), 2),
            merchant_category=MerchantCategory.FOOD,
            timestamp=datetime.now(UTC) - timedelta(hours=offset),
        )
        for offset in range(80)
    ]
    fraud_detector.train(normal_transactions)

    transactions = [
        TransferEdge(
            source_id="user_a",
            target_id="merchant_grocery",
            amount=90.0,
            merchant_category=MerchantCategory.FOOD,
        ),
        TransferEdge(
            source_id="user_b",
            target_id="merchant_risky",
            amount=50000,
            merchant_category=MerchantCategory.HIGH_RISK,
        ),
    ]
    for tx in transactions:
        graph.record_transaction(tx)

    print("\nFraud predictions (1=normal, -1=anomaly):")
    for tx in transactions:
        prediction = fraud_detector.detect_anomaly(tx)
        print(f" - {tx.source_id} -> {tx.target_id}: {prediction}")


if __name__ == "__main__":
    run_demo()
