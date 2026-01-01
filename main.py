"""Simulation entrypoint demonstrating RAG and fraud detection integration."""
from __future__ import annotations

import random
from datetime import date, datetime, timedelta

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
    """Build a sample graph and run RAG + ML checks."""

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

    # Ingest sample Korean policy texts into the RAG store
    policy_rag = PolicyRAG()
    policy_rag.ingest_policies(
        [
            "신혼부부 전세자금 대출 조건: 소득 합산 1억원 이하, 무주택 기준, 보증기관 보증 필요.",
            "생애최초 주택구입 혜택: 취득세 감면 및 저리 대출 지원, 주택가격 6억원 이하.",
            "주거안정 월세 대출: 월세 70만원 이하, 보증금 1억원 이하 세입자 대상.",
            "다자녀 가구 주택 구입 지원: 자녀 3명 이상 가구 대상 추가 대출 한도 제공.",
        ]
    )

    # Create normal historical transactions and train the Isolation Forest
    normal_transactions: list[TransferEdge] = []
    base_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    for i in range(100):
        tx = TransferEdge(
            source_id=user_a.id if i % 2 == 0 else user_b.id,
            target_id=grocery.id,
            amount=round(max(random.normalvariate(80, 15), 10), 2),
            merchant_category=grocery.category,
            timestamp=base_time + timedelta(hours=i % 24),
        )
        graph.record_transaction(tx)
        normal_transactions.append(tx)

    fraud_detector = FraudDetector(graph)
    fraud_detector.train(normal_transactions)

    # Evaluate a normal transaction and an outlier
    grocery_tx = TransferEdge(
        source_id=user_a.id,
        target_id=grocery.id,
        amount=90.0,
        merchant_category=grocery.category,
    )
    graph.record_transaction(grocery_tx)

    suspicious_tx = TransferEdge(
        source_id=user_b.id,
        target_id=risky_merchant.id,
        amount=50000,
        merchant_category=risky_merchant.category,
    )
    graph.record_transaction(suspicious_tx)

    # Run RAG query and fraud detection
    rag_results = policy_rag.search_policies("우리 부부 자산 상태에 맞는 전세 대출 찾아줘")

    normal_prediction = fraud_detector.detect_anomaly(grocery_tx, merchant=grocery)
    suspicious_prediction = fraud_detector.detect_anomaly(suspicious_tx, merchant=risky_merchant)

    # Output results
    print("RAG search results:")
    for policy in rag_results:
        print(" -", policy)

    print("\nFraud predictions (1=normal, -1=anomaly):")
    print("Normal transaction:", normal_prediction)
    print("Suspicious transaction:", suspicious_prediction)


if __name__ == "__main__":
    run_simulation()

