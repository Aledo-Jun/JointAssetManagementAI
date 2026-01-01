"""Graph manager built on NetworkX for the joint asset management graph."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Tuple

import networkx as nx

from models import Asset, Household, MemberEdge, Merchant, OwnsEdge, TransferEdge, User


class JointGraph:
    """Manage household graphs using a NetworkX MultiDiGraph."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_user(self, user: User) -> None:
        """Add a user node to the graph."""

        self.graph.add_node(user.id, type="User", data=user)

    def add_household(self, household: Household) -> None:
        """Add a household node to the graph."""

        self.graph.add_node(household.id, type="Household", data=household)

    def add_asset(self, asset: Asset) -> None:
        """Add an asset node to the graph."""

        self.graph.add_node(asset.id, type="Asset", data=asset)

    def add_merchant(self, merchant: Merchant) -> None:
        """Add a merchant node to the graph."""

        self.graph.add_node(merchant.id, type="Merchant", data=merchant)

    def merge_users_into_household(self, household_id: str, member_edges: Iterable[MemberEdge]) -> None:
        """Connect users to a household with MEMBER_OF edges."""

        for member in member_edges:
            self.graph.add_edge(member.user_id, household_id, key="MEMBER_OF", role=member.role)

    def connect_ownership(self, ownership_edges: Iterable[OwnsEdge]) -> None:
        """Connect users to assets with OWNS edges."""

        for edge in ownership_edges:
            self.graph.add_edge(edge.user_id, edge.asset_id, key="OWNS", ownership_ratio=edge.ownership_ratio)

    def record_transaction(self, transfer: TransferEdge) -> None:
        """Record a TRANSFERS edge between two nodes."""

        self.graph.add_edge(
            transfer.source_id,
            transfer.target_id,
            key="TRANSFERS",
            amount=transfer.amount,
            timestamp=transfer.timestamp,
            merchant_category=transfer.merchant_category,
        )

    def get_household_embedding(self, household_id: str) -> List[float]:
        """Generate a mock embedding summarizing household state."""

        household_node = self.graph.nodes.get(household_id)
        if not household_node:
            raise ValueError(f"Household {household_id} does not exist")

        # Count members connected via MEMBER_OF edges
        members = [u for u, v, k in self.graph.edges(keys=True) if k == "MEMBER_OF" and v == household_id]
        member_count = len(members)

        # Aggregate user income bracket distribution
        income_counter = Counter()
        for member_id in members:
            user_data: User = self.graph.nodes[member_id].get("data")
            if user_data:
                income_counter[user_data.income_bracket.value] += 1

        # Sum asset balances for assets owned by household members
        asset_balance = 0.0
        for owner, asset_id, key, data in self.graph.edges(keys=True, data=True):
            if key == "OWNS" and owner in members:
                asset_obj: Asset = self.graph.nodes[asset_id].get("data")
                if asset_obj:
                    asset_balance += asset_obj.balance * data.get("ownership_ratio", 0)

        household_data: Household = household_node.get("data")
        total_assets = household_data.total_asset + asset_balance if household_data else asset_balance

        # Construct a simple numeric vector for downstream engines
        embedding = [
            float(member_count),
            float(total_assets),
            float(income_counter.get("high", 0)),
        ]
        return embedding

    def get_transactions(self, source_id: str | None = None) -> List[Tuple[str, str, dict]]:
        """Return a list of transaction edges for auditing or engine input."""

        transactions = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if key == "TRANSFERS" and (source_id is None or u == source_id):
                transactions.append((u, v, data))
        return transactions

