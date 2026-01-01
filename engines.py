"""Engines for policy retrieval (RAG) and fraud detection (ML)."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Iterable, List, Sequence

from sklearn.ensemble import IsolationForest

from graph_manager import JointGraph
from models import Asset, Household, Merchant, TransferEdge, User


def _l2_normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def _mean_vectors(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    accum = [0.0 for _ in range(width)]
    for vector in vectors:
        for idx, value in enumerate(vector):
            accum[idx] += value
    return [value / len(vectors) for value in accum]


@dataclass
class TextEmbedder:
    """Lightweight text embedder using hashed token counts."""

    output_dim: int = 32
    seed: int = 17

    def embed(self, text: str) -> List[float]:
        # TODO(deploy): Replace hashing with OpenAI embeddings for production-grade text vectors.
        tokens = re.findall(r"\w+", text.lower())
        vector = [0.0 for _ in range(self.output_dim)]
        for token in tokens:
            token_hash = _stable_hash(f"{self.seed}:{token}")
            bucket = token_hash % self.output_dim
            sign = 1.0 if (token_hash // self.output_dim) % 2 == 0 else -1.0
            vector[bucket] += sign
        return _l2_normalize(vector)


@dataclass
class GraphEmbedder:
    """Simple message-passing embedder for household graphs."""

    output_dim: int = 32
    steps: int = 2
    blend: float = 0.6
    seed: int = 29

    def embed_household(self, graph: JointGraph, household_id: str) -> List[float]:
        # TODO(deploy): Swap this message-passing mock for a torch_geometric GNN encoder.
        node_features = self._initialize_features(graph)
        for _ in range(self.steps):
            node_features = self._propagate(graph, node_features)

        household_feature = node_features.get(household_id)
        if household_feature is None:
            raise ValueError(f"Household {household_id} does not exist")

        neighbors = self._neighbors(graph, household_id)
        if neighbors:
            neighbor_features = _mean_vectors([node_features[n] for n in neighbors])
            blended = [
                self.blend * household_feature[i] + (1.0 - self.blend) * neighbor_features[i]
                for i in range(len(household_feature))
            ]
        else:
            blended = household_feature

        return _l2_normalize(self._project(blended))

    def _initialize_features(self, graph: JointGraph) -> dict[str, List[float]]:
        features: dict[str, List[float]] = {}
        for node_id, data in graph.graph.nodes(data=True):
            node_type = data.get("type")
            node_data = data.get("data")
            base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if node_type == "User" and isinstance(node_data, User):
                base[0] = 1.0
                base[4] = min(node_data.age / 100.0, 1.0)
                base[5] = float({"low": 0.0, "middle": 0.5, "high": 1.0}.get(node_data.income_bracket.value, 0.0))
            elif node_type == "Household" and isinstance(node_data, Household):
                base[1] = 1.0
                base[4] = min(node_data.total_asset / 1_000_000.0, 1.0)
            elif node_type == "Asset" and isinstance(node_data, Asset):
                base[2] = 1.0
                base[4] = min(node_data.balance / 1_000_000.0, 1.0)
            elif node_type == "Merchant":
                base[3] = 1.0
                if hasattr(node_data, "risk_score") and node_data.risk_score is not None:
                    base[4] = float(node_data.risk_score)
            features[node_id] = base
        return features

    def _neighbors(self, graph: JointGraph, node_id: str) -> List[str]:
        predecessors = set(graph.graph.predecessors(node_id))
        successors = set(graph.graph.successors(node_id))
        return list(predecessors | successors)

    def _propagate(
        self, graph: JointGraph, node_features: dict[str, List[float]]
    ) -> dict[str, List[float]]:
        updated: dict[str, List[float]] = {}
        for node_id, feature in node_features.items():
            neighbors = self._neighbors(graph, node_id)
            if neighbors:
                neighbor_features = _mean_vectors([node_features[n] for n in neighbors])
                updated[node_id] = [
                    self.blend * feature[i] + (1.0 - self.blend) * neighbor_features[i]
                    for i in range(len(feature))
                ]
            else:
                updated[node_id] = feature[:]
        return updated

    def _project(self, vector: Sequence[float]) -> List[float]:
        projected = [0.0 for _ in range(self.output_dim)]
        for source_idx, value in enumerate(vector):
            for target_idx in range(self.output_dim):
                weight = self._weight(source_idx, target_idx)
                projected[target_idx] += value * weight
        return projected

    def _weight(self, source_idx: int, target_idx: int) -> float:
        hashed = _stable_hash(f"{self.seed}:{source_idx}:{target_idx}")
        return ((hashed % 2000) / 1000.0) - 1.0


class PolicyRAG:
    """RAG engine backed by in-memory vectors for policy retrieval."""

    def __init__(self, vector_dim: int = 32, persist_directory: str | None = None) -> None:
        _ = persist_directory
        # TODO(deploy): Replace in-memory vectors with a real vector store (e.g., Chroma/FAISS/PGVector).
        self.text_embedder = TextEmbedder(output_dim=vector_dim)
        self.graph_embedder = GraphEmbedder(output_dim=vector_dim)
        self._policies: list[tuple[str, List[float]]] = []

    def ingest_policies(self, documents: list[str]) -> None:
        """Index plain-text policy documents into the vector database."""

        # TODO(deploy): Add chunking + metadata and persist policy vectors for retrieval at scale.
        for document in documents:
            vector = self.text_embedder.embed(document)
            self._policies.append((document, vector))

    def search_policies(self, query: str, k: int = 3) -> List[str]:
        """Query the vector store for the most relevant policy snippets."""

        # TODO(deploy): Replace with vector-store similarity search and metadata filtering.
        query_vector = self.text_embedder.embed(query)
        return self._rank(query_vector, k=k)

    def search_policies_for_household(self, graph: JointGraph, household_id: str, k: int = 3) -> List[str]:
        """Retrieve policy snippets using a household graph embedding as the query."""

        # TODO(deploy): Route to a real RAG pipeline with graph-conditioned retrieval + reranking.
        query_vector = self.graph_embedder.embed_household(graph, household_id)
        return self._rank(query_vector, k=k)

    def _rank(self, query_vector: Sequence[float], k: int) -> List[str]:
        scored = [
            (sum(a * b for a, b in zip(query_vector, vector)), text) for text, vector in self._policies
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in scored[:k]]


class FraudDetector:
    """Fraud detector using graph-aware Isolation Forest anomaly scoring."""

    def __init__(self, graph: JointGraph, random_state: int | None = 42) -> None:
        self.graph = graph
        self.model = IsolationForest(random_state=random_state)

    def _extract_features(self, transaction: TransferEdge) -> List[float]:
        """Map a transaction edge to ML-ready graph features.

        Features capture:
        - amount: raw transfer amount
        - source_node_degree / target_node_degree: connectivity in the graph to detect new merchants or users
        - hour_of_day: temporal pattern for usual spending hours
        """

        source_degree = float(self.graph.graph.degree(transaction.source_id))
        target_degree = float(self.graph.graph.degree(transaction.target_id))
        hour_of_day = float(transaction.timestamp.hour)
        return [float(transaction.amount), source_degree, target_degree, hour_of_day]

    def train(self, normal_transactions: Iterable[TransferEdge]) -> None:
        """Fit the Isolation Forest on historical, non-fraudulent transactions."""

        feature_matrix = [self._extract_features(tx) for tx in normal_transactions]
        if not feature_matrix:
            raise ValueError("No transactions provided for training")
        self.model.fit(feature_matrix)

    def detect_anomaly(self, transaction: TransferEdge, merchant: Merchant | None = None) -> int:
        """Predict whether a transaction is anomalous.

        Returns 1 for normal transactions and -1 for anomalies (IsolationForest convention).
        The merchant parameter is accepted for API compatibility with future enrichments.
        """

        if not hasattr(self.model, "estimators_"):
            raise RuntimeError("FraudDetector model is not trained. Call train() first.")

        features = self._extract_features(transaction)
        prediction = int(self.model.predict([features])[0])
        return prediction
