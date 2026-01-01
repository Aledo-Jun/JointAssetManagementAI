"""Engines for policy retrieval (RAG) and fraud detection (ML)."""
from __future__ import annotations

from typing import Iterable, List

from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from sklearn.ensemble import IsolationForest

from graph_manager import JointGraph
from models import Merchant, TransferEdge


class PolicyRAG:
    """RAG engine backed by a Chroma vector store and OpenAI embeddings."""

    def __init__(self, persist_directory: str | None = None) -> None:
        # Chroma can operate fully in-memory; a persist directory enables local caching.
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(
            collection_name="household_policy_rag",
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
            client_settings=Settings(anonymized_telemetry=False),
        )

    def ingest_policies(self, documents: list[str]) -> None:
        """Index plain-text policy documents into the vector database."""

        # LangChain handles splitting into chunks via add_texts; chunking keeps retrieval granular.
        self.vector_store.add_texts(documents)

    def search_policies(self, query: str, k: int = 3) -> List[str]:
        """Query the vector store for the most relevant policy snippets."""

        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]


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

