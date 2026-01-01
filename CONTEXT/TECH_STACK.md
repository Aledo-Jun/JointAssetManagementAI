Tech Stack & Constraints
Backend & Logic
 * Language: Python 3.10+
 * Graph Library: NetworkX (for prototype/skeleton) -> 추후 Neo4j or PyG로 마이그레이션 예정.
 * Data Model: Pydantic for data validation.
 * RAG Interface: LangChain style abstraction (Mock classes required).
Architecture Pattern
 * Service Layer Pattern: 로직과 데이터 접근 분리.
 * Manager Classes: GraphManager, AdvisorEngine, PolicyRAG.
