Graph Data Schema (Heterogeneous)
Nodes (Entities)
| Node Type | Attributes | Description |
|---|---|---|
| User | id, name, age, income_bracket | 개별 사용자 (남편, 아내) |
| Household | id, formation_date, total_asset | 두 유저를 묶는 가구/공동체 노드 |
| Asset | id, type (Account, RealEstate, Stock), balance | 자산 노드 |
| Merchant | id, category (Food, Travel), risk_score | 거래처/가맹점 |
Edges (Relationships)
| Edge Type | Source | Target | Attributes | Description |
|---|---|---|---|---|
| MEMBER_OF | User | Household | role | 가구 구성원 관계 |
| OWNS | User | Asset | ownership_ratio | 자산 소유 관계 |
| TRANSFERS | User/Asset | Merchant/Asset | amount, timestamp | 자금 이체/결제 (Time-sensitive) |
