| strategy | retrieval_coverage | precision_at_k | recall_at_k | mrr | ndcg |
| --- | --- | --- | --- | --- | --- |
| hybrid(alpha=0.0) | 0.3514 | 0.2527 | 0.6486 | 0.5302 | 0.5597 |
| hybrid(alpha=0.25) | 0.4324 | 0.3041 | 0.7838 | 0.6865 | 0.7099 |
| hybrid(alpha=0.5) | 0.5270 | 0.3626 | 0.9189 | 0.8018 | 0.8312 |
| hybrid(alpha=0.75) | 0.5405 | 0.3554 | 0.9189 | 0.8311 | 0.8530 |
| hybrid(alpha=1.0) | 0.5946 | 0.3455 | 0.8919 | 0.8050 | 0.8265 |
| hybrid(alpha=0.75)+rerank | 0.4595 | 0.2599 | 0.7568 | 0.5923 | 0.6337 |

**Winner (nDCG, recall tie-break): hybrid(alpha=0.75)**
