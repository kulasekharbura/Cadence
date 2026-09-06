# RAG Pipeline Evaluation Report

## 1. Objective and Limitations
The objective of this evaluation was to measure and baseline the performance of the RAG implementation. 

**Important Limitations:**
- The evaluation was conducted on a deterministic **20-question benchmark** against **3 deterministic PDFs** (`cadence_history.pdf`, `cadence_geography.pdf`, and `employee_handbook.pdf`).
- The dataset consisted of **17 answerable questions** and **3 unanswerable questions**.
- Retrieval and generation were evaluated separately.
- **Answer Correctness** was evaluated using predefined deterministic/rule-based exact-match keyword criteria, not semantic equivalence.
- **Groundedness** was defined as a composite heuristic of Citation Correctness, Citation Completeness, and Answer Correctness—not via human or LLM-as-a-judge semantic evaluation.
- These results demonstrate the system's correct behavior under strict testing constraints but **should not be generalized as universal factual accuracy** across arbitrary documents.

## 2. Retrieval Parameter Sweep

We tested combinations of `top_k` ∈ [3, 5, 8] and `distance_threshold` ∈ [0.4, 0.5, 0.6] using PostgreSQL pgvector.

| Configuration | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 | MRR |
|---|---|---|---|---|
| k=3, t=0.4 | 14/17 | 16/17 | 16/17 | 0.882 |
| k=3, t=0.5 | 14/17 | 16/17 | 16/17 | 0.882 |
| k=3, t=0.6 | 14/17 | 16/17 | 16/17 | 0.882 |
| k=5, t=0.4 | 14/17 | 16/17 | 16/17 | 0.882 |
| **k=5, t=0.5** | **14/17** | **16/17** | **17/17** | **0.894** |
| **k=5, t=0.6** | **14/17** | **16/17** | **17/17** | **0.894** |
| k=8, t=0.4 | 14/17 | 16/17 | 16/17 | 0.882 |
| k=8, t=0.5 | 14/17 | 16/17 | 17/17 | 0.894 |
| k=8, t=0.6 | 14/17 | 16/17 | 17/17 | 0.894 |

### Distance Threshold Calibration
The retrieval sweep showed that `top_k=5` with thresholds `0.5` and `0.6` achieved identical maximum benchmark performance. Production was subsequently set to `0.6` because live diagnosis identified a relevant resume chunk at cosine distance 0.5273 that was rejected by the previous 0.5 threshold.

## 3. Generation Evaluation Results

On this deterministic 20-question benchmark, the system achieved 100% under the defined evaluation criteria.

- **Answer Correctness**: 17/17 (100%) - All factual questions contained the expected ground-truth keywords.
- **No-Answer Accuracy**: 3/3 (100%) - All unanswerable questions were correctly refused using the mandated refusal language.
- **Citation Completeness**: 17/17 (100%) - All expected ground-truth documents and pages were present in the LLM's final citations.
- **Citation Correctness**: 17/17 (100%) - All sources cited mapped to a chunk_id that was actually present in the context.
- **Groundedness**: 20/20 (100%) - All answers were correctly grounded in the provided documents according to the composite heuristic.
