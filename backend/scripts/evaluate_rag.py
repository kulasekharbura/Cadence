import asyncio
import json
import os
import datetime
from pathlib import Path
from unittest.mock import patch
from typing import List, Dict, Any

from app.core.config import settings
from app.db.session import async_session_maker
from app.models.document import Document
from app.models.chunk import DocumentChunk
from app.models.conversation import Conversation
from app.services.embedding.gemini_provider import GeminiEmbeddingProvider
from app.services.retrieval_service import RetrievalService

def create_pdf(path: str, pages_text: List[str]):
    # Since fitz is deprecated, we will just use PyMuPDF directly or mocking it if we need, but the previous code used fitz
    import pymupdf
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 50), text)
    doc.save(path)
    doc.close()

# The 20 questions
QUESTIONS = [
    # A. Direct Factual
    {"id": "q1", "cat": "A", "q": "What is the capital of Cadence?", "expected": ["nova"], "doc": "cadence_history.pdf", "page": 1, "is_answerable": True},
    {"id": "q2", "cat": "A", "q": "What year was the Great Accord signed?", "expected": ["1492"], "doc": "cadence_history.pdf", "page": 2, "is_answerable": True},
    {"id": "q3", "cat": "A", "q": "What is the primary currency of Cadence?", "expected": ["caden"], "doc": "cadence_geography.pdf", "page": 1, "is_answerable": True},
    {"id": "q4", "cat": "A", "q": "How many hours of PTO do employees get in their first year?", "expected": ["120"], "doc": "employee_handbook.pdf", "page": 1, "is_answerable": True},
    {"id": "q5", "cat": "A", "q": "Who is the CEO of Cadence Corp?", "expected": ["elara"], "doc": "employee_handbook.pdf", "page": 2, "is_answerable": True},
    
    # B. Multi-page
    {"id": "q6", "cat": "B", "q": "Describe the events of the Silent Winter.", "expected": ["blizzard"], "doc": "cadence_history.pdf", "page": 3, "is_answerable": True},
    {"id": "q7", "cat": "B", "q": "What are the core values listed on page 3 of the handbook?", "expected": ["integrity"], "doc": "employee_handbook.pdf", "page": 3, "is_answerable": True},
    {"id": "q8", "cat": "B", "q": "What is the population of the Northern Province?", "expected": ["2.5"], "doc": "cadence_geography.pdf", "page": 2, "is_answerable": True},
    {"id": "q9", "cat": "B", "q": "What mountain range separates the East and West provinces?", "expected": ["obsidian"], "doc": "cadence_geography.pdf", "page": 3, "is_answerable": True},
    {"id": "q10", "cat": "B", "q": "When was the first university established in Cadence?", "expected": ["1520"], "doc": "cadence_history.pdf", "page": 4, "is_answerable": True},

    # C. Multi-document
    {"id": "q11", "cat": "C", "q": "What is the primary export of Cadence?", "expected": ["lumina"], "doc": "cadence_geography.pdf", "page": 4, "is_answerable": True},
    {"id": "q12", "cat": "C", "q": "Are employees allowed to work remotely from the Northern Province?", "expected": ["yes", "manager"], "doc": "employee_handbook.pdf", "page": 4, "is_answerable": True}, 
    
    # D. No-answer
    {"id": "q13", "cat": "D", "q": "Who won the Cadence sports championship in 2020?", "expected": ["not", "mention"], "doc": None, "page": None, "is_answerable": False},
    {"id": "q14", "cat": "D", "q": "What is the recipe for Cadence pie?", "expected": ["not", "provide"], "doc": None, "page": None, "is_answerable": False},
    {"id": "q15", "cat": "D", "q": "How much revenue did Cadence Corp make last year?", "expected": ["not", "mention"], "doc": None, "page": None, "is_answerable": False},
    
    # E. Ambiguous/Distractor
    {"id": "q16", "cat": "E", "q": "Where is the secondary office located?", "expected": ["solaris"], "doc": "employee_handbook.pdf", "page": 5, "is_answerable": True},
    {"id": "q17", "cat": "E", "q": "What is the name of the modern river, not the historical one?", "expected": ["silverstream"], "doc": "cadence_geography.pdf", "page": 5, "is_answerable": True},
    
    # F. Follow-up
    {"id": "q18", "cat": "F", "q": "What is its population?", "expected": ["1.2"], "doc": "cadence_geography.pdf", "page": 1, "is_answerable": True},
    
    # A few more standard questions to round up to 20
    {"id": "q19", "cat": "A", "q": "What is the dress code?", "expected": ["business", "casual"], "doc": "employee_handbook.pdf", "page": 6, "is_answerable": True},
    {"id": "q20", "cat": "A", "q": "Who leads the HR department?", "expected": ["marcus"], "doc": "employee_handbook.pdf", "page": 6, "is_answerable": True},
]

DOCS = {
    "cadence_history.pdf": [
        "The capital of Cadence is Nova.",
        "The Great Accord was signed in 1492, uniting the provinces.",
        "During the Silent Winter of 1503, a massive blizzard covered the land for six months.",
        "The first university, Cadence Academy, was established in 1520. Historical exports included wheat.",
        "The Old River dried up centuries ago."
    ],
    "cadence_geography.pdf": [
        "The primary currency of Cadence is the Caden. The capital Nova has a population of 1.2 million.",
        "The Northern Province is the coldest region with a population of 2.5 million.",
        "The Obsidian Peaks mountain range separates the East and West provinces.",
        "Today, the primary export of Cadence is Lumina crystals, mined in the East.",
        "The modern Silverstream river provides water to the central regions."
    ],
    "employee_handbook.pdf": [
        "Employees receive 120 hours of PTO in their first year.",
        "Our CEO is Elara Vance. The Primary office is located at Nova Tower.",
        "Our core values are Integrity, Innovation, and Inclusion.",
        "Remote work is allowed from the Northern Province with manager approval.",
        "The secondary office is located at Solaris Plaza.",
        "The dress code is business casual. The HR department is led by Marcus Thorne."
    ]
}

async def generate_dataset():
    tmp_dir = Path("/tmp/eval_docs")
    tmp_dir.mkdir(exist_ok=True)
    
    doc_paths = {}
    for filename, pages in DOCS.items():
        path = tmp_dir / filename
        create_pdf(str(path), pages)
        doc_paths[filename] = str(path)
    return doc_paths

async def get_or_create_doc(filepath: str):
    from httpx import AsyncClient, ASGITransport
    from app.main import app
    filename = Path(filepath).name
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with open(filepath, "rb") as f:
            res = await client.post("/api/v1/documents/upload", files={"file": (filename, f, "application/pdf")})
            res.raise_for_status()
            doc_id = res.json()["id"]
            
        for _ in range(60):
            res = await client.get(f"/api/v1/documents/{doc_id}")
            if res.json()["processing_status"] == "READY":
                return res.json()
            await asyncio.sleep(0.5)
            
        raise Exception(f"Doc {filename} failed to process")

async def establish_ground_truth(db, doc_id_map):
    for q in QUESTIONS:
        if q["is_answerable"]:
            doc_id = doc_id_map[q["doc"]]
            res = await db.execute(DocumentChunk.__table__.select().where(
                (DocumentChunk.document_id == doc_id) & 
                (DocumentChunk.page_number == q["page"])
            ))
            chunks = res.fetchall()
            q["relevant_chunk_ids"] = [c.id for c in chunks]
        else:
            q["relevant_chunk_ids"] = []

async def evaluate_retrieval(db, top_k, threshold, q_embeddings, doc_id_map):
    retrieval_service = RetrievalService(db)
    metrics = {
        "recall_1": 0.0, "recall_3": 0.0, "recall_5": 0.0,
        "hit_1": 0, "hit_3": 0, "hit_5": 0,
        "mrr": 0.0,
        "failures": []
    }
    
    answerable_count = 0
    for i, q in enumerate(QUESTIONS):
        if not q["is_answerable"]:
            continue
        
        answerable_count += 1
        q_text = q["q"]
        if q["id"] == "q18":
            q_text = "What is the capital of Cadence? " + q_text 

        query_emb = q_embeddings[q["id"]]
        
        # Patch EmbeddingService to reuse the pre-computed embeddings
        async def mock_embed(chunks):
            return [{"embedding": query_emb}]
            
        with patch.object(retrieval_service.embedding_service, 'generate_embeddings_for_chunks', side_effect=mock_embed):
            retrieved_chunks = await retrieval_service.retrieve_chunks(
                query=q_text,
                document_ids=list(doc_id_map.values()),
                top_k=top_k,
                distance_threshold=threshold
            )
        
        retrieved_ids = [c.chunk_id for c in retrieved_chunks]
        rel_ids = q["relevant_chunk_ids"]
        
        for k in [1, 3, 5]:
            k_ids = retrieved_ids[:k]
            overlap = set(k_ids).intersection(set(rel_ids))
            if len(overlap) > 0:
                metrics[f"hit_{k}"] += 1
            metrics[f"recall_{k}"] += len(overlap) / len(rel_ids) if len(rel_ids) > 0 else 0
        
        for rank_idx, r_id in enumerate(retrieved_ids):
            if r_id in rel_ids:
                metrics["mrr"] += 1.0 / (rank_idx + 1)
                break

    # Normalize recall and MRR
    if answerable_count > 0:
        for k in [1, 3, 5]:
            metrics[f"recall_{k}"] /= answerable_count
        metrics["mrr"] /= answerable_count
        
    return metrics

async def evaluate_generation(db, top_k, threshold, q_embeddings, doc_id_map):
    # We create one conversation with all 3 docs for all tests except if we need filtering.
    res = await db.execute(Conversation.__table__.insert().values(title="Eval").returning(Conversation.id))
    conv_id = res.scalar()
    
    from app.models.conversation import ConversationDocument
    for doc_id in doc_id_map.values():
        await db.execute(ConversationDocument.__table__.insert().values(conversation_id=conv_id, document_id=doc_id))
    await db.commit()
    
    metrics = {
        "answer_correctness": 0,
        "groundedness": 0,
        "citation_correctness": 0,
        "citation_completeness": 0,
        "total_answerable": sum(1 for q in QUESTIONS if q["is_answerable"]),
        "total_unanswerable": sum(1 for q in QUESTIONS if not q["is_answerable"]),
        "failures": []
    }
    
    retrieval_service = RetrievalService(db)
    
    for i, q in enumerate(QUESTIONS):
        query_emb = q_embeddings[q["id"]]
        async def mock_embed(chunks):
            return [{"embedding": query_emb}]
            
        with patch.object(retrieval_service.embedding_service, 'generate_embeddings_for_chunks', side_effect=mock_embed):
            retrieved_chunks = await retrieval_service.retrieve_chunks(
                query=q["q"],
                document_ids=list(doc_id_map.values()),
                top_k=top_k,
                distance_threshold=threshold
            )
        retrieved_ids = [str(c.chunk_id) for c in retrieved_chunks]
        
        with patch("app.core.config.settings.RETRIEVAL_TOP_K", top_k), \
             patch("app.core.config.settings.RETRIEVAL_DISTANCE_THRESHOLD", threshold):
            
            try:
                from httpx import AsyncClient, ASGITransport
                from app.main import app
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test", timeout=30.0) as client:
                    chat_req = {"conversation_id": str(conv_id), "message": q["q"]}
                    
                    import asyncio
                    await asyncio.sleep(4)  # Rate limit mitigation for Gemini Free Tier
                    
                    chat_res = await client.post("/api/v1/chat", json=chat_req)
                    chat_res.raise_for_status()
                    answer_data = chat_res.json()
                    
                    ans_text = answer_data["answer"].lower()
                    sources = answer_data.get("sources", [])
                
                if q["is_answerable"]:
                    correct = all(kw.lower() in ans_text for kw in q["expected"])
                    if correct:
                        metrics["answer_correctness"] += 1
                        
                    valid_citations = True
                    for src in sources:
                        if src["chunk_id"] not in retrieved_ids:
                            valid_citations = False
                    if valid_citations and len(sources) > 0:
                        metrics["citation_correctness"] += 1
                        
                    cited_expected = False
                    for src in sources:
                        if src["document_id"] == str(doc_id_map[q["doc"]]) and src["page_number"] == q["page"]:
                            cited_expected = True
                    if cited_expected:
                        metrics["citation_completeness"] += 1
                        
                    if correct and valid_citations and cited_expected:
                        metrics["groundedness"] += 1
                        
                    if not correct or not valid_citations or not cited_expected:
                        metrics["failures"].append({
                            "question": q["q"],
                            "type": "generation/citation",
                            "top_k": top_k,
                            "threshold": threshold,
                            "generated": ans_text,
                            "expected": q["expected"]
                        })
                        
                else:
                    refused = any(kw.lower() in ans_text for kw in ["i don't know", "not mention", "cannot answer", "information is not", "no information", "does not contain", "sorry", "does not provide", "i do not have"])
                    if refused and len(sources) == 0:
                        metrics["groundedness"] += 1
                        metrics["answer_correctness"] += 1
                    else:
                        metrics["failures"].append({
                            "question": q["q"],
                            "type": "hallucination",
                            "generated": ans_text,
                            "top_k": top_k,
                            "threshold": threshold
                        })
                        
            except Exception as e:
                metrics["failures"].append({
                    "question": q["q"],
                    "type": f"error: {str(e)}"
                })

    return metrics

async def main():
    try:
        doc_paths = await generate_dataset()
        
        async with async_session_maker() as db:
            doc_id_map = {}
            for doc_name, path in doc_paths.items():
                print(f"Uploading {doc_name}...")
                data = await get_or_create_doc(path)
                doc_id_map[doc_name] = data["id"]
                
            print("Establishing Ground Truth...")
            await establish_ground_truth(db, doc_id_map)
            
            provider = GeminiEmbeddingProvider()
            q_texts = [q["q"] if q["id"] != "q18" else "What is the capital of Cadence? " + q["q"] for q in QUESTIONS]
            q_embs = await provider.get_embeddings(q_texts)
            q_embeddings = {q["id"]: emb for q, emb in zip(QUESTIONS, q_embs)}
            
            results = {"retrieval": {}, "generation": {}}
            
            # PHASE A: Retrieval Sweep
            best_mrr = -1
            best_config = None
            
            for k in [3, 5, 8]:
                for t in [0.4, 0.5, 0.6]:
                    key = f"sweep_k{k}_t{t}"
                    print(f"Running Retrieval Sweep {key}...")
                    metrics = await evaluate_retrieval(db, k, t, q_embeddings, doc_id_map)
                    results["retrieval"][key] = metrics
                    print(f"{key} Done. MRR: {metrics['mrr']}")
                    if metrics["mrr"] > best_mrr:
                        best_mrr = metrics["mrr"]
                        best_config = (k, t)
            
            print(f"Best Configuration determined: top_k={best_config[0]}, threshold={best_config[1]}")
            
            # PHASE B: Generation on Best Config
            print(f"Skipping Generation Phase as per Stage 1 restrictions.")
            # gen_metrics = await evaluate_generation(db, best_config[0], best_config[1], q_embeddings, doc_id_map)
            # results["generation"]["best_config"] = {
            #     "top_k": best_config[0],
            #     "threshold": best_config[1],
            #     "metrics": gen_metrics
            # }
            
            with open("phase_13_results.json", "w") as f:
                json.dump(results, f, indent=2)
                
            # Create CSV files
            import csv
            with open("retrieval_results.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Config", "MRR", "Hit@1", "Hit@3", "Hit@5", "Recall@1", "Recall@3", "Recall@5"])
                for key, m in results["retrieval"].items():
                    writer.writerow([key, m["mrr"], m["hit_1"], m["hit_3"], m["hit_5"], m["recall_1"], m["recall_3"], m["recall_5"]])
                    
    except Exception as e:
        print(f"Global Evaluation Failure: {e}")

if __name__ == "__main__":
    asyncio.run(main())
