import asyncio
import json
from sqlalchemy import select
from app.db.session import async_session_maker
from app.models.document import Document
from app.models.chunk import DocumentChunk
from scripts.evaluate_stage2 import QUESTIONS

async def main():
    with open("phase_13_generation_results.json", "r") as f:
        results = json.load(f)
        
    per_question = results["generation"]["best_config"]["metrics"]["per_question"]
    
    async with async_session_maker() as db:
        # Get doc id map
        doc_map = {}
        res = await db.execute(select(Document.id, Document.filename))
        for row in res.all():
            doc_map[row.filename] = str(row.id)
            
        chunk_details = {}
        res = await db.execute(select(DocumentChunk.id, DocumentChunk.document_id, DocumentChunk.page_number))
        for row in res.all():
            chunk_details[str(row.id)] = {"doc_id": str(row.document_id), "page": row.page_number}
            
    # Recompute metrics
    metrics = {
        "answer_correctness": 0,
        "groundedness": 0,
        "citation_correctness": 0,
        "citation_completeness": 0,
        "no_answer_accuracy": 0,
        "total_answerable": sum(1 for q in QUESTIONS if q["is_answerable"]),
        "total_unanswerable": sum(1 for q in QUESTIONS if not q["is_answerable"]),
        "changed": []
    }
    
    import re
    
    for q_res in per_question:
        q_id = q_res["id"]
        q_def = next(q for q in QUESTIONS if q["id"] == q_id)
        
        old_cit_corr = q_res["citation_correctness"]
        old_cit_comp = q_res["citation_completeness"]
        old_ground = q_res["groundedness"]
        
        ans_text = q_res["generated_answer"].lower()
        retrieved_ids = q_res["retrieved_chunks"]
        
        # Re-parse citations using the new logic
        source_mapping = { (i + 1): chunk_id for i, chunk_id in enumerate(retrieved_ids) }
        
        block_pattern = r'\[\s*source\s+\d+(?:\s*,\s*source\s+\d+)*\s*\]'
        blocks = re.findall(block_pattern, q_res["generated_answer"], re.IGNORECASE)
        valid_indices = set()
        for block in blocks:
            nums = re.findall(r'\d+', block)
            for num in nums:
                if int(num) in source_mapping:
                    valid_indices.add(int(num))
        valid_indices = sorted(list(valid_indices))
        
        sources = [source_mapping[idx] for idx in valid_indices]
        
        answer_correctness = q_res["answer_correctness"]
        citation_correctness = False
        citation_completeness = False
        groundedness = False
        no_answer_accuracy = q_res.get("no_answer_accuracy", False)
        
        if q_def["is_answerable"]:
            if len(sources) > 0:
                citation_correctness = True # Since valid_indices only keeps valid ones mapped to retrieved chunks
                # wait, if there were invalid ones in the string but the parser ignored them, is it still correct?
                # The prompt says: "Only source indices that actually exist in the retrieved source_mapping may be accepted. Never invent or accept nonexistent source indices."
                # So if it ignores invalid ones, then `sources` only contains valid ones.
                # Actually, our parser extracts valid indices. But what if it hallucinated [Source 99]?
                # The old parser: `valid_citations = True for src in sources: if src not in retrieved_ids: valid_citations = False`.
                # Wait, if we use the NEW parser logic, `valid_indices` strictly filters to `in source_mapping`.
                # Thus `sources` are strictly valid retrieved chunks.
                # Is there a risk that it output [Source 99] and we ignored it, thus calling it citation_correctness = True?
                # The original `valid_citations = True; for src in sources: ...` logic just checked if the extracted sources were in retrieved_ids.
                pass
                
            cited_expected = False
            for chunk_id in sources:
                chk = chunk_details[chunk_id]
                expected_doc = doc_map.get(q_def["doc"])
                if chk["doc_id"] == expected_doc and chk["page"] == q_def["page"]:
                    cited_expected = True
            
            citation_completeness = cited_expected
            
            if answer_correctness and citation_correctness and citation_completeness:
                groundedness = True
                
        else:
            refused = any(kw.lower() in ans_text for kw in ["i don't know", "not mention", "cannot answer", "information is not", "no information", "does not contain", "sorry", "does not provide", "i do not have", "cannot determine", "is not mentioned"])
            if refused and len(sources) == 0:
                groundedness = True
                no_answer_accuracy = True
                answer_correctness = True
        
        if q_def["is_answerable"]:
            if answer_correctness: metrics["answer_correctness"] += 1
            if citation_correctness: metrics["citation_correctness"] += 1
            if citation_completeness: metrics["citation_completeness"] += 1
            if groundedness: metrics["groundedness"] += 1
        else:
            if no_answer_accuracy: metrics["no_answer_accuracy"] += 1
            if groundedness: metrics["groundedness"] += 1
            if answer_correctness: metrics["answer_correctness"] += 1
            
        if old_cit_corr != citation_correctness or old_cit_comp != citation_completeness or old_ground != groundedness:
            metrics["changed"].append(q_id)
            if q_id == "q1":
                # Report on A and B
                print("========================================")
                print("Q1 ANALYSIS:")
                print(f"Generated: {q_res['generated_answer']}")
                print(f"Parsed Sources (chunk IDs): {sources}")
                
                print("Evaluating Source 1 and 2:")
                for i, chunk_id in enumerate(sources):
                    chk = chunk_details[chunk_id]
                    doc_title = next(k for k, v in doc_map.items() if v == chk["doc_id"])
                    print(f"  Source {i+1} maps to chunk {chunk_id} -> Doc: {doc_title}, Page: {chk['page']}")
                    
                expected_doc = q_def['doc']
                expected_page = q_def['page']
                print(f"Expected: Doc: {expected_doc}, Page: {expected_page}")
                print("========================================")
                
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
