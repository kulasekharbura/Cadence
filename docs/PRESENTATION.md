# Presentation Outline & Script Plan

*Note: This document is a detailed presentation script and slide plan designed to serve as preparation for the actual PowerPoint presentation. It is not the final presentation deliverable itself.*

## Slide 1: Title Slide
- **Visual**: Project Name ("AI-Powered Document Q&A System") and Team Name / Cadence Project context.
- **Talking Points**: Introduce the project, the primary goal (extracting accurate, verifiable answers from complex documents), and the team.

## Slide 2: The Problem Statement
- **Visual**: Bullet points contrasting standard LLM behavior (hallucinations, unsourced claims) vs Enterprise requirements (verifiability, scoped context).
- **Talking Points**: 
  - Off-the-shelf LLMs hallucinate when they lack context.
  - Users need to trust the system. Trust requires transparency: showing exactly *where* the answer came from.
  - Multi-document confusion: asking about an assessment shouldn't retrieve data from a resume.

## Slide 3: Solution & Key Features
- **Visual**: Feature list icons: Strict Scoping, Grounded Citations, Deterministic RAG.
- **Talking Points**:
  - We built a custom RAG pipeline using Google's Gemini models.
  - **Strict Scoping**: Conversations are explicitly bound to selected documents. No context leakage.
  - **Grounded Citations**: Our LLM provides `[Source N]` tags that our backend validates and natively connects to the exact underlying text chunk. 

## Slide 4: System Architecture (Diagram)
- **Visual**: Include the primary System Architecture Diagram from `DESIGN.md`.
- **Talking Points**:
  - Highlight the React + FastAPI stack.
  - Point out that we bypassed heavy frameworks like LangChain in favor of a lean, custom implementation.
  - Mention PostgreSQL with `pgvector` for scalable, deterministic vector storage.

## Slide 5: The RAG & Citation Flow (Diagram)
- **Visual**: Include the Sequence Diagram from `DESIGN.md`.
- **Talking Points**:
  - Walk the audience through the flow: User asks a question -> query embedded -> vector distance search (`<=>`) limited strictly to 0.6 threshold -> context fed to Gemini -> parsed citations persisted directly to `message_sources`.
  - Emphasize that the frontend dynamically resolves these citations to exact file names and page numbers.

## Slide 6: Evaluation & Testing
- **Visual**: High-level test metrics (Backend 63/63, Frontend 13/13) and a brief mention of Phase 13 Eval.
- **Talking Points**:
  - Show that we built this with comprehensive automated and regression testing.
  - Mention our deterministic 20-question benchmark where we achieved 100% groundedness under our defined heuristic criteria.
  - Discuss how manual QA verified that clicking a citation strictly reveals the underlying PDF chunk.

## Slide 7: Live Demo (Script)
- **Visual**: Switch to browser (localhost:5173).
- **Demo Script**:
  1. **Upload**: Upload `Resume_kulasekhar.pdf` and `Assessment Guidelines.pdf`.
  2. **Conversation**: Select *both* documents and start a conversation.
  3. **Disambiguation Query**: Ask "What is the CGPA?"
  4. **Verification**: Show the generated answer. Click the citation chip (`Resume_kulasekhar.pdf · p.X`) and show the Source Details panel opening with the exact chunk.
  5. **Second Query**: Ask "What are the assessment requirements?" and verify the citation points exclusively to the Assessment Guidelines.
  6. **Deletion**: Delete the conversation to demonstrate `ON DELETE CASCADE` backend hygiene.

## Slide 8: Q&A
- **Visual**: "Questions?"
- **Talking Points**: Open the floor to the audience. Be prepared to discuss fallback behavior (`gemini-3.5-flash-lite`, `gemini-3.7-flash`), pgvector indexing, embedding batching, and the custom word-boundary chunking logic.
