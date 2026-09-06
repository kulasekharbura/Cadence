# System Testing Document

## 1. Automated Test Suites

The project maintains two primary automated test suites for the frontend and backend. Both suites execute successfully on the current implementation.

### Backend Verification (`pytest`)
- **Framework**: Pytest `9.1.1`
- **Total Tests**: 63
- **Passing**: 63 (100%)
- **Failing**: 0
- **Scope**: The backend suite covers API endpoint routing, Document extraction mock-ups (including normal PyMuPDF extraction and fallback Tesseract OCR extraction logic), PgVector similarity search logic, conversation cascading deletions, embedding batching tests for the Gemini API limit, and E2E endpoints integrating the database with test containers.

### Frontend Verification (`vitest`)
- **Framework**: Vitest `5.0.0`
- **Total Tests**: 13
- **Passing**: 13 (100%)
- **Failing**: 0
- **Production Build**: Passing
- **Scope**: Covers React components, custom TanStack query hooks, API service definitions, edge cases for citation parsing (`chatUtils.tsx`), and proper rendering of `ReactMarkdown` with custom citation chips matching `#source-N` anchors.

*(Note: Code coverage metrics were not officially generated or tracked during this phase.)*

---

## 2. Manual QA and Validation

Manual Quality Assurance was strictly performed against the final browser UI to ensure that full-stack integrations (especially regarding dynamic source citation rendering) functioned properly.

### Verified Scenarios

The citation UX was manually verified in the browser through the following scenarios:
1. **Multi-Document Disambiguation**: 
   - Asked the system: *"What is the CGPA?"* -> Verified the citation correctly mapped to `Resume_kulasekhar.pdf`.
   - Asked the system: *"What degree is he pursuing?"* -> Verified the citation correctly mapped to `Resume_kulasekhar.pdf`.
   - Verified that clicking the source chip successfully opened the Source Details panel.
   - Verified that the exact correct page number and exact retrieved chunk content were displayed.
   - Asked the system for assessment requirements -> Verified the citation correctly mapped to `Assessment Guidelines.pdf`.
   - Verified multi-document citation behavior dynamically updating per message.
2. **Historical Context**:
   - Verified historical per-message citation behavior accurately persisted and displayed when navigating through older messages.
