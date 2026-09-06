import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import async_session_maker
from app.models.chunk import DocumentChunk
from app.models.document import Document
import app.models.conversation
from sqlalchemy import select
from app.services.embedding_service import EmbeddingService
from app.schemas.chunk import DocumentChunk as ChunkSchema

resume_text = """
Kulasekhar
Email: kulas@example.com
Phone: 123-456-7890

Education
B.Tech Information Technology
XYZ University
CGPA: 8.92 till the 6th semester.

Skills
Python, JavaScript, React, Node.js, SQL, PostgreSQL, Docker, Git.
Machine Learning, Natural Language Processing, Vector Databases.

Experience
Software Engineering Intern at Tech Corp (May 2024 - Aug 2024)
- Developed a full-stack web application using React and Node.js.
- Implemented a scalable database schema using PostgreSQL.
- Optimized API response times by 30% using Redis caching.
- Collaborated with a team of 5 engineers using Agile methodologies.

Projects
1. AI Document Q&A System (Cadence)
- Built an AI-powered document Q&A system using RAG architecture.
- Integrated Gemini API for natural language generation.
- Utilized pgvector for efficient similarity search.
- Designed a responsive frontend using React and Tailwind CSS.

2. E-commerce Platform
- Created a robust e-commerce platform with user authentication and payment gateway integration.
- Managed state using Redux and handled side effects with Redux Saga.

Extracurricular Activities
- Vice President of the Computer Science Club.
- Organized hackathons and coding competitions for over 200 students.
- Volunteered at local coding bootcamps to teach web development to high school students.

Certifications
- AWS Certified Developer - Associate
- Google Cloud Professional Cloud Architect

Languages
- English (Fluent)
- Spanish (Intermediate)

Volunteer Work and Hobbies
I have volunteered at various local community centers to help teach computer literacy to the elderly. I strongly believe in giving back to the community.
In my free time, I enjoy reading science fiction novels, playing chess, and hiking in the mountains. I also participate in local competitive programming competitions.
I have a passion for open-source software and frequently contribute to various GitHub projects related to machine learning and web development.
I have attended several tech conferences, including PyCon and React Conf, where I networked with industry professionals and learned about the latest trends in software engineering.
I am always looking for new challenges and opportunities to grow as a developer and a leader.
My career goal is to become a software architect and lead a team of talented engineers to build innovative products that make a positive impact on society.
I am highly motivated, detail-oriented, and a quick learner. I thrive in fast-paced environments and am always eager to take on new responsibilities.
I have strong communication and interpersonal skills, which allow me to work effectively with cross-functional teams and stakeholders.
I am confident that my technical skills, combined with my passion for software engineering, make me a strong candidate for any software engineering role.
I am available for relocation and am open to both remote and on-site opportunities.
I have a proven track record of delivering high-quality software on time and within budget.
I am committed to writing clean, maintainable, and scalable code.
I am constantly seeking feedback to improve my skills and become a better developer.
I believe that continuous learning is essential for success in the fast-paced tech industry.
I am excited about the opportunity to contribute to a dynamic and innovative team.
"""

async def main():
    async with async_session_maker() as db:
        # Create Dummy Document
        doc = Document(filename="Resume_kulasekhar_mock.pdf", file_path="/fake/path", file_size=1024, page_count=1)
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        
        # Embed the resume chunk
        es = EmbeddingService()
        chunk_schema = ChunkSchema(document_id=str(doc.id), page_number=1, chunk_index=0, content=resume_text)
        embedded_result = await es.generate_embeddings_for_chunks([chunk_schema])
        chunk_vector = embedded_result[0]["embedding"]
        
        db_chunk = DocumentChunk(
            document_id=doc.id,
            page_number=1,
            chunk_index=0,
            content=resume_text,
            embedding=chunk_vector
        )
        db.add(db_chunk)
        await db.commit()
        await db.refresh(db_chunk)
        
        print(f"Created chunk {db_chunk.id} for mock resume")
        
        # Now query it
        query = "What degree is Kulasekhar pursuing?"
        q_schema = ChunkSchema(document_id=str(doc.id), page_number=1, chunk_index=0, content=query)
        q_embedded = await es.generate_embeddings_for_chunks([q_schema])
        query_vector = q_embedded[0]["embedding"]
        
        distance_col = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
        stmt = select(distance_col).where(DocumentChunk.id == db_chunk.id)
        result = await db.execute(stmt)
        distance = result.scalar()
        
        print(f"\n--- Degree Retrieval Test ---")
        print(f"Relevant chunk_id: {db_chunk.id}")
        print(f"Chunk content length: {len(resume_text)} characters")
        print(f"Query: '{query}'")
        print(f"Cosine distance: {distance:.4f}")
        
        from app.core.config import settings
        threshold = settings.RETRIEVAL_DISTANCE_THRESHOLD
        print(f"Current threshold: {threshold}")
        print(f"Rejected by threshold? {distance > threshold}")
        
        # Cleanup
        await db.delete(doc)
        await db.commit()

if __name__ == "__main__":
    asyncio.run(main())
