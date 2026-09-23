from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


COLLECTION_NAME = "documents"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(
    host="qdrant",
    port=6333
)


def read_document(file_path):
    path = Path(file_path)

    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    raise ValueError("Unsupported file type")


def chunk_text(text, chunk_size=500):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks


def ingest_document(file_path):

    text = read_document(file_path)

    chunks = chunk_text(text)

    embeddings = model.encode(chunks)

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embeddings.shape[1],
            distance=Distance.COSINE
        )
    )

    points = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={"text": chunk}
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return len(chunks)


def search_documents(query, limit=3):

    query_embedding = model.encode(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=limit
    ).points

    return [result.payload["text"] for result in results]