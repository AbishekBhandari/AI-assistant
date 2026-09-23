from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)


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

    raise ValueError(
        f"Unsupported file type: {path.suffix}"
    )


def chunk_text(text, chunk_size=500):
    words = text.split()

    chunks = []

    for i in range(
        0,
        len(words),
        chunk_size
    ):
        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def ingest_documents(directory="documents"):

    directory = Path(directory)

    documents = (
        list(directory.glob("*.pdf"))
        + list(directory.glob("*.txt"))
    )

    if not documents:
        raise ValueError(
            "No documents found."
        )

    all_chunks = []

    for document in documents:

        print(
            f"Reading: {document.name}",
            flush=True
        )

        text = read_document(document)

        chunks = chunk_text(text)

        for chunk in chunks:

            all_chunks.append({
                "text": chunk,
                "source": document.name
            })

    print(
        f"Total chunks: {len(all_chunks)}",
        flush=True
    )

    texts = [
        item["text"]
        for item in all_chunks
    ]

    embeddings = model.encode(
        texts
    )

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embeddings.shape[1],
            distance=Distance.COSINE
        )
    )

    points = []

    for i, (item, embedding) in enumerate(
        zip(all_chunks, embeddings)
    ):

        points.append(
            PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "text": item["text"],
                    "source": item["source"]
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return len(all_chunks)


def search_documents(
    query,
    limit=2,
    source=None
):
    """
    Search all documents or restrict the search
    to a specific source document.
    """

    query_embedding = model.encode(query)

    query_filter = None

    if source:

        query_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(
                        value=source
                    )
                )
            ]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        query_filter=query_filter,
        limit=limit
    ).points

    return [
        {
            "text": result.payload["text"],
            "source": result.payload["source"]
        }
        for result in results
    ]