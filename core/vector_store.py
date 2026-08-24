import hashlib  # Used to create a short unique fingerprint for each transcript

from langchain_chroma import Chroma  # Import Chroma vector database
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Import text splitter to divide transcript into smaller chunks
from langchain_core.documents import Document  # Import Document class to store text with metadata

try:  # The embeddings class moved to its own package
    from langchain_huggingface import HuggingFaceEmbeddings  # New location
except ImportError:  # Old location, still works but warns
    from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = "vector_db"  # Folder where the vector database will be stored
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # HuggingFace embedding model name

CHUNK_SIZE = 500  # Target characters per chunk
CHUNK_OVERLAP = 50  # Keep 50 characters overlapping between chunks (plain-text path only)

# Bumped whenever the stored metadata changes shape. Old collections keep their
# old name, so they are simply never found again instead of being loaded with
# missing timestamp fields.
COLLECTION_VERSION = "v2"

_embeddings = None  # Store the embedding model so it loads only once


def get_embeddings():
    """Load the embedding model once and reuse it for every call."""
    global _embeddings  # Access the global embedding model variable

    if _embeddings is None:  # Load the model only if it hasn't been loaded yet
        print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,  # Select the embedding model
            model_kwargs={"device": "cpu"},  # Run the model on CPU
        )
        print("Embedding model loaded.")

    return _embeddings  # Return the loaded embedding model


def _segments_text(segments) -> str:
    """Join timed segments into one plain string."""
    if isinstance(segments, str):  # Already plain text
        return segments
    return " ".join(seg["text"] for seg in segments).strip()


def make_collection_name(segments) -> str:
    """
    Create a unique collection name for this specific video.

    The same video always produces the same name, and two different videos
    can never produce the same name. This keeps every video's vectors in
    its own separate collection.
    """
    text = _segments_text(segments)  # Accepts timed segments or plain text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]  # Short fingerprint
    return f"video_{COLLECTION_VERSION}_{digest}"  # Chroma requires a leading letter


def group_segments(segments: list, max_chars: int = CHUNK_SIZE) -> list:
    """
    Group consecutive timed segments into chunks of about max_chars characters.

    Why not RecursiveCharacterTextSplitter here? Because it works on one long
    string and would lose track of which segment each piece came from. Grouping
    whole segments instead means every chunk keeps the start time of its first
    segment and the end time of its last one, so we always know where in the
    video the chunk came from.

    The last segment of each group is carried into the next group, so a sentence
    sitting on the boundary appears in both - the same idea as CHUNK_OVERLAP.
    """
    docs = []  # Finished Document objects
    current = []  # Segments collected for the chunk being built
    current_len = 0  # Characters collected so far

    for seg in segments:
        text = seg.get("text", "").strip()

        if not text:  # Skip silent segments
            continue

        if current and current_len + len(text) > max_chars:  # This chunk is full
            docs.append(_make_doc(current, len(docs)))

            current = [current[-1]]  # Carry one segment over as the overlap
            current_len = len(current[0]["text"])

        current.append(seg)
        current_len += len(text)

    # Flush the leftover. If it holds only the carried-over segment, its text is
    # already inside the previous chunk, so we skip it - unless it is the only
    # chunk we have.
    if current and (len(current) > 1 or not docs):
        docs.append(_make_doc(current, len(docs)))

    return docs


def _make_doc(group: list, index: int) -> Document:
    """Turn one group of segments into a Document carrying its time range."""
    return Document(
        page_content=" ".join(seg["text"].strip() for seg in group),
        metadata={
            "chunk_index": index,
            "start": float(group[0]["start"]),  # Seconds into the video
            "end": float(group[-1]["end"]),
        },
    )


def _docs_from_text(transcript: str) -> list:
    """Fallback for plain text with no timing: split it and store start = 0."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return [
        Document(
            page_content=chunk,
            metadata={"chunk_index": i, "start": 0.0, "end": 0.0},
        )
        for i, chunk in enumerate(splitter.split_text(transcript))
    ]


def build_vector_store(segments) -> Chroma:
    """
    Embed the video and store it in its own Chroma collection.

    "segments" is the timed list from transcribe_all. A plain string is still
    accepted, but then no timestamps can be stored.
    """
    embeddings = get_embeddings()  # Load the embedding model
    collection_name = make_collection_name(segments)  # Unique collection for this video

    vector_store = Chroma(  # Open (or create) the collection for this video
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    if vector_store.get(limit=1)["ids"]:  # Check whether this video was already embedded
        print(f"Reusing existing vector store: {collection_name}")
        return vector_store  # Skip re-embedding - saves a lot of time

    print(f"Building vector store: {collection_name}")

    if isinstance(segments, str):  # No timing information available
        docs = _docs_from_text(segments)
    else:
        docs = group_segments(segments)  # Timed chunks

    vector_store.add_documents(docs)  # Embed the chunks and store them in this collection
    print(f"Stored {len(docs)} timed chunks in {collection_name}.")

    return vector_store  # Return the ready vector store


def load_vector_store(collection_name: str) -> Chroma:
    """Open an existing collection by name, without re-embedding anything."""
    return Chroma(
        collection_name=collection_name,  # Which video's collection to load
        embedding_function=get_embeddings(),  # Embedding model used for searching
        persist_directory=CHROMA_DIR,  # Location of the stored vector database
    )


def get_retriever(vector_store: Chroma, k: int = 4):
    """Create a retriever that returns the k most similar transcript chunks."""
    return vector_store.as_retriever(
        search_type="similarity",  # Use similarity search
        search_kwargs={"k": k},  # Return the top k most similar chunks
    )


def delete_collection(collection_name: str) -> None:
    """Delete one video's vectors from disk."""
    load_vector_store(collection_name).delete_collection()
    print(f"Deleted collection: {collection_name}")

'''
─── Notes ───────────────────────────────────────────────────────────────────
> "vector_store.py" prepares the transcript for the RAG system. 
> It groups the timed segments into chunks of about 500 characters, converts those chunks into numerical embeddings using a HuggingFace model, and stores them in ChromaDB.
> Each chunk keeps its start and end time in its metadata. That is what lets the chat answer say "this came from 3:42" and link straight to that moment.
> Each video gets its OWN collection, named from a hash of its transcript, so one video's content can never leak into another video's answers. The name also carries a version tag, so collections built before timestamps existed are simply never reused.
> Timed segments -> grouped chunks (+ start/end) -> Embeddings -> ChromaDB -> Retriever
'''