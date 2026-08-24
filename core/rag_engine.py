from langchain_core.output_parsers import StrOutputParser  # Import parser to convert AI response into plain text
from langchain_core.prompts import ChatPromptTemplate  # Import prompt template to create AI prompts
from langchain_core.runnables import (  # Import LCEL building blocks
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_mistralai import ChatMistralAI  # Import Mistral AI chat model

from core.config import MISTRAL_API_KEY, MISTRAL_MODEL, require_key  # Central settings
from core.timestamps import format_timestamp  # Turn seconds into "3:42"
from core.vector_store import (  # Import vector store functions
    build_vector_store,
    get_retriever,
    load_vector_store,
    make_collection_name,
)

RETRIEVER_K = 4  # How many transcript chunks to send to the model as context

# Written once, so the UI can recognise this exact sentence and hide the
# timestamp chips when the answer is not in the video.
NOT_FOUND = "I could not find this information in the video."

# The system prompt is written ONCE here and reused by every chain builder below,
# so the different paths can never drift apart.
RAG_SYSTEM_PROMPT = """You are an expert video assistant. Answer the user's question
based ONLY on the video transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the video."

Do not use any knowledge of your own, even if you are confident it is correct.
Always be concise and precise. If quoting someone, mention it clearly.

Each passage below begins with its timestamp in the video, like [3:42]. When it
helps the user find the moment, mention that timestamp in your answer. Never
invent a timestamp that is not shown below.

Context from the video transcript:
{context}"""


def get_llm():
    """Create and return the Mistral AI model used for answering questions."""
    return ChatMistralAI(
        model=MISTRAL_MODEL,  # Model name from .env
        mistral_api_key=require_key(MISTRAL_API_KEY, "MISTRAL_API_KEY"),  # Fail early if key missing
        temperature=0,  # 0 = stick to the transcript, do not get creative
    )


def format_docs(docs) -> str:
    """
    Combine retrieved chunks into one block of text, each labelled with its time.

    The label is what lets the model say "at 3:42 she explains ..." instead of
    just repeating the words with no idea where they came from.
    """
    parts = []

    for doc in docs:
        start = doc.metadata.get("start", 0)  # Older collections may not have it
        parts.append(f"[{format_timestamp(start)}] {doc.page_content}")

    return "\n\n".join(parts)


def collect_sources(docs) -> list:
    """
    Turn the retrieved chunks into a small list the UI can render as links.

    Duplicate seconds are removed and the list is sorted, so the chips appear in
    the order the moments happen in the video.
    """
    seen = {}

    for doc in docs:
        seconds = float(doc.metadata.get("start", 0) or 0)
        key = int(seconds)  # One chip per second, so near-identical chunks merge

        if key not in seen:
            seen[key] = {"seconds": seconds, "label": format_timestamp(seconds)}

    return [seen[key] for key in sorted(seen)]


def _make_chain(retriever):
    """
    Build the LCEL RAG pipeline from any retriever.

    Both build_rag_chain() and load_rag_chain() call this, so the prompt and
    pipeline only exist in one place.

    The chain now returns a dictionary instead of a string:
        {"docs": [...], "question": "...", "answer": "..."}
    We need the docs alongside the answer, because the timestamps live in their
    metadata and the old chain threw them away as soon as the prompt was built.
    """
    prompt = ChatPromptTemplate.from_messages(  # Create prompt for RAG question answering
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}"),  # Insert user's question
        ]
    )

    answer_chain = (  # Produces just the answer text from the retrieved docs
        RunnableLambda(
            lambda x: {"context": format_docs(x["docs"]), "question": x["question"]}
        )
        | prompt  # Create the final prompt
        | get_llm()  # Send prompt to Mistral AI
        | StrOutputParser()  # Extract plain text response
    )

    return (
        RunnableParallel(  # Retrieve chunks and keep the question side by side
            docs=retriever,
            question=RunnablePassthrough(),
        )
        | RunnablePassthrough.assign(answer=answer_chain)  # Add the answer, keep the docs
    )


def collection_name_for(segments) -> str:
    """
    Return the vector store collection name for a video.

    Save this value if you want to rebuild the chat later with load_rag_chain()
    instead of re-embedding everything again.
    """
    return make_collection_name(segments)


def build_rag_chain(segments):
    """Embed the video (or reuse it if already embedded) and return a RAG chain."""
    vector_store = build_vector_store(segments)  # Create/open this video's collection
    retriever = get_retriever(vector_store, k=RETRIEVER_K)  # Create retriever
    return _make_chain(retriever)  # Return the ready RAG pipeline


def load_rag_chain(collection_name: str):
    """
    Rebuild a RAG chain for a video that was already embedded earlier.

    No downloading, no transcribing, no embedding - it just reopens the
    existing collection. Pass the name you got from collection_name_for().
    """
    vector_store = load_vector_store(collection_name)  # Open the existing collection

    if not vector_store.get(limit=1)["ids"]:  # Make sure it actually contains data
        raise ValueError(
            f"Collection '{collection_name}' is empty or does not exist. "
            "Run build_rag_chain(segments) first."
        )

    retriever = get_retriever(vector_store, k=RETRIEVER_K)  # Create retriever
    return _make_chain(retriever)  # Return the ready RAG pipeline


def ask_question(rag_chain, question: str):
    """
    Send one question through the RAG pipeline.

    Returns (answer, sources), where sources is a list of
    {"seconds": 222.0, "label": "3:42"} records the UI turns into links.
    """
    print(f"Question : {question}")  # Display the user's question in the terminal

    result = rag_chain.invoke(question)  # Send question through the RAG pipeline

    answer = result["answer"].strip()  # The generated answer text
    sources = collect_sources(result["docs"])  # Where that answer came from

    if NOT_FOUND.lower().rstrip(".") in answer.lower():
        sources = []  # Nothing was actually used, so do not offer misleading links

    print(f"Answer : {answer}")  # Display the generated answer

    if sources:
        print(f"Sources : {', '.join(s['label'] for s in sources)}")

    return answer, sources  # Return both

'''
─── Notes ───────────────────────────────────────────────────────────────────
> "rag_engine.py" implements the Retrieval-Augmented Generation pipeline. 
> It takes the user's question, retrieves the most relevant transcript chunks from ChromaDB, labels each one with its timestamp, puts them into a prompt as context, sends the prompt to Mistral AI, and returns an answer based only on the transcript. ask_question() now returns (answer, sources). 
> The sources are the moments in the video the chunks came from, which the UI turns into clickable YouTube links.
> build_rag_chain(segments)        -> embed (or reuse) then chat
> load_rag_chain(collection_name)  -> reopen an already-embedded video and chat instantly
> Question -> Retrieve timed chunks -> Mistral AI -> Answer + timestamps
'''