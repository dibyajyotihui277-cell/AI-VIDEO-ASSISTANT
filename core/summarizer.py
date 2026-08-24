from langchain_core.output_parsers import StrOutputParser  # Convert AI response into plain text
from langchain_core.prompts import ChatPromptTemplate  # Create AI prompts
from langchain_core.runnables import RunnableLambda, RunnablePassthrough  # LCEL building blocks
from langchain_mistralai import ChatMistralAI  # Mistral AI chat model
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Split long transcripts

from core.config import MISTRAL_API_KEY, MISTRAL_MODEL, require_key  # Central settings

# 12000 characters is roughly 3000 tokens, which mistral-small handles easily.
# The old value was 3000, which meant about 4x more API calls than necessary.
CHUNK_SIZE = 12000  # Maximum characters per portion
CHUNK_OVERLAP = 500  # Overlap so a sentence split across two portions is not lost

TITLE_SAMPLE = 1500  # Characters taken from the start, middle and end for the title


def get_llm():
    """Create and return the Mistral AI model used for summarising."""
    return ChatMistralAI(
        model=MISTRAL_MODEL,  # Model name from .env
        mistral_api_key=require_key(MISTRAL_API_KEY, "MISTRAL_API_KEY"),  # Fail early if key missing
        temperature=0.3,  # A little freedom, so summaries read naturally
        max_retries=5,  # Retry automatically if the API rate-limits us
    )


def split_transcript(transcript: str) -> list:
    """Split a long transcript into portions small enough for one API call."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(transcript)


# ─── Prompts ─────────────────────────────────────────────────────────────────
SUMMARY_MAP = (
    "You are an expert video analyst. You are reading ONE PORTION of a longer "
    "video transcript, not the whole thing.\n\n"
    "Summarise what is discussed in this portion concisely.\n"
    "Only describe what is actually said. Do NOT add background knowledge, "
    "context or conclusions of your own."
)

SUMMARY_REDUCE = (
    "You are an expert video summariser. Below are summaries of consecutive "
    "portions of the SAME video.\n\n"
    "Combine them into one clear summary of the whole video:\n"
    "- Group related points under short bold headings\n"
    "- Use concise bullet points under each heading\n"
    "- Remove repetition between portions\n"
    "- Use ONLY information present in the summaries below; do NOT add anything new\n\n"
    "Start directly with the first heading. Do not write an introductory sentence."
)

TITLE_PROMPT = (
    "Based on this video transcript, write a short, specific title of at most "
    "8 words describing what the video is actually about.\n\n"
    "Rules:\n"
    "- Do NOT use the words 'video', 'transcript', 'discussion' or 'meeting'\n"
    "- Be specific about the actual subject, not generic\n"
    "- Return ONLY the title, with no quotation marks and nothing else"
)


def build_chain(system_prompt: str):
    """Build a reusable LangChain pipeline for one system prompt."""
    return (
        RunnablePassthrough()  # Pass the input text into the pipeline
        | RunnableLambda(lambda x: {"text": x})  # Convert text into {"text": text}
        | ChatPromptTemplate.from_messages(  # Create prompt from system prompt and text
            [
                ("system", system_prompt),
                ("human", "{text}"),
            ]
        )
        | get_llm()  # Send prompt to Mistral AI
        | StrOutputParser()  # Extract plain text response
    )


def summarize(transcript: str) -> str:
    """Summarise a transcript of any length using map-reduce."""
    chunks = split_transcript(transcript)  # Split transcript into portions
    print(f"Summarising {len(chunks)} portion(s)...")

    map_chain = build_chain(SUMMARY_MAP)  # Pipeline for the map step
    partials = []  # Collect one summary per portion

    for i, chunk in enumerate(chunks, start=1):  # MAP: one call per portion
        print(f"  → summary: portion {i}/{len(chunks)}")
        partials.append(map_chain.invoke(chunk).strip())

    if len(partials) == 1:  # Short video - the single summary is already the answer
        return partials[0]

    reduce_chain = build_chain(SUMMARY_REDUCE)  # Pipeline for the reduce step
    return reduce_chain.invoke("\n\n".join(partials)).strip()


def _sample_for_title(transcript: str) -> str:
    """
    Take a sample from the start, middle and end of the transcript.

    Using only the opening (as the old code did) gives bad titles on long
    videos, because the first minutes are usually an intro or sponsor read
    that says nothing about the real subject.
    """
    if len(transcript) <= TITLE_SAMPLE * 3:
        return transcript  # Short enough to use whole

    middle_start = (len(transcript) // 2) - (TITLE_SAMPLE // 2)

    return (
        transcript[:TITLE_SAMPLE]
        + "\n...\n"
        + transcript[middle_start: middle_start + TITLE_SAMPLE]
        + "\n...\n"
        + transcript[-TITLE_SAMPLE:]
    )


def generate_title(transcript: str) -> str:
    """Generate a short title describing what the video is about."""
    title = build_chain(TITLE_PROMPT).invoke(_sample_for_title(transcript)).strip()
    return title.strip('"').strip("'").strip()  # Drop quotes the model sometimes adds


# ─── Notes ───────────────────────────────────────────────────────────────────
# "summarizer.py" generates the video summary and title using Mistral AI.
# Because a transcript can be far longer than one API call allows, the summary
# runs as a map-reduce: the transcript is split into portions, each portion is
# summarised separately, then the partial summaries are combined into one final
# summary grouped under headings.
#
# The title is generated from a sample of the start, middle and end of the
# transcript, so a long intro cannot mislead it.
#
# Transcript -> portions -> summarise each -> combine -> Summary + Title