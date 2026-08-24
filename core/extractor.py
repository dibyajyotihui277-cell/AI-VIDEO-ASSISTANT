# Key takeaways , topics covered & notable quotes extractor

from langchain_core.output_parsers import StrOutputParser  # Convert AI response into plain text
from langchain_core.prompts import ChatPromptTemplate  # Create AI prompts
from langchain_core.runnables import RunnableLambda, RunnablePassthrough  # LCEL building blocks
from langchain_mistralai import ChatMistralAI  # Mistral AI chat model
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Split long transcripts

from core.config import MISTRAL_API_KEY, MISTRAL_MODEL, require_key  # Central settings

# One chunk of about 12000 characters is roughly 3000 tokens.
# mistral-small can handle far more than that, so this leaves plenty of safety
# headroom while keeping the number of API calls as low as possible.
CHUNK_SIZE = 12000  # Maximum characters per portion
CHUNK_OVERLAP = 500  # Overlap so a sentence split across two portions is not lost

NONE_MARKER = "NONE"  # The map step replies with this word when a portion has nothing


def get_llm():
    """Create and return the Mistral AI model used for extraction."""
    return ChatMistralAI(
        model=MISTRAL_MODEL,  # Model name from .env
        mistral_api_key=require_key(MISTRAL_API_KEY, "MISTRAL_API_KEY"),  # Fail early if key missing
        temperature=0.2,  # Low temperature = precise, repeatable extraction
        max_retries=5,  # Retry automatically if the API rate-limits us
    )


def split_transcript(transcript: str) -> list:
    """Split a long transcript into portions small enough for one API call."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(transcript)


def build_chain(system_prompt: str):
    """Build a reusable LangChain pipeline for one system prompt."""
    return (
        RunnablePassthrough()  # Pass the input text into the pipeline
        | RunnableLambda(lambda x: {"text": x})  # Convert text into {"text": text}
        | ChatPromptTemplate.from_messages(  # Create prompt using system prompt and text
            [
                ("system", system_prompt),
                ("human", "{text}"),
            ]
        )
        | get_llm()  # Send prompt to Mistral AI
        | StrOutputParser()  # Extract plain text response
    )


def _is_empty(result: str) -> bool:
    """True if the model said this portion contains nothing to extract."""
    cleaned = result.strip().strip(".").strip().upper()
    return cleaned == "" or cleaned == NONE_MARKER


def _map_reduce(
    transcript: str,
    map_prompt: str,
    reduce_prompt: str,
    empty_message: str,
    label: str,
) -> str:
    """
    Run an extraction over a transcript of any length.

    MAP    - run the extraction prompt on each portion separately
    FILTER - throw away the portions that returned NONE
    REDUCE - merge the remaining partial lists into one clean final list
    """
    chunks = split_transcript(transcript)  # Split transcript into portions
    print(f"Extracting {label} from {len(chunks)} portion(s)...")

    map_chain = build_chain(map_prompt)  # Pipeline for the map step
    partials = []  # Collect the non-empty results

    for i, chunk in enumerate(chunks, start=1):  # MAP: one call per portion
        print(f"  → {label}: portion {i}/{len(chunks)}")
        result = map_chain.invoke(chunk).strip()

        if not _is_empty(result):  # FILTER: skip portions that found nothing
            partials.append(result)

    if not partials:  # Nothing found anywhere in the transcript
        return empty_message

    reduce_chain = build_chain(reduce_prompt)  # Pipeline for the reduce step
    # We always run reduce, even for a single portion, so the output format is
    # identical no matter how long the video was.
    return reduce_chain.invoke("\n\n".join(partials)).strip()


# ─── Prompts: key takeaways ──────────────────────────────────────────────────
TAKEAWAYS_MAP = (
    "You are an expert video analyst. You are reading ONE PORTION of a longer "
    "video transcript, not the whole thing.\n\n"
    "Extract the key takeaways from this portion - the specific points a viewer "
    "would actually want to remember.\n\n"
    "Rules:\n"
    "- Use ONLY what is said in this portion\n"
    "- Do NOT invent anything, and do NOT add your own knowledge of the subject\n"
    "- Do NOT pad the list with vague statements to make it longer\n"
    "- Write each takeaway as one clear sentence\n"
    "- Use a dash list, NOT numbers\n"
    f"If this portion is only greetings, music or filler, reply with exactly: {NONE_MARKER}"
)

TAKEAWAYS_REDUCE = (
    "Below are key takeaways extracted from consecutive portions of the SAME video.\n\n"
    "Merge them into one final list:\n"
    "- Remove duplicates and near-duplicates\n"
    "- Combine takeaways that make the same point, keeping the clearer wording\n"
    "- Keep them in roughly the order they appeared\n"
    "- Do NOT invent anything that is not in the text below\n\n"
    "Format as a numbered list, one sentence each."
)


# ─── Prompts: topics covered ─────────────────────────────────────────────────
TOPICS_MAP = (
    "You are reading ONE PORTION of a longer video transcript, not the whole thing.\n\n"
    "List the topics discussed in this portion, in the order they come up.\n\n"
    "Rules:\n"
    "- A topic is a short phrase of 2 to 6 words, not a full sentence\n"
    "- Only list topics actually discussed in this portion\n"
    "- Do NOT invent topics the speaker never raised\n"
    "- Use a dash list, NOT numbers\n"
    f"If nothing identifiable is discussed, reply with exactly: {NONE_MARKER}"
)

TOPICS_REDUCE = (
    "Below are topics extracted from consecutive portions of the SAME video.\n\n"
    "Merge them into one final outline:\n"
    "- Remove duplicates and near-duplicates\n"
    "- Where several entries belong together, group them under a broader topic\n"
    "- Keep every entry short; do NOT invent topics not in the text below\n\n"
    "Format as a numbered list. Use bold for a grouped topic and indented dashes "
    "for the subtopics under it."
)


# ─── Prompts: notable quotes ─────────────────────────────────────────────────
QUOTES_MAP = (
    "You are reading ONE PORTION of a longer video transcript, not the whole thing.\n\n"
    "Extract the most notable things actually SAID in this portion - statements that "
    "are memorable or surprising, or that carry a specific claim, number, name or date.\n\n"
    "Rules:\n"
    "- Quote the words as they appear in the transcript, wrapped in double quotation marks\n"
    "- Do NOT paraphrase, polish, shorten or correct the wording\n"
    "- If the speaker is clearly identifiable, add ' - Speaker Name' after the quote\n"
    "- Extract at most 3 from this portion, choosing the most substantive\n"
    "- Do NOT invent a quote under any circumstances\n"
    "- Use a dash list, NOT numbers\n"
    f"If nothing stands out, reply with exactly: {NONE_MARKER}"
)

QUOTES_REDUCE = (
    "Below are quotes extracted from consecutive portions of the SAME video.\n\n"
    "Merge them into one final list:\n"
    "- Remove duplicates\n"
    "- Keep every quote EXACTLY as written; do NOT rewrite, shorten or tidy the wording\n"
    "- Keep at most 6, choosing the most substantive\n"
    "- Do NOT invent anything not in the text below\n\n"
    "Format as a numbered list."
)


# ─── Public functions ────────────────────────────────────────────────────────
def extract_takeaways(transcript: str) -> str:
    """Extract the key points of the video, from a transcript of any length."""
    return _map_reduce(
        transcript,
        TAKEAWAYS_MAP,
        TAKEAWAYS_REDUCE,
        "No clear takeaways found.",
        "key takeaways",
    )


def extract_topics(transcript: str) -> str:
    """Extract an outline of the topics covered, from a transcript of any length."""
    return _map_reduce(
        transcript,
        TOPICS_MAP,
        TOPICS_REDUCE,
        "No topics found.",
        "topics covered",
    )


def extract_quotes(transcript: str) -> str:
    """Extract notable verbatim quotes, from a transcript of any length."""
    return _map_reduce(
        transcript,
        QUOTES_MAP,
        QUOTES_REDUCE,
        "No notable quotes found.",
        "notable quotes",
    )


# ─── Notes ───────────────────────────────────────────────────────────────────
# "extractor.py" extracts structured information from the transcript using
# Mistral AI: key takeaways, the topics covered, and notable verbatim quotes.
#
# These three work for any video, which is why they replaced the original
# action items / key decisions / open questions. Those three only made sense for
# a recorded meeting and came back empty on ordinary videos.
#
# Because a transcript can be far longer than one API call allows, each
# extraction runs as a map-reduce: the transcript is split into portions, each
# portion is extracted separately, portions with nothing to report are dropped,
# and the remaining partial lists are merged into one clean, de-duplicated list.
#
# Transcript -> portions -> extract each -> merge -> Takeaways + Topics + Quotes