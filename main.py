from utils.audio_processor import prepared_audio  # Prepares audio chunks, deletes them after
from core.transcriber import transcribe_all, segments_to_text  # Audio -> timed segments -> text
from core.summarizer import summarize, generate_title  # Generate video summary and title
from core.extractor import extract_takeaways, extract_topics, extract_quotes  # Extract video information
from core.rag_engine import build_rag_chain, ask_question  # Build RAG and ask questions

# No load_dotenv() here - core/config.py already loads .env with an absolute path
# before any setting is read.


def run_pipeline(source: str, language: str = "english") -> dict:
    print("Starting AI Video Assistant")  # Show that the AI processing pipeline has started

    with prepared_audio(source) as chunks:  # Temp WAV files are deleted after this block
        segments = transcribe_all(chunks, language)  # Timed segments for the whole video

    transcript = segments_to_text(segments)  # Plain text for the summary and extraction
    print(f"Raw transcription (first 300 characters): {transcript[:300]}")  # Quick verification

    title = generate_title(transcript)  # Generate a short title from the transcript

    summary = summarize(transcript)  # Generate a summary of the video

    takeaways = extract_takeaways(transcript)  # Extract the key points of the video
    topics = extract_topics(transcript)  # Extract an outline of the topics covered
    quotes = extract_quotes(transcript)  # Extract notable verbatim quotes

    rag_chain = build_rag_chain(segments)  # Segments, so every chunk keeps its timestamp

    return {  # Return all processed video information as a dictionary
        "title": title,
        "segments": segments,
        "transcript": transcript,
        "summary": summary,
        "takeaways": takeaways,
        "topics": topics,
        "quotes": quotes,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":  # Run this code only when main.py is executed directly

    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()  # Ask for a URL or file path
    language = input("Language (english/hinglish): ").strip() or "english"  # Default to English
    result = run_pipeline(source, language)  # Run the complete pipeline

    print("\n" + "=" * 60)  # Print a separator for better terminal readability
    print(f"📌 Title: {result['title']}")  # Display the generated title
    print(f"\n📋 Summary:\n{result['summary']}")  # Display the summary
    print(f"\n💡 Key Takeaways:\n{result['takeaways']}")  # Display extracted takeaways
    print(f"\n🗂️ Topics Covered:\n{result['topics']}")  # Display extracted topics
    print(f"\n🗣️ Notable Quotes:\n{result['quotes']}")  # Display extracted quotes
    print("=" * 60)  # Print another separator

    # Phase 2 — Chat with your video via RAG
    print("\n💬 Chat with this video (type 'exit' to quit)\n")  # Start the interactive RAG chat
    rag_chain = result["rag_chain"]  # Get the RAG chain from the pipeline

    while True:  # Keep asking questions until the user chooses to exit
        question = input("You: ").strip()  # Get a question from the user

        if question.lower() in ["exit", "quit", "q"]:  # Exit on exit, quit or q
            print("👋 Goodbye!")  # Display goodbye message
            break  # Stop the while loop

        if not question:  # Ignore empty questions and ask again
            continue

        answer, sources = ask_question(rag_chain, question)  # Answer plus its timestamps
        print(f"\n🤖 Assistant: {answer}")  # Display the AI-generated answer

        if sources:  # Show where in the video the answer came from
            print(f"   🕐 In the video at: {', '.join(s['label'] for s in sources)}")

        print()

'''
─── Notes ───────────────────────────────────────────────────────────────────
> "main.py" is the command-line controller of the application. 
> It takes the user's input, sends it through the audio processing and transcription pipeline, generates the title and summary, extracts takeaways, topics and quotes, builds the RAG pipeline, and finally lets the user chat with the video.
> app.py is the Streamlit version of the same pipeline. main.py is useful for testing the pipeline without the UI in the way.
> Answers now come with the timestamps they were taken from.
'''