# 🎬 AI Video Assistant

**Paste a YouTube link. Get a title, a summary, key takeaways, topics and notable quotes — then chat with the video and click any answer to jump to the exact moment it came from.**

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/LangChain-LCEL-1C3C3C" alt="LangChain">
  <img src="https://img.shields.io/badge/Mistral%20AI-mistral--small-FF7000" alt="Mistral AI">
  <img src="https://img.shields.io/badge/Whisper-local-412991?logo=openai&logoColor=white" alt="Whisper">
  <img src="https://img.shields.io/badge/ChromaDB-vector%20store-4B32C3" alt="ChromaDB">
</p>

---

## Screenshots

### 🏠 Home Screen

<p align="center">
  <img width="1919" height="893" alt="Screenshot 2026-08-22 183350" src="https://github.com/user-attachments/assets/383f2076-7828-4bef-9d77-b4835a84669f" />
</p>

### 💡 Key Takeaways, Topics and Quotes<img width="1901" height="914" alt="Screenshot 2026-08-23 113826" src="https://github.com/user-attachments/assets/521addcc-2d4b-4467-a78b-54a4da60ae50" />


<p align="center">
  <img width="1915" height="902" alt="Screenshot 2026-08-23 113905" src="https://github.com/user-attachments/assets/27b24c9b-4fd5-4108-9315-4a0c615cfea3" />
</p>

### 💬 RAG Chat with Timestamp Citations

<p align="center">
  <img width="1901" height="914" alt="Screenshot 2026-08-23 113826" src="https://github.com/user-attachments/assets/35ac35f0-4c7c-480d-bab7-219817348b29" />
</p>

---


## What problem this solves

Watching a 40-minute video to find the one thing you actually need is slow, and asking a general chatbot about a video means it will happily answer from its own training data instead of from the video — confidently, and sometimes wrongly.

This project does the opposite. It transcribes the real audio, stores it, and answers **only** from that transcript. When it cannot find something, it says so:

> I could not find this information in the video.

That single sentence is the point of the whole design. An assistant that admits what it does not know is far more useful than one that always has an answer.

---

## Features

**Any video source.** A YouTube URL — including `youtu.be`, `/watch?v=`, `/shorts/`, `/embed/` and `/live/` forms — or a local audio or video file from your machine.

**Two transcription engines.** English audio goes to a locally-run OpenAI Whisper model, so no audio ever leaves your machine and there is no per-minute cost. Hindi-English mixed speech ("Hinglish") goes to Sarvam AI's `saaras:v2.5`, which transcribes and translates to English in one step.

**Structured analysis.** A generated title, a summary grouped under headings, the key takeaways, an outline of topics covered, and notable verbatim quotes.

**Grounded chat.** Retrieval-augmented generation over the transcript at `temperature=0`, with prompts that forbid the model from using outside knowledge.

**Clickable timestamp citations.** Every answer cites the moments it drew from, as links straight into the video.

**Works on long videos.** Summarisation and extraction run as map-reduce over the transcript, so a two-hour video is handled the same way as a two-minute one.

---

## How it works

```text
YouTube URL  or  local file
         │
         ▼
   utils/audio_processor.py
     ├─ yt-dlp download  ──or──  pydub convert
     ├─ normalise to WAV, mono, 16 kHz
     └─ split into 10-minute AudioChunk(path, start)
                                      ↑
                              start = offset in video
         │
         ▼
   core/transcriber.py
     └─ Whisper (english)  or  Sarvam AI (hinglish)
         │
         ▼
   timed segments:
   [{ "start": 214.7, "end": 219.2, "text": "..." }, ...]
         │
         ├──► segments_to_text()
         │         ├─► core/summarizer.py
         │         │      └─► title + summary
         │         │
         │         └─► core/extractor.py
         │                └─► takeaways · topics · quotes
         │
         └──► core/vector_store.py
                  ├─ group segments into ~500-character chunks
                  ├─ keep each chunk's start / end in metadata
                  └─ embed with all-MiniLM-L6-v2
                       └─► ChromaDB
                              │
                              ▼
                       core/rag_engine.py
                              │
                              ├─ question
                              ├─ retrieve top 4 chunks
                              ├─ label each "[3:42] ..."
                              ├─ Mistral AI, temperature 0
                              └─ answer + timestamp sources
                                      │
                                      ▼
                                   app.py
                                      │
                                      └─► ▶ 3:42  ▶ 4:13  ▶ 5:03
```

### The detail that makes timestamps work

Whisper reports segment times **relative to the file you give it**. Because the audio is split into 10-minute chunks before transcription, a sentence 30 seconds into chunk 2 is reported as `0:30` — but in the real video it happens at `10:30`.

So `chunk_audio()` returns an `AudioChunk(path, start)` record rather than just a path, and the transcriber adds that offset to every time Whisper produces. Without it, every timestamp after the first ten minutes would be silently wrong — the worst kind of bug, because the links would still look perfectly plausible.

---

## Project structure

```text
VIDEO_AI/
├── app.py                    Streamlit UI — the main entry point
├── main.py                   CLI version of the same pipeline, useful for testing
├── test.py                   Offline checks for the timestamp maths (no API calls)
│
├── core/
│   ├── config.py             Loads .env, then exposes every setting
│   ├── transcriber.py        Audio chunks → timed segments
│   ├── summarizer.py         Transcript → title + summary (map-reduce)
│   ├── extractor.py          Transcript → takeaways · topics · quotes (map-reduce)
│   ├── vector_store.py       Segments → embedded chunks in ChromaDB
│   ├── rag_engine.py         Question → grounded answer + timestamps
│   └── timestamps.py         Seconds ↔ "3:42" ↔ YouTube deep link
│
├── utils/
│   └── audio_processor.py    Download / convert / chunk audio, with cleanup
│
├── screenshots/              README screenshots
├── downloades/               Temporary audio (git-ignored, auto-deleted)
├── vector_db/                Persisted ChromaDB collections (git-ignored)
├── .env                      Your API keys (git-ignored)
└── requirements.txt
```

---

## Getting started

### Prerequisites

* **Python 3.10 or newer**

* **FFmpeg**, on your PATH. Whisper and pydub both depend on it.

  * Windows: `winget install Gyan.FFmpeg`
  * macOS: `brew install ffmpeg`
  * Linux: `sudo apt install ffmpeg`

* **Deno** — only if YouTube downloads fail with `HTTP Error 403: Forbidden`. YouTube requires solving a small JavaScript puzzle, and yt-dlp needs a JS runtime to do it.

  * Windows: `winget install DenoLand.Deno`

### Install

```bash
git clone https://github.com/dibyajyotihui277-cell/ai-video-assistant.git
cd ai-video-assistant

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt
```

### Configure

Copy `.env.example` to `.env` and fill in your keys:

```env
MISTRAL_API_KEY=your_mistral_key_here
MISTRAL_MODEL=mistral-small-latest

SARVAM_API_KEY=your_sarvam_key_here
SARVAM_STT_MODEL=saaras:v2.5

WHISPER_MODEL=small
```

A [Mistral API key](https://console.mistral.ai/) is required. A [Sarvam AI key](https://dashboard.sarvam.ai/) is optional and only used for Hinglish audio. Whisper runs locally and needs no key — the model downloads itself on first use.

### Run

```bash
streamlit run app.py
```

For the command-line version:

```bash
python main.py
```

For offline timestamp checks:

```bash
python test.py
```

---

## Notable engineering decisions

**Grounding is enforced, not requested.** The RAG prompt runs at `temperature=0`, forbids outside knowledge even when the model is confident, and defines one exact fallback sentence. That sentence is a constant in `rag_engine.py`, so the UI can recognise it and suppress the timestamp chips — because if nothing was found, offering links would be misleading.

**One vector collection per video.** The collection name is a SHA-256 hash of the transcript, so the same video always maps to the same collection and two different videos can never share one. An earlier version used a single shared collection, which meant one video's content could surface in another video's answers. The name also carries a version tag, so when the stored metadata shape changes, old collections are simply never reused rather than being loaded with missing fields.

**Map, filter, then reduce.** A transcript can easily exceed what one API call accepts. Summarisation and extraction split it into 12,000-character portions, process each one independently, **discard the portions that report nothing**, and merge the rest into one de-duplicated result. The filter step matters: without it, a video with three quiet stretches produces a final list padded with "nothing found here" noise.

**Segment-aware chunking for the vector store.** The obvious tool, `RecursiveCharacterTextSplitter`, works on one long string and loses track of which segment each piece came from — which would throw away every timestamp. Instead, `group_segments()` accumulates whole segments up to ~500 characters, so each chunk keeps the start time of its first segment and the end time of its last, and carries one segment of overlap into the next chunk.

**The RAG chain returns documents, not just text.** The original chain piped the retriever straight into the prompt, so the retrieved documents — and their timestamps — were gone by the time the answer came back. It now uses `RunnableParallel` with `RunnablePassthrough.assign`, which keeps the documents alongside the generated answer.

**One place loads the environment.** Every module imports its settings from `core/config.py`, which calls `load_dotenv()` above its constants. Previously each module read `os.getenv()` at import time, so whichever module Python happened to import first could read a key before `.env` had been loaded and silently get `None`.

**Temporary audio cleanup is guaranteed.** `prepared_audio()` is a context manager with cleanup in a `finally` block, so WAV files are deleted even when transcription raises. Audio files are large, and a pipeline that leaks them fills a disk quickly.

**Escape first, then convert markdown.** LLM output is placed inside styled HTML cards. It is escaped with `html.escape` and only then converted from markdown, because escaping leaves markdown syntax characters untouched. Doing it the other way round would either break the layout or destroy the formatting.

**Temperature matched to the task.** Grounded question answering runs at `0`, extraction at `0.2`, summarisation at `0.3`. Extraction that invents a plausible-sounding quote is worse than extraction that returns nothing.

---

## Known limitations

These are real and worth stating plainly.

**Hinglish timestamps are coarse.** Sarvam's synchronous API returns no timing data at all, so each 25-second piece sent to it becomes one segment. Hinglish timestamps are therefore accurate to roughly 25 seconds, where Whisper gives a few seconds.

**Retrieved is not the same as used.** The citation chips show the passages the search retrieved, which is why the row is labelled *Sources in video* rather than something stronger. The model may have leaned on some of those passages more than others.

**Broad questions see a narrow slice.** Retrieval returns the four chunks nearest to the question, and neighbouring chunks of a transcript are semantically similar to each other. So a whole-video question like "what are the main lessons?" often retrieves four near-neighbours from one short stretch rather than four passages spread across the video. The Key Takeaways card is more complete for that kind of question, because the extractor reads the entire transcript. Diversity-aware retrieval such as MMR would improve this.

**Cloud hosting is awkward.** YouTube blocks downloads from datacenter IP addresses, and Whisper needs PyTorch plus enough memory to load a model, so free hosting tiers are a poor fit. The project is built to run locally.

**Chat has no memory.** Each question is answered independently, so follow-ups like "explain that more" have no idea what "that" refers to.

---

## Tech stack

| Layer                     | Choice                            | Why                                                      |
| ------------------------- | --------------------------------- | -------------------------------------------------------- |
| UI                        | Streamlit                         | Fast to build, and the whole app stays in Python         |
| Audio download            | yt-dlp                            | Actively maintained, handles YouTube's changes           |
| Audio processing          | pydub + FFmpeg                    | Format conversion and chunking                           |
| Speech to text (English)  | OpenAI Whisper, local             | Free, private, and gives segment-level timestamps        |
| Speech to text (Hinglish) | Sarvam AI `saaras:v2.5`           | Transcribes and translates code-mixed speech in one call |
| LLM                       | Mistral AI `mistral-small-latest` | Strong instruction-following at low cost                 |
| Orchestration             | LangChain LCEL                    | Composable chains, and parallel branches when needed     |
| Embeddings                | `all-MiniLM-L6-v2`                | Small, fast, runs on CPU                                 |
| Vector store              | ChromaDB                          | Persists to disk with no server to run                   |

---

## Roadmap

* Diversity-aware retrieval (MMR) so broad questions cover the whole video
* Conversation memory for follow-up questions
* Clickable timestamps throughout the full transcript view
* An embedded player that seeks in place instead of opening a new tab
* Speaker diarisation

---

## Acknowledgements

[OpenAI Whisper](https://github.com/openai/whisper) for local speech recognition · [Sarvam AI](https://www.sarvam.ai/) for Indic speech-to-text-translate · [Mistral AI](https://mistral.ai/) for the language model · [LangChain](https://www.langchain.com/) for orchestration · [Chroma](https://www.trychroma.com/) for the vector store · [yt-dlp](https://github.com/yt-dlp/yt-dlp) for making the audio accessible.

---

## License

MIT
