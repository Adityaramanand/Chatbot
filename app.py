

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from faster_whisper import WhisperModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pyannote.audio import Pipeline
from pydub import AudioSegment
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
import os
from PIL import Image
import json
import librosa
import torch
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

#=================================================
# LOAD ENV VARIABLES
#=================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

IMAGE_FOLDER = "static/images"

os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)

PDF_FOLDER="static/pdfs"

os.makedirs(
    PDF_FOLDER,
    exist_ok =True
)
AUDIO_FOLDER = "static/audio"

os.makedirs(
    AUDIO_FOLDER,
    exist_ok=True
)

# MEMORY_FILE = "chat_history.json"
SESSION_FILE = "sessions.json"
# ==================================================
# GEMINI API CONFIG
# ==================================================



genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)


# ==================================================
# FLASK SETUP
# ==================================================

app = Flask(__name__)

vectorstore = None
if not os.path.exists(
    SESSION_FILE
):

    with open(
        SESSION_FILE,
        "w"
    ) as f:

        json.dump(
            {},
            f
        )

# if not os.path.exists(MEMORY_FILE):

#     with open(
#         MEMORY_FILE,
#         "w"
#     ) as f:

#         json.dump(
#             [],          single memory for all users, not ideal but works for demo
#             f
#         )

# try:

#     with open(
#         MEMORY_FILE,
#         "r"
#     ) as f:

#         chat_history = json.load(f)
        

# except:

#     chat_history = []

# ==================================================
# MEMORY FUNCTIONS
# ==================================================

# def save_memory():

#     with open(
#         MEMORY_FILE,
#         "w"
#     ) as f:

#         json.dump(
#             chat_history,   same just one memory is used for all users, not ideal but works for demo
#             f,
#             indent=4
#         )
CORS(app)
# ==================================================
# SESSION MEMORY FUNCTIONS
# ==================================================


def load_sessions():

    with open(
        SESSION_FILE,
        "r"
    ) as f:

        return json.load(f)


def save_sessions(data):

    with open(
        SESSION_FILE,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )
# ==================================================
# WHISPER MODEL
# ==================================================

print("\nLoading Whisper Model...\n")

whisper_model = WhisperModel(
    "base",
    compute_type="int8"
)

print("\nWhisper Loaded Successfully!\n")

# ==================================================
# DIARIZATION MODEL
# ==================================================

print("\nLoading Speaker Diarization Model...\n")

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)

print("\nSpeaker Diarization Loaded!\n")

# # ==================================================
# # LOAD PDF
# # ==================================================

# print("\nLoading PDF...\n")

# loader = PyPDFLoader(
#     "data/Policy Sample.pdf"
# )

# documents = loader.load()

# print(f"Total Pages Loaded: {len(documents)}")


# # ==================================================
# # SPLIT INTO CHUNKS
# # ==================================================

# print("\nSplitting into chunks...\n")

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=100
# )

# chunks = splitter.split_documents(documents)

# print(f"Total Chunks Created: {len(chunks)}")


# ==================================================
# LOAD EMBEDDING MODEL
# ==================================================

print("\nLoading Embedding Model...\n")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded Successfully!")


# # ==================================================
# # CREATE VECTOR DATABASE
# # ==================================================

# print("\nCreating FAISS Vector Database...\n")

# vectorstore = FAISS.from_documents(
#     chunks,
#     embedding_model
# )

# print("FAISS Vector Database Created Successfully!")


# ==================================================
# HOME ROUTE
# ==================================================

@app.route("/")
def home():

    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():

    try:

        audio = request.files["audio"]

        audio_folder = "static/audio"

        os.makedirs(
            audio_folder,
            exist_ok=True
        )

        audio_path = os.path.join(
            audio_folder,
            "voice.webm"
        )

        audio.save(audio_path)

        segments, info = whisper_model.transcribe(
            audio_path,
            beam_size=5
        )
        print("Detected Language:", info.language)
        text = ""

        for segment in segments:

            text += segment.text

        return jsonify({
            "text": text.strip()
        })

    except Exception as e:

        print(e)

        return jsonify({
            "text": "",
            "error": str(e)
        })
    
# ==================================================
# IMAGE CHAT ROUTE
# ==================================================

@app.route("/image-chat", methods=["POST"])
def image_chat():

    try:

        image = request.files["image"]

        image_path = os.path.join(
            IMAGE_FOLDER,
            image.filename
        )

        image.save(image_path)

        question = request.form.get(
            "question",
            "Explain this image"
        )

        img = Image.open(
            image_path
        )

        response = model.generate_content(
            [
                question,
                img
            ]
        )

        return jsonify({
            "response": response.text
        })

    except Exception as e:

        print(e)

        return jsonify({
            "response": str(e)
        })
    
#===================================================
# MULTI-PDF UPLOAD ROUTE
#===================================================
@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():

    global vectorstore

    try:

        pdf = request.files["pdf"]

        pdf_path = os.path.join(
            PDF_FOLDER,
            pdf.filename
        )

        pdf.save(pdf_path)

        loader = PyPDFLoader(
            pdf_path
        )

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(
            docs
        )

        if vectorstore is None:

            vectorstore = FAISS.from_documents(
                chunks,
                embedding_model
            )

        else:

            vectorstore.add_documents(
                chunks
            )

        return jsonify({
            "message":
            f"{pdf.filename} uploaded successfully. Ask me anything about it."
        })

    except Exception as e:

        print(e)

        return jsonify({
            "message": str(e)
        })

@app.route(
    "/upload-audio",
    methods=["POST"]
)
def upload_audio():

    try:

        audio = request.files["audio"]

        audio_path = os.path.join(
            AUDIO_FOLDER,
            audio.filename
        )

        audio.save(audio_path)
        print("Audio upload started")

        waveform_np, sample_rate = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )

        waveform = torch.from_numpy(
            waveform_np
        ).float().unsqueeze(0)
        
        output = diarization_pipeline(
            {
                "waveform": waveform,
                "sample_rate": sample_rate
            }
        )

        segments_data = []

        print(type(output))
        print(dir(output))

        annotation = output.speaker_diarization
        print(annotation)
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments_data.append(
                {
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end
                }
                )

        audio_file = AudioSegment.from_file(
            audio_path
        )

        conversation = []

        for i, segment in enumerate(
            segments_data
        ):

            start_ms = int(
                segment["start"] * 1000
            )

            end_ms = int(
                segment["end"] * 1000
            )

            chunk = audio_file[
                start_ms:end_ms
            ]

            chunk_path = os.path.join(
                AUDIO_FOLDER,
                f"chunk_{i}.wav"
            )

            chunk.export(
                chunk_path,
                format="wav"
            )

            whisper_segments, info = (
                whisper_model.transcribe(
                    chunk_path
                )
            )

            text = ""

            for s in whisper_segments:

                text += s.text

            conversation.append(
                {
                    "speaker":
                    segment["speaker"],

                    "text":
                    text.strip()
                }
            )

            if os.path.exists(
                chunk_path
            ):
                os.remove(
                    chunk_path
                )

        import re

        dialogue_text = ""

        for item in conversation:
            text = item["text"]

            text = re.sub(
            r"(Speaker\s+\d+),",
            r"\n\1:",
            text
            )

            dialogue_text += text + "\n\n"


        global vectorstore

        doc = Document(
        page_content=dialogue_text
        )

        if vectorstore is None:
            vectorstore = FAISS.from_documents(
                [doc],
                embedding_model
                )

        else:
            vectorstore.add_documents(
                [doc]
                )
        summary_prompt = f"""
Analyze this conversation.

Conversation:

{dialogue_text}

Provide:

1. Main Topic

2. Key Discussion Points

3. Important Decisions

4. Action Items

5. Summary
"""

        summary = model.generate_content(
            summary_prompt
        ).text

        return jsonify(
            {
                "dialogue":
                dialogue_text,

                "summary":
                summary
            }
        )

    except Exception as e:

        print(e)

        return jsonify(
            {
                "dialogue": "",
                "summary": str(e)
            }
        )
# ==================================================
# CLEAR MEMORY ROUTE
# ==================================================
# @app.route("/clear-memory", methods=["POST"])
# def clear_memory():
#     # global chat_history
#     # chat_history = []
#     # save_memory()
#     return jsonify({"message": "Memory cleared successfully."})
# ==================================================
# CLEAR MEMORY ROUTE SESSION BASED
# ==================================================
@app.route(
    "/clear-memory",
    methods=["POST"]
)
def clear_memory():

    data = request.get_json()

    session_id = data.get(
        "session_id"
    )

    sessions = load_sessions()

    sessions[session_id] = []

    save_sessions(
        sessions
    )

    return jsonify(
        {
            "message":
            "Session memory cleared."
        }
    )
# ==================================================
# CHAT ROUTE
# ==================================================

@app.route(
    "/chat",
    methods=["POST", "OPTIONS"]
)
def chat():

    # global chat_history

    if request.method == "OPTIONS":

        return "", 200

    data = request.get_json()

    query = data.get(
        "message"
        )

    session_id = data.get(
        "session_id"
        )

    sessions = load_sessions()

    if not session_id:
        return jsonify(
        {
            "response":
            "Session ID missing."
        }
    )

    if session_id not in sessions:
        sessions[session_id] = []

    save_sessions(
        sessions
    )

    chat_history = sessions[session_id]

    # =========================================
    # STORE USER MESSAGE
    # =========================================

    chat_history.append(
        {
            "role": "user",
            "content": query
        }
        )
    sessions[session_id] = chat_history

    save_sessions(
        sessions
        )

    # chat_history = chat_history[-100:]

    # save_memory()

    # =========================================
    # CHECK PDF EXISTS
    # =========================================

    if vectorstore is None:

        return jsonify(
            {
                "response":
                "Please upload a PDF first."
            }
        )

    # =========================================
    # RETRIEVE SIMILAR CHUNKS
    # =========================================

    results = vectorstore.similarity_search(
        query,
        k=3
    )

    # =========================================
    # BUILD PDF CONTEXT
    # =========================================

    context = ""

    for doc in results:

        context += doc.page_content

        context += "\n\n"

    # =========================================
    # BUILD MEMORY CONTEXT
    # =========================================

    memory_context = ""

    for msg in chat_history[-10:]:

        memory_context += (
            f"{msg['role']}: "
            f"{msg['content']}\n"
        )

    # =========================================
    # CREATE PROMPT
    # =========================================

    prompt = f"""
Answer the question using only the PDF Context.

Keep responses short and natural.

If the answer is a simple fact:
return only the fact.

If the answer requires explanation:
provide a brief explanation.

If not found:
"The information is not available in the document."

PDF Context:

{context}

Question:

{query}
"""

    # =========================================
    # GEMINI RESPONSE
    # =========================================

    response = model.generate_content(
        prompt
    )

    final_answer = response.text

    # =========================================
    # STORE BOT RESPONSE
    # =========================================

    chat_history.append(
        {
            "role": "assistant",
            "content": final_answer
        }
    )
    sessions[session_id] = chat_history

    save_sessions(
        sessions
    )

    # chat_history = chat_history[-100:]

    # save_memory()

    # =========================================
    # RETURN RESPONSE
    # =========================================

    return jsonify(
        {
            "response": final_answer
        }
    )
# ==================================================
# RUN APP
# ==================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
