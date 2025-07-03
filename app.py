import streamlit as st
import pdfplumber
import numpy as np
import re
from io import StringIO
import os # For environment variables

# Import the Google Generative AI library
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- ADDED THIS LINE TO RESOLVE NameError for SentenceTransformer ---
from sentence_transformers import SentenceTransformer, util
# --- ADDED THIS LINE FOR KeyBERT (if not already present) ---
from keybert import KeyBERT

# --- Constants ---
MIN_PARAGRAPH_WORDS_FOR_QA = 10

# --- Configure Gemini API ---
# It's best practice to load API keys from Streamlit secrets or environment variables
# For Streamlit Cloud: Use st.secrets["GEMINI_API_KEY"]
# For local: Set GEMINI_API_KEY as an environment variable or create a .streamlit/secrets.toml file
# Example secrets.toml:
# GEMINI_API_KEY = "YOUR_API_KEY_HERE"

try:
    # Attempt to get API key from Streamlit secrets (for deployment)
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
except (AttributeError, KeyError):
    # Fallback to environment variable (for local development)
    gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("Gemini API key not found. Please set it in Streamlit secrets or as an environment variable (GEMINI_API_KEY).")
    st.stop() # Stop the app if API key is missing

genai.configure(api_key=gemini_api_key)

# ----------------------
# Load models (Now simpler, just the GenerativeModel)
# ----------------------
@st.cache_resource(show_spinner=False)
def load_gemini_model(model_name="gemini-1.5-flash"): # Use gemini-1.5-flash for speed/cost, or gemini-1.5-pro for quality
    """
    Loads the specified Gemini GenerativeModel.
    Caches the model to prevent reloading on app reruns.
    """
    # Configure safety settings (optional, but good practice)
    # Adjust as needed based on your application's requirements
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    return genai.GenerativeModel(model_name=model_name, safety_settings=safety_settings)

@st.cache_resource(show_spinner=False)
def load_embedder():
    """
    Loads the Sentence Transformer model for embeddings.
    Still useful for semantic search of passages.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

# KeyBERT is a local model, no change needed unless you want a Gemini-based alternative
@st.cache_resource(show_spinner=False)
def load_keybert():
    """Loads the KeyBERT model for keyword extraction."""
    return KeyBERT()

# ----------------------
# Core functions
# ----------------------
def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from a PDF file, performing basic cleanup to improve readability.
    (This function remains the same as it's PDF processing, not an LLM task)
    """
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}. Please ensure it's a valid PDF.")
        return ""

    text = re.sub(r'(?<=[a-zA-Z])(?=[A-Z])', ' ', text)
    text = re.sub(r'\.(?=[^\s])', '. ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def summarize_with_gemini(text, gemini_model, summary_words):
    """
    Summarizes text using the Gemini model.
    No chunking needed for Gemini due to larger context window (but still good for very large docs).
    """
    if not text:
        return "No text provided for summarization."

    # Gemini 1.5 Flash has a very large context window (1 million tokens),
    # so simple summarization prompts usually work without explicit chunking for most PDFs.
    # If PDFs are truly massive (hundreds/thousands of pages), you might still need to chunk
    # and summarize iteratively, then summarize the summaries, or use the Gemini File API.

    prompt = f"""Summarize the following document concisely, focusing on its main points.
    The summary should be approximately {summary_words} words long.

    Document:
    {text}
    """
    try:
        response = gemini_model.generate_content(prompt)
        # Check if response.text is directly available
        if hasattr(response, 'text'):
            return response.text
        # Fallback for structured responses or other types
        elif hasattr(response, 'parts') and response.parts:
            return "".join(part.text for part in response.parts if hasattr(part, 'text'))
        else:
            return "Gemini model did not return a readable summary."
    except genai.types.BlockedPromptException as e:
        st.warning(f"Summarization blocked due to safety concerns. Please review the content. Details: {e}")
        return "Content blocked by safety filters."
    except Exception as e:
        st.error(f"Error calling Gemini API for summarization: {e}")
        return "Failed to generate summary due to an API error."

def extract_keyphrases(text, keybert_model, n_phrases=10):
    """
    Extracts key phrases using KeyBERT. (This can remain as KeyBERT is local and often fast enough for this specific task)
    """
    if not text:
        return []
    try:
        keyphrases = keybert_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1,2),
            stop_words='english',
            top_n=n_phrases
        )
        return [kp[0] for kp in keyphrases]
    except Exception as e:
        st.error(f"Error extracting key phrases: {e}")
        return []

def find_best_passage(question, paragraphs, embedder, doc_embeddings):
    """
    Finds the most semantically similar passage using sentence embeddings.
    (This remains the same, as embedding is efficient and necessary for semantic search)
    """
    if not paragraphs or not question:
        return "No relevant passage found."

    try:
        question_embedding = embedder.encode(question, convert_to_tensor=True)
        cosine_scores = util.cos_sim(question_embedding, doc_embeddings)[0]
        top_idx = np.argmax(cosine_scores)
        return paragraphs[top_idx]
    except Exception as e:
        st.error(f"Error finding best passage for Q&A: {e}")
        return "Could not find a relevant passage due to an internal error."

def answer_question_with_gemini(question, context, gemini_model):
    """
    Answers a question based on a given context using the Gemini model.
    """
    if not question or not context:
        return "Please provide both a question and context."

    prompt = f"""Based on the following text, answer the question. If the answer is not in the text, state that you don't know or that the information is not provided.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    try:
        response = gemini_model.generate_content(prompt)
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'parts') and response.parts:
            return "".join(part.text for part in response.parts if hasattr(part, 'text'))
        else:
            return "Gemini model did not return a readable answer."
    except genai.types.BlockedPromptException as e:
        st.warning(f"Answer generation blocked due to safety concerns. Please rephrase your question or review the content. Details: {e}")
        return "Content blocked by safety filters."
    except Exception as e:
        st.error(f"Error calling Gemini API for Q&A: {e}")
        return "Failed to generate answer due to an API error."

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="📑 PDF Analyzer AI", layout="wide")
st.title("📑 Intelligent PDF Analyzer")
st.caption("Summarize long documents, extract key points, and ask questions from your PDF.")

with st.sidebar:
    st.header("⚙️ Configuration")
    # Removed chunk_words for Gemini summarization as it handles larger contexts
    # If you have extremely large PDFs (1000s of pages) you might need to re-introduce chunking logic
    # and use Gemini's File API for truly massive documents.
    summary_output_words = st.slider("Summary Max Length (words)", 50, 500, 200, help="Sets the approximate maximum word count for the generated summary.")
    n_keyphrases = st.slider("Number of key phrases", 5, 30, 10, help="Controls how many key phrases are extracted.")
    st.markdown("---")
    st.markdown("Developed with ❤️ by Your Farhan Sial (Powered by Google Gemini)")

uploaded_file = st.file_uploader("🚀 Upload your PDF document", type=["pdf"], help="Please upload a PDF file (e.g., reports, articles, research papers).")

if uploaded_file:
    # --- 1. Extract Text ---
    with st.spinner("Extracting text from PDF... This may take a moment for large files."):
        text = extract_text_from_pdf(uploaded_file)

    if not text or len(text.split()) < 50: # Minimum words for meaningful analysis
        st.warning("No readable text found in the uploaded PDF or the text is too short for analysis. Please try another file.")
    else:
        st.success("✅ PDF text extracted successfully! Ready for analysis.")

        with st.expander("📄 View Extracted Text (first 20,000 characters)"):
            st.text_area("Extracted Document Text",
                         text[:20000] + ("..." if len(text) > 20000 else ""),
                         height=300,
                         disabled=True,
                         help="This is the raw text extracted from your PDF.")

        # --- 2. Load Models ---
        with st.spinner("Loading Google Gemini model... (first time might be slow)"):
            gemini_model = load_gemini_model()
        with st.spinner("Loading embedding model for Q&A..."):
            embedder = load_embedder()
        with st.spinner("Loading KeyBERT model for key phrases..."):
            keybert_model = load_keybert()

        # --- 3. Process Summary & Key Phrases ---
        st.markdown("---")
        st.header("✨ Document Insights")
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner(f"Generating summary (up to {summary_output_words} words) with Gemini..."):
                final_summary = summarize_with_gemini(text, gemini_model, summary_output_words)
            st.subheader("📝 Summary")
            st.info(final_summary)
        with col2:
            with st.spinner(f"Extracting {n_keyphrases} key phrases..."):
                keyphrases = extract_keyphrases(text, keybert_model, n_keyphrases)
            st.subheader("📌 Key Phrases")
            if keyphrases:
                st.success(", ".join(keyphrases))
            else:
                st.warning("Could not extract key phrases. The document might be too short or lack distinct terms.")

        # --- 4. Q&A Section ---
        st.markdown("---")
        st.subheader("💬 Ask questions from this document")
        # Split text into paragraphs for Q&A, filtering out very short ones
        # This helps in finding a precise context for the answer
        paragraphs = [p.strip() for p in re.split(r'(?<=[.!?])\s+', text) if len(p.strip().split()) > MIN_PARAGRAPH_WORDS_FOR_QA]

        if not paragraphs:
            st.warning("Not enough distinct passages found in the document for effective Q&A. Please ensure your PDF contains well-structured paragraphs.")
        else:
            # Embed document paragraphs only once for semantic search
            if 'doc_embeddings' not in st.session_state:
                with st.spinner("Embedding document passages for semantic search... (This is a one-time process for the current document)"):
                    st.session_state.doc_embeddings = embedder.encode(paragraphs, convert_to_tensor=True)

            question = st.text_input(
                "Type your question here:",
                placeholder="e.g., What is the main conclusion of this report?",
                help="Ask a specific question about the content of your PDF."
            )

            if question:
                with st.spinner("Finding best passage & generating answer with Gemini..."):
                    top_passage = find_best_passage(question, paragraphs, embedder, st.session_state.doc_embeddings)
                    # Check if find_best_passage returned an error message
                    if "No relevant passage found." in top_passage or "Could not find a relevant passage due to an internal error." in top_passage:
                         st.warning(top_passage)
                    else:
                        answer_text = answer_question_with_gemini(question, top_passage, gemini_model)
                        st.markdown("**📝 Most Relevant Passage Found:**")
                        st.info(top_passage)
                        st.markdown("**💡 Answer:**")
                        st.success(answer_text)
            else:
                st.info("Enter a question above to get an answer directly from the document content.")

        # --- 5. Download Section ---
        st.markdown("---")
        st.subheader("⬇️ Download Analysis Results")
        with StringIO() as output:
            output.write("--- DOCUMENT SUMMARY ---\n" + final_summary + "\n\n")
            output.write("--- KEY PHRASES ---\n" + ", ".join(keyphrases) + "\n\n")
            # Only add Q&A if a question was asked and an answer was generated
            if question and 'answer_text' in locals() and answer_text:
                output.write(f"--- QUESTION & ANSWER ---\nQuestion: {question}\nAnswer: {answer_text}\n\n")
            output.write("--- ORIGINAL EXTRACTED TEXT (Partial) ---\n" + text[:20000] + ("..." if len(text) > 20000 else ""))

            st.download_button(
                label="📥 Download Analysis Report",
                data=output.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}_analysis_report.txt",
                mime="text/plain",
                help="Download a text file containing the summary, key phrases, and Q&A (if applicable)."
            )
else:
    st.info("Upload a PDF document using the file uploader above to begin your intelligent analysis.")