📑 Intelligent PDF Analyzer


🚀 Overview
The Intelligent PDF Analyzer is a powerful Streamlit web application that leverages the capabilities of Google's Gemini Pro model, Sentence Transformers, and KeyBERT to provide insightful analysis of PDF documents. This tool is designed to help users quickly understand the core content of long documents by:

Extracting text from PDF files.

Generating concise summaries using Google Gemini.

Identifying key phrases and keywords.

Answering specific questions directly from the document's content using a combination of semantic search and Google Gemini.

Whether you're dealing with research papers, reports, articles, or any other text-heavy PDF, this application streamlines the process of extracting critical information.

✨ Features
PDF Text Extraction: Robustly extracts text from uploaded PDF documents.

Gemini-Powered Summarization: Utilizes the advanced Google Gemini (1.5 Flash by default for speed) to generate high-quality, concise summaries of documents.

Key Phrase Extraction: Identifies the most relevant keywords and phrases using KeyBERT.

Intelligent Q&A: Ask natural language questions about your document, and the AI will find the most relevant passage and provide an answer.

Intuitive User Interface: Built with Streamlit for a clean and easy-to-use experience.

Downloadable Analysis: Export a text report containing the summary, key phrases, and Q&A.

Secure API Key Handling: Integrates with Streamlit Secrets for secure management of your Gemini API key.

🛠️ Technologies Used
Streamlit: For building the interactive web application.

Google Gemini API (google-generativeai): For advanced text summarization and question-answering.

pdfplumber: For extracting text from PDF documents.

sentence-transformers: For generating embeddings used in semantic search for Q&A context retrieval.

keybert: For extracting key phrases from the document.

numpy: For numerical operations, particularly in embedding comparisons.

re (Regular Expressions): For text cleaning and processing.

⚙️ Setup and Installation
Follow these steps to get the PDF Analyzer up and running on your local machine.

1. Clone the Repository
2. Create a Virtual Environment (Recommended)
Bash

python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
3. Install Dependencies
Install all necessary Python packages using pip:

Bash

pip install -r requirements.txt
requirements.txt content:

streamlit
pdfplumber
google-generativeai
sentence-transformers
keybert
numpy
4. Obtain a Google Gemini API Key
Go to Google AI Studio (or Google Cloud Console).

Sign in with your Google account.

Create a new project if you don't have one.

Enable the Generative Language API.

Generate an API key.

5. Configure Your API Key (Streamlit Secrets)
For secure handling of your API key, use Streamlit's secrets management:

In the root directory of your project (the DOCUMENTSUMMRIZER folder), create a folder named .streamlit (if it doesn't already exist).

Inside the .streamlit folder, create a file named secrets.toml.

Open secrets.toml and add your Gemini API key in the following format:

Ini, TOML

# .streamlit/secrets.toml
GEMINI_API_KEY = "YOUR_ACTUAL_GEMINI_API_KEY_HERE"
Make sure to replace "YOUR_ACTUAL_GEMINI_API_KEY_HERE" with the key you obtained from Google AI Studio.

🏃 How to Run the Application
Once everything is set up, run the Streamlit app from your terminal:

Bash

streamlit run app3.py
This command will open the application in your default web browser.

🚀 Deployment (Optional)
You can deploy this application to Streamlit Cloud for easy sharing.
When deploying to Streamlit Cloud, you'll need to add your GEMINI_API_KEY as a secret directly in the Streamlit Cloud app settings (under "Advanced settings" -> "Secrets").

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

📧 Contact
Farhan Shafique - farhansial64@gmail.com

