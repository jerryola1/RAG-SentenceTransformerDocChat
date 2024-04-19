# RAG-SentenceTransformerDocChat

## Overview
RAG-SentenceTransformerDocChat is a document-based chatbot application that leverages the Sentence Transformer model and Retriever-Augmented Generation (RAG) to extract and utilize context from various document types such as PDFs, TXT, or JSON files. This tool provides a unique way to interact with document contents, making it easier to retrieve useful information via a chat interface.

## Features
- **Document Processing**: Supports PDF, TXT, and JSON files, extracting text for further processing.
- **Contextual Chat**: Utilizes extracted text to provide context-based responses within a chatbot interface.
- **AI-Driven Responses**: Employs advanced NLP techniques, including Sentence Transformers and RAG, to enhance the relevance and accuracy of responses.

## Installation

To get started with RAG-SentenceTransformerDocChat, clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/RAG-SentenceTransformerDocChat.git
cd RAG-SentenceTransformerDocChat
pip install -r requirements.txt

## Usage
To run the application, execute the following command:
```bash
python app.py

Navigate to http://localhost:7860 to access the Gradio interface where you can upload documents and ask questions based on their content.

## How It Works
The application processes uploaded files to extract text, which is then used to generate embeddings with Sentence Transformers. These embeddings serve as input for the RAG model to fetch the most relevant information based on the user's query.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
* Sentence Transformers for NLP modeling.
* OpenAI for the Ollama API.
* Fitz for PDF processing

## Contact
For any questions or suggestions, please reach out to olagunjujeremiah@gmail.com.