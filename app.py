import torch
from sentence_transformers import SentenceTransformer, util
import os
# import PyPDF2
from openai import OpenAI
import fitz
import json
import gradio as gr
import re

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Configuration for the Ollama API client
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='mistral'
    # api_key=os.getenv('OPENAI_API_KEY', 'default_key_if_none')
)

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
    
# Function to save embeddings to a file
def save_embeddings(embeddings, path):
    """Save tensor embeddings to a file."""
    with open(path, 'wb') as f:
        torch.save(embeddings, f)

# Function to load embeddings from a file
def load_embeddings(path):
    """Load tensor embeddings from a file if they exist."""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return torch.load(f)
    return None


def process_file_content(file_path, file_type):
    try:
        """Extract text from the provided file based on its type (PDF, TXT, JSON)."""
        text = ''
        if file_type == 'pdf':
            doc = fitz.open(file_path)  # Use file path directly with fitz
            text = ' '.join([page.get_text() for page in doc if page.get_text()])
        elif file_type == 'txt':
            with open(file_path, 'r', encoding="utf-8") as file:
                text = file.read()
        elif file_type == 'json':
            with open(file_path, 'r', encoding="utf-8") as file:
                data = json.load(file)
                text = json.dumps(data, ensure_ascii=False)
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        print(f"Failed to process file {file_path} of type {file_path}: {str(e)}")
        return ""


def get_relevant_context(user_input, vault_embeddings, vault_content, model):
    """Simple context retrieval based on cosine similarity between input and document content embeddings."""
    input_embedding = model.encode([user_input])
    cos_scores = util.cos_sim(input_embedding, vault_embeddings)[0]
    top_indices = torch.topk(cos_scores, k=1)[1].tolist()
    return [vault_content[idx] for idx in top_indices]



# Function to interact with the Ollama model
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, model):
    # Get relevant context from the vault
    relevant_context = get_relevant_context(user_input, vault_embeddings, vault_content, model)
    if relevant_context:
        # Convert list to a single string with newlines between items
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)
    
    # Prepare the user's input by concatenating it with the relevant context
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = context_str + "\n\n" + user_input

    # Create a message history including the system message and the user's input with context
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input_with_context}
    ]
    # Send the completion request to the Ollama model
    response = client.chat.completions.create(
        model="mistral",
        messages=messages
    )
    # Return the content of the response from the model
    return response.choices[0].message.content



def chat_with_document(uploaded_file, user_input):
    try:
        # Check the type of uploaded_file to handle both possible cases
        if isinstance(uploaded_file, tuple):
            # This block assumes uploaded_file is a tuple (file_name, file_content)
            file_name, file_content = uploaded_file
        else:
            # If uploaded_file is just bytes (for binary upload), you might need additional info to handle it
            file_name = "uploaded_file.pdf"  # You might want to generate or get the name dynamically based on the use case
            file_content = uploaded_file  # Directly use the bytes

        file_type = 'pdf' if file_name.endswith('.pdf') else 'txt' if file_name.endswith('.txt') else 'json'
        
        # Save the file to disk
        file_path = f"/tmp/{file_name}"
        with open(file_path, "wb") as f:
            f.write(file_content)

        text = process_file_content(file_path, file_type)
        embeddings = model.encode([text])  # Convert text to embeddings
        vault_embeddings = torch.tensor(embeddings)  # Convert embeddings to tensor

        response = ollama_chat(user_input, "You are a helpful assistant that is an expert at extracting the most useful information from a given text", vault_embeddings, [text], model)
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"


# Define the interface
model = SentenceTransformer("all-MiniLM-L6-v2")
iface = gr.Interface(
    fn=chat_with_document,
    inputs=[gr.File(type="binary", label="Upload PDF, TXT, or JSON"), gr.Textbox(label="Your question:")],
    outputs=gr.Text(label="Response"),
    title="Document-Based Chatbot",
    description="Upload a document and ask questions about its content. Supports PDF, TXT, and JSON files."
)
iface.launch()