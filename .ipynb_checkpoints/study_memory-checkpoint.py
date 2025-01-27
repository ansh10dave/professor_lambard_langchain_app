from flask import Flask, render_template, request, jsonify, session
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'newton'  # Required for session management

# Load Meta's LLaMA 3.2 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "D:/Huggingface/llama-3.2-3B-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = "PAD"

# Function to generate answers using LLaMA
def generate_answer(query, context):
    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    output = model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, max_new_tokens=500)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer_start_index = generated_text.lower().rfind('answer:') + len('Answer:')

    # Extract only the answer part
    answer = generated_text[answer_start_index:].strip()
    return answer

# Load PDF documents using LangChain community PDF loader
def load_documents(pdf_paths):
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            # Adding chapter information in metadata
            doc.metadata["chapter"] = path.split(".")[0]
        documents.extend(docs)
    return documents

# Initialize documents
pdf_files = ['fluid-mechanics.pdf', 'thermodynamics.pdf', 'waves.pdf', 'gravitation.pdf']
documents = load_documents(pdf_files)

# Conversational retrieval chain using FAISS and LLaMA
def get_answer_from_retriever(query, chapter=None):
    if chapter:
        # Filter documents by chapter
        chapter_docs = [doc for doc in documents if doc.metadata.get("chapter") == chapter]
        
        if not chapter_docs:
            return "No relevant content found for the selected chapter."

        # Create a temporary vector store for the chapter
        chapter_texts = [doc.page_content for doc in chapter_docs]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        texts = []
        for text in chapter_texts:
            texts.extend(text_splitter.split_text(text))

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        chapter_vector_store = FAISS.from_texts(texts, embeddings)

        # Retrieve relevant documents
        results = chapter_vector_store.similarity_search(query, k=3)
        context = "\n".join(results)
    else:
        # Default: Retrieve from all documents if no chapter is specified
        context = "\n".join([doc.page_content for doc in documents[:3]])

    return generate_answer(query, context)

@app.route('/')
def index():
    return render_template('index.html')  # Homepage for selecting chapter

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    chapter = data.get('chapter')
    question = data.get('question')

    if not chapter or not question:
        return jsonify({"error": "Chapter and question are required"}), 400

    # Initialize conversation history if not already
    if 'conversation_history' not in session:
        session['conversation_history'] = []

    # If there's already a history, append the new question and answer to it
    context = ""
    if session['conversation_history']:
        # Get the full history of Q&A to pass to the model
        context = "\n".join(session['conversation_history'])

    # Generate answer using the context and the current question
    answer = generate_answer(question, context)

    # Append the current question and answer to the history (this will not be printed in the response)
    session['conversation_history'].append(f"Question: {question}")
    session['conversation_history'].append(f"Answer: {answer}")

    # Now only return the latest answer (not the full history)
    return jsonify({"answer": answer})
    
    # data = request.json
    # chapter = data.get('chapter')
    # question = data.get('question')

    # if not chapter or not question:
    #     return jsonify({"error": "Chapter and question are required"}), 400

    # # Initialize conversation history if not already
    # if 'conversation_history' not in session:
    #     session['conversation_history'] = []

    # # Add the current question to the conversation history
    # session['conversation_history'].append(f"Question: {question}")

    # # Combine previous conversation with the current question to provide context
    # context = "\n".join(session['conversation_history'])

    # # Generate answer using the context and question
    # answer = generate_answer(question, context)

    # # Add the answer to the conversation history
    # session['conversation_history'].append(f"Answer: {answer}")

    # return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
