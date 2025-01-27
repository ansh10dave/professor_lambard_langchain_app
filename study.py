from flask import Flask, render_template, request, jsonify
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
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
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

##################### This case is only for when chapter is not selected for retrieval. ###############################
##################### If it is selected, temporary vector stores are created in get_answer_from_retreiver() function. ##########################
# Create a vector store (FAISS) for document retrieval. 
# def create_vector_store(documents):
#     #embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Using Sentence-Transformers model for embeddings
#     document_texts = [doc.page_content for doc in documents]
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

#     texts = []
#     for text in document_texts:
#         texts.extend(text_splitter.split_text(text)) 
    
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     #embeddings = embedder.encode(document_texts, convert_to_tensor=False)
    
#     vector_store = FAISS.from_texts(texts, embeddings)
#     return vector_store

# Initialize documents and vector store
pdf_files = ['ch9.pdf', 'ch11.pdf', 'ch14.pdf']
documents = load_documents(pdf_files)
# vector_store = create_vector_store(documents)

# Conversational retrieval chain using FAISS and LLaMA
def get_answer_from_retriever(query, chapter=None):
    if chapter:
        # Filter documents by chapter
        chapter_docs = [doc for doc in documents if doc.metadata.get("chapter") == chapter]
        
        # Debug: Check filtered documents
        print(f"Filtered Documents for Chapter {chapter}: {len(chapter_docs)}")

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
        results = vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in results])
    
    return generate_answer(query, context)
    # Retrieve relevant documents from FAISS
    # results = vector_store.similarity_search(query, k=3)
    # context = "\n".join([doc.page_content for doc in results])  # Join top-k retrieved docs
    # return generate_answer(query, context)

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

    # Simulate retrieving relevant content for the chapter (e.g., from loaded PDFs)
    context = " ".join([doc.page_content for doc in documents if chapter in doc.metadata.get("title", "")])

    # Generate answer using the context and question
    answer = generate_answer(question, context)

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True) 