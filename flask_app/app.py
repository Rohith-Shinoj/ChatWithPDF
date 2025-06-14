import os
import re
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from transformers import pipeline

# Constants
UPLOAD_FOLDER = './uploads'
INDEX_FOLDER = './vectorstores'
ALLOWED_EXTENSIONS = {'pdf'}
EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-xl"

# Setup
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "HUGGINGFACEHUB_API_TOKEN"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Prompt template
prompt_template = """
You are a helpful assistant that answers questions using only the provided text chunk from a PDF document.

PDF Chunk:
{context}

User Question: {question}

Based only on the content of this chunk, provide a direct and concise answer to the question. 
If the chunk does not contain the answer, respond with "Not found in this chunk."
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_pdf_text(text):
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'\n+', ' ', text)
    return text

def process_pdf_and_save_index(pdf_path, index_path):
    loader = PyPDFLoader(file_path=pdf_path)
    document = loader.load()
    for i in range(len(document)):
        document[i].page_content = clean_pdf_text(document[i].page_content)

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    chunks = text_splitter.split_documents(document)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local(index_path)

# Routes
@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('upload_pdf'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_pdf():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            index_path = os.path.join(INDEX_FOLDER, filename.rsplit('.', 1)[0])

            file.save(pdf_path)
            process_pdf_and_save_index(pdf_path, index_path)

            return redirect(url_for('ask_questions', index_name=filename.rsplit('.', 1)[0]))
    return render_template('upload.html')

@app.route('/ask/<index_name>', methods=['GET', 'POST'])
def ask_questions(index_name):
    index_path = os.path.join(INDEX_FOLDER, index_name)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    generator = pipeline("text2text-generation", model=GEN_MODEL)
    llm = HuggingFacePipeline(pipeline=generator)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    if request.method == 'POST':
        question = request.form['question']
        if question.strip().lower() == "exit":
            return "Thanks for using chatPDF"
        answer = qa.run(question)
        return render_template("ask.html", pdf=index_name, question=question, answer=answer)

    return render_template("ask.html", pdf=index_name)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
