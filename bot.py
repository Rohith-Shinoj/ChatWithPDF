import warnings
warnings.filterwarnings("ignore")

import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import fitz

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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" 
prompt_template = """
You are an AI assistant trained to answer questions from PDF documents.
Use only the provided text chunk to answer the question.

PDF Chunk:
{context}

User Question: {question}

Answer based only on the chunk above and rephrase to summarize or make your answer clearer. Do not make assumptions.
If the answer is not available, respond with "The document does not provide that information so that you do not hallucinate.
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

def clean_pdf_text(text):
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'\n+', ' ', text)
    return text

def extract_images_from_pdf(pdf_path, save_dir="pdf_images"):
    os.makedirs(save_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            img_filename = f"{save_dir}/page_{i+1}_img_{img_index+1}.{image_ext}"
            with open(img_filename, "wb") as f:
                f.write(image_bytes)
    doc.close()


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def get_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

def extract_text_from_image(image_path):
    return pytesseract.image_to_string(Image.open(image_path))

if __name__ == "__main__":
    pdf_path = "sample.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    document = loader.load()
    for i in range(len(document)):
        document[i].page_content = clean_pdf_text(document[i].page_content)
    # print(document)

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    doc = text_splitter.split_documents(documents=document)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
    vectorstore = FAISS.from_documents(documents=doc, embedding=embeddings)
    vectorstore.save_local("QML-Learnings")
    
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    
    llm = HuggingFacePipeline(pipeline=generator)

    new_vectorstore = FAISS.load_local("QML-Learnings", embeddings,allow_dangerous_deserialization=True)
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=new_vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
    )

    user_query = ""
    while True:
        user_query = input("\nAsk (enter Exit to quit): ")
        if user_query.lower()!= "exit":
            result = qa.run(user_query)
            print("\nAnswer:", result)
        else:
            print("Thanks for using chatPDF")
            break