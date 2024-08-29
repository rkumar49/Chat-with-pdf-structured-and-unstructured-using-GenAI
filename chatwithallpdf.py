import os
import subprocess
from PIL import Image
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain_openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your api key"

# Set the path to the Poppler binaries
poppler_path = r"C:\Users\kumar\Downloads\Release-24.07.0-0\poppler-24.07.0\Library\bin"

# Path to Tesseract executable (Update if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def convert_pdf_to_images(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    command = [
        os.path.join(poppler_path, "pdftoppm"),
        "-jpeg",  # Output format
        pdf_path,  # Input PDF file
        os.path.join(output_dir, "output")  # Output filename base
    ]
    try:
        subprocess.run(command, check=True)
        print("PDF conversion to images successful.")
    except subprocess.CalledProcessError as e:
        print(f"Error during PDF to image conversion: {e}")

def ocr_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    try:
        # Attempt to extract text using PyPDF2
        pdfreader = PdfReader(pdf_path)
        raw_text = ''
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
        if raw_text.strip():
            return raw_text
        else:
            # Fallback to OCR if PyPDF2 does not extract any text
            temp_images_dir = "temp_images"
            convert_pdf_to_images(pdf_path, temp_images_dir)
            ocr_text = ''
            for filename in os.listdir(temp_images_dir):
                if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                    image_path = os.path.join(temp_images_dir, filename)
                    ocr_text += ocr_image(image_path)
            return ocr_text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ''

# Load and extract text
pdf_path = "E50115-B3464-S505-E.pdf"
raw_text = extract_text_from_pdf(pdf_path)

print(f"Extracted text length: {len(raw_text)}")  # Debugging print

# Split the text into manageable chunks
class SimpleTextSplitter:
    def __init__(self, chunk_size=800, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

text_splitter = SimpleTextSplitter(chunk_size=800, chunk_overlap=200)
texts = text_splitter.split_text(raw_text)

print(f"Number of text chunks: {len(texts)}")  # Debugging print

# Check if texts are empty
if not texts:
    print("No text chunks available. Exiting.")
    exit()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Create a FAISS vector store from the text chunks
try:
    document_search = FAISS.from_texts(texts, embeddings)
    print("FAISS vector store created successfully.")  # Debugging print
except Exception as e:
    print(f"Error creating FAISS vector store: {e}")

# Set up the QA chain
try:
    llm = OpenAI()
    retriever = document_search.as_retriever()
    
    # Create RetrievalQA chain
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
except Exception as e:
    print(f"Error setting up QA chain: {e}")

# Test the chain with a sample query
test_query = "What is the focus of the document?"
try:
    test_response = chain.run({"query": test_query})
    print(f"Test Response: {test_response}")
except Exception as e:
    print(f"Error during QA chain execution: {e}")

# Interactively ask the user for queries
while True:
    query = input("Enter your query (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    try:
        response = chain.run({"query": query})
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error during QA chain execution: {e}")
