import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set persist directory
persist_directory = 'db'

# Load documents
herman_loader = DirectoryLoader('./docs/herman/', glob="*.pdf", show_progress=True)
herman_docs = herman_loader.load()

# Initialize embeddings and text splitter
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150, length_function=len, is_separator_regex=False)

# Split documents and generate embeddings
herman_docs_split = text_splitter.split_documents(herman_docs)

# Create Chroma instances and persist embeddings
hermanDB = Chroma.from_documents(herman_docs_split, embeddings, persist_directory=os.path.join(persist_directory, 'herman'))
hermanDB.persist()