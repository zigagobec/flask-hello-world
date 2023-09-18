import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Load the herman database
hermanDB = Chroma(persist_directory=os.path.join('db', 'herman'), embedding_function=embeddings)
herman_retriever = hermanDB.as_retriever(search_kwargs={"k": 5})