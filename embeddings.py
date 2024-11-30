import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma # we will use chroma as a way to centrally maintain all out text file embeddings into a non relational database
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

embeddingFunction = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)

vector_store = Chroma(
    collection_name="LMIProject",
    embedding_function=embeddingFunction,
    persist_directory="./chroma_langchain_db",
)

def documentChunker(text):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 500,
      chunk_overlap = 50,
  )

  document_chunks = text_splitter.create_documents([text])
  return document_chunks


def chunkWithMetadata(document_chunk, text_source):
  listOfDocuments = []
  counter = 1
  for chunk in document_chunk:
      curr_document = Document(
          page_content=chunk.page_content,
          metadata={
              "source": text_source['source'],
              "article_title": text_source["article_title"],
              "article_author": text_source["article_author"],
              "article_date": text_source["article_date"]
          },
          id=counter
      )
      counter += 1
      listOfDocuments.append(curr_document)

  return listOfDocuments


def insertIntoChromaDB(documentList, idsList):
    vector_store.add_documents(documents=documentList, ids=idsList)
