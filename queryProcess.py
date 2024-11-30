from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from embeddings import embeddingFunction
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

def queryProcess():
    try:
        db = Chroma(collection_name="LMIProject",
                    persist_directory="chroma_langchain_db",
                    embedding_function=embeddingFunction)

        print("Please enter in your query: ")
        query = str(input())

        print("How many documents would you prefer for a similarity search: ")
        topKNum = int(input())

        topResults = db.similarity_search(query=query, k=topKNum)

        similarity_search_results = []

        for eachResult in topResults:
            page_content = eachResult.page_content
            metadata = eachResult.metadata

            author = metadata['article_author']
            article_date = metadata['article_date']
            article_title = metadata['article_title']
            source = metadata['source']

            docChunkObject = {
                "content": page_content,
                "author_name": author,
                "article_date": article_date,
                "article_title": article_title,
                "source": source
            }

            similarity_search_results.append(docChunkObject)

        return query, similarity_search_results

    except Exception as e:
        print(f"An Error Occurred {e}")


