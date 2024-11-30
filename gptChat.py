from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


def generateGPTResponse(query, listOfTopKDocuments):
    documentsReformatted = "\n\n --- \n\n".join([f"""DOCUMENT {i + 1}: \n Document Content:\n {document['content']} \n Document Author: {document["author_name"]} \n Document Date: {document["article_date"]} \n Document Title: {document["article_title"]} \n Document Source: {document["source"]}""" for i, document in enumerate(listOfTopKDocuments)])

    PROMPT_TEMPLATE = f"""
    You are a RAG-AI ChatBot with access to three articles. When given a query, respond only with information relevant 
    to the content of these articles. Ensure your answers are concise, contextually accurate, and supported by source 
    citations. Use the provided metadata—author, article date, article title, and source—to attribute information 
    appropriately in the response. Format citations in a clear and professional style.  
    
    **Context**: {documentsReformatted}

    **Query**: {query}  
    """

    curr_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    curr_prompt = curr_template.format(context=documentsReformatted, question=query)

    curr_model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

    response = curr_model.invoke(curr_prompt)

    print("Response: \n "+ response.content)
    print()




