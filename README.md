# RAGModel
This is a part of the onboarding project for ML@UVA's LMI project

In order to setup the project properly and run on your terminal, please follow these steps:

1. Clone this repository on IDE of choice
2. Run pip install -r requirements.txt after installing pip and python to get all packages and libraries
3. Create a .env file within main directory and add api key for OPEN_API_KEY = apikey code 
4. Go to main.py and run file to start the Q & A Chatbot.

The Project workflow is as follows:

1. main.py is run and processes the 3 txt files located in the textFiles folder
2. Then on embeddings.py the documentChunker method takes our full texts and makes a list of digestable document chunks using LangChain's Recursive Character Text Splitter and overlapping text for consecutive chunks
3. After, on embeddings.py, each of the list of documents is fed through a custom metadata modifier, so sources are cited with author name, article date, article title, and url/source.
4. Then after being inserted into a temporary ChromaDB vector database collection called "LMIProject", we enter the real loop.
5. First on queryProcess.py, the user enters a prompt and how many top k documents they want to search for and those are fed into chromadb's similarity search function.
6. After returning both the inputted query and a list of all the top k documents and associated metadata, we input these into another function called generateGPTResponse.
7. On gptChat.py, we use the generateGPTResponse function to organize our documents and each metadata, feed those into a prompt template, and use the gpt-4o-mini model to return a response.
8. After how many ever times the user wants to ask queries on the given texts, the user can exit using a simple command of q.

Used Technologies and Libraries:
1. Langchain and subsidiaries
2. Os
3. Python + Pycharm IDE
4. Git
5. Pandas
6. ChromaDB and subsidiaries
7. uuid
8. OpenAI API Key for embeddings and ChatCompletion.
