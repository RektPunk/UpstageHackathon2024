# Code Insights: From Upstage Global AI Week Online Hackathon Winners

Hello Upstage Global AI Week Participants! 🙌  
We were overwhelmed by the passion and effort you showed during the first round of qualifiers for the Upstage Global AI Week. There were so many great ideas, and we can still see you all working day and night to overcome the technical challenges that arose. 👨‍💻👩‍💻

Being a tech support, answering questions and finding solutions together has given me a tremendous opportunity for growth, and your enthusiasm has motivated and inspired me to keep going as a developer. 🙏

In this Code Insight article, we'd like to share some of the interesting coding patterns we found while analysing the code of the 30 teams that made it through the first round. We'll also provide useful code snippets and relevant recipes from the Upstage Cookbook, which we hope will help you in your own development.

So let's dive into the code of the developers who made it through the first gateway to winning Upstage Global AI Week. Let's get started!

## Common patterns using the Upstage Chat API

### Upstage Reference
- [Upstage Docs Chat](https://developers.upstage.ai/docs/apis/chat)

### Example 1: Python example using OpenAI Client

```python
import os
import openai

client = openai.Client(api_key=os.getenv("UPSTAGE_API_KEY"))

def summarize_article(article_text):
    stream = client.chat.completions.create(
        model="solar-1-mini-chat",
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert in summarizing articles. Below is a news article. 
                Your task is to summarize the key information, including all relevant facts, figures, and notable observations.
                Ensure that no important details are omitted. Your summary should be clear, concise, and no longer than 200 words, focusing only on the most significant points.
                If you find the article lacks sufficient information for a summary, you MUST return an empty response.
                """,
            },
            {
                "role": "user",
                "content": f"""
                This is the article you are to summarize:
                {article_text}
                """,
            },
        ],
        stream=False,
    )
    return stream.choices[0].message.content
```

This example shows how to summarise a news article using Upstage's OpenAI-compatible Chat API. In the system message, you give the AI the role of article summarisation expert, and in the user message, you tell it the content of the article to summarise. This allows you to clearly and concisely summarise the key information in the article in 200 characters or less.

### Example 2: Example using Langchain's Chat Upstage

```python
from langchain.chat_models import ChatUpstage
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage

class LegalAssistant:
    def __init__(self):
        upstage_api_key = "Your API key"
        self.chat = ChatUpstage(
            api_key=upstage_api_key,
            temperature=0.1,
            max_tokens=300,
        )
        self.prompt_template = PromptTemplate(
            input_variables=["legal_question"],
            template="""You are a helpful legal assistant. Answer the following legal question to the best of your ability:
            {legal_question}""",
        )

    def answer_question(self, legal_question: str):
        prompt = self.prompt_template.format(legal_question=legal_question)
        messages = [SystemMessage(content="You are a helpful legal assistant."), HumanMessage(content=prompt)]
        response = self.chat.invoke(messages).content
        return response
```

This example shows how to use the Chat Upstage class in the Langchain framework to answer a legal question. It uses PromptTemplate to create a prompt that combines a system message and a user question, and calls the Upstage Chat API with the chat.invoke() method. This allows you to get responses from the AI assistant for different legal scenarios.

## Extra RAG patterns using Upstage Retriever

Today, we'll take a look at some code from the Upstage Global AI Week first round qualifiers that leverages Retriever to create Retrieval-Augmented Generation (RAG) patterns, a technique that retrieves documents relevant to a query and provides additional information to generate more accurate and richer responses.

### Upstage Reference
- [Upstage Cookbook 05_1 ChromaDB.ipynb](https://github.com/UpstageAI/cookbook/blob/main/cookbooks/upstage/Solar-Full-Stack-LLM-101/05_1_ChromaDB.ipynb)
- [Upstage Cookbook 08 RAG.ipynb](https://github.com/UpstageAI/cookbook/blob/main/cookbooks/upstage/Solar-Full-Stack-LLM-101/08_RAG.ipynb)
- [Upstage Docs Embedding](https://developers.upstage.ai/docs/apis/embeddings)


### Example 1: Python example with Langchain, Upstage Chat Embedding

```python
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import UpstageEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import UpstageChat
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

loader = UnstructuredPDFLoader("data/product_manual.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
db = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=UpstageChat(), 
    chain_type="stuff", 
    retriever=db.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template="""
            Please provide a detailed answer to the following question based on the context provided.
            
            Question: {question}
            
            Context:
            {context}
            
            Answer:
            """,
            input_variables=["question", "context"]
        )
    }
)

query = "How do I troubleshoot the error code E23 on my product?"
result = qa(query)

print(result["result"])
print(result["source_documents"])
```

This example is an implementation of RAG leveraging the Langchain framework and Upstage's Chat Embedding. The main steps are as follows

1. load a PDF document with UnstructuredPDFLoader, and split the text into chunks with RecursiveCharacterTextSplitter.
2. embed the text chunks with UpstageEmbeddings, and store them in the Chroma vector database. 
3. Create a RetrievalQA chain, connecting the UpstageChat LLM and the Chroma searcher.
4. Take a user query, retrieve relevant documents, and generate a response based on the retrieved documents.

This allows you to find a solution to an error code in the product manual and present it to the user.

### Example 2: Simple Retriever Python example with Chroma

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import UpstageEmbeddings
from langchain.llms import UpstageChat
from langchain.chains import ConversationalRetrievalChain

persist_directory = './.cache/db'

embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

retriever = vectordb.as_retriever()

qa = ConversationalRetrievalChain.from_llm(
    UpstageChat(), 
    retriever=retriever,
    return_source_documents=True
)

chat_history = []

while True:
    query = input("Human: ")
    
    result = qa({"question": query, "chat_history": chat_history})
    
    print(f"Assistant: {result['answer']}")
    print(f"Source Documents: {result['source_documents']}")
    
    chat_history.append((query, result["answer"]))
```

This example implements a simple interactive RAG system using the Chroma vector database and the UpstageChat LLM. The main steps are as follows

1. define embedding functions with UpstageEmbeddings, and initialise the Chroma database.
2. create a ConversationalRetrievalChain, and connect the UpstageChat LLM and the Chroma searcher.
3. Take user input, create a query with previous dialogue history, search for relevant documents, and generate a response.
4. Output the generated response and retrieved documents, and add the current question-answer pair to the dialogue history.

This allows you to create an interactive system that can dynamically respond to a user's question while taking into account the context of the previous dialogue.

### Example 3: Python example of implementing a RAG based on the Predibase API

```python
import os
import pandas as pd
from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader, 
    ServiceContext,
    StorageContext,
    load_index_from_storage
)
from llama_index.vector_stores import ChromaVectorStore
from llama_index.embeddings import UpstageEmbedding
from langchain.llms import Predibase

os.environ['PREDIBASE_API_TOKEN'] = 'YOUR_API_KEY'
os.environ['UPSTAGE_API_KEY'] = 'YOUR_API_KEY'

df = pd.read_csv('data/jeju_reviews.csv')
df['text'] = df['poi_name'] + ' - ' + df['reviews'] 

documents = [doc for doc in df['text']]

embed_model = UpstageEmbedding(model="solar-embedding-1-large")
service_context = ServiceContext.from_defaults(embed_model=embed_model)

vector_store = ChromaVectorStore(documents, service_context=service_context)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context)

index.storage_context.persist()

llm = Predibase(
    model_name="solar-1-mini-chat-240612",
    adapter_id="jeju-info-model",
    adapter_version=1,
    api_token=os.environ.get("PREDIBASE_API_TOKEN"),
    temperature=0.3,
    max_new_tokens=600,
)

query_engine = index.as_query_engine(
    service_context=ServiceContext.from_defaults(llm=llm),
    similarity_top_k=3
)

query = "What are the top attractions in Jeju Island according to the reviews?"
response = query_engine.query(query)

print(response)
```

This example uses Predibase APIs and Upstage Embedding to implement a RAG system for Jeju Island tourist review data. The main steps are as follows

1. Load and preprocess the Jeju Island tourist review CSV data.
2. Embed the review text using UpstageEmbedding and ChromaVectorStore and store it in a vector database.
3. Create a GPTVectorStoreIndex and connect the Predibase LLM to the query engine.
4. Input a user query, retrieve relevant reviews based on similarity, and generate responses using Predibase LLM.

This allows us to extract and summarise information that meets the user query from the vast data of Jeju Island tourism reviews.

Retriever enables us to quickly retrieve information relevant to a query from vast amounts of data to generate more accurate and richer responses. 

## Upstage RAG with Reranking to leverage different patterns

Let's take a look at examples of how Retrieval-Augmented Generation (RAG) uses Reranking techniques such as BM25 to generate more accurate responses by selecting the most important documents from the retrieved documents and using them as context.

### Upstage Reference
- [Upstage Cookbook 09 Smart RAG.ipynb](https://github.com/UpstageAI/cookbook/blob/main/cookbooks/upstage/Solar-Full-Stack-LLM-101/09_Smart_RAG.ipynb)

### Example 1: RAG example with the addition of Langchain-based BM25

```python
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def load_documents(pdf_files):
    loaders = []
    docs = []

    for i, pdf_file in enumerate(pdf_files, start=1):
        print(f"Processing file {i}: {pdf_file}")
        loader = UnstructuredPDFLoader(pdf_file)
        loaders.append(loader)
        doc = loader.load()
        docs.append(doc)
        print(f"File {i} processed successfully.")

    print("All files have been processed.")
    return docs

def process_documents(docs, queries):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    embeddings = OpenAIEmbeddings()
    llm = OpenAI()
    
    prompt_template = PromptTemplate(
        template="""
        Please provide the most relevant answer from the given context.
        
        Question: {question}
        
        Context:
        {context}
        
        Answer:
        """,
        input_variables=["question", "context"]
    )
    
    results = {}
    for i, doc in enumerate(docs):
        splits = text_splitter.split_documents(doc)
        
        bm25_retriever = BM25Retriever.from_documents(splits)
        vector_store = FAISS.from_documents(splits, embeddings)
        
        retriever = bm25_retriever
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        
        doc_results = {}
        for query in queries:
            result = qa_chain({"query": query})
            doc_results[query] = result["result"]
            print(f"Document {i}, Query: {query}")
            print(result["result"])
            print("---")
        
        results[i] = doc_results
    
    return results
```

This example is an adaptation of BM25 Retriever to RAG using the Langchain framework. The main steps are as follows

1. load a PDF document with UnstructuredPDFLoader, and split the text into chunks with RecursiveCharacterTextSplitter.
2. embed the text chunks with OpenAIEmbeddings and save them to the FAISS vector database.
3. Create a BM25Retriever to use the BM25 algorithm for retrieval. 
4. Create a RetrievalQA chain, connecting the OpenAI LLM and BM25 Retriever.
5. Take a user query, retrieve relevant documents with BM25, and generate a response based on the retrieved documents.

The BM25 algorithm calculates the similarity between the query and documents and selects the most relevant documents to generate a more accurate response.

### Example 2: Example using Predibase-served models and Faiss Vector DB, MultiQuery Retriever, etc.

```python
import os
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.llms import Predibase
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
os.environ["PREDIBASE_API_KEY"] = "YOUR_API_KEY" 

def combine_text(row):
    return f"Product Name: {row['Product Namel']}\nIngredients: {row['Ingredientsl']}\nEfficacy: {row['Efficacy']}\nDosing: {row['Dosage']}"

def load_dataframe(csv_file):
    df = pd.read_csv(csv_file)
    df["text"] = df.apply(combine_text, axis=1)
    return df

def create_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore

def create_retrievers(vectorstore, documents):
    multiquery_retriever = MultiQueryRetriever.from_llm(
        llm=Predibase(model_name="solar-1-mini-chat-240612", adapter_id="health-qa/1", api_key=os.environ["PREDIBASE_API_KEY"]),
        retriever=vectorstore.as_retriever()
    )
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[multiquery_retriever, bm25_retriever],
        weights=[0.7, 0.3],
        search_type="mmr"
    )
    
    return ensemble_retriever

def create_qa_chain(retriever, prompt_template):
    llm = Predibase(model_name="solar-1-mini-chat-240612", adapter_id="health-qa/1", api_key=os.environ["PREDIBASE_API_KEY"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    
    return qa_chain

csv_file = "data/health_supplements.csv"
df = load_dataframe(csv_file)

loader = DataFrameLoader(df, page_content_column="text")
documents = loader.load()

vectorstore = create_vectorstore(documents)
retriever = create_retrievers(vectorstore, documents)

prompt_template = PromptTemplate(
    template="""
    Give the best answer to the question based on the following context
    
    Question: {query}
    
    Context:
    {context}
    
    Answer:
    """,
    input_variables=['query', 'context']
)

qa_chain = create_qa_chain(retriever, prompt_template)

query = 'What are the benefits of propolis?'
result = qa_chain({'query': query})

print(result['result'])
print(result['source_documents'])
```

This example implements a RAG system for nutraceutical data using the Solar model served by Predibase, FAISS vector database, MultiQuery Retriever, BM25 Retriever, etc. The main steps are as follows:

1. Load and preprocess the dietary supplement CSV data.
2. Split the text into chunks with RecursiveCharacterTextSplitter, embed with OpenAIEmbeddings, and store in FAISS vector store.
3. Create a MultiQuery Retriever and a BM25 Retriever and combine them with EnsembleRetriever. Use the MMR method to increase the diversity of the search results.
4. Load the Solar model served by Predibase and create a RetrievalQA chain.
5. Take a user query, retrieve relevant documents with EnsembleRetriever, and generate responses with the Solar model.

By ensembling MultiQuery Retriever and BM25 Retriever, you can improve search performance and generate more comprehensive and accurate responses by ensuring the diversity of search results in an MMR fashion.

In the above, we have seen various patterns using Reranking in Upstage RAG. You can improve search quality through techniques such as BM25 and MultiQuery Retriever, and build a more powerful RAG system by combining various retrievers in an ensemble.

## RAG patterns using Upstage Layout Analysis and OCR APIs

Let's take a look at examples of using Upstage Layout Analysis and the OCR API to extract information from documents and perform Retrieval-Augmented Generation (RAG) based on the extracted information.

### Upstage Reference
- [Upstage Cookbook 07 LA_CAG.ipynb](https://github.com/UpstageAI/cookbook/blob/main/cookbooks/upstage/Solar-Full-Stack-LLM-101/07_LA_CAG.ipynb)
- [Upstage Docs Layout Analysis](https://developers.upstage.ai/docs/apis/layout-analysis)
- [Upstage Docs OCR](https://developers.upstage.ai/docs/apis/document-ocr)

### Example 1: RAG example using Upstage Layout Analysis and Embedding

```python
import os
from typing import List
from fastapi import UploadFile, HTTPException
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

class DocumentService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0.7)
        self.vector_store_dir = "./vector_store"
        os.makedirs(self.vector_store_dir, exist_ok=True)

    async def process_pdf(self, file: UploadFile, collection_name: str) -> List[str]:
        loader = UnstructuredPDFLoader(file.file)
        pages = loader.load_and_split()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(pages)
        
        vector_store = Chroma.from_documents(docs, self.embeddings, collection_name=collection_name)
        vector_store.persist(self.vector_store_dir)
        
        return [doc.page_content for doc in docs]

    async def retrieve_and_answer(self, query: str, collection_name: str) -> str:
        vector_store = Chroma(collection_name=collection_name, embedding_function=self.embeddings, persist_directory=self.vector_store_dir)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        result = qa_chain({"query": query})
        return result["result"]

    async def handle_upload(self, file: UploadFile, collection_name: str):
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")
        
        extracted_text = await self.process_pdf(file, collection_name)
        return {"message": f"PDF processed successfully. Extracted {len(extracted_text)} text chunks."}

    async def handle_query(self, query: str, collection_name: str):
        answer = await self.retrieve_and_answer(query, collection_name)
        return {"query": query, "answer": answer}
```

This example uses FastAPI and Langchain to extract text from a PDF file with Upstage Layout Analysis, embed the extracted text and store it in a Chroma vector database, and then retrieve relevant documents for a user's question and generate an answer using a RetrievalQA chain.

The main process is as follows:

1. Extract text from the PDF file using Upstage Layout Analysis in the `process_pdf` method, and split the text into chunks with RecursiveCharacterTextSplitter.
2. Embed the text chunks using OpenAIEmbeddings and store them in the Chroma vector database.
3. In the `retrieve_and_answer` method, retrieve relevant documents based on similarity from the Chroma vector database for the user question.
4. Use the RetrievalQA chain to generate answers to the question based on the retrieved documents.

This allows you to effectively search and utilise the content of PDF documents.

### Example 2: Example of food allergy analysis using Upstage Layout Analysis

```python
import os
import re
from typing import List, Dict
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel

class AllergenInfo(BaseModel):
    name: str
    level: float
    category: str

class AllergenReport(BaseModel):
    allergens: List[AllergenInfo]

class FoodAnalysisService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0.7)
        self.vector_store_dir = "./vector_store"
        os.makedirs(self.vector_store_dir, exist_ok=True)

    def load_allergen_report(self, file_path: str) -> AllergenReport:
        loader = UnstructuredPDFLoader(file_path)
        pages = loader.load_and_split()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(pages)
        
        vector_store = Chroma.from_documents(docs, self.embeddings)
        vector_store.persist(self.vector_store_dir)
        
        allergen_info = []
        for doc in docs:
            allergens = self.extract_allergens(doc.page_content)
            allergen_info.extend(allergens)
        
        return AllergenReport(allergens=allergen_info)

    def extract_allergens(self, text: str) -> List[AllergenInfo]:
        pattern = r"(\w+)\s*:\s*(\d+\.\d+)\s*(\w+)"
        matches = re.findall(pattern, text)
        
        allergens = []
        for match in matches:
            name = match[0]
            level = float(match[1])
            category = match[2]
            allergens.append(AllergenInfo(name=name, level=level, category=category))
        
        return allergens

    def analyze_product(self, product_name: str, ingredients: List[str]) -> Dict[str, str]:
        prompt = PromptTemplate(
            input_variables=["product_name", "ingredients"],
            template="""
            Analyze the suitability of the ingredients in the product "{product_name}" based on the allergen report.
            
            Ingredients: {ingredients}
            
            For each ingredient, determine its suitability as follows:
            - If the ingredient is not found in the allergen report or has a "Normal" level, classify it as "Suitable".
            - If the ingredient has a "Borderline" level, classify it as "Caution".
            - If the ingredient has an "Elevated" level, classify it as "Avoid".
            - If there is insufficient information, classify it as "Unknown".
            
            Provide the analysis in the following format:
            Ingredient: Suitability
            """
        )
        
        allergen_report = self.load_allergen_report("allergen_report.pdf")
        
        query = prompt.format(product_name=product_name, ingredients=", ".join(ingredients))
        
        vector_store = Chroma(embedding_function=self.embeddings, persist_directory=self.vector_store_dir)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        result = qa_chain({"query": query})
        
        suitability = {}
        for line in result["result"].split("\n"):
            if ":" in line:
                ingredient, suit = line.split(":")
                suitability[ingredient.strip()] = suit.strip()
        
        return suitability
```

This example uses Upstage Layout Analysis to implement a service that extracts information from a food allergy report PDF and analyses the ingredient compliance of a product based on the extracted information.

The main steps are as follows:

1. Extract text from an allergy report PDF using Upstage Layout Analysis in the `load_allergen_report` method, and split the text into chunks using RecursiveCharacterTextSplitter.
2. Parse the allergen information from the extracted text using regular expressions and store it as an `AllergenInfo` object.
3. In the `analyse_product` method, take in the product name and ingredient list and analyse the suitability of each ingredient based on the allergy report.
4. Create an analysis query using PromptTemplate to retrieve relevant information from the Chroma vector database.
5. Use the RetrievalQA chain to determine ingredient suitability based on the retrieved information and return the results.

This allows you to leverage food allergy reports to automatically analyse the ingredient suitability of products.

### Example 3: Example of creating a quiz using the Upstage OCR API

```python
import os
import requests
import json
from typing import List, Dict
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class QuizGenerator:
    def __init__(self):
        self.llm = OpenAI(temperature=0.7)
        self.upstage_api_key = os.environ["UPSTAGE_API_KEY"]

    def extract_text_from_image(self, image_path: str) -> str:
        url = "https://api.upstage.ai/v1/document-ai/ocr"
        headers = {"Authorization": f"Bearer {self.upstage_api_key}"}
        
        with open(image_path, "rb") as image_file:
            files = {"document": image_file}
            response = requests.post(url, headers=headers, files=files)
        
        ocr_result = response.json()
        text = " ".join(page["text"] for page in ocr_result["pages"])
        return text

    def generate_quiz(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Generate a quiz based on the following text:
            
            {text}
            
            Create 5 multiple-choice questions, each with 4 options and only one correct answer.
            Provide the quiz in the following JSON format:
            {{
                "questions": [
                    {{
                        "question": "Question text",
                        "options": [
                            "Option 1",
                            "Option 2",
                            "Option 3",
                            "Option 4"
                        ],
                        "answer": "Correct option"
                    }},
                    ...
                ]
            }}
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        quiz_json = chain.run(text)
        quiz = json.loads(quiz_json)
        
        return quiz

    def generate_quiz_from_image(self, image_path: str) -> Dict[str, List[Dict[str, str]]]:
        text = self.extract_text_from_image(image_path)
        quiz = self.generate_quiz(text)
        return quiz
```

This example implements the ability to extract text from an image using the Upstage OCR API and create a quiz based on the extracted text.

The main steps are as follows

1. Call the Upstage OCR API in the `extract_text_from_image` method to extract text from an image file. 2.
2. generate a quiz based on the extracted text in the `generate_quiz` method.
3. define a quiz generation prompt using PromptTemplate, and pass the prompt to the OpenAI model via LLMChain to generate the quiz.
4. The generated quiz is returned in JSON format, with each question consisting of a question, choices, and an answer.

This allows you to extract text from images and effectively utilise that textual data.

These are just a few examples of different RAG patterns using the Upstage Layout Analysis and OCR APIs. By extracting and structuring information from documents, you can transform vast amounts of unstructured data into a searchable form and perform RAG on it to provide more accurate and richer responses to user queries.

## More examples using the Upstage Translate API

Let's take a look at some examples that leverage the Upstage Translate API to implement various translation features. These examples demonstrate how you can leverage the Translate API to convert text into multiple languages.

### Upstage Reference
- [Upstage Cookbook 11 summary_writeing_translation.ipynb](https://github.com/UpstageAI/cookbook/blob/main/cookbooks/upstage/Solar-Full-Stack-LLM-101/11_summary_writing_translation.ipynb)
- [Upstage Docs Translate](https://developers.upstage.ai/docs/apis/translation)

### Example 1: Translating a response to a user request

```python
import os
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage

class TranslationService:
    def __init__(self):
        self.api_key = os.getenv("UPSTAGE_API_KEY")
        self.translator_ko2en = ChatUpstage(api_key=self.api_key, model="solar-1-mini-translate-koen")
        self.translator_en2ko = ChatUpstage(api_key=self.api_key, model="solar-1-mini-translate-enko")

    def translate_ko2en(self, message: str) -> str:
        messages = [HumanMessage(content=message)]
        response = self.translator_ko2en.invoke(messages)
        return response.content

    def translate_en2ko(self, message: str) -> str:
        messages = [HumanMessage(content=message)]
        response = self.translator_en2ko.invoke(messages)
        return response.content

    def handle_translation_request(self, message: str, target_language: str) -> str:
        if target_language == "en":
            return self.translate_ko2en(message)
        elif target_language == "ko":
            return self.translate_en2ko(message)
        else:
            raise ValueError("Unsupported target language")

# Example usage
if __name__ == "__main__":
    service = TranslationService()
    original_text = l'Hi, we're testing our translation service.'
    translated_text = service.handle_translation_request(original_text, "en")
    print(f"Translated to English: {translated_text}")
```

This example is a service that uses the Upstage Translate API to perform a translation between Korean and English. The main steps are as follows

1. The `TranslationService` class initialises a translation model using the Upstage Translate API. 2.
2. The `translate_en2en` method translates a Korean message into English, and the `translate_en2en` method translates an English message into Korean.
3. The `handle_translation_request` method calls the appropriate translation method based on the translation language requested by the user.

This allows the user to easily translate the entered text into the desired language.

### Example 2: Example of a service to help with prescriptions.

```python
import os
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage

class MedicalTranslationService:
    def __init__(self):
        self.api_key = os.getenv("UPSTAGE_API_KEY")
        self.translator_ko2en = ChatUpstage(api_key=self.api_key, model="solar-1-mini-translate-koen")
        self.translator_en2ko = ChatUpstage(api_key=self.api_key, model="solar-1-mini-translate-enko")

    def translate_medical_ko2en(self, message: str) -> str:
        messages = [HumanMessage(content=message)]
        response = self.translator_ko2en.invoke(messages)
        return response.content

    def translate_medical_en2ko(self, message: str) -> str:
        messages = [HumanMessage(content=message)]
        response = self.translator_en2ko.invoke(messages)
        return response.content

    def provide_medical_advice(self, question: str, language: str = "en") -> str:
        if language == "ko":
            question_translated = self.translate_medical_ko2en(question)
        else:
            question_translated = question

        advice = f"Medical advice for: {question_translated}"

        if language == "ko":
            advice_translated = self.translate_medical_en2ko(advice)
            return advice_translated
        else:
            return advice

if __name__ == "__main__":
    service = MedicalTranslationService()
    medical_question = "의사 선생님, 두통이 심할 때 어떻게 해야 하나요?"
    advice = service.provide_medical_advice(medical_question, language="ko")
    print(f"Medical Advice: {advice}")
```

This example is a service that provides medical advice, which uses the translation feature to translate user questions into English, generate medical advice, and then translate it back into Korean and serve it to the user. The main steps are as follows:

1. The `MedicalTranslationService` class initialises a translation model using the Upstage Translate API.
2. the `translate_medical_en2en` method translates Korean medical questions into English, and the `translate_medical_en2ko` method translates the medical advice generated in English into Korean.
3. The `provide_medical_advice` method translates the user's question, generates medical advice, and returns the translated advice.

This allows the user to enter a medical question and receive translated advice.

We've seen a variety of examples using the Upstage Translate API, which can help you implement multilingual capabilities to better serve your global audience. We encourage you to develop creative applications using the Upstage Translate API for a variety of translation scenarios.

We've seen many examples of how you can use the Upstage API, and we hope that your passion and creativity will inspire you to use these technologies to create more innovative projects. We look forward to seeing your work at the final hackathon, and we'll be cheering you on as you continue to develop. Thank you for your journey with Upstage, and we hope to see you again with successful projects. Cheers!
🚀
