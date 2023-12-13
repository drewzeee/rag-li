import langchain
from langchain.embeddings import CacheBackedEmbeddings,HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.llms import Ollama
from langchain.retrievers import BM25Retriever,EnsembleRetriever
from langchain.document_loaders import DirectoryLoader
from langchain.llms import HuggingFacePipeline
from langchain.cache import InMemoryCache
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import prompt
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain import hub
import chainlit as cl
import csv
import os
from langchain.retrievers import ContextualCompressionRetriever
from langchain.embeddings import CohereEmbeddings
from configparser import ConfigParser
from langchain.retrievers.document_compressors import CohereRerank
from cohere import Client

import nltk
from llama_index.node_parser import SentenceWindowNodeParser

DATA_PATH="/home/ubuntu/OneDrive/Projects/oracle/SOURCE_DOCUMENTS/"
DB_PATH = "storage"

env_config = ConfigParser()
# Retrieve the cohere api key from the environmental variables
def read_config(parser: ConfigParser, location: str) -> None:
 assert parser.read(location), f"Could not read config {location}"
#
CONFIG_FILE = os.path.join(".", ".env")
read_config(env_config, CONFIG_FILE)
api_key = env_config.get("cohere", "api_key").strip()
os.environ["COHERE_API_KEY"] = api_key

loader = DirectoryLoader(DATA_PATH)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128, separators="\n\n\n")
texts=text_splitter.split_documents(documents)
print(f"number of chunks in storage : {len(texts)}")
store = LocalFileStore("./cache/")
embed_model_id = 'embed-english-light-v3.0'
core_embeddings_model = CohereEmbeddings(model=embed_model_id)
embedder = CacheBackedEmbeddings.from_bytes_store(core_embeddings_model,store,namespace=embed_model_id)
vectorstore = FAISS.from_documents(texts,embedder)
bm25_retriever = BM25Retriever.from_documents(texts)
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k":4})
bm25_retriever.k=4
ensemble_retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[0.3,0.7])
langchain.llm_cache = InMemoryCache()
#Cohere Reranker
 #
compressor = CohereRerank(client=Client(api_key=os.getenv("COHERE_API_KEY")),user_agent='langchain')
 #
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
base_retriever=ensemble_retriever,
)
#PROMPT_TEMPLATE = hub.pull("drewzy/rag-zephyr")
#input_variables = ['context', 'question']
#custom_prompt = PromptTemplate(template=PROMPT_TEMPLATE,
#                            input_variables=input_variables)
handler = StdOutCallbackHandler()
def load_llm():
    llm = Ollama(
        model="neural-chat",
        temperature=0,
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm
def retrieval_qa_with_sources_chain(llm,vectorstore):
    qa_with_sources_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever = compression_retriever,
        callbacks=[handler],
        #chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )
    return qa_with_sources_chain

def qa_bot(): 
    llm=load_llm() 
    DB_PATH = "storage"
    vectorstore = FAISS.from_documents(texts,embedder)
    qa = retrieval_qa_with_sources_chain(llm,vectorstore)
    return qa 

@cl.on_chat_start
async def start():
 chain=qa_bot()
 msg=cl.Message(content="Firing up the research info bot...")
 await msg.send()
 msg.content= "Good day, friend! How can I help you??"
 await msg.update()
 cl.user_session.set("chain",chain)

@cl.on_message
async def main(message):
 chain=cl.user_session.get("chain")
 cb = cl.AsyncLangchainCallbackHandler(
 stream_final_answer=True,
 answer_prefix_tokens=["FINAL", "ANSWER"]
 )
 cb.answer_reached=True
 await cl.sleep(2)
 res=await chain.acall(message.content, callbacks=[cb])
 #print(f"response: {res}")
 answer=res["result"]
     # Log the interaction to a CSV file
 with open('chatlog.csv', mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write header if file is empty
    if file.tell() == 0:
        writer.writerow(["User Message", "Bot Response"])
    # Write the data
    writer.writerow([message.content, answer])
# answer=answer.replace(".",".\n")
 # Append Sources to respons
 sources=res["source_documents"]
 text_elements = []  # type: List[cl.Text]
 if sources:
        for source_idx, source_doc in enumerate(sources):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]
        if source_names:
            answer += f"\n\nSources: {', '.join(source_names)}"
        else:
            answer += "\n\nNo sources found"
 await cl.Message(content=answer, elements=text_elements).send()
