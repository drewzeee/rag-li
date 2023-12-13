import csv
import logging
import os
import sys

from llama_index.llms import Ollama
from llama_index.callbacks.base import CallbackManager
from llama_index import (
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.text_splitter import SentenceSplitter
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.embeddings.cohereai import CohereEmbedding
from configparser import ConfigParser
from llama_index.retrievers import BaseRetriever, BM25Retriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.vector_stores import FaissVectorStore

import chainlit as cl
import faiss

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


env_config = ConfigParser()


# Retrieve the cohere api key from the environmental variables
def read_config(parser: ConfigParser, location: str) -> None:
    assert parser.read(location), f"Could not read config {location}"


#
CONFIG_FILE = os.path.join(".", ".env")
read_config(env_config, CONFIG_FILE)
cohere_api_key = env_config.get("cohere", "api_key").strip()
os.environ["COHERE_API_KEY"] = cohere_api_key
DATA_PATH = (
    "C:/Users/andrew/OneDrive - Entegration Inc/Projects/oracle/SOURCE_DOCUMENTS/"
)
DB_PATH = "./storage"


@cl.on_chat_start
async def start():
    # load documents
    documents = SimpleDirectoryReader(DATA_PATH).load_data()

    # initialize service context (set chunk size)
    llm = Ollama(
        temperature=0,
        model="neural-chat",
    )

    embed_model = CohereEmbedding(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        model_name="embed-english-v3.0",
        input_type="search_query",
    )
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        text_splitter=SentenceSplitter(
            separator="\n\n", chunk_size=512, chunk_overlap=30
        ),
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )

    nodes = service_context.node_parser.get_nodes_from_documents(documents)

    if not os.path.exists(DB_PATH):
        vector_store = FaissVectorStore(
            faiss.IndexFlatL2(1024)
        )  # cohere embeddings have 1024 dimensions

        # initialize storage context (by default it's in-memory)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        storage_context.docstore.add_documents(nodes)
        index = VectorStoreIndex(
            nodes, storage_context=storage_context, service_context=service_context
        )
        index.storage_context.persist(persist_dir=DB_PATH)
    else:
        vector_store = FaissVectorStore.from_persist_dir(DB_PATH)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=DB_PATH
        )
        index = load_index_from_storage(
            storage_context, service_context=service_context
        )

    # retireve the top N most similar nodes using embeddings
    vector_retriever = index.as_retriever(similarity_top_k=3)
    # retireve the top N most similar nodes using bm25
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=3)

    class HybridRetriever(BaseRetriever):
        def __init__(self, vector_retriever, bm25_retriever):
            self.vector_retriever = vector_retriever
            self.bm25_retriever = bm25_retriever
            super().__init__()

        def _retrieve(self, query, **kwargs):
            bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
            vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

            # combine the two lists of nodes
            all_nodes = []
            node_hashes = set()
            for n in bm25_nodes + vector_nodes:
                if n.node.hash not in node_hashes:
                    all_nodes.append(n)
                    node_hashes.add(n.node.hash)
            return all_nodes

    index.as_retriever(similarity_top_k=3)
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

    cohere_rerank = CohereRerank(api_key=os.getenv("COHERE_API_KEY"), top_n=6)

    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[cohere_rerank],
        service_context=service_context,
        streaming=True,
    )
    cl.user_session.set("query_engine", query_engine)
    await cl.Message(author="Seraphina", content="Hello! How may I help you? ").send()


def log_chat(input_message, output_message):
    with open("chatlog.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([input_message.content, output_message.content])


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")
    response = await cl.make_async(query_engine.query)(message.content)
    response_message = cl.Message(content="")

    for token in response.response_gen:
        await response_message.stream_token(token=token)

    if response.response_txt:
        response_message.content = response.response_txt
    await response_message.send()

    log_chat(message, response_message)
