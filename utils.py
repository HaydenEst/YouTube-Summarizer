from langchain.document_loaders import YoutubeLoader
from langchain.prompts import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain
import pinecone as pin
import openai
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from itertools import chain
import os



# This function extracts the transcripts from the given youtube video
def raw_documents(yt_url):
    loader = YoutubeLoader.from_youtube_url(
            yt_url, 
            add_video_info=True
        )
    raw_docs = loader.load()
    return raw_docs

# This function split and stores the raw documents
def store_information(yt_url, chunk_size, chunk_overlap):
    raw_docs = raw_documents(yt_url)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap,
                                                   separators=["\n\n", ".", "\n", " ", ""])
    documents = text_splitter.split_documents(raw_docs)
    return documents

# This function splits the transcripts extracted from raw_documents, and split them.
# After the split, it uses the load_summarize_chain to create a summary of the entire video.
# This necassary to bypass the token limit that openai has, so that we can create
# summaries of longer videos!
def summarize_video(yt_url, apikey):
    llm = OpenAI(temperature=0, model="babbage-002", openai_api_key=apikey)
    documents = store_information(yt_url, 4000, 100)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary=chain.run(documents)
    return summary

# create and store embeddings for similarity searches
def vector_store(yt_url, api_key):
    documents = store_information(yt_url, 500, 50)
    db = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=api_key))
    return db

def pretty_print_docs(docs):
    return f"\n{'-' * 100}\n".join([f"Document {i+1}\n\n" + d.page_content for i, d in enumerate(docs)])

def search_query(yt_url, query, apikey):
    db = vector_store(yt_url, apikey)
    # Wrap our vectorstore
    llm2 = OpenAI(temperature=0, openai_api_key=apikey)
    compressor = LLMChainExtractor.from_llm(llm2)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=db.as_retriever()
    )
    compressed_docs = compression_retriever.get_relevant_documents(query)
    response = pretty_print_docs(compressed_docs)
    return response
