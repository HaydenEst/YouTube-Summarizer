from langchain.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
import openai
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import pydantic
import tiktoken



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
    error = False
    try:
        llm = OpenAI(temperature=0, model="text-davinci-003", openai_api_key=apikey)
        #llm = OpenAI(temperature=0, model="babbage-002", openai_api_key=apikey)
        documents = store_information(yt_url, 4000, 100)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(documents)
        return [summary, error]
    except openai.AuthenticationError:
        invalid_key_message = "Your OpenAI API key is invalid. Please try another."
        error = True
        return [invalid_key_message, error]
    except pydantic.ValidationError:
        invalid_key_message = "Please input your OpenAI API key. Access it in the sidebar by clicking the arrow in the top left corner of the page."
        error = True
        return [invalid_key_message, error]
    except ValueError:
        invalid_key_message = "Please input a valid YouTube url and OpenAI API Key. Input your API key in the sidebar by clicking the arrow in the top left corner of the page."
        error = True
        return [invalid_key_message, error]


# create and store embeddings for similarity searches
def vector_store(yt_url, api_key):
    try:
        documents = store_information(yt_url, 500, 0)
        db = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=api_key))
        return db
    except AttributeError:
        invalid_key_message = "Your OpenAI API key is invalid. Please try another."
        return invalid_key_message

#def pretty_print_docs(docs):
#    return f"\n{'-' * 100}\n".join([f"Document: {i+1}\n\n" + d.page_content for i, d in enumerate(docs)])

def search_query(yt_url, query, apikey):
    db = vector_store(yt_url, apikey)
    if db is None:
        return "Failed to retrieve data. Please make sure your API key is correct."
    # Wrap our vectorstore
    llm2 = OpenAI(temperature=0, openai_api_key=apikey)
    compressor = LLMChainExtractor.from_llm(llm2)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=db.as_retriever()
    )
    compressed_docs = compression_retriever.get_relevant_documents(query)
    # response = pretty_print_docs(compressed_docs)
    docs_list = []
    for item in compressed_docs:
        start_index = str(item).find("page_content=") + len("page_content=")
        end_index = str(item).find("metadata")
        start_index != -1 and end_index != -1
        str_doc = str(item)[start_index:end_index]
        docs_list.append(str_doc)

    return docs_list
