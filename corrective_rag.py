from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader


from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field



from dotenv import load_dotenv
load_dotenv()
dir_path = "./data"
#https://github.com/athina-ai/rag-cookbooks/blob/main/agentic_rag_techniques/corrective_rag.ipynb
llm_ollama = ChatOllama(
    model="hf.co/dearxoasis/llama3.2-typhoon2-offical_doc-3b:Q8_0",
            # model="hf.co/dearxoasis/llama-3-typhoon2_dearx:latest",
            temperature=0,
         
)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "mps"},
)

loader = DirectoryLoader(
            dir_path,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True,
            recursive=True,
    )
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

vector_store = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory="./chroma_db"
)



#Retriever
retriever = vector_store.as_retriever()


#Document Grader
class grade(BaseModel):
    binary_score:str = Field(description="คะแนนการเกี่ยวข้องของเอกสารที่ให้มาตอบ 'ใช่' หรือ 'ไม่ใช่'")





structured_llm = llm_ollama.with_structured_output(grade)

print(structured_llm.invoke("คุณคือใคร"))
