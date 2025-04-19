from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI,OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate,PromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader,UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_community.tools import TavilySearchResults

#Graph 
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
#load env 
from dotenv import load_dotenv
import os
load_dotenv()



#get all files in folder data_markdown
files = "./data_markdown"
PATHUMMA_LLM_8B = "hf.co/dearxoasis/Pathumma_LLM_local:Q8_0"
TYPHOON_3B = "hf.co/dearxoasis/llama3.2-typhoon2-offical_doc-3b:Q8_0"
llm_ollama = ChatOllama(
    model=PATHUMMA_LLM_8B,
            # model="hf.co/dearxoasis/llama-3-typhoon2_dearx:latest",
            temperature=0,
            
         
)
typhoon_70b = ChatOpenAI(
    model="scb10x/llama3.1-typhoon2-70b-instruct",
    base_url="https://openrouter.ai/api/v1",
    api_key = "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",

)



#load directory
loader = DirectoryLoader(
    files, 
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader,
    recursive=True
)

documents = loader.load()

chunker = RecursiveCharacterTextSplitter(
    chunk_size = 1024,
    chunk_overlap = 250,
    length_function = len,
    separators=["\n\n", "\n", ". ", " ", ""] 
)


doc_split = chunker.split_documents(documents)


#TODO:  try to use Rerank


#create vector store
# vectorstore = Chroma.from_documents(
#     documents=documents,
#     embedding=OpenAIEmbeddings(),
#     # persist_directory="chroma_db",
# )
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
retriever = FAISS.from_documents(
    documents=doc_split,
    embedding=CohereEmbeddings(model="embed-multilingual-v3.0")

).as_retriever(search_kwargs={"k":3})

#create retriever
# retriever = vectorstore.as_retriever()




#Document Grader 
class GradeDocument(BaseModel):
    binary_score:str = Field(description="คะแนนความเกี่ยวข้อง ตอบเพียง 'ใช่' หรือ 'ไม่ใช่' เท่านั้น")
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
structured_llm = llm.with_structured_output(GradeDocument)
system_prompt =  """คุณเป็นผู้ประเมินความเกี่ยวข้องของเอกสารที่ค้นพบกับคำถามของผู้ใช้ \n
    การประเมินไม่จำเป็นต้องเข้มงวดมากนัก เป้าหมายคือการกรองเอกสารที่ไม่เกี่ยวข้องออกไป \n
    หากเอกสารมีคำสำคัญหรือความหมายเชิงความหมายที่เกี่ยวข้องกับคำถามของผู้ใช้ ให้ประเมินว่าเกี่ยวข้อง \n
    ให้คะแนนแบบทวิภาค 'ใช่' หรือ 'ไม่ใช่' เพื่อระบุว่าเอกสารนั้นเกี่ยวข้องกับคำถามหรือไม่"""


grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Retrieved document: \n\n {docs} \n\n User question: {question}"),
    ]
)


retrieval_grader = grade_prompt | structured_llm




 # Create system and human message templates
system_prompt_msg = """คุณเป็นผู้เชี่ยวชาญในการเขียนหนังสือราชการบันทึกข้อความ
    โครงสร้างตัวอย่างการตอบคำถาม(ต้องตอบตามโครงสร้างเท่านั้น):
    1. ส่วนหัวเรื่อง (หน่วยงาน, ที่, วันที่)
    2. ชื่อเรื่อง
    3. คำขึ้นต้น (เรียน)
    4. เนื้อเรื่อง 
        - เรื่องเดิม ต้องขึ้นต้นด้าย ตามบันทึก
        - ข้อพิจารณา คือ ข้อมูลที่เป็นรูปแบบความคิดเห็น หรือ ข้อมูลที่เป็นรูปแบบการขออนุมัติ  (ถ้ามี) 
        - ถ้าเป็นการขออนุมัติ ให้เป็น ข้อพิจารณา เท่านั้น
        
    5. (ลงชื่อ, ตำแหน่ง)
   
    ห้ามเขียน [ระบุข้อมูล] 
    - ตอบเป็นภาษาไทยทางการเท่านั้น
    -เขียนบันทึกข้อความโดยมีโครงสร้างหนังสือตามด้านบนและเว้นบรรทัดให้มีระเบียบ
    กรุณาสร้างหนังสือราชการโดยใช้รูปแบบ Markdown และรักษารูปแบบหนังสืออย่างเคร่งครัด:"""
# prompt = PromptTemplate(
#     template=template,
#     input_variables=["context", "question"]
# )
main_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_msg),
    ("human", "ตัวอย่างบันทึกข้อความ: \n\n {context} \n\n คำถาม: {question}")
])


def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])


rag_chain = main_prompt | llm_ollama | StrOutputParser()


# web search
web_search_tool = TavilySearchResults(k=3)




def retrieve(state):
    print(f"Retrieving documents")
    question = state["question"]


    #Retrueve documents retriever
    documents = retriever.invoke(question)

    return {"documents":documents,"question":question}


def generate(state):
    print(f"Generating response")
    question = state["question"]
    documents = state["documents"]

    generation = rag_chain.invoke({"context":documents,"question":question})

    return {"generation":generation,"question":question}

def grade_documents(state):

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    #Score each doc 
    filtered_docs = []
    web_search = "ไม่ใช่"
    web_search_needed = True
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question":question,"docs":doc.page_content}
            )
        grade = score.binary_score
        print(f"grade: {grade}")
        if grade == "ใช่":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
            web_search_needed = False
            # break
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "ใช่"
            # continue

    return {"filtered_docs":filtered_docs,"question":question,"web_search":web_search,"web_search_needed":web_search_needed}


def web_search(state):
    print("--- Web Search ---")
    question = state["question"]
    documents = state["documents"]


    template = """
    คุณเป็นผู้เชี่ยวชาญในการค้นหาข้อมูลเอกสารที่เกี่ยวข้องกับคำถามของผู้ใช้
    คำขอ: {question}

    คำตอบ:

    """

    prompt_search = PromptTemplate(
        template=template,
        input_variables=["question"]
    )

   #web search 
    search_query = prompt_search.format(question=question)
    web_search_results = web_search_tool.invoke(search_query)

    web_results = "\n".join([d["content"] for d in web_search_results])

    web_results = Document(page_content=web_results) 
    documents.append(web_results)

    return {"documents":documents,"question":question}

def decide_to_generate(state):

    print("---ASSESS GRADED DOCUMENTS ---")
    state["question"]
    web_search = state["web_search"]
    documents = state["documents"]
    web_search_needed = state["web_search_needed"] 

    if len(documents)  > 0 and not web_search_needed:
        print("---DECISION: GENERATE---")
        return "generate"
    elif web_search_needed or web_search == "ใช่":
        print("---DECISION: WEB SEARCH---")
        return "web_search"
    else:
        print("---DECISION: NO ACTION---")
        return "no_action"








#Graph 
class GraphState(TypedDict):
    question:str
    generation:str 
    web_search:str 
    documents:List[str]


workflow = StateGraph(GraphState)

#Add nodes 
workflow.add_node("retrieve",retrieve)
workflow.add_node("grade_documents",grade_documents)
workflow.add_node("generate",generate)
workflow.add_node("web_search_node",web_search)





#Build edges 
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search_node",
        "generate": "generate"
    }
)

workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)




app = workflow.compile()



from pprint import pprint

input_state_1 = {"question":"เขียนหนังสือบันทึกข้อความ ขออนุมัติเดินทางตรวจตามแผน"}
input_state_2 = {"question":"เขียนหนังสือเชิญ ประชุมกับคณะกรรมการ ITU โดยมีรายละเอียด การป้องกัน แก๊ง call center ในประเทศเพื่อนบ้าน"}
for output in app.stream(input_state_1):
    for k,v in output.items():
        pprint(f"Node: '{k}':")
    pprint("\n---\n")
print(v['generation'])
