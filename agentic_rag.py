from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from langchain_community.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from pydantic import BaseModel, Field
from langchain import hub
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
import pprint


from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode


import os


from dotenv import load_dotenv
load_dotenv()
### https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/#nodes-and-edges
dir_path = "./data"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
llm_ollama = ChatOllama(
    model="hf.co/dearxoasis/llama3.2-typhoon2-offical_doc-3b:Q8_0",
            # model="hf.co/dearxoasis/llama-3-typhoon2_dearx:latest",
            temperature=0,
         
)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "mps" },
)

loader = DirectoryLoader(
            dir_path,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True,
            recursive=True,
    )



documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

doc_spilts = text_splitter.split_documents(documents)

#add to vectorDB
vector_store = Chroma.from_documents(
    documents=doc_spilts,
    collection_name="rag_collection",
    embedding=embeddings,
)


retriever = vector_store.as_retriever(
    # search_type="similarity_score_threshold",
    # search_kwargs={
    #     "k": 5,  # Return top 5 documents
    #     "score_threshold": 0.5,  # Only return documents with similarity score above 0.7
    # }
)
retriever_tool = create_retriever_tool(
    retriever,
    name="search_thai_documents",
    description="ค้นหาข้อมูลในคลังเอกสารราชการไทยใช้เครื่องมือนี้เฉพาะเมื่อต้องการข้อมูลเฉพาะจากเอกสารราชการไทยเท่านั้นหากไม่พบเอกสารที่เกี่ยวข้อง ให้ระบุอย่างชัดเจนว่าไม่มีข้อมูลนี้ในคลังเอกสาร"
)
tools = [retriever_tool]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def grade_documents(state) -> Literal["generate","rewrite"]:


    print("---CHECK RELEVANCE---")


    class grade(BaseModel):

        binary_score:str = Field(description="คะแนนความเกี่ยวข้อง ตอบเพียง 'ใช่' หรือ 'ไม่ใช่' เท่านั้น")

    model = ChatOpenAI(model="gpt-4o",temperature=0)
    

    llm_with_tools = model.with_structured_output(grade)

    # prompt = PromptTemplate(
    # template="""คุณเป็นผู้ประเมินความเกี่ยวข้องของเอกสารที่ค้นพบกับคำถามของผู้ใช้

    # นี่คือเอกสารที่ค้นพบ:
    # {context}

    # นี่คือคำถามของผู้ใช้:
    # {question}

    # กรุณาประเมินความเกี่ยวข้องอย่างเข้มงวด โดยพิจารณาว่าเอกสารนี้มีข้อมูลที่ตรงกับคำถามหรือไม่
    
    # เอกสารจะถือว่าเกี่ยวข้องเฉพาะเมื่อ:
    # - มีคำสำคัญ (Keywords) ที่ตรงกับคำถามโดยตรง
    # - มีความหมายเชิงความเข้าใจ (Semantic Meaning) ที่ตอบคำถามได้อย่างชัดเจน
    # - มีเนื้อหาที่สามารถนำไปใช้ตอบคำถามได้อย่างเฉพาะเจาะจง

    # ให้ตอบว่า 'ไม่ใช่' หากพบว่า:
    # - คำถามเกี่ยวกับการคำนวณทางคณิตศาสตร์ (เช่น 1+1, 5*3)
    # - เอกสารมีเพียงความเกี่ยวข้องเล็กน้อยหรือทั่วไปเกินไป
    # - คำถามเกี่ยวกับความรู้ทั่วไปที่ไม่เกี่ยวข้องกับเอกสารราชการ
    # - คำถามที่ขอความคิดเห็นส่วนตัวหรือคำแนะนำ
    # - คำถามเกี่ยวกับหัวข้อที่ไม่น่าจะมีในเอกสารราชการ
    
    # ให้คะแนนแบบ binary เท่านั้น:
    # - ตอบ 'ใช่' ถ้าเอกสารเกี่ยวข้องโดยตรงและมีประโยชน์ต่อการตอบคำถาม
    # - ตอบ 'ไม่ใช่' ถ้าเอกสารไม่เกี่ยวข้องหรือเกี่ยวข้องเพียงเล็กน้อย""",

     # Prompt
    prompt = PromptTemplate(
    template="""คุณเป็นผู้ประเมินความเกี่ยวข้องของเอกสารที่ค้นพบกับคำถามของผู้ใช้
    คำถามเกี่ยวกับการคำนวณทางคณิตศาสตร์ (เช่น 1+1, 5*3) ให้ตอบ 'ไม่ใช่' ทันที
    นี่คือเอกสารที่ค้นพบ:
    {context}

    นี่คือคำถามของผู้ใช้:
    {question}

    
    ให้คะแนนแบบ binary โดยตอบเพียง 'ใช่' หรือ 'ไม่ใช่' เพื่อระบุว่าเอกสารเกี่ยวข้องกับคำถามหรือไม่""",
    input_variables=["context", "question"],
)

    
    chain = prompt | llm_with_tools

    message = state["messages"]
    last_message = message[-1]

    question = message[0].content
    docs = last_message.content


    scoreed_result = chain.invoke({"question":question,"context":docs})

    score = scoreed_result.binary_score

    print(score)

    if score == "ใช่":
        print("---RELEVANT---")
        return "generate"
    else:
        print("---NOT RELEVANT---")
        return "rewrite"









def agent(state):

    print("--CALL AGENT--")

    messages = state["messages"]
    model = llm_ollama.bind_tools(tools)
    prompt_template = PromptTemplate(
        template="""
        คุณเป็นผู้เชี่ยวชาญในการเขียนหนังสือราชการ
        ให้ตอบตามโครงสร้าง:
         โครงสร้างเอกสารที่ต้องการ:
        1. ส่วนหัวเรื่อง (หน่วยงาน, ที่, วันที่)
        2. ชื่อเรื่อง
        3. คำขึ้นต้น (เรียน)
        4. เนื้อเรื่อง 
            - เรื่องเดิม
            - ข้อเท็จจริง คือ ข้อมูลที่เป็นรูปแบบรายงานผล  (ถ้ามี)
            - ข้อพิจารณา คือ ข้อมูลที่เป็นรูปแบบความคิดเห็น หรือ ข้อมูลที่เป็นรูปแบบการขออนุมัติ  (ถ้ามี)
            ให้วิเคราห์แล้วเขียน ข้อเท็จจริง หรือ ข้อพิจารณา อย่างใดอย่างหนึ่ง
        5. (ลงชื่อ, ตำแหน่ง)
        ให้ตอบเป็นภาษาราชการไทยเท่านั้น
       คำถาม:
        {messages}
        """,
        input_variables=["messages"]
    )
    response = model.invoke(prompt_template.invoke({"messages":messages}))

    return {"messages": [response]}



def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")

    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f"""\n
    คุณคือนักเขียนหนังสือราชการมีความรู้ความสามารถในการวิเคราะห์คำถามและปรับปรุงตอบเป็นหนังสือราชการ

    พิจารณาคำถามต่อไปนี้และวิเคราะห์ความตั้งใจและความหมายที่แท้จริง:
    \n ------- \n
    {question}
    ถ้าคำถามมีข้อมูลไม่ชัดให้สมมุติรายละเอียดของคำถามเพิ่มเติม
    โดยใส่ขอมูลโครงสร้างของหนังสือราชการเพิ่มเติม
    โครงสร้างเอกสารที่ต้องการ:
        1. ส่วนหัวเรื่อง (หน่วยงาน, ที่, วันที่)
        2. ชื่อเรื่อง
        3. คำขึ้นต้น (เรียน)
        4. เนื้อเรื่อง 
            - เรื่องเดิม (ควรละเอียดและสมจริง)
            - ข้อเท็จจริง หรือ ข้อพิจารณา ให้วิเคราะห์ความสำคัญของข้อมูลที่มีความสำคัญต่อการตอบคำถาม(ควรละเอียดและสมจริง)
            - ถ้าเป็นการขออนุมัติ ให้เป็น ข้อพิจารณา เท่านั้น    
        5. (ลงชื่อ, ตำแหน่ง)

    \n ------- \n
    
   
    
    """,
        )
    ]
    
   
    response = llm_ollama.invoke(msg)
    
    return {"messages": [response]}




def generate(state):

    print("--GENERATE---")

    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    #Prompt 
    # prompt = hub.pull("rlm/rag-prompt")
    template = """คุณเป็นผู้เชี่ยวชาญในการเขียนหนังสือราชการ
        
        ข้อมูลอ้างอิง (เรียงตามความเกี่ยวข้อง):
        {context}

        โครงสร้างเอกสารที่ต้องการ:
        1. ส่วนหัวเรื่อง (หน่วยงาน, ที่, วันที่)
        2. ชื่อเรื่อง
        3. คำขึ้นต้น (เรียน)
        4. เนื้อเรื่อง 
            - เรื่องเดิม
            - ข้อเท็จจริง คือ ข้อมูลที่เป็นรูปแบบรายงานผล  (ถ้ามี)
            - ข้อพิจารณา คือ ข้อมูลที่เป็นรูปแบบความคิดเห็น หรือ ข้อมูลที่เป็นรูปแบบการขออนุมัติ  (ถ้ามี) 
            - ถ้าเป็นการขออนุมัติ ให้เป็น ข้อพิจารณา เท่านั้น
            
        5. (ลงชื่อ, ตำแหน่ง)

        คำขอ: {question}

        กรุณาสร้างหนังสือราชการโดยใช้รูปแบบ Markdown และรักษารูปแบบราชการอย่างเคร่งครัด:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    #llm

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")

    rag_chain = prompt | llm_ollama | StrOutputParser()


    response = rag_chain.invoke({"context":docs,"question":question})

    return {"messages": [response]}





print("*" * 20 + "Prompt[rlm/rag-prompt]" + "*" * 20)
prompt = hub.pull("rlm/rag-prompt").pretty_print()




#Graph


workflow = StateGraph(AgentState)




workflow.add_node("agent",agent)
retrieve = ToolNode([retriever_tool])
workflow.add_node("retriever",retrieve)
workflow.add_node("rewrite",rewrite)
workflow.add_node("generate",generate)


workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retriever",
        END:END,
    },
)





workflow.add_conditional_edges(
    "retriever",
    grade_documents,
        
)



workflow.add_edge("generate",END)
workflow.add_edge("rewrite","agent")




#compile
graph = workflow.compile()

from IPython.display import Image, display

try:
    picture =display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    picture.save("graph.png")
except Exception:
    # This requires some extra dependencies and is optional
    pass



inputs = {
    
        "messages":[
            ("user","เขียนหนังสือราชการขออนุมัติเดินทาง จังหวัดชัยภูมิ ตรวจตามแผนประจำปี"),
        ]
    }
# inputs = {
#     "messages": [
#         ("user", "เขียนหนังสือบันทึกข้อความ สรุปการประชุม โดยมีข้อมูล : พบการลักลอบลากสายผิดกฏหมาย โดยสมมุติรายละเอียดของข้อมูลขึ้นมาเอง"),
#     ]
# } 
# inputs = {
#     "messages": [
#         ("user", ""),
#     ]
# }

for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=80, depth=None)
    pprint.pprint("\n---\n")






