from langchain.prompts import PromptTemplate
from llama_index import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.vector_stores import FaissVectorStore
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.schema import Document
from llama_index.node_parser import UnstructuredElementNodeParser

from src.utils import get_model
from src.fields import (
    auto_summarization_field, auto_summarization_attributes,
    keyword_generation_field, keyword_generation_attributes,
    question_bank_generation_field, question_bank_generation_attributes,
)

import streamlit as st
import os
import faiss
import time
from pypdf import PdfReader

st.set_page_config(page_title="Academic Navigator Dashboard", page_icon=":card_index_dividers:", initial_sidebar_state="expanded", layout="wide")

st.title(":card_index_dividers: Academic Navigator Dashboard")
st.info("""
Begin by uploading your study material in PDF format. Afterward, click on 'Process Document'. Once the document has been processed, select one of the tasks tap on 'Commence' and the system will start its magic. After a brief wait, you'll be presented with your desired output derived from your study material.
""")

def process_pdf(pdf):
    file = PdfReader(pdf)

    document_list = []
    for page in file.pages:
        document_list.append(Document(text=str(page.extract_text())))

    node_paser = UnstructuredElementNodeParser()
    nodes = node_paser.get_nodes_from_documents(document_list, show_progress=True)
    
    return nodes

def get_vector_index(nodes, vector_store):
    print(nodes)
    llm = get_model("openai")
    if vector_store == "faiss":
        d = 1536
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        service_context = ServiceContext.from_defaults(llm=llm) 
        index = VectorStoreIndex(nodes, 
            service_context=service_context,
            storage_context=storage_context
        )
    elif vector_store == "simple":
        index = VectorStoreIndex.from_documents(nodes)

    return index

def generate_output(engine, output_name, output_format):
    with open("prompts/initial.prompt", "r") as f:
        template = f.read()

    prompt_template = PromptTemplate(
        template=template,
        input_variables=['output_name', 'output_format']
    )

    formatted_input = prompt_template.format(output_name=output_name,  output_format=output_format)
    print(formatted_input)
    response = engine.query(formatted_input)
    return response.response
    
def report_output(engine,  fields_to_include, section_num):

    fields = None
    attribs = None

    if section_num == 1:
        fields = auto_summarization_field
        attribs = auto_summarization_attributes
    elif section_num == 2:
        fields = keyword_generation_field
        attribs = keyword_generation_attributes
    elif section_num == 3:
        fields = question_bank_generation_field
        attribs = question_bank_generation_attributes

    ins = {}
    for i, field in enumerate(attribs):
        if i < len(fields_to_include) and fields_to_include[i]:
            response = generate_output(engine,field,  str({field: fields[field]}))
            ins[field] = response

    return {
        "outputs": ins
    }

def get_query_engine(engine):
    llm = get_model("openai")
    service_context = ServiceContext.from_defaults(llm=llm)

    query_engine_tools = [
        QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(
                name="PaperProbe",
                description=f"Smart study companion.",
            ),
        ),
    ]

    s_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        service_context=service_context
    )
    return s_engine

for output in auto_summarization_attributes:
    if output not in st.session_state:
        st.session_state[output] = None

for output in keyword_generation_attributes:
    if output not in st.session_state:
        st.session_state[output] = None

for output in question_bank_generation_attributes:
    if output not in st.session_state:
        st.session_state[output] = None

if "end_time" not in st.session_state:
    st.session_state.end_time = None


if "process_doc" not in st.session_state:
        st.session_state.process_doc = False


st.sidebar.info("""
You can get your OpenAI API key [here](https://openai.com/blog/openai-api)
""")
OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if not OPENAI_API_KEY:
    st.error("Please enter your OpenAI API Key")

if OPENAI_API_KEY:
    pdfs = st.sidebar.file_uploader("Upload your study material in PDF format", type="pdf")
    st.sidebar.info("""
    - Welcome to Paperprobe - Your smart study companion!
    """)

    if st.sidebar.button("Process Document"):
        with st.spinner("Processing Document..."):
            nodes = process_pdf(pdfs)
            st.session_state.index = get_vector_index(nodes, vector_store="faiss")
            st.session_state.process_doc = True
            

        st.toast("Document Processsed!")

    if st.session_state.process_doc:

        col1, col2 = st.columns([0.25, 0.75])

        with col1:
            st.write("""
                ### Select Tasks
            """)
            
            with st.expander("**Task-1**", expanded=True):
                task1 = st.toggle("Auto-Summarization")


                auto_summarization_field_list = [task1]

            with st.expander("**Task-2**", expanded=True):
                task2 = st.toggle("Keyword Generation")


                keyword_generation_field_list = [task2]

            with st.expander("**Task-3**", expanded=True):
                task3 = st.toggle("Question Bank Generation")

                question_bank_generation_field_list = [task3]


        with col2:
            if st.button("Commence"):
                engine = get_query_engine(st.session_state.index.as_query_engine(similarity_top_k=3))
                start_time = time.time()

                with st.status("**Generating...**"):


                    if any(auto_summarization_field_list):
                        st.write("Task1...")

                        for i, output in enumerate(auto_summarization_attributes):
                            if st.session_state[output]:
                                auto_summarization_field_list[i] = False

                        response = report_output(engine, auto_summarization_field_list, 1)

                        for key, value in response["outputs"].items():
                            st.session_state[key] = value

                    if any(keyword_generation_field_list):
                        st.write("Task2...")

                        for i, output in enumerate(keyword_generation_attributes):
                            if st.session_state[output]:
                                keyword_generation_field_list[i] = False
                        response = report_output(engine, keyword_generation_field_list, 2)

                        for key, value in response["outputs"].items():
                            st.session_state[key] = value


                    if any(question_bank_generation_field_list):
                        st.write("Task3...")

                        for i, output in enumerate(question_bank_generation_attributes):
                            if st.session_state[output]:
                                question_bank_generation_field_list[i] = False
                        
                        response = report_output(engine, question_bank_generation_field_list, 3)

                        for key, value in response["outputs"].items():
                            st.session_state[key] = value

                    st.session_state["end_time"] = "{:.2f}".format((time.time() - start_time))

                    st.toast("Task Accomplished!")
            
            if st.session_state.end_time:
                st.write("Task Completion Time: ", st.session_state.end_time, "s")
        
            tab1, tab2, tab3 = st.tabs(["Auto-Summarization", "Keyword Generation", "Question Bank Generation"])           
                
            with tab1:
                st.write("## Summary and Overview")
                try: 
                    if task1:
                        if st.session_state['auto_summarization']:
                            
                            st.write(st.session_state['auto_summarization'])
                        else:
                            st.error("Auto Summarization task has not been accomplished")
                except:
                    st.error("This task has not been accomplished")
            
            with tab2:
                st.write("## Keywords")
                try:
                    if task2:
                        if st.session_state["keyword_generation"]:
                            
                            st.write(st.session_state["keyword_generation"])
                        else:
                            st.error("Keywords have not been generated")
                except:
                    st.error("This task has not been accomplished")

            with tab3:
                st.write("## Question Bank")

                try:
                    if task3:
                        if st.session_state["question_bank_generation"]:
                            
                            st.write(st.session_state["question_bank_generation"])
                        else:
                            st.error("Question Bank has not been generated")
                except:
                    st.error("This task has not been accomplished")


