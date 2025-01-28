import os
from dotenv import load_dotenv
import requests
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain.vectorstores import FAISS    
from langchain.embeddings import OllamaEmbeddings

from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import streamlit as st
from uuid import uuid4

load_dotenv()
os.environ["LANGCHAIN_API_KEY"]=st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=st.secrets["LANGCHAIN_PROJECT"]
os.environ['GROQ_API_KEY'] =st.secrets["GROQ_API"]

# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
# os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API")

GROQ_API_KEY = st.secrets["GROQ_API"]
GROQ_API_URL = "https://api.groq.com/chat"

def query_groq_llama(session_id, query):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {"session_id": session_id, "query": query}
    response = requests.post(GROQ_API_URL, json=payload, headers=headers)
    return response.json()


embedding_model = OllamaEmbeddings(model='llama3.1:8b')
faiss_db = FAISS.load_local("vector_database_3", embedding_model,allow_dangerous_deserialization=True)


llm = ChatGroq(model="llama-3.3-70b-versatile")
prompt_template = ChatPromptTemplate.from_template(
    (
        "You are AiVin, an AI assistant specializing in portfolio insights, project experiences, "
        "and professional expertise. The entire context provided is related a person called Vineeth. Only use the provided context to answer questions.\n\n"
        "Context: {context}\n\n"
        "Question: {input}\n\n"
        "Answer: If sufficient information is present in the context, provide a response. "
        "If not, say 'Not enough information'."
    )
)
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt_template,
)

retrieval_chain = create_retrieval_chain(
    retriever=faiss_db.as_retriever(),
    combine_docs_chain=document_chain,
)
# Define the RetrievalQA chain with the prompt template


# Initialize Streamlit session
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid4())



# Streamlit App
st.set_page_config(page_title="Portfolio & Chatbot", layout="wide", initial_sidebar_state="expanded", page_icon="🤖",)

# Sidebar Section
st.sidebar.title("Vineeth S",)
# st.sidebar.title("My Portfolio")
st.sidebar.header("Contact Me")
st.sidebar.write(
    "Mobile: +91 9164488928\n\n"
    "Email: vineethvini8888@gmail.com"
    )
# About Me Section in Sidebar
st.sidebar.header("About Me")
st.sidebar.write(
    "Dynamic and results-oriented Data Scientist with 3.5+ years of experience in building advanced AI-driven solutions for predictive maintenance, inventory optimization, and digital twin development. Adept at leveraging statistical modeling, machine learning, and natural language processing to deliver data-backed insights and operational efficiencies. Proven track record in designing custom algorithms and deploying models in production environments using cutting-edge platforms."
)
st.sidebar.header("Organizations")
st.sidebar.write(
    """
    - **Tata Consultancy Services**: Data Scientist, 2021-2024
    - **Asset Integrity Engineering**: Data Scientist, 2024-Present
    """
)
# Skills Section in Sidebar
st.sidebar.header("Skills")
st.sidebar.write(
    """
    - **Programming Languages**: Python, SQL
    - **Machine Learning**: Regression Algorithms, Classification algorithms, Clustering, XGBoost, Scikit-Learn
    - **Deep Learning**: Neural Networks,ANN, RNN, LSTM, Transformers
    - **Natural Language Processing**: NLTK, Tf-Idf, Word2Vec
    - **Data Analysis Tools**:Pandas, NumPy, Matplotlib, Plotly, Json
    - **AI Tools**: LangChain, Hugging face, GROQ, FAISS, TensorFlow
    - **Other Skills**: Project management, Communication, Problem-solving
    """
)

# Projects Section in Sidebar
st.sidebar.header("Projects")
st.sidebar.write(
    """
    - **Warehouse Optimization**: Developed a statistical model to optimize inventory levels and reduce carrying costs.
    - **Mota Engil Maintenance Optimization**: Reduced fuel consumption through decision windows for truck categories.
    - **Fleet Matching**: Matched dump truck vehicles to optimal mining pit locations to improve performance.
    - **CORAL** - No Code AI Platform to build end to end models.
    - **Asset Twin**: Created digital twins for predictive maintenance of industrial assets.
    - **Proccess Twin**: Developed digital twins for process optimization in carbonation plants.
    """
)
st.sidebar.header("Education")
st.sidebar.write(
    """
    - **Bachelor of Engineering (B.E.)**: Electronics and Communication Engineering, Visvesvaraya Technological University, 2017-2021
    """
)
st.sidebar.header("Certifications")
st.sidebar.write(
    """
    - **Python for Data Science**:NPTEL
    - **Machine Learning**: Coursera
    - **Complete Generative AI Course With Langchain and Huggingface**: Udemy
    - **Algorithms and Data Structures in Python**: Udemy
    """
)
# Main Page Title
st.title("Chat with AiVin")

# Cards showcasing sample questions in a single row using HTML and CSS (flexbox)
st.markdown("""
    <style>
        .cards-container {
            display: flex;
            justify-content: space-between;
            gap: 20px;
        }
        .card {
            background-color: 	#669999;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            width: 30%;
        }
        .card h3 {
            margin: 0;
            font-size: 18px;
            font-weight: bold;
        }
        .card p {
            margin-top: 10px;
        }
       
    </style>
     <style>
        [data-testid="stSidebar"] {
            background-color:	#e0ebeb; /* Light blue-gray color */
            padding: 10px;
            color: #000000;
        }
        /* Adjust text and header styles in the sidebar */

    </style>
    <div class="cards-container">
        <div class="card">
            <h3>Sample Question 1</h3>
            <p>Explain about warehouse optimization project</p>
        </div>
        <div class="card">
            <h3>Sample Question 2</h3>
            <p>Total year of experience?</p>
        </div>
        <div class="card">
            <h3>Sample Question 3</h3>
            <p>How are you?</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Text input for user query
st.text("")
user_input = st.text_input("Ask a question to AiVin:")
# Cards showcasing sample questions using HTML and CSS


if user_input:
    response = retrieval_chain.invoke({"input": user_input, "session_id": st.session_state["session_id"]})
    st.write("AiVin:", response['answer'])
