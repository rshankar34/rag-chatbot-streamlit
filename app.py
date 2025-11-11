"""RAG PDF Chatbot - Streamlit Cloud + AWS Edition"""
import streamlit as st
import uuid
import tempfile
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from aws_utils import S3Manager, DynamoDBChat

#debugging info
st.sidebar.write(f"🔍 langchain-openai: {langchain_openai.__version__}")
st.sidebar.write(f"🔍 openai: {openai.__version__}")

# Page config
st.set_page_config(page_title="RAG PDF Chatbot", page_icon="☁️", layout="wide")

# Initialize AWS services
@st.cache_resource
def init_aws():
    """Initialize AWS services with explicit region"""
    region = st.secrets.get("AWS_REGION", "eu-north-1")
    
    s3 = S3Manager(
        bucket_name=st.secrets["S3_BUCKET_NAME"],
        region=region
    )
    
    dynamo = DynamoDBChat(
        table_name="ChatHistory",
        region=region
    )
    
    return s3, dynamo

s3_manager, chat_db = init_aws()

# Initialize embeddings
@st.cache_resource
def init_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

embeddings = init_embeddings()

# Session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    history = chat_db.get_history(st.session_state.session_id)
    st.session_state.messages = [
        {"role": msg['role'], "content": msg['content']} 
        for msg in history
    ]
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# Helper functions
def process_pdf(file_bytes, filename):
    """Process PDF and add to vector store"""
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    # Extract text
    reader = PdfReader(tmp_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Clean up
    os.unlink(tmp_path)
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    
    # Create/update vector store
    texts_with_meta = [(chunk, {"source": filename}) for chunk in chunks]
    
    if st.session_state.vector_store is None:
        st.session_state.vector_store = FAISS.from_texts(
            [t[0] for t in texts_with_meta],
            embeddings,
            metadatas=[t[1] for t in texts_with_meta]
        )
    else:
        new_store = FAISS.from_texts(
            [t[0] for t in texts_with_meta],
            embeddings,
            metadatas=[t[1] for t in texts_with_meta]
        )
        st.session_state.vector_store.merge_from(new_store)
    
    return True

def get_qa_chain():
    """Create QA chain with GPT-4o-mini"""
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=500,
            api_key=st.secrets["OPENAI_API_KEY"]
        )
        
        template = """Use the context to answer the question. If you don't know the answer, say so clearly.

Context: {context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever(
                search_kwargs={"k": 4}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain
        
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None

# UI
st.title("☁️ RAG PDF Chatbot")
st.caption("Upload PDFs, ask questions powered by AI")

# Sidebar
with st.sidebar:
    st.header("📄 PDF Management")
    
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if st.button("📤 Upload & Process", type="primary"):
        if uploaded_files:
            progress = st.progress(0)
            for idx, file in enumerate(uploaded_files):
                # Upload to S3
                s3_key = s3_manager.upload_pdf(file.getvalue(), file.name)
                
                if s3_key:
                    # Process PDF
                    process_pdf(file.getvalue(), file.name)
                
                progress.progress((idx + 1) / len(uploaded_files))
            
            st.success(f"✅ Processed {len(uploaded_files)} PDF(s)!")
            progress.empty()
        else:
            st.warning("Upload PDFs first")
    
    st.divider()
    
    if st.button("📋 Show S3 PDFs"):
        pdfs = s3_manager.list_pdfs()
        if pdfs:
            st.write(f"**{len(pdfs)} PDF(s):**")
            for pdf in pdfs:
                st.text(f"• {pdf.split('/')[-1]}")
        else:
            st.info("No PDFs yet")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat"):
        chat_db.clear_history(st.session_state.session_id)
        st.session_state.messages = []
        st.success("Cleared!")
        st.rerun()

# Chat
st.subheader("💬 Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about your PDFs..."):
    if not st.session_state.vector_store:
        st.error("⚠️ Upload PDFs first!")
    else:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        chat_db.save_message(st.session_state.session_id, "user", prompt)
        
        # AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Create QA chain
                    qa_chain = get_qa_chain()
                    
                    # Check if chain was created successfully
                    if qa_chain is None:
                        st.error("❌ Failed to create QA chain. Check OpenAI API key and package versions.")
                        st.stop()
                    
                    # Get response
                    result = qa_chain.invoke({"query": prompt})
                    answer = result['result']
                    
                    # Add sources
                    if 'source_documents' in result and result['source_documents']:
                        sources = set([doc.metadata.get('source', 'Unknown') 
                                     for doc in result['source_documents']])
                        answer += f"\n\n📄 *Sources: {', '.join(sources)}*"
                    
                    st.write(answer)
                    
                    chat_db.save_message(st.session_state.session_id, "assistant", answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.error("Please check your OpenAI API key and internet connection.")
