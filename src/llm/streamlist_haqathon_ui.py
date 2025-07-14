import streamlit as st
import fitz
from docx import Document
import re
import toolkit
from rag_pipeline import RAGIndexer

# Main layout
# st.title("Cryptalyze")
st.markdown(
    "<h1 style='text-align: center; font-size: 4em'>Cryptalyze</h1>",
    unsafe_allow_html=True,
)
# st.caption("Securely Analyze Your Most Important Docs")
st.markdown(
    "<p style='text-align: center; color: grey; font-size: 1.5em;'>Securely Analyze Your Most Important Docs",
    unsafe_allow_html=True,
)


def strip_think_tags(response: str) -> str:
    response = re.sub(r".*?</think>", "", response, flags=re.DOTALL)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    response = re.sub(r"</?think>", "", response)
    return response


with st.container(border=True):
    category = st.radio(
        "Choose Category", options=["Medical", "Finance"], horizontal=True
    )

# Initialize RAG Indexer
indexer = RAGIndexer()

# Sidebar layout
with st.sidebar:
    st.title(f"Upload {category} Documents")

    # File uploader for multiple files
    uploaded_files = st.file_uploader(
        "Choose documents to analyze",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
    )
    if uploaded_files is not None:
        data = list()
        for file in uploaded_files:

            if "pdf" in file.type:
                with fitz.open(stream=file.read(), filetype="pdf") as doc:
                    text = ""
                    for page in doc:
                        temp = page.get_text()
                        text += re.sub(r"\s+", " ", temp).strip()
                data = text
            elif (
                file.type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                doc = Document(file)
                full_text = "\n".join([para.text for para in doc.paragraphs])
                full_text = re.sub(r"\s+", " ", full_text).strip()
                data = full_text
            else:
                txt = file.read().decode("utf-8")
                txt = re.sub(r"\s+", " ", txt).strip()
                data = txt

            indexer.index_all(group=category, data=data)
            st.success(f"Documents Uploaded Successfully!")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]


# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    model_response = toolkit.query_llm_with_rag(prompt, group=category)
    cleaned_response = strip_think_tags(model_response)

    st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
    st.chat_message("assistant").write(cleaned_response)
