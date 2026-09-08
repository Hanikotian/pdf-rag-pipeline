"""
Streamlit web application for the PDF RAG system.

Application flow:

    User uploads PDF
            ↓
    PDF saved temporarily
            ↓
    RAGPipeline indexes document
            ↓
    FAISS vector index
            ↓
    User asks questions
            ↓
    Retrieval + reranking + Gemini
            ↓
    Answer + source pages

The application layer is intentionally thin.

The actual RAG logic remains inside src/.
"""

import tempfile
from pathlib import Path

import streamlit as st

from src.pipeline import RAGPipeline


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
)


# =========================================================
# APPLICATION CONFIGURATION
# =========================================================

# The uploaded PDF will be temporarily stored here.
UPLOAD_DIRECTORY = Path(
    "data/uploads"
)

# The generated FAISS indexes will be stored here.
INDEX_DIRECTORY = Path(
    "data/indexes/app"
)


# =========================================================
# PAGE HEADER
# =========================================================

st.title("📄 PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions about its contents. "
    "The answers are generated using the information "
    "retrieved from your document."
)


# =========================================================
# SESSION STATE
# =========================================================

# Streamlit reruns the script whenever the user interacts
# with the application.
#
# Session state allows us to preserve the RAG pipeline
# between those reruns.

if "pipeline" not in st.session_state:

    st.session_state.pipeline = None


if "document_name" not in st.session_state:

    st.session_state.document_name = None


if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# =========================================================
# PDF UPLOAD
# =========================================================

st.header("1. Upload your PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
)


# =========================================================
# PROCESS UPLOADED PDF
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # Only process the file when a new document is uploaded.
    # -----------------------------------------------------

    if (
        st.session_state.document_name
        != uploaded_file.name
    ):

        # Clear previous conversation because we are
        # switching to a different document.
        st.session_state.chat_history = []

        # Make sure the upload directory exists.
        UPLOAD_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -------------------------------------------------
        # Save the uploaded PDF.
        #
        # Streamlit UploadedFile behaves like a file object,
        # so getvalue() gives us its raw bytes.
        # -------------------------------------------------

        pdf_path = (
            UPLOAD_DIRECTORY
            / uploaded_file.name
        )

        with open(
            pdf_path,
            "wb",
        ) as file:

            file.write(
                uploaded_file.getvalue()
            )

        # -------------------------------------------------
        # Create a unique index directory for this
        # application session/document.
        #
        # tempfile gives us a unique directory name.
        # -------------------------------------------------

        document_index_directory = (
            INDEX_DIRECTORY
            / Path(
                uploaded_file.name
            ).stem
        )

        try:

            # -------------------------------------------------
            # Display progress to the user.
            # -------------------------------------------------

            with st.status(
                "Processing your PDF...",
                expanded=True,
            ) as status:

                st.write(
                    "📖 Reading PDF..."
                )

                # -------------------------------------------------
                # Create the RAG pipeline.
                # -------------------------------------------------

                pipeline = RAGPipeline(
                    index_directory=(
                        document_index_directory
                    )
                )

                # -------------------------------------------------
                # Build the complete index:
                #
                # PDF
                # ↓
                # ingestion
                # ↓
                # chunks
                # ↓
                # embeddings
                # ↓
                # FAISS
                # -------------------------------------------------

                st.write(
                    "🧩 Creating document chunks..."
                )

                pipeline.index_document(
                    pdf_path=pdf_path
                )

                st.write(
                    "🧠 Building semantic index..."
                )

                st.write(
                    f"✅ Indexed "
                    f"{pipeline.number_of_chunks} chunks."
                )

                # Store the working pipeline in session
                # state so future questions can reuse it.
                st.session_state.pipeline = (
                    pipeline
                )

                st.session_state.document_name = (
                    uploaded_file.name
                )

                status.update(
                    label="PDF ready!",
                    state="complete",
                    expanded=False,
                )

            st.success(
                f"'{uploaded_file.name}' is ready "
                "for questions."
            )

        except Exception as exc:

            # -------------------------------------------------
            # Do not expose a giant Python traceback to the
            # end user.
            # -------------------------------------------------

            st.session_state.pipeline = None
            st.session_state.document_name = None

            st.error(
                "We could not process this PDF."
            )

            st.exception(exc)


# =========================================================
# DOCUMENT STATUS
# =========================================================

if (
    st.session_state.pipeline is not None
    and st.session_state.document_name is not None
):

    st.divider()

    st.header("2. Ask questions")

    st.info(
        f"📄 Current document: "
        f"**{st.session_state.document_name}**"
    )

    st.caption(
        f"Indexed chunks: "
        f"{st.session_state.pipeline.number_of_chunks}"
    )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # Display source pages underneath assistant
        # responses.
        if (
            message["role"] == "assistant"
            and message.get("pages")
        ):

            st.caption(
                "Sources: "
                + ", ".join(
                    f"Page {page}"
                    for page in message["pages"]
                )
            )


# =========================================================
# QUESTION INPUT
# =========================================================

if st.session_state.pipeline is not None:

    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        # -----------------------------------------------------
        # Display the user's question immediately.
        # -----------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        # -----------------------------------------------------
        # Generate RAG answer.
        # -----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Searching the document..."
            ):

                try:

                    response = (
                        st.session_state.pipeline.ask(
                            question=question
                        )
                    )

                    # -----------------------------------------
                    # Display answer.
                    # -----------------------------------------

                    st.markdown(
                        response.answer
                    )

                    # -----------------------------------------
                    # Display source pages.
                    # -----------------------------------------

                    pages = (
                        response.page_numbers
                    )

                    if pages:

                        st.caption(
                            "Sources: "
                            + ", ".join(
                                f"Page {page}"
                                for page in pages
                            )
                        )

                    else:

                        st.caption(
                            "Sources: No source pages "
                            "were returned."
                        )

                    # -----------------------------------------
                    # Save assistant response to history.
                    # -----------------------------------------

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": response.answer,
                            "pages": pages,
                        }
                    )

                except Exception as exc:

                    error_message = (
                        "I couldn't answer that question "
                        "right now. Please try again."
                    )

                    st.error(
                        error_message
                    )

                    # Log the actual exception inside
                    # the expandable Streamlit error
                    # section for development.
                    with st.expander(
                        "Technical details"
                    ):

                        st.exception(
                            exc

                        )

else:

    st.info(
        "Upload a PDF above to start asking questions."
    )