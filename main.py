"""
Streamlit application for the PDF RAG system.

Application workflow:

    User uploads PDF
            ↓
    PDF is saved locally
            ↓
    Document-specific FAISS index
            ↓
    RAGPipeline
            ↓
    User asks questions
            ↓
    Retrieval + reranking + Gemini
            ↓
    Grounded answer + page citations

The application layer intentionally does NOT implement:

- PDF extraction
- chunking
- embeddings
- vector search
- reranking
- LLM prompting

Those responsibilities belong to the src/ package.

This file is responsible only for:

- User interface
- File upload
- Application state
- Starting/loading the RAG pipeline
- Displaying answers and citations
"""


# =========================================================
# IMPORTS
# =========================================================

import hashlib
from pathlib import Path

import streamlit as st

from src.pipeline import RAGPipeline


# =========================================================
# CONFIGURATION
# =========================================================

# Directory where uploaded PDFs will be stored.
UPLOAD_DIRECTORY = Path(
    "data/uploads"
)


# Directory where document-specific FAISS
# indexes will be stored.
INDEX_DIRECTORY = Path(
    "data/indexes/uploads"
)


# Maximum size of an uploaded PDF.
#
# Streamlit's default upload limit is normally sufficient,
# but we define one here so the application has an explicit
# configuration.
MAX_UPLOAD_SIZE_MB = 50


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
)


# =========================================================
# APPLICATION TITLE
# =========================================================

st.title("📄 PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions about its contents. "
    "Answers are generated using information retrieved "
    "from your document."
)


# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

# Streamlit reruns this script whenever the user interacts
# with the application.
#
# Session state allows us to remember information between
# those reruns.

if "pipeline" not in st.session_state:

    st.session_state.pipeline = None


if "document_name" not in st.session_state:

    st.session_state.document_name = None


if "document_hash" not in st.session_state:

    st.session_state.document_hash = None


if "index_directory" not in st.session_state:

    st.session_state.index_directory = None


if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def calculate_file_hash(
    file_bytes: bytes,
) -> str:
    """
    Create a SHA-256 hash for an uploaded PDF.

    Why do we need this?

    Two PDFs can have the same filename but completely
    different contents.

    Using the file hash allows us to create a unique
    index for each actual document.

    Example:

        report.pdf
            ↓
        SHA-256 hash
            ↓
        data/indexes/uploads/<hash>/

    This prevents different PDFs from accidentally
    sharing the same FAISS index.
    """

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


def get_index_directory(
    document_hash: str,
) -> Path:
    """
    Return the FAISS index directory for a document.
    """

    return (
        INDEX_DIRECTORY /
        document_hash
    )


def get_page_count(
    pipeline: RAGPipeline,
) -> int:
    """
    Estimate the number of pages represented in the
    indexed chunks.

    The current RAGPipeline exposes the vector store,
    whose chunks contain page numbers.

    We therefore find the highest page number appearing
    in the chunks.

    This avoids introducing another public API only for
    displaying the page count.
    """

    if not pipeline.vector_store:

        return 0

    pages = set()

    for chunk in pipeline.vector_store.chunks:

        for page_number in chunk.page_numbers:

            pages.add(
                page_number
            )

    if not pages:

        return 0

    return max(pages)


def index_uploaded_pdf(
    uploaded_file,
):
    """
    Save and index an uploaded PDF.

    If the document has already been indexed, the existing
    FAISS index is loaded instead of rebuilding it.

    Returns
    -------
    RAGPipeline
        Ready-to-use RAG pipeline.

    str
        SHA-256 document hash.

    Path
        Index directory.
    """

    # -----------------------------------------------------
    # Read uploaded file.
    # -----------------------------------------------------

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:

        raise ValueError(
            "The uploaded PDF is empty."
        )

    # -----------------------------------------------------
    # Validate file size.
    # -----------------------------------------------------

    file_size_mb = (
        len(file_bytes)
        / (
            1024 * 1024
        )
    )

    if (
        file_size_mb
        > MAX_UPLOAD_SIZE_MB
    ):

        raise ValueError(
            f"The PDF is too large. "
            f"Maximum allowed size is "
            f"{MAX_UPLOAD_SIZE_MB} MB."
        )

    # -----------------------------------------------------
    # Validate extension.
    # -----------------------------------------------------

    filename = uploaded_file.name

    if not filename.lower().endswith(
        ".pdf"
    ):

        raise ValueError(
            "Please upload a PDF file."
        )

    # -----------------------------------------------------
    # Calculate document identity.
    # -----------------------------------------------------

    document_hash = calculate_file_hash(
        file_bytes
    )

    # -----------------------------------------------------
    # Determine storage locations.
    # -----------------------------------------------------

    UPLOAD_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    INDEX_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = (
        UPLOAD_DIRECTORY /
        f"{document_hash}.pdf"
    )

    index_directory = (
        get_index_directory(
            document_hash
        )
    )

    # -----------------------------------------------------
    # Save the uploaded PDF.
    #
    # We use the hash as the filename internally.
    # The original filename is still displayed to
    # the user.
    # -----------------------------------------------------

    if not pdf_path.exists():

        pdf_path.write_bytes(
            file_bytes
        )

    # -----------------------------------------------------
    # Create the RAG pipeline.
    # -----------------------------------------------------

    pipeline = RAGPipeline(
        index_directory=index_directory
    )

    # -----------------------------------------------------
    # Reuse an existing index if available.
    #
    # This is important because Streamlit reruns the
    # application frequently.
    # -----------------------------------------------------

    index_file = (
        index_directory /
        "index.faiss"
    )

    chunks_file = (
        index_directory /
        "chunks.json"
    )

    if (
        index_file.exists()
        and chunks_file.exists()
    ):

        pipeline.load_existing_index()

    else:

        pipeline.index_document(
            pdf_path=pdf_path
        )

    return (
        pipeline,
        document_hash,
        index_directory,
    )


# =========================================================
# PDF UPLOAD SECTION
# =========================================================

st.header("1. Upload your PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help=(
        "Upload a research paper, technical document, "
        "report, contract, manual, or any other PDF."
    ),
)


# =========================================================
# HANDLE PDF UPLOAD
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # Calculate the hash before doing anything expensive.
    #
    # This lets us determine whether the user uploaded
    # a new document or the same document again.
    # -----------------------------------------------------

    uploaded_bytes = (
        uploaded_file.getvalue()
    )

    current_hash = calculate_file_hash(
        uploaded_bytes
    )

    # -----------------------------------------------------
    # Detect whether this is a new document.
    # -----------------------------------------------------

    is_new_document = (
        st.session_state.document_hash
        != current_hash
    )

    # -----------------------------------------------------
    # Only index/load the document when the uploaded
    # document actually changes.
    # -----------------------------------------------------

    if is_new_document:

        try:

            with st.spinner(
                "Processing your PDF..."
            ):

                (
                    pipeline,
                    document_hash,
                    index_directory,
                ) = index_uploaded_pdf(
                    uploaded_file
                )

            # -------------------------------------------------
            # Store the ready pipeline in session state.
            # -------------------------------------------------

            st.session_state.pipeline = (
                pipeline
            )

            st.session_state.document_name = (
                uploaded_file.name
            )

            st.session_state.document_hash = (
                document_hash
            )

            st.session_state.index_directory = (
                index_directory
            )

            # -------------------------------------------------
            # Start a fresh conversation for the new PDF.
            # -------------------------------------------------

            st.session_state.messages = []

            st.success(
                "PDF is ready. You can now ask questions."
            )

        except Exception as exc:

            st.error(
                "Unable to process this PDF."
            )

            with st.expander(
                "Technical details"
            ):

                st.exception(
                    exc
                )

    # -----------------------------------------------------
    # Show uploaded document information.
    # -----------------------------------------------------

    if (
        st.session_state.pipeline
        is not None
    ):

        pipeline = (
            st.session_state.pipeline
        )

        page_count = get_page_count(
            pipeline
        )

        st.info(
            f"📄 Current document: "
            f"**{st.session_state.document_name}**"
        )

        # -------------------------------------------------
        # Document statistics.
        # -------------------------------------------------

        col1, col2, col3 = st.columns(
            3
        )

        with col1:

            st.metric(
                "Pages",
                page_count,
            )

        with col2:

            st.metric(
                "Indexed chunks",
                pipeline.number_of_chunks,
            )

        with col3:

            st.metric(
                "Status",
                "Ready",
            )


# =========================================================
# QUESTION / ANSWER SECTION
# =========================================================

st.divider()

st.header("2. Ask questions")


# =========================================================
# CHECK WHETHER A DOCUMENT IS READY
# =========================================================

if (
    st.session_state.pipeline
    is None
):

    st.warning(
        "Upload a PDF above to start asking questions."
    )

else:

    pipeline = (
        st.session_state.pipeline
    )

    # -----------------------------------------------------
    # Display previous conversation.
    # -----------------------------------------------------

    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            # -------------------------------------------------
            # Display source pages for assistant messages.
            # -------------------------------------------------

            if (
                message["role"]
                == "assistant"
                and message.get(
                    "pages"
                )
            ):

                st.caption(
                    "Sources: "
                    + ", ".join(
                        f"Page {page}"
                        for page in message["pages"]
                    )
                )

    # -----------------------------------------------------
    # Chat input.
    # -----------------------------------------------------

    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        question = question.strip()

        if not question:

            st.warning(
                "Please enter a question."
            )

        else:

            # -------------------------------------------------
            # Add user question to conversation history.
            # -------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            # -------------------------------------------------
            # Display user question immediately.
            # -------------------------------------------------

            with st.chat_message(
                "user"
            ):

                st.markdown(
                    question
                )

            # -------------------------------------------------
            # Generate answer.
            # -------------------------------------------------

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Thinking..."
                ):

                    try:

                        response = (
                            pipeline.ask(
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
                        # Display page citations.
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
                                "No source pages were returned."
                            )

                        # -----------------------------------------
                        # Store assistant response.
                        # -----------------------------------------

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response.answer,
                                "pages": pages,
                            }
                        )

                    except Exception as exc:

                        # -----------------------------------------
                        # Friendly message for the user.
                        # -----------------------------------------

                        st.error(
                            "I couldn't answer that question "
                            "right now. Please try again."
                        )

                        # -----------------------------------------
                        # Keep technical details available
                        # for development/debugging.
                        # -----------------------------------------

                        with st.expander(
                            "Technical details"
                        ):

                            st.exception(
                                exc
                            )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "About"
    )

    st.write(
        "This application uses a Retrieval-Augmented "
        "Generation (RAG) pipeline to answer questions "
        "from uploaded PDFs."
    )

    st.markdown(
        """
### RAG pipeline

**1. PDF ingestion**

Extract text from the document.

**2. Chunking**

Split the document into searchable chunks.

**3. Embeddings**

Convert chunks into numerical vectors.

**4. FAISS**

Retrieve semantically relevant chunks.

**5. Reranking**

Use a Cross-Encoder to improve relevance.

**6. Gemini**

Generate a grounded answer using the retrieved
document context.

**7. Citations**

Return the pages supporting the answer.
"""
    )

    # -----------------------------------------------------
    # Clear conversation button.
    # -----------------------------------------------------

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()