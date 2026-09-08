"""
PDF RAG Assistant - Streamlit UI

This application provides:
- PDF upload
- Automatic document indexing
- Document statistics
- RAG-based question answering
- Conversation history
- Source/page references

All RAG functionality remains inside src/.
"""

import hashlib
import html
from pathlib import Path
from uuid import uuid4

import streamlit as st

from src.pipeline import RAGPipeline


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# APPLICATION DIRECTORIES
# =========================================================

UPLOAD_DIRECTORY = Path("data/uploads")
INDEX_DIRECTORY = Path("data/indexes/app")


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

/* =====================================================
   GLOBAL PAGE
   ===================================================== */

.block-container {
    width: 92vw !important;
    max-width: 1750px !important;

    padding-top: 4.5rem !important;
    padding-bottom: 2.5rem !important;

    padding-left: 0 !important;
    padding-right: 0 !important;
}


/* Remove unnecessary top spacing */



/* =====================================================
   APPLICATION HEADER
   ===================================================== */

.app-header {
    margin-bottom: 1.5rem;
}

.app-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1.15;
    margin: 0;
}

.app-subtitle {
    margin-top: 0.4rem;
    font-size: 0.88rem;
    line-height: 1.5;
    opacity: 0.52;
}


/* =====================================================
   SECTION LABEL
   ===================================================== */

.section-label {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    opacity: 0.43;
    margin-bottom: 0.55rem;
}


/* =====================================================
   UPLOAD AREA
   ===================================================== */

[data-testid="stFileUploader"] {
    margin-bottom: 1.15rem;
}

[data-testid="stFileUploaderDropzone"] {
    min-height: 82px !important;

    border-radius: 9px !important;

    border: 1px dashed rgba(140, 140, 140, 0.38) !important;

    background: rgba(255, 255, 255, 0.012) !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    padding: 0.4rem !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] > div {
    font-size: 0.76rem !important;
}


/* =====================================================
   DOCUMENT CARD
   ===================================================== */

.document-card {
    border: 1px solid rgba(140, 140, 140, 0.18);

    border-radius: 10px;

    padding: 0.9rem 1rem;

    background: rgba(255, 255, 255, 0.018);

    margin-bottom: 1.45rem;
}

.document-header {
    display: flex;

    align-items: center;

    justify-content: space-between;

    gap: 1rem;

    margin-bottom: 0.75rem;
}

.document-name {
    font-size: 0.91rem;

    font-weight: 650;

    line-height: 1.3;

    word-break: break-word;
}

.document-meta {
    margin-top: 0.18rem;

    font-size: 0.68rem;

    opacity: 0.42;
}


/* =====================================================
   STATUS
   ===================================================== */

.status-badge {
    display: inline-flex;

    align-items: center;

    gap: 0.35rem;

    border: 1px solid rgba(34, 197, 94, 0.3);

    border-radius: 999px;

    padding: 0.22rem 0.55rem;

    font-size: 0.63rem;

    font-weight: 650;

    color: rgb(34, 197, 94);

    white-space: nowrap;
}

.status-dot {
    width: 5px;
    height: 5px;

    border-radius: 50%;

    background: rgb(34, 197, 94);
}


/* =====================================================
   DOCUMENT STATISTICS
   ===================================================== */

.stat-container {
    display: grid;

    grid-template-columns:
        repeat(3, minmax(0, 1fr));

    gap: 0.55rem;
}

.stat {
    border: 1px solid rgba(140, 140, 140, 0.13);

    border-radius: 7px;

    padding: 0.55rem 0.7rem;

    background: rgba(255, 255, 255, 0.01);
}

.stat-label {
    font-size: 0.57rem;

    font-weight: 650;

    text-transform: uppercase;

    letter-spacing: 0.09em;

    opacity: 0.42;

    margin-bottom: 0.18rem;
}

.stat-value {
    font-size: 0.95rem;

    font-weight: 650;

    letter-spacing: -0.02em;
}


/* =====================================================
   CONVERSATION HEADER
   ===================================================== */

.conversation-header {
    margin-bottom: 0.7rem;
}

.conversation-title {
    font-size: 1.05rem;

    font-weight: 650;

    letter-spacing: -0.025em;
}

.conversation-description {
    margin-top: 0.18rem;

    font-size: 0.73rem;

    opacity: 0.44;
}


/* =====================================================
   CHAT CONTAINER
   ===================================================== */

.chat-area {
    width: 100%;
}


/* =====================================================
   REMOVE STREAMLIT CHAT AVATARS
   ===================================================== */

[data-testid="stChatMessageAvatar"] {
    display: none !important;
}

[data-testid="stChatMessage"] {
    padding-left: 0 !important;
    padding-right: 0 !important;

    margin-bottom: 0.45rem !important;
}


/* =====================================================
   USER MESSAGE
   ===================================================== */

.user-message {
    width: 100%;

    padding: 0.78rem 0.95rem;

    border-radius: 8px;

    background: rgba(255, 255, 255, 0.045);

    border: 1px solid rgba(140, 140, 140, 0.09);

    font-size: 0.87rem;

    line-height: 1.5;

    margin-bottom: 0.6rem;
}


/* =====================================================
   ASSISTANT MESSAGE
   ===================================================== */

.assistant-message {
    width: 100%;

    padding: 0.15rem 0.95rem 0.7rem 0.95rem;

    font-size: 0.88rem;

    line-height: 1.65;
}

.assistant-message p {
    margin-bottom: 0.45rem;
}


/* =====================================================
   SOURCE
   ===================================================== */

.source-reference {
    margin-top: 0.35rem;

    font-size: 0.66rem;

    opacity: 0.42;

    letter-spacing: 0.005em;
}


/* =====================================================
   CHAT INPUT
   ===================================================== */

[data-testid="stChatInput"] {
    margin-top: 0.35rem;
}

[data-testid="stChatInput"] textarea {
    min-height: 50px !important;

    border-radius: 9px !important;

    font-size: 0.84rem !important;
}


/* =====================================================
   CLEAR BUTTON
   ===================================================== */

.clear-button-container {
    margin-top: -0.15rem;
}


/* =====================================================
   EMPTY STATE
   ===================================================== */

.empty-state {
    text-align: center;

    padding-top: 1.2rem;

    font-size: 0.74rem;

    opacity: 0.38;
}


/* =====================================================
   STREAMLIT STATUS
   ===================================================== */

[data-testid="stStatusWidget"] {
    border-radius: 8px;
}


/* =====================================================
   ALERTS
   ===================================================== */

[data-testid="stAlert"] {
    border-radius: 8px;

    font-size: 0.76rem;
}


/* =====================================================
   BUTTONS
   ===================================================== */

.stButton > button {
    min-height: 30px;

    border-radius: 7px;

    font-size: 0.7rem;

    font-weight: 550;
}


/* =====================================================
   MOBILE
   ===================================================== */

@media (max-width: 900px) {

    .block-container {
    width: 100% !important;
    max-width: none !important;

    padding-top: 4.5rem !important;
    padding-bottom: 2.5rem !important;

    padding-left: 3.5rem !important;
    padding-right: 3.5rem !important;

    margin: 0 !important;
}

    .app-title {
        font-size: 1.65rem;
    }

    .stat-container {
        grid-template-columns: 1fr;
    }

    .document-header {
        align-items: flex-start;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================================================
# APPLICATION HEADER
# =========================================================

st.markdown(
    """
<div class="app-header">
    <div class="app-title">PDF RAG Assistant</div>
    <div class="app-subtitle">
        Ask questions about any PDF and get answers grounded
        in the document's content with source references.
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# UPLOAD DOCUMENT
# =========================================================

st.markdown(
    '<div class="section-label">Upload document</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    label_visibility="collapsed",
    help="Upload a PDF to create a searchable knowledge base.",
)


# =========================================================
# PROCESS DOCUMENT
# =========================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    # -----------------------------------------------------
    # Hash the actual file.
    #
    # This means:
    # - Same PDF uploaded again = no unnecessary indexing.
    # - Different PDFs with the same filename = safe.
    # -----------------------------------------------------

    document_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

    if (
        st.session_state.document_hash
        != document_hash
    ):

        # -------------------------------------------------
        # Reset conversation for new document.
        # -------------------------------------------------

        st.session_state.chat_history = []

        UPLOAD_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        INDEX_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -------------------------------------------------
        # Generate unique internal ID.
        # -------------------------------------------------

        document_id = uuid4().hex

        pdf_path = (
            UPLOAD_DIRECTORY
            / f"{document_id}.pdf"
        )

        document_index_directory = (
            INDEX_DIRECTORY
            / document_id
        )

        # -------------------------------------------------
        # Save PDF.
        # -------------------------------------------------

        try:

            with open(
                pdf_path,
                "wb",
            ) as file:

                file.write(file_bytes)

        except OSError as exc:

            st.error(
                "The document could not be saved."
            )

            with st.expander(
                "Technical details"
            ):

                st.exception(exc)

            st.stop()

        # -------------------------------------------------
        # Build index.
        # -------------------------------------------------

        try:

            with st.status(
                "Preparing document...",
                expanded=True,
            ) as status:

                st.write(
                    "Reading PDF content..."
                )

                pipeline = RAGPipeline(
                    index_directory=(
                        document_index_directory
                    )
                )

                pipeline.index_document(
                    pdf_path=pdf_path
                )

                status.update(
                    label="Document ready",
                    state="complete",
                    expanded=False,
                )

            # ---------------------------------------------
            # Save application state.
            # ---------------------------------------------

            st.session_state.pipeline = pipeline

            st.session_state.document_name = (
                uploaded_file.name
            )

            st.session_state.document_id = (
                document_id
            )

            st.session_state.document_hash = (
                document_hash
            )

            # -------------------------------------------------
            # IMPORTANT:
            #
            # Don't immediately force another rerun here.
            #
            # The previous version used st.rerun(), which could
            # leave the browser at an awkward scroll position.
            # -------------------------------------------------

        except Exception as exc:

            st.session_state.pipeline = None

            st.session_state.document_name = None

            st.session_state.document_id = None

            st.session_state.document_hash = None

            st.error(
                "The document could not be processed."
            )

            with st.expander(
                "Technical details"
            ):

                st.exception(exc)


# =========================================================
# DOCUMENT INFORMATION
# =========================================================

if (
    st.session_state.pipeline is not None
    and st.session_state.document_name is not None
):

    pipeline = st.session_state.pipeline

    st.markdown(
        '<div class="section-label">Indexed document</div>',
        unsafe_allow_html=True,
    )

    safe_document_name = html.escape(
        st.session_state.document_name
    )

    document_card = f"""
<div class="document-card">

<div class="document-header">

<div>
<div class="document-name">
{safe_document_name}
</div>

<div class="document-meta">
Searchable knowledge base
</div>
</div>

<div class="status-badge">
<span class="status-dot"></span>
Ready
</div>

</div>


<div class="stat-container">

<div class="stat">
<div class="stat-label">
Pages
</div>

<div class="stat-value">
{pipeline.page_count}
</div>
</div>


<div class="stat">
<div class="stat-label">
Chunks
</div>

<div class="stat-value">
{pipeline.number_of_chunks}
</div>
</div>


<div class="stat">
<div class="stat-label">
Characters
</div>

<div class="stat-value">
{pipeline.character_count:,}
</div>
</div>

</div>

</div>
"""

    st.markdown(
        document_card,
        unsafe_allow_html=True,
    )


# =========================================================
# CONVERSATION
# =========================================================

if (
    st.session_state.pipeline is not None
    and st.session_state.pipeline.is_ready
):

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    conversation_col, clear_col = st.columns(
        [10, 1]
    )

    with conversation_col:

        st.markdown(
            """
<div class="conversation-header">

<div class="conversation-title">
Conversation
</div>

<div class="conversation-description">
Ask questions about the uploaded document.
Answers are generated from retrieved document content.
</div>

</div>
""",
            unsafe_allow_html=True,
        )

    with clear_col:

        if st.session_state.chat_history:

            if st.button(
                "Clear",
                use_container_width=True,
            ):

                st.session_state.chat_history = []

                st.rerun()


    # =====================================================
    # CHAT HISTORY
    # =====================================================

    for message in st.session_state.chat_history:

        role = message["role"]

        content = message["content"]

        # -------------------------------------------------
        # USER MESSAGE
        # -------------------------------------------------

        if role == "user":

            safe_content = html.escape(
                content
            )

            st.markdown(
                f"""
<div class="user-message">
{safe_content}
</div>
""",
                unsafe_allow_html=True,
            )

        # -------------------------------------------------
        # ASSISTANT MESSAGE
        # -------------------------------------------------

        else:

            st.markdown(
                f"""
<div class="assistant-message">
{content}
</div>
""",
                unsafe_allow_html=True,
            )

            pages = message.get(
                "pages",
                [],
            )

            if pages:

                page_text = ", ".join(
                    f"Page {page}"
                    for page in pages
                )

                st.markdown(
                    f"""
<div class="source-reference">
Source · {page_text}
</div>
""",
                    unsafe_allow_html=True,
                )


    # =====================================================
    # QUESTION INPUT
    # =====================================================

    question = st.chat_input(
        "Ask anything about your document..."
    )


    # =====================================================
    # PROCESS QUESTION
    # =====================================================

    if question:

        # -------------------------------------------------
        # Add user question to history.
        # -------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        # -------------------------------------------------
        # Display user question immediately.
        # -------------------------------------------------

        safe_question = html.escape(
            question
        )

        st.markdown(
            f"""
<div class="user-message">
{safe_question}
</div>
""",
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # Generate answer.
        # -------------------------------------------------

        with st.spinner(
            "Searching document..."
        ):

            try:

                response = (
                    st.session_state.pipeline.ask(
                        question=question
                    )
                )

                # -----------------------------------------
                # Assistant answer
                # -----------------------------------------

                st.markdown(
                    f"""
<div class="assistant-message">
{response.answer}
</div>
""",
                    unsafe_allow_html=True,
                )

                # -----------------------------------------
                # Sources
                # -----------------------------------------

                pages = response.page_numbers

                if pages:

                    page_text = ", ".join(
                        f"Page {page}"
                        for page in pages
                    )

                    st.markdown(
                        f"""
<div class="source-reference">
Source · {page_text}
</div>
""",
                        unsafe_allow_html=True,
                    )

                # -----------------------------------------
                # Store response.
                # -----------------------------------------

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "pages": pages,
                    }
                )

            except Exception as exc:

                st.error(
                    "I couldn't answer that question. "
                    "Please try again."
                )

                with st.expander(
                    "Technical details"
                ):

                    st.exception(exc)


# =========================================================
# EMPTY STATE
# =========================================================

else:

    st.markdown(
        """
<div class="empty-state">
Upload a PDF to create your searchable document knowledge base.
</div>
""",
        unsafe_allow_html=True,
    )