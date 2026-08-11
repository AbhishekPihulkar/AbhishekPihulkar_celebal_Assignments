"""
app.py
------
DriveWise UI. Run with:  streamlit run app.py

Prereqs:
  1. Put brochure PDFs in data/, named like hyundai_creta_2026.pdf
  2. python ingest.py
  3. python vector_store.py
  4. Set GROQ_API_KEY (env var or .env file)
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

from retriever import retrieve, list_available_brands_models
from generator import generate_answer
from logger import log_query, Timer

load_dotenv()

st.set_page_config(page_title="DriveWise", page_icon="🚗", layout="centered")
st.title("🚗 DriveWise")
st.caption("Ask questions about a car, grounded in its official brochure.")

# --- Sidebar: brand/model selection (metadata filter driver) ---
try:
    available = list_available_brands_models()
except Exception:
    available = []

if not available:
    st.warning(
        "No indexed brochures found. Run `python ingest.py` then "
        "`python vector_store.py` after adding PDFs to data/."
    )
    st.stop()

brands = sorted({b for b, m in available})
st.sidebar.header("Select vehicle")
selected_brand = st.sidebar.selectbox("Brand", brands)
models_for_brand = sorted({m for b, m in available if b == selected_brand})
selected_model = st.sidebar.selectbox("Model", models_for_brand)

st.sidebar.divider()
st.sidebar.caption(
    "DriveWise retrieves only from the selected vehicle's brochure "
    "(metadata filtering), re-ranks results, and cites its sources."
)

# --- Chat history ---
current_selection = (selected_brand, selected_model)
if st.session_state.get("active_selection") != current_selection:
    st.session_state.active_selection = current_selection
    st.session_state.messages = [{
        "role": "assistant",
        "content": f"Hello! 👋 How can I help you with the {selected_brand} {selected_model}?",
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"- **{s['brand']} {s['model']}** — {s['section'].title()} "
                        f"(Page: {s['page']}, Chunk: `{s['chunk_id']}`)"
                    )

# --- Chat input ---
query = st.chat_input(f"Ask about the {selected_brand} {selected_model}...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving brochure sections..."):
            diagnostics = None
            try:
                with Timer() as total_t:
                    chunks, diagnostics = retrieve(query, selected_brand, selected_model)

                    with Timer() as gen_t:
                        result = generate_answer(query, chunks)

                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Sources"):
                        for s in result["sources"]:
                            st.markdown(
                                f"- **{s['brand']} {s['model']}** — {s['section'].title()} "
                                f"(Page: {s['page']}, Chunk: `{s['chunk_id']}`)"
                            )

                log_query(
                    selected_brand, selected_model, query,
                    status="success", diagnostics=diagnostics,
                    generation_time_ms=gen_t.elapsed * 1000,
                    total_time_ms=total_t.elapsed * 1000,
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")
                log_query(
                    selected_brand, selected_model, query,
                    status="failed", diagnostics=diagnostics, error=str(e),
                )
