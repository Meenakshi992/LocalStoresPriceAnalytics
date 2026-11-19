# -------------------------
# Part-5 
# -------------------------

# This module isolates all sidebar widgets—including the PIN/Postal 
# code filter, radius slider, chunk size/overlap controls, live ChromaDB 
# chunk count metric, and flyer uploader.

import os
import streamlit as st
from modules.doc_processor import ingest_document_chunks

def render_sidebar(collection, api_key: str) -> dict:
    """Renders all sidebar controls and returns user configuration settings."""
    with st.sidebar:
        st.header("📍 1. PIN / Postal Code Filter")
        pincode = st.text_input("Enter Postal / ZIP / PIN Code", value="V3T 2W1")
        max_radius = st.slider("Search Radius (km)", min_value=1.0, max_value=15.0, value=8.0, step=0.5)
        forecast_days = st.slider("ML Price Forecast Horizon (Days)", min_value=3, max_value=14, value=7)

        st.divider()
        st.header("🧩 2. Chunking & Vector DB Stats")
        chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=1500, value=600, step=100)
        chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=300, value=100, step=50)

        total_vectors = collection.count()
        flyer_records = collection.get(where={"category": "Uploaded Flyer"})
        flyer_ids = flyer_records["ids"]

        c1, c2 = st.columns(2)
        c1.metric("Total Vectors", total_vectors)
        c2.metric("Flyer Chunks", len(flyer_ids))

        # Button to delete only uploaded flyer chunks while keeping Live Store Feed
        if flyer_ids:
            if st.button("🗑️ Clear Uploaded Flyers", use_container_width=True):
                collection.delete(ids=flyer_ids)
                st.toast("Cleared all uploaded flyer chunks from ChromaDB!")
                st.rerun()

        st.divider()
        st.header("📄 3. Upload Store Flyer / Doc")
        flyer_store = st.selectbox(
            "Tag Store Name",
            ["Walmart Supercentre", "No Frills", "Real Canadian Superstore", "Save-On-Foods", "Other"]
        )
        uploaded_doc = st.file_uploader("Upload Flyer (PDF or TXT)", type=["pdf", "txt"])

        if uploaded_doc is not None:
            if st.button("Embed & Index Document", use_container_width=True):
                with st.spinner("Chunking & embedding with text-embedding-3-small..."):
                    stats = ingest_document_chunks(
                        collection=collection,
                        uploaded_file=uploaded_doc,
                        store_tag=flyer_store,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                st.success(f"Indexed **{stats['chunks_added']}** chunks from `{uploaded_doc.name}`!")
                st.rerun()

    return {
        "api_key": api_key,
        "pincode": pincode,
        "max_radius": max_radius,
        "forecast_days": forecast_days,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
