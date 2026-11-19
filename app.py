# Geo-spatial Grocery Price Comparison App
# Author-Meenakshi Sharma
# Start-date- 14 Nov. 2026

# -------------------------
# Part-1
# -------------------------
# This clean entry point brings together sidebar.py, 
# vector_db.py, and price_model.py into a 3-tab interactive dashboard.

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI
import os
from modules.sidebar import render_sidebar
from modules.vector_db import get_chroma_collection, resolve_pincode, seed_nearby_stores_to_db
from modules.price_model import generate_historical_prices, train_and_forecast_prices

st.set_page_config(page_title="CartGap AI — Modular ML & RAG Capstone", layout="wide")
st.title("🛒 CartGap AI: Geospatial Price Gap, ML Forecast & Document RAG")


def unpack_chroma_query(res: dict):
    """Safely unwraps ChromaDB's nested [[...]] query outputs into flat lists."""
    docs_raw = res.get("documents") or []
    metas_raw = res.get("metadatas") or []
    docs = docs_raw if (docs_raw and isinstance(docs_raw, list)) else docs_raw
    metas = metas_raw if (metas_raw and isinstance(metas_raw, list)) else metas_raw
    return docs, metas


# Step 1: Get OpenAI API Key first
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("🔑 Enter OpenAI API Key", type="password")
    if not api_key:
        st.warning("Please enter your `OPENAI_API_KEY` in the sidebar to start.")
        st.stop()

# Step 2: Initialize ChromaDB BEFORE rendering sidebar
@st.cache_resource
def load_db(key: str):
    return get_chroma_collection(api_key=key)


collection = load_db(api_key)
llm_client = OpenAI(api_key=api_key)

# Step 3: Render Sidebar with live collection & API key
config = render_sidebar(collection=collection, api_key=api_key)

# Step 4: Resolve PIN/Postal Code & Load Nearest Branch per Store
coords, address = resolve_pincode(config["pincode"])
user_lat, user_lon = coords
price_df = seed_nearby_stores_to_db(collection, user_lat, user_lon)
nearby_df = price_df[price_df["Distance_km"] <= config["max_radius"]]
hist_df = generate_historical_prices(nearby_df)

st.caption(f"📍 **Location:** {address} | **Radius:** {config['max_radius']} km | **Total Vectors in DB:** `{collection.count()}`")

# --- Main Dashboard Tabs ---
tab_gap, tab_ml, tab_rag = st.tabs([
    "📊 Grocery List & Price Gap Charts",
    "📈 Daily ML Price Prediction",
    "💬 Specific Store & Flyer RAG Chat"
])

# --- TAB 1: Multi-Store Price Gap Charts ---
with tab_gap:
    st.subheader("Compare Your Grocery List Across Nearby Stores")
    user_list = st.text_input(
        "Enter comma-separated grocery items:",
        value="milk, eggs, bananas, chicken, bread",
    )

    if nearby_df.empty:
        st.warning(
            f"No stores found within {config['max_radius']} km. "
            "Increase the Search Radius slider in the sidebar."
        )
    elif user_list.strip():
        keywords = [x.strip().lower() for x in user_list.split(",") if x.strip()]

        matched_frames = []
        for kw in keywords:
            hits = nearby_df[
                nearby_df["Item"].str.lower().str.contains(kw, regex=False)
                | nearby_df["Category"].str.lower().str.contains(kw, regex=False)
            ].copy()
            if not hits.empty:
                hits["Query_Item"] = kw.title()
                matched_frames.append(hits)

        if not matched_frames:
            st.info("No matching items found. Try: milk, eggs, bananas, chicken, bread, or cheese.")
        else:
            basket_df = pd.concat(matched_frames).drop_duplicates(subset=["Query_Item", "Store"])
            totals = (
                basket_df.groupby(["Store", "Branch", "Distance_km"], as_index=False)["Price"]
                .sum()
                .rename(columns={"Price": "Total_Cost"})
                .sort_values("Total_Cost")
            )
            min_cost = totals["Total_Cost"].min()
            totals["Price_Gap ($)"] = (totals["Total_Cost"] - min_cost).round(2)
            totals["Price_Gap (%)"] = (((totals["Total_Cost"] - min_cost) / min_cost) * 100).round(1)

            c1, c2 = st.columns([1.15, 1.25])
            with c1:
                best = next(iter(totals.to_dict(orient="records")))
                st.success(
                    f"Cheapest Store: {best['Store']} ({best['Branch']} - {best['Distance_km']} km) "
                    f"- ${best['Total_Cost']:.2f}"
                )
                st.dataframe(totals, use_container_width=True, hide_index=True)

            with c2:
                fig_gap = go.Figure()
                fig_gap.add_trace(go.Bar(
                    x=totals["Store"],
                    y=totals["Total_Cost"],
                    name="Total Cost ($)",
                    text=totals["Total_Cost"].apply(lambda v: f"${v:.2f}"),
                    textposition="auto",
                    marker_color="#1F77B4",
                ))
                fig_gap.add_trace(go.Bar(
                    x=totals["Store"],
                    y=totals["Price_Gap ($)"],
                    name="Price Gap ($)",
                    text=totals["Price_Gap ($)"].apply(lambda v: f"+${v:.2f}" if v > 0 else "$0.00"),
                    textposition="outside",
                    marker_color="#D62728",
                ))
                fig_gap.update_layout(
                    barmode="overlay",
                    title="Total Basket Cost & Store Price Gap ($)",
                    yaxis_title="Cost ($)",
                )
                st.plotly_chart(fig_gap, use_container_width=True)

            fig_items = px.bar(
                basket_df,
                x="Query_Item",
                y="Price",
                color="Store",
                barmode="group",
                text_auto=".2f",
                title="Item-by-Item Price Breakdown",
            )
            st.plotly_chart(fig_items, use_container_width=True)
# --- TAB 2: Daily ML Price Forecast ---
with tab_ml:
    st.subheader("🤖 Daily Grocery Price Prediction (Ridge Time-Series Model)")
    if not nearby_df.empty:
        col_a, col_b = st.columns(2)
        sel_store = col_a.selectbox("Select Store", nearby_df["Store"].unique())
        sel_item = col_b.selectbox("Select Product", nearby_df["Item"].unique())

        result = train_and_forecast_prices(hist_df, sel_store, sel_item, forecast_days=config["forecast_days"])
        best_day = result["best_day"]

        st.info(
            f"📅 **Best Predicted Day to Buy:** **{best_day['Date'].strftime('%A, %b %d')}** at **${best_day['Predicted_Price']:.2f}** "
            f"| Model Validation MAE: **±${result['mae']:.3f}**"
        )

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=result["history"]["Date"], y=result["history"]["Price"],
            mode="lines+markers", name="Historical Daily Price"
        ))
        fig_fc.add_trace(go.Scatter(
            x=result["forecast"]["Date"], y=result["forecast"]["Predicted_Price"],
            mode="lines+markers", name="ML Predicted Price", line=dict(dash="dash", color="orange", width=3)
        ))
        fig_fc.update_layout(title=f"{config['forecast_days']}-Day Forecast: {sel_item} ({sel_store})", yaxis_title="Price ($)")
        st.plotly_chart(fig_fc, use_container_width=True)

# --- TAB 3: Specific Store & Uploaded Document RAG Chat ---
# --- TAB 3: Specific Store & Uploaded Document RAG Chat ---
with tab_rag:
    st.subheader("💬 Ask Specific Store Prices or Query Uploaded Flyers")

    search_mode = st.radio(
        "Select Knowledge Source:",
        options=["Uploaded Flyers Only", "Both (Flyers + Live Store Feed)", "Live Store Feed Only"],
        index=0,
        horizontal=True
    )

    query = st.text_input("Ask a question (e.g., 'List all meat and produce deals in the flyer' or 'When is the flyer valid?'):")

    if query:
        docs, metas = [], []
        uploaded_check = collection.get(where={"category": "Uploaded Flyer"})

        # Mode 1: Uploaded Flyers Only
        if search_mode == "Uploaded Flyers Only":
            if not uploaded_check["ids"]:
                st.warning("⚠️ No flyer uploaded yet! Please upload a PDF or TXT flyer in the sidebar and click **Embed & Index Document**.")
                st.stop()

            total_flyer_chunks = len(uploaded_check["ids"])
            broad_keywords = ["all", "list", "every", "summarize", "summary", "items", "deals", "what is in"]
            is_broad_query = any(kw in query.lower() for kw in broad_keywords)

            # If it's a broad summary question OR the flyer is <= 25 chunks, send ALL flyer chunks in order!
            if is_broad_query or total_flyer_chunks <= 25:
                paired = sorted(
                    zip(uploaded_check["documents"], uploaded_check["metadatas"]),
                    key=lambda item: item[1].get("chunk_index", 0) if isinstance(item[1], dict) else 0
                )
                docs = [doc_text for doc_text, _ in paired]
                metas = [meta_dict for _, meta_dict in paired]
            else:
                n_flyer = min(12, total_flyer_chunks)
                flyer_res = collection.query(
                    query_texts=[query],
                    n_results=n_flyer,
                    where={"category": "Uploaded Flyer"}
                )
                docs, metas = unpack_chroma_query(flyer_res)

        # Mode 2: Live Store Feed Only
        elif search_mode == "Live Store Feed Only":
            live_res = collection.query(query_texts=[query], n_results=6, where={"source": "Live Store Feed"})
            docs, metas = unpack_chroma_query(live_res)

        # Mode 3: Both Sources Combined
        else:
            live_res = collection.query(query_texts=[query], n_results=5, where={"source": "Live Store Feed"})
            l_docs, l_metas = unpack_chroma_query(live_res)
            docs.extend(l_docs)
            metas.extend(l_metas)

            if uploaded_check["ids"]:
                n_flyer = min(12, len(uploaded_check["ids"]))
                flyer_res = collection.query(query_texts=[query], n_results=n_flyer, where={"category": "Uploaded Flyer"})
                f_docs, f_metas = unpack_chroma_query(flyer_res)
                docs.extend(f_docs)
                metas.extend(f_metas)

        # Deduplicate chunks while preserving order
        seen = set()
        unique_pairs = []
        for d, m in zip(docs, metas):
            if isinstance(m, dict) and d not in seen:
                seen.add(d)
                unique_pairs.append((d, m))

        context = "\n".join(
            [f"- [Source: {m.get('source', 'Unknown')} | Store: {m.get('store', 'Unknown')}] {d}" for d, m in unique_pairs]
        )

        resp = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are CartGap AI. Answer the user's question thoroughly using ONLY the provided context. "
                        "When asked for items or deals in a flyer, list ALL matching categories and products found across all chunks, "
                        "not just the first item. Cite the exact document source name and store."
                    )
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        st.markdown("### 🤖 Answer")
        first_choice = next(iter(resp.choices))
        st.write(first_choice.message.content)

        with st.expander(f"🔍 View Retrieved Context Chunks Sent to LLM ({len(unique_pairs)} chunks)"):
            st.text(context)
