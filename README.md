# 🛒 LocalStoresPriceAnalytics: Geospatial Grocery Price Comparison, ML Forecaster & Document RAG

**LocalStoresPriceAnalytics** is an end-to-end AI/ML capstone application built with **Python**, **Streamlit**, **ChromaDB**, **OpenAI (`text-embedding-3-small` & `gpt-4o-mini`)**, and **Scikit-Learn**. It helps shoppers compare multi-store grocery basket price gaps around their postal/PIN code, forecast daily item prices, and query uploaded store flyers using Retrieval-Augmented Generation (RAG).

---

## 🌟 Key Features

* **📍 Geospatial Store Discovery (PIN / Postal Code Filter):** Converts any Canadian Postal Code, US ZIP Code, or PIN code into latitude/longitude coordinates (`geopy`) and filters nearby grocery stores within a customizable kilometer radius.
* **📊 Semantic Grocery List & Price Gap Charts:** Uses **OpenAI `text-embedding-3-small` (1,536-dimensional embeddings)** to match user-entered grocery lists against store catalogs, calculating total basket costs and visualizing exact dollar/percentage price gaps with interactive **Plotly** charts.
* **📈 Daily ML Price Prediction:** Engineers calendar and weekly flyer promotional features to train a **Ridge Regression** time-series model, displaying validation **MAE (Mean Absolute Error)** and recommending the cheapest upcoming day to buy.
* **💬 Store & Uploaded Flyer RAG Chatbot:** Lets users upload weekly store flyers (`.pdf` or `.txt`), chunks text with configurable `chunk_size` and `chunk_overlap`, indexes vectors into **ChromaDB**, and answers specific store/item price questions with source citations.

---

## 🏗️ System Architecture

```text
+-----------------------------------------------------------------------+

|                         Streamlit Frontend UI                         |
|  [Sidebar: PIN Code, Radius, Chunk Controls, Flyer PDF Upload]        |
|  [Tab 1: Basket Gap Charts] [Tab 2: ML Forecast] [Tab 3: Store RAG]   |
+-------------------+-----------------------------+---------------------+

                    |                             |
                    v                             v
+-----------------------------------+   +-------------------------------+

|       Vector & RAG Pipeline       |   |   ML Forecasting Pipeline     |
| 1. PyMuPDF + RecursiveSplitter    |   | 1. 90-Day Price History       |
| 2. OpenAI text-embedding-3-small  |   | 2. Calendar & Promo Features  |
| 3. ChromaDB (HNSW Cosine Index)   |   | 3. Ridge Regression + MAE     |
| 4. GPT-4o-mini Grounded QA        |   | 4. 7-to-14 Day Forecast       |
+-----------------------------------+   +-------------------------------+
```

---

## 📂 Repository Structure

```text
/workspaces/LocalStoresPriceAnalytics/
├── app.py                  # Main Streamlit dashboard & multi-tab UI
   modules
   ├── sidebar.py              # Sidebar filters (PIN/Postal code, chunk sliders, uploader)
   ├── vector_db.py            # ChromaDB + text-embedding-3-small setup & store seeding
   ├── doc_processor.py        # PDF/TXT parsing, text chunking, & vector ingestion
   ├── price_model.py          # Time-series feature engineering & Ridge ML model
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation & setup guide
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com
cd localstorespriceanalytics
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
source /workspaces/LocalStoresPriceAnalytics/.venv/bin/activate     
# Installed requirements.txt into the project’s .venv  
/workspaces/LocalStoresPriceAnalytics/.venv/bin/python -m pip install -r /workspaces/LocalStoresPriceAnalytics/requirements.txt
```

### 3. Set Your OpenAI API Key
```bash
export OPENAI_API_KEY="sk-your-api-key-here"   # On Windows PowerShell: \$env:OPENAI_API_KEY="sk-..."
```

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```

---

## 🧪 Bootcamp ML Concepts Demonstrated

| Domain | Implementation in LocalStoresPriceAnalytics |
| :--- | :--- |
| **Embeddings & Vector Search** | `text-embedding-3-small` (1,536 dims) + `ChromaDB` HNSW cosine similarity index |
| **Document Preprocessing** | `PyMuPDF` (`fitz`) text extraction + `RecursiveCharacterTextSplitter` with tunable overlap |
| **Retrieval-Augmented Generation** | Grounded `gpt-4o-mini` prompting with metadata source citations (`source`, `store`) |
| **Supervised Time-Series ML** | Day-of-week & promotional cycle feature engineering + `Ridge` regression + `MAE` validation |
| **Geospatial Analytics** | `Nominatim` geocoding + `geodesic` distance calculation in kilometers |

---

## 🔮 Future Enhancements

* **Live Flyer Ingestion:** Integrate automated scraping or public retail APIs to pull daily promotional pricing automatically.
* **Hybrid Search & Reranking:** Combine dense ChromaDB vector search with sparse **BM25** keyword scoring for exact SKU/brand matching.
* **RAG Evaluation Suite:** Add automated **RAGAS** metrics (Faithfulness & Context Precision) to benchmark retrieval accuracy.
