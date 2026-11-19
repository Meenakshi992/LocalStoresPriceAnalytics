# Copilot instructions for LocalStoresPriceAnalytics

## Project overview
This repository is a Streamlit-based grocery intelligence app that combines geospatial store discovery, vector search, document RAG, and price forecasting. The main app is `app.py`, which orchestrates the UI and backend services for:

- comparing basket costs across nearby grocery stores
- finding the cheapest nearby option by postal/PIN code
- forecasting daily prices with a Ridge regression model
- answering questions from uploaded store flyers with ChromaDB + OpenAI embeddings

## Architecture and responsibilities
- `app.py`: main dashboard entry point. Keep the app shell, tab layout, and orchestration logic here.
- `modules/sidebar.py`: sidebar controls for postal code, radius, forecast horizon, chunk configuration, and document upload.
- `modules/vector_db.py`: ChromaDB setup, geocoding, and nearest-store catalog seeding. Preserve metadata keys such as `source`, `store`, `branch`, `distance_km`, `category`, `item_name`, and `price`.
- `modules/doc_processor.py`: PDF/TXT extraction and chunking for uploaded flyers. Keep chunk metadata consistent with RAG queries.
- `modules/price_model.py`: synthetic historical price generation and forecast model training.
- `cartgap_openai_chroma_db/`: persistent local ChromaDB storage. Do not delete or repurpose it lightly.

## Development workflow
- Use the repo-local Python environment in `.venv` or the project environment configured by the user.
- Keep dependencies aligned with the project stack: `streamlit`, `pandas`, `plotly`, `scikit-learn`, `chromadb`, `openai`, `geopy`, `PyMuPDF`, `langchain-text-splitters`.
- Prefer minimal, surgical edits. The app is modular; changes should stay in the most relevant module rather than being scattered across the codebase.
- When adding new functionality, keep the app flow consistent: sidebar config -> ChromaDB collection -> nearby-store pricing -> forecast or RAG response.

## Runtime and environment requirements
- The app requires `OPENAI_API_KEY` to be available at runtime. Do not hardcode secrets or API keys.
- Favor `os.getenv("OPENAI_API_KEY")` and existing Streamlit sidebar patterns instead of introducing new secret storage.
- Respect the existing user-facing patterns for warnings and setup flows in `app.py`.
- The project targets Python 3.14 compatibility; keep syntax and dependencies compatible with that baseline.

## Data and model conventions
- Live store feed documents and uploaded flyers are intentionally stored in the same Chroma collection but separated by metadata (`source`, `category`, `store`). Preserve that distinction in queries and deletes.
- Do not break the `render_sidebar()` contract or the `train_and_forecast_prices()` return shape unless you update all call sites in `app.py`.
- Keep price/metadata keys stable: query logic depends on `Store`, `Branch`, `Distance_km`, `Item`, `Price`, `source`, `category`, and `chunk_index`.
- For RAG behavior, keep retrieved chunks grounded in the provided context and cite source/store information when answering questions.

## UI expectations
- Preserve the multi-tab Streamlit dashboard structure in `app.py`.
- Keep interactive controls and explanatory text user-friendly and consistent with the existing app tone.
- When changing visuals or metrics, ensure they still match the underlying `DataFrame` / model outputs and do not introduce silent mismatches.

## Validation guidance
- Validate Python syntax with a focused command such as `python -m compileall app.py modules` after changes.
- For functional changes, prefer the smallest relevant validation, such as running the Streamlit app or a focused import check rather than broad suite runs.
- If a change affects vector search or document ingestion, confirm the metadata schema still matches the app’s query logic before finishing.

## Do not do
- Do not commit or expose secrets, API keys, or personal data.
- Do not add broad framework churn or unrelated refactors.
- Do not break the separation between the live store feed and uploaded flyer data.
- Do not bypass the existing app architecture by embedding model logic directly in the UI when it belongs in `modules/`.
