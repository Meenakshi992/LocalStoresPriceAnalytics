import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from geopy.distance import geodesic
from geopy.geocoders import Nominatim


def get_chroma_collection(api_key: str, db_path: str = "./cartgap_openai_chroma_db"):
    """Initializes persistent ChromaDB with OpenAI text-embedding-3-small."""
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="grocery_flyers_openai_emb",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def resolve_pincode(pincode: str):
    """Converts a Postal Code / ZIP / PIN code into (lat, lon) and formatted address."""
    geolocator = Nominatim(user_agent="cartgap_ai_single_branch_v1")
    clean_code = pincode.strip().upper()
    try:
        location = geolocator.geocode(clean_code, timeout=5)
        if location:
            return (location.latitude, location.longitude), location.address
    except Exception:
        pass
    return (49.1896, -122.8489), "Surrey Central, BC (Fallback Location)"


def seed_nearby_stores_to_db(collection, user_lat: float, user_lon: float) -> pd.DataFrame:
    """Finds the single closest branch per grocery store brand and updates ChromaDB."""
    all_branches = [
        {"Store": "Walmart Supercentre", "Branch": "Surrey Central", "lat": 49.1882, "lon": -122.8455, "mult": 0.92},
        {"Store": "Walmart Supercentre", "Branch": "Guildford", "lat": 49.1891, "lon": -122.8045, "mult": 0.95},
        {"Store": "Walmart Supercentre", "Branch": "Metrotown", "lat": 49.2276, "lon": -123.0076, "mult": 0.95},
        {"Store": "Real Canadian Superstore", "Branch": "King George", "lat": 49.1935, "lon": -122.8442, "mult": 0.99},
        {"Store": "Real Canadian Superstore", "Branch": "Grandview Hwy", "lat": 49.2582, "lon": -123.0385, "mult": 0.99},
        {"Store": "Save-On-Foods", "Branch": "Whalley Blvd", "lat": 49.1965, "lon": -122.8420, "mult": 1.08},
        {"Store": "Save-On-Foods", "Branch": "Kingsway Burnaby", "lat": 49.2248, "lon": -122.9990, "mult": 1.08},
    ]

    # Calculate true distance from user's postal code to every branch
    for b in all_branches:
        b["Distance_km"] = round(geodesic((user_lat, user_lon), (b["lat"], b["lon"])).km, 2)

    # Sort by distance and keep ONLY ONE (the closest) branch per Store brand
    closest_by_brand = {}
    for b in sorted(all_branches, key=lambda x: x["Distance_km"]):
        if b["Store"] not in closest_by_brand:
            closest_by_brand[b["Store"]] = b

    stores = list(closest_by_brand.values())

    base_catalog = [
        ("Dairyland 2% Milk (4L)", "Milk", 5.89),
        ("Large White Eggs (12 Pack)", "Eggs", 4.49),
        ("Fresh Cavendish Bananas (1 kg)", "Bananas", 1.79),
        ("Boneless Skinless Chicken Breast (1 kg)", "Chicken Breast", 13.99),
        ("Dempster's 100% Whole Wheat Bread", "Bread", 3.49),
        ("Cracker Barrel Cheddar Cheese (400g)", "Cheese", 6.99),
    ]

    # Clear previous Live Store Feed vectors so old distances never linger
    old_live = collection.get(where={"source": "Live Store Feed"})
    if old_live["ids"]:
        collection.delete(ids=old_live["ids"])

    docs, metadatas, ids, rows = [], [], [], []

    for full_name, category, base_price in base_catalog:
        for s in stores:
            price = round(base_price * s["mult"], 2)
            doc_text = f"{full_name} ({category}) at {s['Store']} ({s['Branch']} branch) costs ${price:.2f}. Distance: {s['Distance_km']} km."
            doc_id = f"{s['Store']}_{category}".replace(" ", "_").lower()

            docs.append(doc_text)
            metadatas.append({
                "source": "Live Store Feed",
                "store": s["Store"],
                "branch": s["Branch"],
                "distance_km": s["Distance_km"],
                "category": category,
                "item_name": full_name,
                "price": price,
            })
            ids.append(doc_id)
            rows.append({
                "Store": s["Store"],
                "Branch": s["Branch"],
                "Distance_km": s["Distance_km"],
                "Category": category,
                "Item": full_name,
                "Price": price,
            })

    if ids:
        collection.upsert(documents=docs, metadatas=metadatas, ids=ids)

    return pd.DataFrame(rows)
