import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_enterprise_dataset(total_rows=1000000):
    print(f"🚀 Initializing Enterprise Narrative Engine (2023-2026)...")
    
    # 1. THE MEGA-CATALOG
    BRANDS_CATALOG = {
        "Apple": ["iPhone 14", "iPhone 15 Pro", "iPhone 16 Air", "iPhone 17 Pro Max"],
        "Samsung": ["Galaxy S23", "Galaxy S24 Ultra", "Galaxy S25 Ultra", "Galaxy Z Fold 7", "Galaxy A56 5G"],
        "Google": ["Pixel 8a", "Pixel 9 Pro XL", "Pixel 10 Pro Fold", "Pixel 11"],
        "OnePlus": ["OnePlus 12", "OnePlus 13", "OnePlus Open 2", "OnePlus Nord 6"],
        "Xiaomi": ["Xiaomi 14 Ultra", "Xiaomi 15 Pro", "Redmi Note 15 Pro+", "Poco F7 Pro"],
        "Motorola": ["Razr Plus 2025", "Edge 60 Ultra", "Moto G85 5G"],
        "Nothing": ["Phone (2)", "Phone (3a) Pro", "CMF Phone 2"],
        "Vivo": ["Vivo X200 Pro", "Vivo V60 Pro"],
        "Oppo": ["Oppo Find X9 Pro", "Oppo Reno 15 Pro"],
        "Realme": ["Realme GT 7 Pro", "Realme 16 Pro Plus"],
        "Sony": ["Xperia 1 VI", "Xperia 10 VI"],
        "Asus": ["ROG Phone 9 Pro", "Zenfone 11 Ultra"],
        "Honor": ["Honor Magic 7 Pro", "Honor 500 Pro"],
        "Infinix": ["Zero 40 Ultra", "GT 20 Pro"],
        "Huawei": ["Mate 70 Pro", "Pura 70 Ultra"]
    }

    VARIANTS = ["128GB", "256GB", "512GB", "1TB"]
    COLORS = ["Titanium", "Obsidian", "Snow", "Astral Blue", "Cosmic Orange"]

    # 2. THE STORYBOARD ENGINE: (Model Key, Variant, Year, Month) -> Story Data
    # Probability Spike: 0.1 to 0.3 for specific narrative events
    STORIES = [
        ("iPhone 17", "128GB", 2026, 1, "Software - AI Storage Full", "Negative", 0.25),
        ("S25 Ultra", "1TB", 2025, 3, "Hardware - Camera Sensor Bug", "Negative", 0.15),
        ("Pixel 10", "512GB", 2025, 11, "Software - Thermal Throttling", "Negative", 0.18),
        ("Z Fold 7", "512GB", 2025, 8, "Hardware - Hinge Fatigue", "Negative", 0.20),
        ("Xiaomi 15", "256GB", 2026, 1, "Software - Bloatware Crash", "Negative", 0.12),
        ("Razr Plus", "256GB", 2025, 5, "Hardware - Display Crease", "Negative", 0.22),
        ("ROG Phone 9", "1TB", 2025, 12, "Hardware - Cooling Fan Failure", "Negative", 0.15),
        ("iPhone 14", "128GB", 2024, 6, "Buyer Remorse - Outdated Tech", "Neutral", 0.08)
    ]

    # 3. VECTORIZED PRE-CALCULATION
    sku_list = []
    for brand, models in BRANDS_CATALOG.items():
        for model in models:
            for var in VARIANTS:
                base_price = 1199 if any(x in model for x in ["Ultra", "Pro Max", "Fold", "Pro XL"]) else 599
                price = base_price + (VARIANTS.index(var) * 140) + random.randint(-30, 30)
                sku_list.append({
                    "brand": brand, "model_base": model, "full_name": f"{brand} {model} ({var}, {random.choice(COLORS)})",
                    "asin": f"B0{random.randint(10,99)}{brand[0]}{random.randint(100,999)}X",
                    "price": price, "market_seg": "Flagship" if price > 900 else "Mid-Tier"
                })

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2026, 1, 11)
    date_range = (end_date - start_date).days

    data_rows = []
    chunk_size = 100000
    for i in range(0, total_rows, chunk_size):
        current_chunk = min(chunk_size, total_rows - i)
        print(f"   - Processing chunk {i//chunk_size + 1}...")

        for _ in range(current_chunk):
            sku = random.choice(sku_list)
            dt = start_date + timedelta(days=random.randint(0, date_range))
            
            is_return = False
            reason, theme, sentiment, action = ("", "N/A", "Positive", "Monitor")
            
            # Narrative Matching
            story = next((s for s in STORIES if s[0] in sku['model_base'] and s[1] == sku['full_name'].split('(')[1].split(',')[0] and s[2] == dt.year and s[3] == dt.month), None)
            
            return_prob = 0.05 # 5% Baseline
            if story: return_prob += story[6]
            
            if random.random() < return_prob:
                is_return = True
                if story:
                    reason, theme, sentiment = story[4], story[4].split(' - ')[0], story[5]
                    action = "Engineering Recall" if theme == "Hardware" else "Urgent OTA Update"
                else:
                    reason = random.choice(["Changed Mind", "Found Better Price", "Logistics - Damaged Box"])
                    theme = "Buyer Remorse" if "Price" in reason else "Logistics"
                    sentiment = "Neutral"
                    action = "Issue Credit"

            data_rows.append({
                "subcategory_code": "501010", "manufacturer_name": sku['brand'],
                "product_type": "CELLULAR_PHONE", "asin": sku['asin'],
                "year": dt.year, "month": dt.month,
                "asp_band": "High" if sku['price'] > 850 else "Standard",
                "customer_id": f"CID-{random.randint(100000, 999999)}",
                "concession_reason": reason if is_return else "",
                "ship_day": dt.strftime('%Y-%m-%d 00:00:00'),
                "concession_creation_day": (dt + timedelta(days=random.randint(1, 14))).strftime('%Y-%m-%d %H:%M:%S') if is_return else "",
                "our_price": sku['price'], "our_price_discount_amt": random.choice([0, 0, 0, 50, 150]),
                "shipped_units": 1, "total_units_conceded": 1.0 if is_return else 0.0,
                "ProductName": sku['full_name'], "Sentiment": sentiment, "ReturnTheme": theme, "SuggestedAction": action
            })

    pd.DataFrame(data_rows).to_csv('data/enriched_data.csv', index=False)
    print(f"✅ Enterprise Dataset Complete: 1,000,000+ rows generated.")

if __name__ == "__main__":
    generate_enterprise_dataset(1000000)