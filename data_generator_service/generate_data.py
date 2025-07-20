import os
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta
import numpy as np
import json
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import time
import re

# --- INITIALIZE FAKER ---
fake = Faker()

# --- CONFIGURATION & NARRATIVE ---
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

PRODUCT = {
    'asin': 'B0CQ44392F', 'product_name': 'Apple iPhone 16 Pro',
    'brand': 'Apple', 'product_type': 'Third-Party', 'msrp': 1299.00
}
START_DATE = datetime(2024, 9, 15)
END_DATE = START_DATE + timedelta(days=365)
NUM_CUSTOMERS = 1500
AVG_REVIEWS_PER_WEEK = 85
LLM_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"

EVENT_TIMELINE = {
    "launch_hype": (datetime(2024, 9, 15), datetime(2024, 10, 19)),
    "flaw_discovery": (datetime(2024, 10, 20), datetime(2024, 12, 15)),
    "holiday_sale": (datetime(2024, 11, 25), datetime(2024, 12, 26)),
    "post_holiday_lull": (datetime(2024, 12, 27), datetime(2025, 1, 9)),
    "software_fix": (datetime(2025, 1, 10), datetime(2025, 2, 28)),
    "competitor_launch": (datetime(2025, 3, 1), datetime(2025, 4, 15)),
    "steady_mid_life": (datetime(2025, 4, 16), datetime(2025, 7, 3)),
    "summer_promo": (datetime(2025, 7, 4), datetime(2025, 7, 20)),
    "end_of_life_decline": (datetime(2025, 7, 21), END_DATE)
}

# --- LLM SETUP ---
try:
    client = OpenAI(
        api_key=os.environ.get("NOVITA_API_KEY"),
        base_url=os.environ.get("NOVITA_API_BASE_URL")
    )
    print(f"LLM Client configured for Novita.ai with model: {LLM_MODEL}")
except Exception as e:
    print(f"Error initializing client: {e}")
    exit()

def generate_reviews_for_week_llm(product_name, num_reviews, rating_distribution, scenario):
    """Generates a realistic mix of reviews based on a main scenario."""
    # This new prompt instructs the LLM to create variety within the batch.
    prompt = f"""
    You are a data generator creating a batch of realistic product reviews for the "{product_name}".
    Your task is to generate a list of exactly {num_reviews} unique reviews.

    The main scenario or event for this week is: **{scenario}**.

    To make the data realistic, please generate a mix of reviews:
    - **Majority of reviews (approx. 70%):** These should focus directly on the main scenario.
    - **Some general reviews (approx. 20%):** These should be more generic, talking about other features like the camera, design, or overall performance, and should NOT mention the main scenario.
    - **A few outlier/contradictory reviews (approx. 10%):** These should be unusual. For example, a review that contradicts the main scenario (e.g., praises a feature that is known to be flawed) or complains about a completely different, niche problem.

    The overall rating distribution for the batch should roughly match: {rating_distribution}.
    Ensure variety in phrasing, tone, and length.

    Return ONLY a single, valid JSON object with one key: "reviews".
    The value for "reviews" should be a LIST of JSON objects. Each object must have "rating", "title", and "text" keys.
    """
    
    # The rest of the function remains the same.
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=1.0, # Increased temperature slightly for more creativity
                timeout=120.0,
            )
            llm_output_str = response.choices[0].message.content
            
            json_start, json_end = llm_output_str.find('{'), llm_output_str.rfind('}')
            if json_start != -1 and json_end != -1:
                json_str = llm_output_str[json_start:json_end+1]
                corrected_json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
                review_data = json.loads(corrected_json_str)
                return review_data.get('reviews', [])
            else:
                raise ValueError("No JSON object found in LLM response.")

        except Exception as e:
            print(f"\nAttempt {attempt + 1} of {MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("All retries failed. Using fallback template for this batch.")
    
    # Fallback logic
    fallback_reviews = []
    for _ in range(num_reviews):
        rating = random.choices(list(rating_distribution.keys()), weights=list(rating_distribution.values()), k=1)[0]
        title = "Good Product" if rating >= 4 else "Could Be Better"
        text = "I am very happy with this product. It works as expected and I would recommend it."
        if rating < 4:
            text = "I was not satisfied with this purchase. It had some issues and did not meet my expectations."
        fallback_reviews.append({"rating": rating, "title": title, "text": text})
    return fallback_reviews

# --- MAIN SCRIPT with Sub-Batching for Higher Reliability ---
print(f"Starting batched 1-year simulation for: {PRODUCT['product_name']}")

output_dir = "final_generated_data"
if not os.path.exists(output_dir): os.makedirs(output_dir)
temp_review_file = f"{output_dir}/temp_reviews.csv"

reviews_list = []
start_week_index = 0
if os.path.exists(temp_review_file):
    print("--- Resuming from previously saved progress ---")
    resumed_df = pd.read_csv(temp_review_file)
    reviews_list = resumed_df.to_dict('records')
    last_date_saved = pd.to_datetime(resumed_df['review_date'].max())
    all_weeks_full = pd.to_datetime(pd.date_range(start=START_DATE, end=END_DATE, freq='W-MON')).tolist()
    for i, week_start in enumerate(all_weeks_full):
        if week_start > last_date_saved:
            start_week_index = i
            break
    print(f"Resuming simulation from week {start_week_index + 1}...")

customers_df = pd.DataFrame([{'customer_id': f"CUS_{fake.uuid4()[:8].upper()}"} for _ in range(NUM_CUSTOMERS)])
customer_ids = customers_df['customer_id'].tolist()
all_weeks = pd.to_datetime(pd.date_range(start=START_DATE, end=END_DATE, freq='W-MON')).tolist()

for i in tqdm(range(start_week_index, len(all_weeks)), desc="Simulating Year (Weekly Batches)", initial=start_week_index, total=len(all_weeks)):
    week_start = all_weeks[i]
    num_reviews_this_week = random.randint(AVG_REVIEWS_PER_WEEK - 20, AVG_REVIEWS_PER_WEEK + 20)
    
    scenario, rating_dist = "General feedback", {5: 0.7, 4: 0.2, 3: 0.1}
    if EVENT_TIMELINE["launch_hype"][0] <= week_start <= EVENT_TIMELINE["launch_hype"][1]:
        scenario, rating_dist = "User is an early adopter, thrilled with the new phone, especially camera and performance.", {5: 0.8, 4: 0.2}
    elif EVENT_TIMELINE["flaw_discovery"][0] <= week_start <= EVENT_TIMELINE["flaw_discovery"][1]:
        scenario, rating_dist = "User is angry and disappointed about a severe battery drain issue.", {1: 0.5, 2: 0.3, 5: 0.1, 4: 0.1}
    elif EVENT_TIMELINE["post_holiday_lull"][0] <= week_start <= EVENT_TIMELINE["post_holiday_lull"][1]:
        scenario, rating_dist = "User is waiting for a fix for the known battery issue.", {2: 0.4, 3: 0.4, 4: 0.2}
    elif EVENT_TIMELINE["software_fix"][0] <= week_start <= EVENT_TIMELINE["software_fix"][1]:
        scenario, rating_dist = "User is very happy and relieved because a software update fixed the battery drain.", {5: 0.9, 4: 0.1}
    elif EVENT_TIMELINE["competitor_launch"][0] <= week_start <= EVENT_TIMELINE["competitor_launch"][1]:
        scenario, rating_dist = "User is reviewing the phone, possibly comparing it to a new competitor that just launched.", {5: 0.6, 4: 0.3, 3: 0.1}
    elif EVENT_TIMELINE["steady_mid_life"][0] <= week_start <= EVENT_TIMELINE["steady_mid_life"][1]:
         scenario, rating_dist = "User is providing a general review of the phone now that it's been on the market for a while.", {5: 0.7, 4: 0.2, 3: 0.1}
    elif EVENT_TIMELINE["end_of_life_decline"][0] <= week_start <= EVENT_TIMELINE["end_of_life_decline"][1]:
        scenario, rating_dist = "User is reviewing the phone late in its lifecycle, mentioning the upcoming model.", {4: 0.5, 5: 0.3, 3: 0.2}
    if EVENT_TIMELINE["holiday_sale"][0] <= week_start <= EVENT_TIMELINE["holiday_sale"][1]:
        scenario += " They are also happy they got a great deal during the holiday sale."
    elif EVENT_TIMELINE["summer_promo"][0] <= week_start <= EVENT_TIMELINE["summer_promo"][1]:
         scenario, rating_dist = "User bought the phone during a summer sale and is happy with the value for money.", {5: 0.7, 4: 0.3}

    SUB_BATCH_SIZE = 20
    reviews_generated_this_week = 0
    while reviews_generated_this_week < num_reviews_this_week:
        reviews_to_generate_now = min(SUB_BATCH_SIZE, num_reviews_this_week - reviews_generated_this_week)
        if reviews_to_generate_now <= 0: break
        
        generated_reviews = generate_reviews_for_week_llm(PRODUCT['product_name'], reviews_to_generate_now, rating_dist, scenario)
        
        for review in generated_reviews:
            reviews_list.append({
                'review_id': f"R_{fake.uuid4()[:10].upper()}", 'asin': PRODUCT['asin'], 'customer_id': random.choice(customer_ids),
                'review_date': fake.date_time_between(start_date=week_start, end_date=week_start + timedelta(days=6)),
                'rating': review.get('rating'), 'review_title': review.get('title'), 'review_text': review.get('text')
            })
        reviews_generated_this_week += len(generated_reviews)
    
    time.sleep(1)

    if (i + 1) % 5 == 0:
        pd.DataFrame(reviews_list).to_csv(temp_review_file, index=False)
        print(f"\n--- Progress saved at week {i + 1}/{len(all_weeks)} ---")

# --- FINAL PROCESSING AND SAVING ---
reviews_df = pd.DataFrame(reviews_list)
reviews_df['review_date'] = pd.to_datetime(reviews_df['review_date'])
print(f"\nGenerated a total of {len(reviews_df)} reviews.")

print("Aggregating reviews and generating correlated sales data...")
weekly_data, sentiment_mapping = [], {5: 0.9, 4: 0.6, 3: 0.0, 2: -0.6, 1: -0.9}
for week_start in all_weeks:
    weekly_reviews = reviews_df[(reviews_df['review_date'] >= week_start) & (reviews_df['review_date'] <= week_start + timedelta(days=6))]
    weekly_data.append({
        'asin': PRODUCT['asin'], 'week_start_date': week_start, 'num_reviews_received': len(weekly_reviews),
        'average_rating_new': round(weekly_reviews['rating'].mean(), 2) if not weekly_reviews.empty else None,
        'sentiment_score_new': round(weekly_reviews['rating'].apply(lambda r: sentiment_mapping.get(r, 0)).mean(), 2) if not weekly_reviews.empty else None
    })
weekly_perf_df = pd.DataFrame(weekly_data)
base_sales_per_week = 2500
weekly_perf_df.set_index('week_start_date', inplace=True)
for week_start in weekly_perf_df.index:
    sales_factor, sentiment_factor, discount = 1.0, 1.0, 0.0
    time_since_launch = (week_start - START_DATE).days
    sales_factor *= max(0.4, 1 - (time_since_launch / (365 * 1.5)))
    if EVENT_TIMELINE["launch_hype"][0] <= week_start <= EVENT_TIMELINE["launch_hype"][1]: sales_factor *= 1.8
    if EVENT_TIMELINE["competitor_launch"][0] <= week_start <= EVENT_TIMELINE["competitor_launch"][1]: sales_factor *= 0.75
    prev_week_start = week_start - timedelta(days=7)
    if prev_week_start in weekly_perf_df.index:
        prev_sentiment = weekly_perf_df.loc[prev_week_start, 'sentiment_score_new']
        if pd.notna(prev_sentiment): sentiment_factor = 1 + (prev_sentiment * 0.7)
    if EVENT_TIMELINE["holiday_sale"][0] <= week_start <= EVENT_TIMELINE["holiday_sale"][1]: discount = 0.15
    if EVENT_TIMELINE["summer_promo"][0] <= week_start <= EVENT_TIMELINE["summer_promo"][1]: discount = 0.10
    discount_factor = 1 + (discount * 2.5)
    units_sold = int(base_sales_per_week * sales_factor * sentiment_factor * discount_factor * random.uniform(0.9, 1.1))
    weekly_perf_df.loc[week_start, 'discount_percentage'] = discount * 100
    weekly_perf_df.loc[week_start, 'total_units_sold'] = max(100, units_sold)
    weekly_perf_df.loc[week_start, 'average_selling_price'] = round(PRODUCT['msrp'] * (1 - discount), 2)
weekly_perf_df.reset_index(inplace=True)
weekly_perf_df['product_name'], weekly_perf_df['product_type'] = PRODUCT['product_name'], PRODUCT['product_type']

final_cols = ['asin', 'product_name', 'product_type', 'week_start_date', 'average_selling_price', 'discount_percentage', 'total_units_sold', 'num_reviews_received', 'average_rating_new', 'sentiment_score_new']
weekly_perf_df[final_cols].to_csv(f"{output_dir}/weekly_product_performance.csv", index=False)
reviews_df.to_csv(f"{output_dir}/reviews.csv", index=False)

if os.path.exists(temp_review_file):
    os.remove(temp_review_file)

print(f"\n✅ Successfully generated flawless datasets in the '{output_dir}' directory.")