from openai import OpenAI

# --- PASTE YOUR PERSONAL API KEY DIRECTLY HERE FOR THIS TEST ---
# Use the key from your Novita.ai account dashboard, NOT the "session_" key
API_KEY = "sk_xDJjFgwGW7hokja4dBBU4MZA113qvX8t8eDuN4Ujnww"
# ----------------------------------------------------------------

client = OpenAI(
    base_url="https://api.novita.ai/v3/openai",
    api_key=API_KEY,
)

print("Running final, direct API key test...")

try:
    chat_completion_res = client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        messages=[
            {
                "role": "user",
                "content": "Hi there!",
            }
        ],
        stream=False,
    )
    print("\nSUCCESS! The API key worked in this direct test.")

except Exception as e:
    print("\nTEST FAILED. The 401 error is confirmed to be an issue with the API key or account.")
    print("Error:", e)