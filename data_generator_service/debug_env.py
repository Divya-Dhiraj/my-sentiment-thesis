import os
from dotenv import load_dotenv

# --- THIS IS THE FIX ---
# Build an explicit path to the .env file in this script's directory
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)
# --------------------

# Paste the correct key from your Novita.ai dashboard here
hardcoded_key = "sk_xDJjFgwGW7hokja4dBBU4MZA113qvX8t8eDuN4Ujnww"

# Load the key from the .env file
key_from_env = os.environ.get("NOVITA_API_KEY")

print("--- Environment Variable Debugger ---")

if key_from_env:
    print(f"Key from .env file (length {len(key_from_env)}): '{key_from_env}'")
    print(f"Hardcoded key      (length {len(hardcoded_key)}): '{hardcoded_key}'")
    print("\nComparing the two keys...")

    if key_from_env == hardcoded_key:
        print("\n✅ SUCCESS: The key in the .env file is correct.")
    else:
        print("\n❌ FAILED: The key in the .env file DOES NOT MATCH the hardcoded key.")
else:
    print("❌ FAILED: Could not load NOVITA_API_KEY from the .env file at all.")

print("-------------------------------------")