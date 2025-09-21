# test_openai.py
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load the environment variables from your .env file
load_dotenv()

# The client will automatically find the OPENAI_API_KEY in your .env file
client = OpenAI()

try:
    print("--- 🤖 Sending a direct request to the OpenAI API... ---")
    
    # This is the official method to call the OpenAI API directly
    response = client.chat.completions.create(
      model="gpt-5", 
      messages=[
        {"role": "user", "content": "Write a one-sentence bedtime story about a brave little robot."}
      ]
    )

    # This is the correct way to access the content from a direct API call
    story = response.choices[0].message.content
    print("\n--- ✨ Story Generated ---")
    print(story)

except Exception as e:
    print(f"\nAn error occurred: {e}")