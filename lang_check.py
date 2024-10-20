import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]

# Invoke the LLM and capture the raw response
try:
    ai_msg = llm.invoke(messages)
    print(ai_msg.content)
except AttributeError as e:
    print(f"An error occurred: {e}")
    # Optionally, print the response object or inspect the API response
    response = llm.invoke_raw(messages)  # Assuming `invoke_raw` fetches the full API response
    print(response)  # This allows you to see what the finish_reason value really is
