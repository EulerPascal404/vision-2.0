from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import base64, httpx
from dotenv import load_dotenv
import os
import getpass

# Load environment variables
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Define the Pydantic model to parse the model's output
class Fruit(BaseModel):
    name: str = Field(description="The name of the fruit shown in the image")
    color: str = Field(description="The color of the fruit shown in the image")
    taste: str = Field(description="The taste of the fruit shown in the image")
    marketing_description: str = Field(description="A marketing description of the fruit shown in the image")

# Create the parser
parser = PydanticOutputParser(pydantic_object=Fruit)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Return the requested response object in {language}.\n'{format_instructions}'\n"),
    ("human", [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
        },
    ]),
])

# Retrieve the encoded image data for an image
image_data = base64.b64encode(httpx.get('https://storage.googleapis.com/vectrix-public/fruit/peach.jpeg').content).decode("utf-8")

# Run the chain and print the result
chain = prompt | model | parser

# Generate format instructions using the Pydantic v2 `model_json_schema()`
format_instructions = parser.pydantic_object.model_json_schema()

print(chain.invoke({
    "language": "English",
    "format_instructions": format_instructions,
    "image_data": image_data
}).json(indent=2))
