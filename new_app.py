from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import PromptTemplate
from huggingface_hub import InferenceClient
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
HUGGINGFACE_API_TOKEN = os.environ.get('HUGGINGFACE_API_TOKEN')

# Download embeddings (Ensure this works correctly)
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-bot"

# Check if index exists before creating
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Set dimension based on your embedding model
        metric="cosine",  # Cosine similarity for sentence-transformers
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Loading the existing index for search
docsearch = LangChainPinecone.from_existing_index(index_name, embeddings)

# Setting up the prompt
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Hugging Face Inference API client
hf_client = InferenceClient(model="meta-llama/Llama-2-7b-chat-hf", token=HUGGINGFACE_API_TOKEN)

# Custom Hugging Face LLM Wrapper implementing the LLM interface
class HuggingFaceLLM(LLM):
    def _call(self, prompt: str, stop: list = None) -> str:
        """Method to generate text from the Hugging Face model using the Inference API."""
        response = hf_client.text_generation(prompt, max_length=512)
        return response['generated_text']

    def _identifying_params(self) -> dict:
        """This method can be used to return model-specific parameters."""
        return {"model": "meta-llama/Llama-2-7b-chat-hf"}

    @property
    def _llm_type(self) -> str:
        """This defines the type of the LLM."""
        return "huggingface_hub"

# Instantiate the custom HuggingFaceLLM
llm = HuggingFaceLLM()

# Set up the RetrievalQA chain with the custom Hugging Face LLM
qa = RetrievalQA.from_chain_type(
    llm=llm,  # Using the custom Hugging Face LLM
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# Default route to serve the chat page
@app.route("/")
def index():
    return render_template('chat.html')

# Route to handle chat input
@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form["msg"]
        print(f"User input: {msg}")
        
        # Use the __call__ method instead of invoke
        result = qa({"query": msg})
        
        # Check if result contains the "result" key
        if "result" in result:
            response_text = result["result"]
        else:
            response_text = "No response generated."

        # Log the result
        print("Response: ", response_text)
        
        # Return the result as a string
        return jsonify({"response": response_text})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Something went wrong!"}), 500
