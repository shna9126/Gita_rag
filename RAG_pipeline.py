import os
import openai
import streamlit as st
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from groq import Groq

# Load the .env file to get the API keys
load_dotenv()

# Get API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_host = os.getenv("PINECONE_HOST")
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if the API keys are loaded correctly
if not openai_api_key or not pinecone_api_key or not groq_api_key:
    st.error("API keys are missing. Please make sure your .env file contains the correct keys.")
else:
    # Set the OpenAI API key
    openai.api_key = openai_api_key

    # Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)

    # Replace with the actual host URL of your index
    index = pc.Index(host=pinecone_host)

    # Initialize Groq client
    groq_client = Groq(api_key=groq_api_key)

    # Define Streamlit interface
    st.title("Bhagvad Gita GPT created by Shivam with Rapa_learn")
    st.sidebar.header("Search Configuration")

    # Input for query
    query = st.text_input("Enter your search query:", "meditation")

    # Number of top results to retrieve
    top_k = st.sidebar.slider("Number of top results:", 1, 10, 3)

    # Function to generate embedding for a query using OpenAI
    def generate_query_embedding(query):
        try:
            client = openai.OpenAI()
            response = client.embeddings.create(
                model="text-embedding-ada-002", 
                input=query
            )
            embedding = response.data[0].embedding
            return embedding  
        except Exception as e:
            st.error(f"Error generating embedding: {e}")
            return None

    # Function to query Pinecone and retrieve the top results
    def query_pinecone_index(embedding, index, top_k=3):
        """Query the Pinecone index with the provided embedding."""
        try:
            response = index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True  # Ensure metadata is included in the response
            )
            return response
        except Exception as e:
            st.error(f"Error querying Pinecone index: {e}")
            return None

    # Function to generate response using Groq API (Llama-based model)
    def generate_response_groq(context, query):
        """Generate a response using the Groq API with a Llama-based model."""
        try:
            prompt = (
                f"You are an expert in Vedic sciences especially on Ramanuja Gita Bhashya. "
                f"Answer the query in detail and accurately so that anybody can understand.\n\n"
                f"Context:\n{context}\n\n"
                f"Query:\n{query}\n\nAnswer:"
            )
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",  
                messages=[
                    {"role": "system", "content": "You are an expert in the topic based on the context provided."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating response with Groq: {e}")
            return None

    # Display results from Pinecone
    def display_human_readable_results(results):
        """Display the results from Pinecone in a human-readable format."""
        if 'matches' in results:
            st.subheader("Top Matches Found:")
            context = ""
            for i, result in enumerate(results['matches']):
                match_id = result['id']
                match_score = result['score']
                match_metadata = result.get('metadata', {})
                commentary_chunks = match_metadata.get('commentary_chunks', 'No commentary chunks available')

                st.markdown(f"### Match {i+1}")
                st.markdown(f"- **ID:** {match_id}")
                st.markdown(f"- **Score:** {match_score}")
                st.markdown(f"- **Commentary Chunks:** {commentary_chunks}")

                # Combine commentary chunks to form context
                context += f"{commentary_chunks}\n\n"
            return context
        else:
            st.info("No matches found.")
            return None



    # Main action
    if st.button("Search"):
        if query.strip():
            st.info("Generating embedding...")
            embedding = generate_query_embedding(query)

            if embedding:
                st.info("Querying Pinecone index...")
                response = query_pinecone_index(embedding, index, top_k)

                if response:
                    st.info("Generating response using Groq...")
                    context = display_human_readable_results(response)
                    if context:
                        generated_response = generate_response_groq(context, query)
                        if generated_response:
                            st.subheader("Generated Response:")
                            st.write(generated_response)
        else:
            st.warning("Please enter a search query.")