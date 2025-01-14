Project Summary: Building a Retrieval-Augmented Generation (RAG) Model for Bhagavad Gita Commentary

During my internship, I successfully developed a Retrieval-Augmented Generation (RAG) model to deliver personalized and contextualized insights from the Bhagavad Gita commentary authored by Ramanujacharya. This innovative project combined cutting-edge AI technologies, including a large language model (LLM), vector database, and a user-friendly deployment setup. Below are the highlights of the project:

Objectives:
	1.	Enable dynamic querying and retrieval of the Bhagavad Gita commentary for personalized use.
	2.	Enhance response accuracy by combining vector-based search with LLM capabilities.
	3.	Personalize user interactions through prompt engineering by considering user experience, preferences, and desired output format.

Technologies Used:
	1.	Groq API - Llama 3.3 (70B Versatile Model):
	•	Utilized for generating human-like and insightful responses from retrieved context.
	•	Provided the capability to generate detailed or concise answers based on user preferences.
	2.	OpenAI Ada-002 Model:
	•	Leveraged for creating high-dimensional embeddings from user queries.
	•	These embeddings were used to map user queries to the most relevant commentary sections.
	3.	Pinecone Vector Database:
	•	Served as the core retrieval mechanism by indexing commentary chunks.
	•	Enabled fast and accurate retrieval of context based on semantic similarity to the user’s query.
	4.	Streamlit for Deployment:
	•	Built a user-friendly interface to allow users to input their queries and preferences (e.g., experience level, purpose, answer format).
	•	Integrated Streamlit widgets for real-time interactions and feedback.
	5.	Google Cloud Platform (GCP):
	•	Deployed the application on GCP for scalability and accessibility.
	•	Ensured smooth integration with APIs and seamless user experience on the cloud-hosted platform.

Key Features:
	1.	Personalization through Prompt Engineering:
	•	Captured basic user details like experience with the Bhagavad Gita, purpose of query, and preferred response format (concise or detailed).
	•	Dynamically adjusted the LLM prompt to cater to these preferences, improving relevance and user satisfaction.
	2.	RAG Workflow:
	•	Combined vector search with generative capabilities for context-driven answers.
	•	Query embeddings generated by OpenAI Ada-002 were matched against the Pinecone database to fetch the most relevant commentary sections.
	•	Retrieved context was passed to the Groq API (Llama model) to generate detailed, human-readable responses.
	3.	Seamless Deployment and Accessibility:
	•	The Streamlit application was designed to be intuitive, making it accessible to users with diverse levels of familiarity with technology and the Bhagavad Gita.
	•	The app was deployed on GCP to ensure high availability, scalability, and performance.

Outcomes:
	1.	Built a robust, personalized, and scalable system for exploring Ramanujacharya’s Bhagavad Gita commentary.
	2.	Enhanced query-to-response relevance by combining vector-based retrieval and generative AI capabilities.
	3.	Delivered an engaging user experience with real-time interaction and customization options.
	4.	Successfully integrated advanced ML models with modern deployment tools to create a production-ready system.

Link to Deployed Application:https://ramanuja-gita.streamlit.app/

This internship project provided invaluable experience in state-of-the-art AI technologies, deployment strategies, and user-focused design principles. It demonstrated the power of combining retrieval, generation, and personalization to make ancient texts accessible and relevant in today’s world.