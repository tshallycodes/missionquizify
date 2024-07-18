from langchain_google_vertexai import VertexAIEmbeddings
import streamlit as st


class EmbeddingClient:
    """
    Task: Initialize the EmbeddingClient class to connect to Google Cloud's VertexAI for text embeddings.

    The EmbeddingClient class should be capable of initializing an embedding client with specific configurations
    for model name, project, and location. Your task is to implement the __init__ method based on the provided
    parameters. This setup will allow the class to utilize Google Cloud's VertexAIEmbeddings for processing text queries.

    Instructions:
    - Carefully initialize the 'self.client' with VertexAIEmbeddings in the __init__ method using the parameters.
    - Pay attention to how each parameter is used to configure the embedding client.
    """

    def __init__(self, model_name, project, location):
        # Initialize the VertexAIEmbeddings client with the provided parameters
        self.client = VertexAIEmbeddings(
            model_name = "textembedding-gecko@003", 
            project = "quizify-428719",
            location = "europe-west2"
        )

    def embed_query(self, query):
        """
        Uses the embedding client to retrieve embeddings for the given query.

        :param query: The text query to embed.
        :return: The embeddings for the query or None if the operation fails.
        """
        try:
            vectors = self.client.embed_query(query)
            return vectors
        except Exception as e:
            print(f"Error embedding query: {e}")
            return None

    def embed_documents(self, documents):
        """
        Retrieve embeddings for multiple documents.

        :param documents: A list of text documents to embed.
        :return: A list of embeddings for the given documents.
        """
        try:
            return self.client.embed_documents(documents)
        except AttributeError:
            print("Method embed_documents not defined for the client.")
            return None

if __name__ == "__main__":
    model_name = "textembedding-gecko@003",
    project = "quizify-428719",
    location = "europe-west2"

    embedding_client = EmbeddingClient(model_name, project, location)
    vectors = embedding_client.embed_query("Hello World!")
    if vectors:
        print(vectors)
        st.write(vectors)
        print("Successfully used the embedding client!")
