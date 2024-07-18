import contextlib
import os
import sys
from google.cloud.aiplatform import base
import streamlit as st
from task_3 import DocumentProcessor
from task_4 import EmbeddingClient
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document  # Import Document class

# Initialize Logger instance
_LOGGER = base.Logger(__name__)

class ChromaCollectionCreator:
    def __init__(self, processor, embed_model):
        self.processor = processor
        self.embed_model = embed_model
        self.db = None

    def create_chroma_collection(self):
        try:
            # Step 1: Check for processed documents
            if len(self.processor.pages) == 0:
                st.error("No documents found!", icon="🚨")
                return

            # Step 2: Split documents into text chunks
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )

            aux_array = list(map(lambda page: page.page_content, self.processor.pages))
            texts = text_splitter.create_documents(aux_array)

            if texts is not None:
                st.success(f"Successfully split pages into {len(texts)} documents!", icon="✅")

            # Step 3: Create the Chroma Collection
            self.db = Chroma.from_documents(texts, self.embed_model.client, persist_directory = os.path.join(os.path.dirname(__file__), '..', 'chroma_db'))


            if self.db:
                st.success("Successfully created Chroma Collection!", icon="✅")
            else:
                st.error("Failed to create Chroma Collection!", icon="🚨")

        except Exception as e:
            _LOGGER.error(f"Error occurred in create_chroma_collection: {str(e)}")
            st.error(f"Error occurred: {str(e)}")

    def as_retriever(self):
        return self.db.as_retriever() if self.db else None

    def query_chroma_collection(self, query) -> Document:
        try:
            if self.db:
                docs = self.db.similarity_search_with_relevance_scores(query)
                if docs:
                    return docs[0]
                else:
                    st.error("No matching documents found!", icon="🚨")
            else:
                st.error("Chroma Collection has not been created!", icon="🚨")
        except Exception as e:
            _LOGGER.error(f"Error occurred in query_chroma_collection: {str(e)}")
            st.error(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    try:
        processor = DocumentProcessor()
        processor.ingest_documents()

        embed_config = {
            "model_name": "textembedding-gecko@003",
            "project": "quizify-428719",
            "location": "europe-west2"
        }

        embed_client = EmbeddingClient(**embed_config)

        chroma_creator = ChromaCollectionCreator(processor, embed_client)

        with st.form("Load Data to Chroma"):
            st.write("Select PDFs for Ingestion, then click Submit")

            submitted = st.form_submit_button("Submit")
            if submitted:
                chroma_creator.create_chroma_collection()
    except Exception as e:
        _LOGGER.error(f"Error in main execution: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
