import streamlit as st
import os
import sys
import json

sys.path.append(os.path.abspath('../../'))
from task_3 import DocumentProcessor
from task_4 import EmbeddingClient
from task_5 import ChromaCollectionCreator
from task_8 import QuizGenerator
from task_9 import QuizManager

def initialize_session_state():
    if "question_bank" not in st.session_state:
        st.session_state["question_bank"] = []
    if "display_quiz" not in st.session_state:
        st.session_state["display_quiz"] = False
    if "question_index" not in st.session_state:
        st.session_state["question_index"] = 0

if __name__ == "__main__":
                             
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "quizify-428719",
        "location": "europe-west2"
    }
    
    initialize_session_state()
    
    if not st.session_state["question_bank"]:
        screen = st.empty()
        with screen.container():
            st.header("Quiz Builder")
            
            # Create a new st.form flow control for Data Ingestion
            with st.form("Load Data to Chroma"):
                st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
                
                processor = DocumentProcessor()
                processor.ingest_documents()
            
                embed_client = EmbeddingClient(**embed_config) 
            
                chroma_creator = ChromaCollectionCreator(processor, embed_client)
                
                topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
                questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
                    
                submitted = st.form_submit_button("Submit")
                
                if submitted:
                    chroma_creator.create_chroma_collection()
                        
                    if len(processor.pages) > 0:
                        st.write(f"Generating {questions} questions for topic: {topic_input}")
                    
                    generator = QuizGenerator(topic_input, questions, chroma_creator)
                    question_bank = generator.generate_quiz()
                    
                    st.session_state["question_bank"] = question_bank
                    st.session_state["display_quiz"] = True
                    st.session_state["question_index"] = 0

                    st.rerun()

    elif st.session_state["display_quiz"]:
        question_bank = st.session_state["question_bank"]

        screen = st.empty()
        with screen.container():
            st.header("Generated Quiz Question: ")
            quiz_manager = QuizManager(question_bank)
                
            # Format the question and display it
            with st.form("MCQ"):
                index_question = quiz_manager.get_question_at_index(st.session_state["question_index"])

                # Unpack choices for radio button
                choices = [f"{choice['key']}) {choice['value']}" for choice in index_question['choices']]
                    
                # Display the Question
                st.write(f"{st.session_state['question_index'] + 1}. {index_question['question']}")
                answer = st.radio("Choose an answer", choices, index=None)
                    
                answer_choice = st.form_submit_button("Submit")
                
                if answer_choice and answer is not None:
                    correct_answer_key = index_question['answer']
                    if answer.startswith(correct_answer_key):
                        st.success("Correct!")
                    else:
                        st.error("Incorrect!")
                    st.write(f"Explanation: {index_question['explanation']}")
                
                st.form_submit_button("Next Question", on_click=lambda: quiz_manager.next_question_index(1))
                st.form_submit_button("Previous Question", on_click=lambda: quiz_manager.next_question_index(-1))
