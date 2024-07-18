import streamlit as st
import os
import sys
import logging
from task_3 import DocumentProcessor
from task_4 import EmbeddingClient
from task_5 import ChromaCollectionCreator
from task_8 import QuizGenerator

# Ensure the correct path is set for imports
sys.path.append(os.path.abspath('../../'))

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# QuizManager class definition
class QuizManager:
    def __init__(self, questions: list):
        self.questions = questions
        self.total_questions = len(questions)

    def get_question_at_index(self, index: int):
        valid_index = index % self.total_questions
        return self.questions[valid_index]

    def handle_submission_and_next_question(self, submitted_answer: str):
        try:
            question = self.get_question_at_index(st.session_state["question_index"])
            correct_answer_key = question['answer']
            
            # Debugging: log the submitted answer and correct answer
            logger.debug(f"Submitted answer: {submitted_answer}")
            logger.debug(f"Correct answer key: {correct_answer_key}")

            if submitted_answer.startswith(correct_answer_key):
                st.session_state["is_correct"] = True
            else:
                st.session_state["is_correct"] = False
            
            st.session_state["show_result"] = True
        except Exception as e:
            logger.error(f"Error in handle_submission_and_next_question: {e}")
            st.error("An error occurred during submission. Please try again.")

    def next_question_index(self, direction=1):
        if "question_index" not in st.session_state:
            st.session_state["question_index"] = 0
        
        current_index = st.session_state["question_index"]
        new_index = (current_index + direction) % self.total_questions
        st.session_state["question_index"] = new_index

# Initialize session state variables
if "question_index" not in st.session_state:
    st.session_state["question_index"] = 0
if "show_result" not in st.session_state:
    st.session_state["show_result"] = False
if "is_correct" not in st.session_state:
    st.session_state["is_correct"] = None
if "question_bank" not in st.session_state:
    st.session_state["question_bank"] = None
if "form_submitted" not in st.session_state:
    st.session_state["form_submitted"] = False

# Test Generating the Quiz
if __name__ == "__main__":
    try:
        embed_config = {
            "model_name": "textembedding-gecko@003",
            "project": "quizify-428719",
            "location": "europe-west2"
        }
        
        screen = st.empty()
        with screen.container():
            st.header("Quiz Builder")
            processor = DocumentProcessor()
            processor.ingest_documents()
        
            embed_client = EmbeddingClient(**embed_config) 
        
            chroma_creator = ChromaCollectionCreator(processor, embed_client)
        
            with st.form("Load Data to Chroma"):
                st.subheader("Quiz Builder")
                st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
                
                topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
                questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
                
                submitted = st.form_submit_button("Submit")
                if submitted:
                    try:
                        logger.debug("Starting Chroma Collection creation.")
                        chroma_creator.create_chroma_collection()
                        logger.debug("Chroma Collection created successfully.")
                        
                        st.write(topic_input)
                        
                        # Test the Quiz Generator
                        logger.debug("Starting quiz generation.")
                        generator = QuizGenerator(topic_input, questions, chroma_creator)
                        st.session_state["question_bank"] = generator.generate_quiz()
                        st.session_state["form_submitted"] = True  # Set a flag to indicate form submission
                        logger.debug("Quiz generated successfully.")
                    except Exception as e:
                        logger.error(f"Error generating quiz: {e}")
                        st.error("An error occurred while generating the quiz. Please try again.")

    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        st.error("An error occurred during initialization. Please try again.")

# Check if question bank is generated and form is submitted
if st.session_state["question_bank"] and st.session_state["form_submitted"]:
    # Clear the flag after processing the form submission
    st.session_state["form_submitted"] = False
    
    screen.empty()
    with st.container():
        st.header("Generated Quiz Question: ")
        
        quiz_manager = QuizManager(st.session_state["question_bank"])
        
        index_question = quiz_manager.get_question_at_index(st.session_state["question_index"])
        
        # Debugging: log the index_question
        logger.debug(f"Retrieved index_question: {index_question}")
        
        with st.form("Multiple Choice Question"):
            question_text = index_question["question"]
            st.write(question_text)
            
            choices = []
            for choice in index_question['choices']:
                key = choice['key']
                value = choice['value']
                choices.append(f"{key}) {value}")
            
            # Debugging: log the constructed choices
            logger.debug(f"Constructed choices: {choices}")
            
            answer = st.radio('Choose the correct answer', choices)
            answer_submitted = st.form_submit_button("Submit Answer")
            
            if answer_submitted:
                submitted_answer = answer.split(') ')[0]  # Extract the key from the selected answer
                quiz_manager.handle_submission_and_next_question(submitted_answer)

        if st.session_state["show_result"]:
            if st.session_state["is_correct"]:
                st.success("Correct!")
            else:
                st.error("Incorrect!")

        next_button = st.button("Next Question")
        prev_button = st.button("Previous Question")
        
        if next_button:
            st.session_state["show_result"] = False
            quiz_manager.next_question_index(1)
            st.rerun()
        
        if prev_button:
            st.session_state["show_result"] = False
            quiz_manager.next_question_index(-1)
            st.rerun()
