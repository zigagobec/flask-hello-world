import os
import openai
import time
from dotenv import load_dotenv
from database import herman_retriever
from prompts import system_message, human_template

load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_page_contents(docs):
    contents = ""
    for i, doc in enumerate(docs, 1):
        contents += f"Document #{i}:\n{doc.page_content}\n\n"
    return contents

import time

def get_openai_response(user_input, max_retries=3, retry_delay=10):
    for attempt in range(max_retries):
        try:
            relevant_docs = herman_retriever.get_relevant_documents(user_input)
            context = get_page_contents(relevant_docs)
            query_with_context = human_template.format(query=user_input, context=context)

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query_with_context},
            ]

            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5,
            top_p=0.2,
            frequency_penalty=0,
            presence_penalty=0
)

            return response['choices'][0]['message']['content']

        except openai.error.ServiceUnavailableError:
            if attempt < max_retries - 1:  # -1 because we start counting from 0
                print(f"Error: Service unavailable. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Error: Service unavailable after multiple retries. Please try again later.")
                return "Sorry, the assistant is currently unavailable. Please try again later."


def process_question(user_input):
    # Generate text
    start_time = time.time()
    response_text = get_openai_response(user_input)
    end_time = time.time()
    print("Assistant: " + response_text)
    print(f"Total time taken: {end_time - start_time} seconds")
    return response_text  # we are only returning the text response.

def run_console():
    while True:
        user_input = input("User: ")
        process_question(user_input)

if __name__ == "__main__":
    run_console()