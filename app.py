import gradio as gr
from openai import OpenAI
import os
import json
from pydantic import BaseModel, ValidationError
from typing import List, Tuple, Optional


class IndexResponse(BaseModel):
    # For using structured outputs
    index: int


class RespondInput(BaseModel):
    # For validating inputs
    message: str
    history: List[Tuple[str, str]]


def llm_response(message, history, output_structure=None):
    # Prepare the conversation history for the API call
    messages = [
        {"role": "system", "content": "You are a thoughtful and helpful assistant."}
    ]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    if output_structure is None:

        chat_completion = client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini",
        )
    else:
        chat_completion = client.beta.chat.completions.parse(
            messages=messages,
            model="gpt-4o-mini",
            response_format=output_structure,
        )

    return chat_completion.choices[0].message.content


def get_closest_question_idx(message, predef_questions):

    q_strs = [f"Index {i}: {q}" for i, q in enumerate(predef_questions)]
    qs_str = "\n".join(q_strs)

    rag_message = f"""Which, if any, of the below pre-defined questions is equivalent or highly similar to the user's question?
    Return the integer index value, such as 0, 1, 2, etc., of the pre-defined question which is equal/highly similar to the user question, or return -1 if none of the pre-defined questions are equal/highly similar to the user question.

    User question: {message}
    Pre-defined questions:
    {qs_str}
    """
    rag_response = llm_response(
        rag_message,
        [],
        output_structure=IndexResponse,
    )

    rag_response_json = json.loads(rag_response)
    best_index = rag_response_json["index"]

    return best_index


def respond(message, history):
    try:
        # Validate inputs using Pydantic
        input_data = RespondInput(message=message, history=history)
    except ValidationError as e:
        return f"Input validation error: {e}"

    # Extract validated data
    message = input_data.message
    history = input_data.history

    # First create a list of the pre-recorded questions from the responses.json file
    predef_questions = []
    predef_answers = []
    with open("responses.json", "r") as f:
        data = json.load(f)
        for item in data["questions"]:
            predef_questions.append(item["question"])
            predef_answers.append(item["answer"])

    best_index = get_closest_question_idx(message, predef_questions)

    if best_index == -1:
        return llm_response(message, history)

    else:
        return predef_answers[best_index]


demo = gr.ChatInterface(
    fn=respond,
    examples=[],
    title="Thoughtful Chatbot",
    description="A chatbot powered by OpenAI's GPT-4o model.",
)

demo.launch()
