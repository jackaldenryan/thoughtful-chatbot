import gradio as gr
from openai import OpenAI
import os


def respond(message, history):
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

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
    )

    return chat_completion.choices[0].message.content


demo = gr.ChatInterface(
    fn=respond,
    examples=[],
    title="Thoughtful Chatbot",
    description="A chatbot powered by OpenAI's GPT-4 model.",
)

demo.launch()
