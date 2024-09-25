# Thoughtful AI Chatbot

A chatbot powered by OpenAI's GPT-4o model to answer pre-defined questions about Thoughtful AI, built with Gradio and Pydantic.

## How it works

1. The chatbot first checks if the user's input matches any pre-defined questions, using a GPT-4o "RAG" prompt.
2. If a match is found, it returns the corresponding pre-defined answer.
3. If no match is found, it uses prompts GPT-4o again to generate a response.

## Setup

1. Install dependencies:
   ```
   pip install gradio openai pydantic
   ```

2. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the script to launch the Gradio interface:

```
python app.py
```

The chatbot will be accessible through a web interface.

