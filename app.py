import gradio as gr


def echo(message, history):
    return message["text"]


demo = gr.ChatInterface(
    fn=echo,
    examples=[],
    title="Thoughtful Chatbot",
    multimodal=True,
)
demo.launch()
