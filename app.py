import gradio as gr
from dotenv import load_dotenv
from implementation.answer import answer_question

load_dotenv(override=True)


def format_context(docs):
    """Format retrieved context for display."""
    result = ""
    for doc in docs:
        source = doc.metadata.get("filename", "Unknown")
        doc_type = doc.metadata.get("doc_type", "General")
        result += f"📄 Source: {source} ({doc_type})\n\n{doc.page_content}\n\n{'-'*50}\n\n"
    return result


def chat(message, history):
    """Handle user input and return assistant response."""
    history = history or []

    answer, docs = answer_question(message, history)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return "", history, format_context(docs)


def main():
    with gr.Blocks(
        title="Fitness & Health Assistant",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .chatbot-container {
            border-radius: 12px;
            border: 1px solid #e0e0e0;
        }
        """
    ) as ui:
        gr.Markdown("""
        # 💪 Fitness & Health Assistant

        Your personal guide to fitness, nutrition, sleep, and wellness. 
        Ask me anything about:
        - 🏋️ Workout routines and exercise
        - 🥗 Nutrition and meal planning
        - 😴 Sleep optimization
        - 🧘 Stress management
        - 💡 General health and wellness

        *Disclaimer: I provide general information and guidance. Always consult healthcare professionals for medical advice.*
        """)

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=500,
                    avatar_images=("🧑", "🏋️")
                )
                message = gr.Textbox(
                    placeholder="Ask about fitness, nutrition, sleep, or health...",
                    show_label=False,
                    lines=2
                )
            with gr.Column(scale=1):
                context_box = gr.Textbox(
                    label="📚 Retrieved Knowledge Base",
                    lines=25,
                    interactive=False
                )

        message.submit(chat, inputs=[message, chatbot], outputs=[message, chatbot, context_box])

        gr.Markdown("""
        ---
        **How this works:** I search through a comprehensive fitness and health knowledge base
        to provide you with accurate, evidence-based information. My responses are designed to be
        practical, encouraging, and tailored to your needs.
        """)

    ui.launch()


if __name__ == "__main__":
    main()