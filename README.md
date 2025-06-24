# 🧠 Mental Health Support Chatbot

A compassionate, AI-powered chatbot trained to provide supportive responses to mental health-related messages.

![Gradio UI](https://img.shields.io/badge/UI-Gradio-blue?style=flat&logo=gradio)
![Model](https://img.shields.io/badge/Model-T5--small-green?style=flat&logo=huggingface)

---

## About the Project

The **Mental Health Support Chatbot** is a domain-specific conversational AI designed to offer compassionate, non-judgmental responses to users experiencing emotional distress, loneliness, or anxiety. Built within the **mental health domain**, it aims to simulate the empathetic tone of a human support listener.
Mental health issues often go undiscussed due to stigma, lack of access to professionals, or fear of being misunderstood. This chatbot was developed to provide a **safe, anonymous first step** toward emotional support — whether for individuals who are struggling, curious, or in need of someone to talk to.

By tailoring the model's training to **mental health-specific conversations**, this project highlights how AI can serve as a supportive tool in well-being and mental health education.

This chatbot was trained on a curated [dataset of mental health-related questions and answers](https://huggingface.co/datasets/Ashokajou51/ESConv_Original) from Hugging Face. Using the T5 Transformer architecture, it generates empathetic, context-aware responses to user inputs like:

> **"I feel so alone lately"**  
> → _"I am so sorry to hear that. I'm sorry that you're feeling so alone lately."_

---

## 💡 Key Features

- Built using the `t5-small` model from Hugging Face Transformers
- Fine-tuned for mental health support conversations using the [ESConv_original](https://huggingface.co/datasets/Ashokajou51/ESConv_Original) dataset
- 🎛Interactive UI powered by **Gradio**. The live interactive version is at [this link](https://74b9d45eac235934a8.gradio.live/)
- Evaluation using BLEU score (with smoothing)
---

## 🖥️ Demo

To launch the chatbot locally in Colab or Python:

```python
import gradio as gr

def chatbot_interface(input_text):
    model_input = f"mental_health: {input_text}"
    response = generate_response(model, tokenizer, model_input)
    return response

gr.Interface(fn=chatbot_interface, inputs="text", outputs="text").launch()

```

## Evaluation

I evaluated on a reserved validation set using batch inference. I got a BLEU score of 0.22

## The fine-tuned model

If you want to load the fine-tuned model to test it out, use the following Python code in your Google collab

```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("Eliane-M/mental_health_chatbot")
tokenizer = AutoTokenizer.from_pretrained("Eliane-M/mental_health_chatbot")
```

## Folder Structure

```
mental_health_chatbot/
│
├── chatbot.ipynb                 # Full training & evaluation notebook
├── mental_health_qa_model/      # Saved model files
├── mental_health_qa_tokenizer/  # Tokenizer config
└── README.md                    
```

NB: The mental_health_qa_model in this repo doesn't contain the trained model due to size constraints that wouldn't allow it to get pushed to github.

This is an exmple of the chatbot conversation
![Screenshot 2025-06-24 223658](https://github.com/user-attachments/assets/3acc7434-e59f-454b-9220-c2f0961073bc)
