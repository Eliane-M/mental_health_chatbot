# 🧠 Mental Health Support Chatbot

A compassionate, AI-powered chatbot trained to provide supportive responses to mental health-related messages.

![Gradio UI](https://img.shields.io/badge/UI-Gradio-blue?style=flat&logo=gradio)
![Model](https://img.shields.io/badge/Model-T5--small-green?style=flat&logo=huggingface)
![License](https://img.shields.io/github/license/Eliane-M/mental_health_chatbot)

---

## About the Project

This chatbot was trained on a curated [dataset of mental health-related questions and answers](https://huggingface.co/datasets/Ashokajou51/ESConv_Original) from hugging face. Using the T5 Transformer architecture, it generates empathetic, context-aware responses to user inputs like:

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
