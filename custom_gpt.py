from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = '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'

def create_index(directory_path):
    max_input_len = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_len, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    
    # Save index to disk
    try:
        index.save_to_disk('index.json')
        print("Index saved to disk")
    except Exception as e:
        print(f"Error saving index to disk: {e}")

    return index

def chat(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="tree_summarize")
    return response.response

interface = gr.Interface(fn=chat,
                     inputs=gr.components.Textbox(lines=5, label="Enter your text"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")

index = create_index("docs")
interface.launch(share=True)
