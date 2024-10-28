import nbformat as nbf
from nbconvert import MarkdownExporter
import pyperclip
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY") # "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"
client = Groq(api_key=api_key) 

def ipynb_to_markdown_string(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = nbf.read(f, as_version=4)
    markdown_exporter = MarkdownExporter()
    markdown, _ = markdown_exporter.from_notebook_node(notebook)
    processed_markdown = process_markdown(markdown)
    return processed_markdown

def process_markdown(markdown):
    lines = markdown.split('\n')
    processed_lines = []
    in_code_block = False
    for line in lines:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            processed_lines.append(line)
        elif in_code_block:
            processed_lines.append(line)
        else:
            processed_lines.append(line + '  ')
    return '\n'.join(processed_lines)

def convert_and_process_with_llm(path, model):
    markdown_content = ipynb_to_markdown_string(path)
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": markdown_content
            }
        ],
        model=model,
        temperature=0.5,
        max_tokens=4096,
        stream=False,  # Changed to False for simplicity
    )

    return response.choices[0].message.content


def run_conversion(path, model):
    result = convert_and_process_with_llm(path, model)
    pyperclip.copy(result)
    return result

if __name__ == "__main__":
    run_conversion(path="Visualization.ipynb", model="llama-3.1-70b-versatile")