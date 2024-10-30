import nbformat as nbf
from nbconvert import MarkdownExporter
import pyperclip
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
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
    
    try:
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
            stream=False,
        )
        return response.choices[0].message.content, markdown_content
    except Exception as e:
        print(f"LLM processing failed: {str(e)}")
        return None, markdown_content

def run_conversion(path, model, fallback_to_markdown=True):
    llm_result, markdown_content = convert_and_process_with_llm(path, model)
    
    # If LLM processing failed and fallback is enabled, copy markdown content
    if llm_result is None and fallback_to_markdown:
        # Copy markdown content to clipboard by default
        pyperclip.copy(markdown_content)
        print("Falling back to markdown content")

    else:
        # If LLM processing succeeded, copy the LLM response
        pyperclip.copy(llm_result)
        print("Successfully copied the content")

if __name__ == "__main__":
    result = run_conversion(
        path="Visualization.ipynb", 
        model="llama-3.1-70b-versatile",
        fallback_to_markdown=True  # Set to False if you don't want fallback
    )
