# Imports
import re
import fitz  
from openai import OpenAI
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter # Smart text splits
import os
import datetime
import json 
# JSON is a new addition, as before the chunks were just printing in terminal with no destination

# Setup
current_directory = os.getcwd()
output_directory = os.path.join(current_directory, "chunked_data") # Sends chunks to the project folder
os.makedirs(output_directory, exist_ok=True) # Makes sure we actually have an output directory
model = "gpt-4o-mini" # For some reason I kept trying to run this code with the wrong model but I fixed it

with open(os.path.join(current_directory, "SecretStuff4.txt"), "r") as file: # API Key
    for lines in file:
        name, value = lines.split("=")
        if name.strip() == "api_key":
            api_key = value.strip()

openai = OpenAI(api_key=api_key)

# Token Tools
encoding = tiktoken.get_encoding("cl100k_base")
MAX_TOKEN_BUDGET = 95000 # Ensures we do not reach OpenAi's token limit

def count_tokens(text):
    return len(encoding.encode(text)) # Returns token count

# Data Cleansing
def extract_text_from_pdf(pdf_path): # Extracts text from PDF
    doc = fitz.open(pdf_path)
    return "\n\n".join([page.get_text("text") for page in doc])

def clean_text(text):
    cleaned_text = re.sub(r"[\u2022\-\*\x95]+", "", text) # Gets rid of bullet points/other special characters
    cleaned_text = re.sub(r"[\n\t\r]+", " ", cleaned_text) # Gets rid of newlines/tabs/carriage returns
    return cleaned_text

def get_optimal_chunk_size(text):
    token_count = count_tokens(text) # Counts tokens in the particular PDF
    return min(3000, max(1000, token_count // 10)) # Decides how big the chunks should be based on token count

def split_text(text):
    chunk_size = get_optimal_chunk_size(text) # Calls the previous chunk size function
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50, # Allows overlap in chunks of context is needed in both
        separators=["\n\n", ". ", "\n", ", ", "; ", ": "] # Splits on spaces/periods/commas/etc.
    )
    return text_splitter.split_text(text)

def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*\n]', '', name).strip().replace(' ', '_')[:100] # Turns chunk name into valid file name

# OpenAI Calls
def generate_chunk_name(chunk_text): # Calls ChatGPT to name chunk
    response = openai.chat.completions.create( #OpenAI Chat Completion
        model=model,
        messages=[ 
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the key theme of this text in 5 words or fewer:\n\n{chunk_text}"}
        ],
        max_tokens=20
    )
    return response.choices[0].message.content.strip()

def summarize_chunk(chunk_text): # Calls ChatGPT to summarize
    response = openai.chat.completions.create(
        model=model,
        messages=[ 
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": (
                f"Please summarize the following text in **2–3 short, concise sentences**. "
                "Focus only on the most important ideas and facts. Do not include extra commentary or explanations.\n\n"
                f"{chunk_text}"
            )}
        ],
        max_tokens=100, 
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


# Chunk Processing
def process_pdf(pdf_path, token_limit=MAX_TOKEN_BUDGET):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_dir = os.path.join(output_directory, base_name)
    os.makedirs(pdf_output_dir, exist_ok=True)

    metadata_path = os.path.join(pdf_output_dir, "metadata.json") # Saves metadata in chunk folder
    
    metadata_records = [] # Empty list for metadata

    text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(text)
    chunks = split_text(cleaned_text)

    print(f"\n📄 Processing '{base_name}' — {len(chunks)} chunks to evaluate...\n")

    for idx, chunk in enumerate(chunks):
        estimated_chunk_tokens = count_tokens(chunk)
        if estimated_chunk_tokens > token_limit:
            print(f"🛑 Token limit exceeded for chunk {idx+1} ({estimated_chunk_tokens} tokens). Skipping this chunk.")
            continue

        temp_name = f"temp_chunk_{idx+1}.txt"
        print(f"✨ Processing chunk {idx+1}/{len(chunks)}...")

        try:
            chunk_name = generate_chunk_name(chunk)
            print(f"🔖 Chunk Name: {chunk_name}")

            chunk_name_clean = sanitize_filename(chunk_name)
            summary = summarize_chunk(chunk)
            print(f"📝 Summary: {summary}")
        except Exception as e: # Buffer if GPT fails
            print(f"❌ Error processing chunk {idx+1}: {e}")
            break

        filename = f"{chunk_name_clean or 'chunk'}_{idx+1:03d}.txt" 
        filepath = os.path.join(pdf_output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Chunk Name: {chunk_name}\n")
            f.write(f"Summary: {summary}\n")
            f.write("\nChunk Content:\n")
            f.write(chunk)

        metadata_records.append({ # Appends metadata to the list
            "chunk_id": idx + 1,
            "chunk_name": chunk_name,
            "filename": filename,
            "original_chunk_text": chunk,
            "summary": summary,
            "timestamp": datetime.datetime.now().isoformat(),
            "token_count": estimated_chunk_tokens
        })

    # Save Metadata
    with open(metadata_path, "w", encoding="utf-8") as f: # Metadata to JSON
        json.dump(metadata_records, f, indent=4)

    print(f"\n✅ Finished '{base_name}' — {len(metadata_records)} chunks saved.\n")

# PDF List
pdf_list = [
    "i1040gi.pdf",
    "Form 8917 (Rev. January 2020) - f8917.pdf",
    "i1040x.pdf",
    "i1042s.pdf",
    "i1098et.pdf",
    "i1099mec.pdf",
    "i8863.pdf",
    "iw2w3.pdf",
    "PATaxLaws.pdf"
]

for pdf in pdf_list:
    process_pdf(pdf)

