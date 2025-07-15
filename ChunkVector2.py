import os
import pinecone 
from openai import OpenAI

# Paths
current_directory = r"C:\Users\paone\source\repos\ChunkVector2\ChunkVector2"
chunks_root = os.path.join(current_directory, "chunked_data")

# OpenAI API Key
with open(os.path.join(current_directory, "SecretStuff5.txt"), "r") as file:
    for line in file:
        name, value = line.split("=")
        if name.strip() == "api_key":
            openai_api_key = value.strip()

# Pinecone API Key
with open(os.path.join(current_directory, "SigmaKey.txt"), "r") as file:
    pinecone_api_key = file.read().strip()

# API Initialization
openai = OpenAI(api_key=openai_api_key)
print("OpenAI API initialized successfully.")
pc = pinecone.Pinecone(api_key=pinecone_api_key, environment="us-west1-gcp")
print("Pinecone initialized successfully.")

# Pinecone Index Initialization
index_name = "omegachunkindex4"
index = pc.Index(index_name)
print(f"Connected to Pinecone index: {index_name}")

# PDF Subfolder List
subfolders = [
    "Form 8917 (Rev. January 2020) - f8917",
    "i1040gi",
    "i1040x",
    "i1042s",
    "i1098et",
    "i1099mec",
    "i8863",
    "iw2w3",
    "PATaxLaws"
]

# Chunk Processing/Uploading
for subfolder in subfolders: # Loop through subfolders
    subfolder_path = os.path.join(chunks_root, subfolder)
    if not os.path.isdir(subfolder_path):
        print(f"[SKIP] Folder not found: {subfolder_path}")
        continue

    chunk_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith(".txt")]) #
    print(f"\nProcessing folder: {subfolder} ({len(chunk_files)} chunks)")

    for i, filename in enumerate(chunk_files): # Loop through chunk files
        file_path = os.path.join(subfolder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            chunk_text = f.read()

        if not chunk_text.strip():
            print(f"  [SKIP] Empty file: {filename}")
            continue

        # Vector Embedding
        response = openai.embeddings.create(
            input=[chunk_text],
            model="text-embedding-ada-002"
        )
        vector = response.data[0].embedding

        # ID Creation
        folder_id = subfolder.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "").replace("-", "")
        vector_id = f"{folder_id}_chunk_{i+1}"

        # Upload to Pinecone
        index.upsert([(vector_id, vector)])
        print(f"  [OK] Uploaded {vector_id}")

print("\n✅ All chunks uploaded successfully.")
