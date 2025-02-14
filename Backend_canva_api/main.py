from fastapi import FastAPI, File, UploadFile, HTTPException
import openai
import pypdf
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Retrieve OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Make sure to define it in the .env file.")

openai.api_key = OPENAI_API_KEY  # Set the OpenAI API key

# Function to extract text from PDFs asynchronously
async def extract_text_from_pdf(pdf_file):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_pdf_extraction, pdf_file)

# Synchronous helper function for PDF extraction
def _sync_pdf_extraction(pdf_file):
    pdf_reader = pypdf.PdfReader(pdf_file)
    text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages if page.extract_text())
    return text

# Function to summarize text using OpenAI (GPT-3.5)
def summarize_text(text):
    try:
        # Use GPT-3.5 with the new OpenAI API
        response = openai.completions.create(
            model="gpt-3.5-turbo",  # Specify GPT-3.5 model
            prompt=f"Summarize this text concisely: {text}",
            max_tokens=300,
            temperature=0.5
        )
        return response['choices'][0]['text'].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API to handle file upload and summarization
@app.post("/summarize/")
async def summarize_file(file: UploadFile = File(...)):
    try:
        if file.content_type == "application/pdf":
            text = await extract_text_from_pdf(file.file)  # Updated async function
        elif file.content_type in ["text/plain", "application/octet-stream"]:
            text = await file.read()
            text = text.decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use PDF or TXT.")

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in the document.")

        summary = summarize_text(text)
        return {"filename": file.filename, "summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
