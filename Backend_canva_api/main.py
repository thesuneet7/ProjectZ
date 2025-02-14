from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import openai
import pypdf
import os
import asyncio

# Initialize FastAPI
app = FastAPI()

# Set your OpenAI API Key from GitHub Secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please configure GitHub Secrets properly.")

# Function to extract text from PDFs asynchronously
async def extract_text_from_pdf(pdf_file):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_pdf_extraction, pdf_file)

# Synchronous helper function for PDF extraction
def _sync_pdf_extraction(pdf_file):
    pdf_reader = pypdf.PdfReader(pdf_file)
    text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages if page.extract_text())
    return text

# Function to summarize text using OpenAI
def summarize_text(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": "Summarize this text concisely."},
                      {"role": "user", "content": text}],
            temperature=0.5,
            max_tokens=300
        )
        return response["choices"][0]["message"]["content"]
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
