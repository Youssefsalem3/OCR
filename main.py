from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import easyocr
import google.generativeai as genai
import numpy as np
from io import BytesIO

app = FastAPI()

# Configure the Gemini API (you should replace this with your real API key)
genai.configure(api_key="AIzaSyBLz93HFhSFOwsG11tFgbns0SJ3S3ODrGk")
model = genai.GenerativeModel('gemini-1.5-pro-latest')

def correct_text_with_gemini(extracted_text):
    prompt = (
        "Correct only the spelling and punctuation mistakes in the following text.\n"
        "Do not add, remove, or change any words.\n"
        "Do not explain anything—just return the corrected version of the text with the same formatting:\n"
        f"{extracted_text}"
    )

    response = model.generate_content(prompt)
    corrected_text = response.text.strip()
    return corrected_text

@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    try:
        # Read image from the uploaded file
        image_bytes = await file.read()
        image = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Convert to grayscale (improves OCR accuracy)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Initialize EasyOCR Reader
        reader = easyocr.Reader(['en'], gpu=True)

        # Perform OCR
        results = reader.readtext(gray, detail=0, paragraph=True)

        # Combine text results
        extracted_text = '\n'.join(results)
        corrected_text = correct_text_with_gemini(extracted_text)

        return JSONResponse(content={"extracted_text": extracted_text, "corrected_text": corrected_text})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# To run locally for testing
# If you are deploying to Railway, remove the below line
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
