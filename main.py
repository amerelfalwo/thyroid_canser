from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.disease.schema import ThyroidInput
from app.disease.model import predict_thyroid
from app.segmentation.model import process_full_pipeline

app = FastAPI(
    title="ThyraX API",
    description="Comprehensive API for Thyroid Cancer Diagnosis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "success", "message": "ThyraX API is running!"}

@app.post("/predict/disease")
def predict_clinical_disease(data: ThyroidInput):
    try:
        input_data = data.model_dump()
        result = predict_thyroid(input_data)
        
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/image")
async def predict_ultrasound_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    try:
        image_bytes = await file.read()
        result = process_full_pipeline(image_bytes)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")