from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from predict import predict_class
from io import BytesIO

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/result/")
async def create_item(image: UploadFile = File(...)):
    image_bytes = await image.read()
    image_stream = BytesIO(image_bytes)
    return {"result": predict_class(image_stream)}