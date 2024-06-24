
from fastapi import FastAPI, UploadFile
from model import ModelAPI

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Load the image database when the app starts
    ModelAPI.load_image_database()

@app.post("/identify")
async def identify_employee(image: UploadFile):
    binary_content = await image.read()
    img = ModelAPI.to_image(binary_content)
    identity = ModelAPI.search(img, return_max=True)
    return {'id': identity}

