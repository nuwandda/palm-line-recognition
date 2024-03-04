from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import fastapi as _fapi

import schemas as _schemas
import services as _services
from io import BytesIO
import base64
import traceback


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to Palm Recognition API"}


# Endpoint to test the backend
@app.get("/api")
async def root():
    return {"message": "Welcome to the Palm Recognition with FastAPI"}


@app.post("/api/recognize/")
async def generate_image(palmCreate: _schemas.PalmCreate = _fapi.Depends()):
    
    try:
        image = await _services.recognize_palm_lines(palmCreate=palmCreate)
    except Exception as e:
        print(traceback.format_exc())
        return {"message": f"{e.args}"}
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_img = base64.b64encode(buffered.getvalue())
    payload = {
        "mime" : "image/jpg",
        "image": encoded_img
        }
    
    return payload
