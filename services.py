import schemas as _schemas
import torch 
import os
# from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import numpy as np
from pkg_resources import parse_version
from palm_lines_recognition import detect_lines
import uuid
import os
import shutil
import base64


TEMP_PATH = 'temp'

def create_temp(temp_id):
    if not os.path.exists(TEMP_PATH + '/' + temp_id):
        os.makedirs(TEMP_PATH + '/' + temp_id)


def delete_folder(temp_id):
    # Try to remove the tree; if it fails, throw an error using try...except.
    try:
        shutil.rmtree(TEMP_PATH + '/' + temp_id)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


async def recognize_palm_lines(palmCreate: _schemas.PalmCreate) -> Image:
    temp_id = str(uuid.uuid4())
    create_temp(temp_id)
    init_image = Image.open(BytesIO(base64.b64decode(palmCreate.encoded_base_img[0]))).save(TEMP_PATH + '/' + temp_id + '/input.jpg')
    result_path = detect_lines.run(TEMP_PATH + '/' + temp_id + '/input.jpg', palmCreate.resize_width, TEMP_PATH + '/' + temp_id)

    return_image = Image.open(result_path)
    delete_folder(temp_id)
    
    return return_image