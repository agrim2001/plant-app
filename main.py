import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app_desc = """<h2></h2>
<br>by Agrim gupta"""

app = FastAPI(title='Plant api', description=app_desc)

MODEL = tf.keras.models.load_model("./model/1")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict/image")
async def predict_api(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="localhost")
