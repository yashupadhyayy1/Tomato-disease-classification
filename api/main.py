# from fastapi import FastAPI, UploadFile,  File
# import uvicorn
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from io import BytesIO
# from PIL import Image
#
#
# app = FastAPI()
#
# MODEL = tensorflow.keras.models.load_model("../my_mode_9806.h5")
# CLASS_NAME = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus','Tomato_healthy']
# #
# # @app.get("/")
# # async def ping():
# #     return "RAM RAM"
#
# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image
#
# @app.get("/ping")
# async def ping():
#     return "Hello i am ping"
#
#
# @app.get("/predict")
# async def predict(
#     file: UploadFile = File(...)
# ):
#    image =  read_file_as_image(await file.read())
#    img_batch = np.expand_dims(image,0)
#    predictions = MODEL.predict(img_batch)
#    predicted_class = CLASS_NAME[np.argmax(predictions[0])]
#    confidence = np.max(predictions[0])
#    print(predicted_class, confidence)
#    return{
#     "class": predicted_class,
#     "confidence": float(predictions)
#     }
#
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)


#
# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
#
# app = FastAPI()
#
# MODEL = tf.keras.models.load_model("../maro")
# CLASS_NAMES = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus','Tomato_healthy']
#
#
# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive"
#
#
# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image
#
#
# @app.post("/predict")
# async def predict(
#         file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, 0)
#
#     predictions = MODEL.predict(img_batch)
#
#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
#     return {
#         'class': predicted_class,
#         'confidence': float(confidence)
#     }
#
#
# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)


from fastapi import FastAPI, UploadFile,  File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware


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


MODEL = tf.keras.models.load_model("../save_models/1")
# MODEL = tf.keras.models.load_model("../save_models/1")

CLASS_NAMES = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus','Tomato_healthy']



@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((128, 128))
    image = np.array(image)
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):

    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    # print(predicted_class+" "+confidence)
    return {
            'class': predicted_class,
            'confidence': float(confidence)
        }
    #

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
