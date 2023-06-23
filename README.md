# Potato Disease Prediction Project

This project aims to predict potato diseases using machine learning techniques. It provides an API for predicting diseases based on input images of potato leaves.

## API

### api/main.py

This file contains the main FastAPI application for the disease prediction API. It uses the following libraries:

- `fastapi`: A modern, fast (high-performance) web framework for building APIs with Python.
- `uvicorn`: A fast ASGI server implementation, used to run the FastAPI application.
- `numpy`: A library for numerical computing in Python.
- `tensorflow`: A popular deep learning library.
- `PIL`: The Python Imaging Library, used for image processing.

The API has the following endpoints:

- **GET /ping**: Returns a simple "Hello, I am alive" message to verify the API's availability.
- **POST /predict**: Accepts an uploaded image file and predicts the potato disease based on the image.

### api/exp.py

This file contains an alternate version of the API implementation. It also uses the FastAPI framework and provides the same endpoints for predicting potato diseases. The libraries used in this file are similar to the ones used in `api/main.py`.

### gcp/main.py

This file implements a Flask application for deploying the disease prediction model on Google Cloud Platform (GCP). It uses the following libraries:

- `google-cloud-storage`: A library for interacting with Google Cloud Storage, used for downloading the model file.
- `tensorflow`: The TensorFlow library for machine learning.
- `PIL`: The Python Imaging Library for image processing.
- `numpy`: A library for numerical computing.

The application has a `predict` function that accepts an image file, downloads the trained model from Google Cloud Storage, performs the prediction, and returns the predicted disease class and confidence score.

## Model Training

The notebook `model_training.ipynb` contains the code for training the potato disease prediction model. The specific details of the model architecture and training process can be found in the notebook.

## Requirements

The `requirements.txt` file lists all the necessary libraries and their versions required to run the project. You can install the dependencies using the following command:

```shell
pip install -r requirements.txt
```

Please make sure to have the correct versions of the libraries to avoid any compatibility issues.

## How to Run

To run the project, follow these steps:

1. Install the required dependencies using the command mentioned above.
2. Start the API server by running one of the following commands in the respective API directories:
   - For `api/main.py`:
     ```
     uvicorn main:app --reload
     ```
   - For `api/exp.py`:
     ```
     uvicorn exp:app --reload
     ```
   - For `gcp/main.py`:
     ```
     python main.py
     ```
3. Once the server is running, you can make requests to the API endpoints for predicting potato diseases.

Note: The provided code snippets assume that the necessary model files and dataset are available in the correct paths.

Feel free to explore and experiment with the project code to understand the disease prediction process in detail.
