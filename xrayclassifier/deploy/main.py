from fastapi import FastAPI, File, UploadFile
import mlflow
from mlflow.tracking import MlflowClient
import os
from skimage.io import imread

relative_model_dev_path = '..'

mlflow.set_tracking_uri(f"sqlite:///{relative_model_dev_path}/mlflow.db")
runs = mlflow.search_runs(experiment_ids=0)
client = MlflowClient()

model_threshold = 0.5

app = FastAPI()


@app.post("/uploadfile/")
async def predict_xray_image(file: UploadFile = File(...)):
    model_path = return_production_model()
    model = mlflow.pyfunc.load_model(os.path.join(relative_model_dev_path,model_path))

    try:
        for (root,_,files) in os.walk('..', topdown=True):
            if file.filename in files:
                image_path = os.path.join(root,file.filename)
                break
        image = imread(image_path, as_gray=True)[::8,::8]
        images_2d_list = (image - image.min())/(image.max() - image.min())
        images_2d_list = images_2d_list.reshape(1, 128, 128, 1)
    except Exception as e:
        error_msg = e
        return {"error_encountered_reading": str(error_msg)}

    try:
        result = model.predict(images_2d_list)
        prob = result[0][0]
        if prob > model_threshold:
            disease = "Cardiomegaly"
        else:
            disease = "Not Cardiomegaly"
    except Exception as e:
        error_msg = e
        return {"error_encountered_predictions": str(error_msg)}

    return {"model_result": disease,"prediction_score":str(prob),"modelpath": model_path}



def return_production_model():
    prod_model_path = None
    for reg_mod in client.list_registered_models():
        for versions in reg_mod.latest_versions:
            if versions.current_stage == 'Production':
                prod_model_path = versions.source
                break
    return prod_model_path