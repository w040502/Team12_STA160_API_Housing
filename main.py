from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import requests
import os
import tempfile

# ----------------------------------------
# 1) Dropbox 下载模型
# ----------------------------------------

DROPBOX_URL = "https://www.dropbox.com/s/xxxxx/county_models.joblib?dl=1"  
# ↑↑↑ 把这个换成你真实的 dl=1 链接

LOCAL_MODEL_PATH = "county_models.joblib"

def download_model():
    # 如果本地已经下载过，就不重复下载
    if os.path.exists(LOCAL_MODEL_PATH):
        print("Model already exists locally.")
        return

    print("Downloading model from Dropbox...")

    r = requests.get(DROPBOX_URL)
    if r.status_code != 200:
        raise Exception("Failed to download model! HTTP " + str(r.status_code))

    with open(LOCAL_MODEL_PATH, "wb") as f:
        f.write(r.content)

    print("Model downloaded → county_models.joblib")

# 下载模型
download_model()

# ----------------------------------------
# 2) 加载模型
# ----------------------------------------
bundle = joblib.load(LOCAL_MODEL_PATH)
COUNTY_MODELS = bundle["models"]
FEATURE_COLUMNS = bundle["feature_columns"]
COUNTIES = bundle["counties"]
LOG_TARGET = bundle["log_target"]
WINSOR_Q = bundle["winsor_q"]

# ----------------------------------------
# 3) FastAPI 初始化
# ----------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 上线后可换为你的网站域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------
# 4) 请求体
# ----------------------------------------
class PredictRequest(BaseModel):
    County: str
    Bedrooms: float | None = None
    Bathrooms: float | None = None
    SquareFeet: float | None = None
    YearBuilt: float | None = None
    Latitude: float | None = None
    Longitude: float | None = None

class PredictResponse(BaseModel):
    predicted_price: float

# ----------------------------------------
# 5) 预测接口
# ----------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):

    county = req.County
    if county not in COUNTY_MODELS:
        return {"predicted_price": -1}

    model = COUNTY_MODELS[county]

    # 生成 dataframe
    row = {col: getattr(req, col, None) for col in FEATURE_COLUMNS}
    X = pd.DataFrame([row]).fillna(0)

    # 模型是 log(y)
    pred_log = model.predict(X)[0]
    pred_price = float(np.expm1(pred_log))

    return PredictResponse(predicted_price=pred_price)

@app.get("/")
def health():
    return {"status": "ok"}
