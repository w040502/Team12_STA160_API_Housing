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

import os
import requests
import hashlib

DROPBOX_URL = "https://dl.dropboxusercontent.com/scl/fi/0w2lumdeb0g2z3rr16u5e/county_models.joblib?rlkey=mzxha28ahre7j0m1esim6sf0q&raw=1"
LOCAL_MODEL_PATH = "county_models.joblib"

def download_model():
    if os.path.exists(LOCAL_MODEL_PATH):
        print("Model already exists locally.")
        return

    print("Downloading model from Dropbox (streaming)...")
    with requests.get(DROPBOX_URL, stream=True, allow_redirects=True, timeout=120) as r:
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "")
        # 如果 Dropbox 还在返回 HTML 页面，直接提示
        if "text/html" in content_type.lower():
            raise RuntimeError(
                "Dropbox returned HTML instead of the model file. "
                "Your link is not a raw direct-download link."
            )

        total = 0
        with open(LOCAL_MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB
                if chunk:
                    f.write(chunk)
                    total += len(chunk)

    if total < 5 * 1024 * 1024:  # 小于 5MB 基本不可能是你的模型
        raise RuntimeError(
            f"Downloaded file too small ({total/1024/1024:.2f} MB). "
            "Likely not the real model file."
        )

    print(f"Model downloaded ✔ ({total/1024/1024:.1f} MB)")

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
