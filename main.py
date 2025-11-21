from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import pandas as pd
import numpy as np
import joblib
import requests
import os

# ----------------------------
# Dropbox download (keep your working version)
# ----------------------------
DROPBOX_URL = "https://dl.dropboxusercontent.com/scl/fi/0w2lumdeb0g2z3rr16u5e/county_models.joblib?rlkey=mzxha28ahre7j0m1esim6sf0q&raw=1"
LOCAL_MODEL_PATH = "county_models.joblib"

def download_model():
    if os.path.exists(LOCAL_MODEL_PATH):
        print("Model already exists locally.")
        return

    print("Downloading model from Dropbox (streaming)...")
    with requests.get(DROPBOX_URL, stream=True, allow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        total = 0
        with open(LOCAL_MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
    print(f"Model downloaded ✔ ({total/1024/1024:.1f} MB)")

download_model()

bundle = joblib.load(LOCAL_MODEL_PATH)
COUNTY_MODELS = bundle["models"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# ✅ schema: allow extra fields
# ----------------------------
class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # 允许前端多传字段也不报错

    County: str

    # 你网页里会传的字段（可选）
    Bedrooms: float | None = None
    Bathrooms: float | None = None
    SquareFeet: float | None = None
    TotalRooms: float | None = None
    CrimeRate: float | None = None
    AvgIncome: float | None = None
    YearBuilt: float | None = None
    Latitude: float | None = None
    Longitude: float | None = None

class PredictResponse(BaseModel):
    predicted_price: float

# ----------------------------
# ✅ predict: align to each county model's features
# ----------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    county = req.County
    if county not in COUNTY_MODELS:
        return PredictResponse(predicted_price=-1)

    model = COUNTY_MODELS[county]

    # 1) 取出所有请求字段
    req_dict = req.model_dump()

    # 2) 只保留数值型（训练时也是 numeric-only）
    X_all = pd.DataFrame([req_dict]).select_dtypes(include=["float64", "int64", "float32", "int32"])

    # 3) ✅ 用该县模型训练时的特征列对齐
    feat_cols = list(model.feature_names_in_)  # 每个县自己的列（SF 不会有 Population）
    X = X_all.reindex(columns=feat_cols, fill_value=0)

    # 4) 预测（模型训练在 log1p(y)）
    pred_log = model.predict(X)[0]
    pred_price = float(np.expm1(pred_log))

    return PredictResponse(predicted_price=pred_price)

@app.get("/")
def health():
    return {"status": "ok"}
