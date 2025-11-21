from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import pandas as pd
import numpy as np
import joblib
import requests
import os

# =========================================================
# 1) Download big files from Dropbox
# =========================================================

MODEL_URL = "https://dl.dropboxusercontent.com/scl/fi/0w2lumdeb0g2z3rr16u5e/county_models.joblib?rlkey=mzxha28ahre7j0m1esim6sf0q&raw=1"
MODEL_PATH = "county_models.joblib"

CSV_URL = "https://www.dropbox.com/scl/fi/eopsl7xl68u42eub8meva/housingvars.csv?rlkey=cp59ajsh04wip925t1rvdmvtq&st=gck71hxr&raw=1"
CSV_PATH = "housingvars.csv"

def download_file(url, path, label):
    if os.path.exists(path):
        print(f"{label} already exists locally.")
        return
    print(f"Downloading {label} from Dropbox (streaming)...")
    with requests.get(url, stream=True, allow_redirects=True, timeout=180) as r:
        r.raise_for_status()
        total = 0
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
    print(f"{label} downloaded ✔ ({total/1024/1024:.1f} MB)")

download_file(MODEL_URL, MODEL_PATH, "Model")
download_file(CSV_URL, CSV_PATH, "Housing CSV")

# =========================================================
# 2) Load models
# =========================================================
bundle = joblib.load(MODEL_PATH)
COUNTY_MODELS = bundle["models"]

# =========================================================
# 3) Load & prepare lookup table from housingvars.csv
# =========================================================
df = pd.read_csv(CSV_PATH)

# 只保留你模型会用的列（如果列名有空格/大小写，下面会做 alias）
# 先统一列名，方便匹配
def norm(c): 
    return str(c).strip().lower()

df.columns = [norm(c) for c in df.columns]

# ---- 列名 alias（把训练列名映射到 csv 可能的写法）
ALIASES = {
    "beds": ["beds","bedrooms","bed"],
    "baths": ["baths","bathrooms","bath"],
    "living space": ["living space","living_space","squarefeet","sqft","living_sqft"],
    "city average household income": ["city average household income","avg_income","medianincome","average household income"],
    "crime rate": ["crime rate","crimerate","crime_rate"],
    "latitude": ["latitude","lat"],
    "longitude": ["longitude","lon","lng"],
    "population": ["population","pop"],
    "num_priv_schools": ["num_priv_schools","private_schools","num_private_schools"],
    "percent_bike": ["percent_bike","pct_bike"],
    "percent_car": ["percent_car","pct_car"],
    "percent_carpool": ["percent_carpool","pct_carpool"],
    "percent_home": ["percent_home","pct_home"],
    "percent_publictr": ["percent_publictr","pct_publictr","percent_transit"],
    "percent_total": ["percent_total","pct_total"],
    "percent_walk": ["percent_walk","pct_walk"]
}

def pick_col(target):
    """find real csv col name for a target feature"""
    for cand in ALIASES[target]:
        cand_n = norm(cand)
        if cand_n in df.columns:
            return cand_n
    return None

# 真实列名（csv版）
COL = {k: pick_col(k) for k in ALIASES.keys()}

# 必要的 ID 列（用来查城市/县）
city_col   = "city"   if "city"   in df.columns else None
county_col = "county" if "county" in df.columns else None

# 做城市级 lookup：对每个 City 取均值（城市统计特征）
city_lookup = None
if city_col:
    city_lookup = df.groupby(city_col).mean(numeric_only=True)

# 做县级 lookup：对每个 County 取均值（fallback 用）
county_lookup = None
if county_col:
    county_lookup = df.groupby(county_col).mean(numeric_only=True)

# =========================================================
# 4) FastAPI init
# =========================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 5) Request schema
#    前端只需要传 Location + 三个房屋变量
# =========================================================
class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    Location: str
    Beds: float
    Baths: float
    LivingSpace: float

class PredictResponse(BaseModel):
    predicted_price: float
    used_county: str | None = None
    used_city: str | None = None

# =========================================================
# 6) Helper: parse county/city from Location
# =========================================================
COUNTY_MAP = {
    "los angeles": "Los Angeles",
    "fresno": "Fresno",
    "san diego": "San Diego",
    "sacramento": "Sacramento",
    "san francisco": "San Francisco",
    "santa clara": "Santa Clara",
    "alameda": "Alameda"
}

def infer_county(location: str):
    loc = location.lower()
    for key, val in COUNTY_MAP.items():
        if key in loc:
            return val
    return None

def infer_city(location: str):
    """粗暴版：用 location 前半段当 city；如果 csv 有 city 列就尝试匹配"""
    if not city_col:
        return None
    loc = location.lower()
    # 直接在 city_lookup index 里找包含关系
    for city in city_lookup.index:
        if str(city).lower() in loc:
            return city
    return None

# =========================================================
# 7) Predict endpoint (auto-fill city stats)
# =========================================================
from fastapi import Query

@app.post("/predict")
def predict(req: PredictRequest, debug: bool = Query(False)):
    location = req.Location.strip()
    county = infer_county(location)

    if county is None or county not in COUNTY_MODELS:
        return {
            "predicted_price": -1,
            "used_county": None,
            "used_city": None,
            "reason": "county not recognized"
        }

    model = COUNTY_MODELS[county]

    # 1) 尝试 city stats（优先）
    city = infer_city(location)
    stats = None
    if city and city_lookup is not None and city in city_lookup.index:
        stats = city_lookup.loc[city]
    elif county_lookup is not None and county in county_lookup.index:
        stats = county_lookup.loc[county]
    else:
        # ✅ 关键改变：找不到 stats 就用空 Series，仍然继续预测
        stats = pd.Series(dtype=float)

    # 2) 组装模型需要的特征
    feat_cols = list(model.feature_names_in_)
    row = {}

    for f in feat_cols:
        f_n = norm(f)

        if f_n == "beds":
            row[f] = float(req.Beds)
        elif f_n == "baths":
            row[f] = float(req.Baths)
        elif f_n == "living space":
            row[f] = float(req.LivingSpace)
        else:
            # stats 里找同名特征
            target_key = None
            for k in ALIASES.keys():
                if norm(k) == f_n:
                    target_key = k
                    break

            if target_key and COL[target_key] and COL[target_key] in stats.index:
                row[f] = float(stats[COL[target_key]])
            else:
                row[f] = 0.0  # ✅ 找不到就补 0

    X = pd.DataFrame([row]).fillna(0)

    pred_log = model.predict(X)[0]
    pred_price = float(np.expm1(pred_log))

    out = {
        "predicted_price": pred_price,
        "used_county": county,
        "used_city": city
    }
    if debug:
        out["final_features_used"] = row
    return out


@app.get("/")
def health():
    return {"status": "ok"}
