"""Private web app: two photos + a few numbers -> body fat % and waist/hip.

  APP_PASSWORD=... venv/bin/uvicorn server:app --app-dir scripts --port 8000

Every submission is saved to its own folder under SUBMISSIONS_DIR (default: ./submissions):
the original photos, the silhouettes, and result.json with the inputs and outputs.
"""
import io
import json
import os
import secrets
import threading
import time
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from PIL import Image

from estimate import Estimator

PASSWORD = os.environ.get("APP_PASSWORD")
if not PASSWORD:
    raise RuntimeError("Set the APP_PASSWORD environment variable before starting the server")

INCH_TO_CM = 2.54
LB_TO_KG = 0.45359237
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 60_000_000
PROJECT = Path(__file__).resolve().parent.parent
INDEX_HTML = PROJECT / "web" / "index.html"
SUBMISSIONS_DIR = Path(os.environ.get("SUBMISSIONS_DIR", PROJECT / "submissions"))
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
security = HTTPBasic()
estimator = Estimator()
inference_lock = threading.Lock()


def require_password(creds: HTTPBasicCredentials = Depends(security)):
    if not secrets.compare_digest(creds.password.encode(), PASSWORD.encode()):
        time.sleep(1)
        raise HTTPException(401, "Wrong password", headers={"WWW-Authenticate": "Basic"})


def read_image(upload: UploadFile):
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Photo is too large (max 15 MB)")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise HTTPException(400, "Could not read that photo; use a JPEG or PNG")
    return img, data


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def new_submission(inputs, front, side):
    """Create the submission folder and save the original photos before any processing."""
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    folder = SUBMISSIONS_DIR / f"{stamp}_{secrets.token_hex(3)}"
    folder.mkdir(mode=0o700)
    for name, (img, data) in (("front", front), ("side", side)):
        ext = (img.format or "jpeg").lower().replace("jpeg", "jpg")
        (folder / f"{name}_original.{ext}").write_bytes(data)
    write_json(folder / "result.json", {"status": "received", "time": stamp, "inputs": inputs})
    return folder


@app.get("/", dependencies=[Depends(require_password)])
def index():
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/estimate", dependencies=[Depends(require_password)])
def estimate(
    front: UploadFile = File(...),
    side: UploadFile = File(...),
    gender: str = Form(...),
    age: float = Form(...),
    height_in: float = Form(...),
    weight_lb: float = Form(...),
):
    if gender not in ("male", "female"):
        raise HTTPException(422, "Gender must be male or female")
    if not 18 <= age <= 59:
        raise HTTPException(422, "The body fat model was trained on adults aged 18 to 59")
    if not 47 <= height_in <= 90:
        raise HTTPException(422, "Height should be in inches, between 47 and 90 (5 ft 10 in = 70)")
    if not 66 <= weight_lb <= 550:
        raise HTTPException(422, "Weight should be in pounds, between 66 and 550")

    height_cm = round(height_in * INCH_TO_CM, 2)
    weight_kg = round(weight_lb * LB_TO_KG, 2)

    front_in, side_in = read_image(front), read_image(side)
    inputs = {
        "gender": gender, "age": age,
        "height_in": height_in, "weight_lb": weight_lb,
        "height_cm": height_cm, "weight_kg": weight_kg,
    }
    folder = new_submission(inputs, front_in, side_in)
    record = {"status": "received", "time": folder.name.split("_")[0], "inputs": inputs}

    try:
        with inference_lock:
            result = estimator.from_photos(front_in[0], side_in[0], gender, age, height_cm, weight_kg)
    except ValueError as e:
        write_json(folder / "result.json", {**record, "status": "error", "error": str(e)})
        raise HTTPException(422, str(e))

    result["masks"][0].save(folder / "front_mask.png")
    result["masks"][1].save(folder / "side_mask.png")
    m = result["measurements"]
    write_json(folder / "result.json", {
        **record,
        "status": "ok",
        "body_fat_pct": result["body_fat"],
        "rfm_body_fat_pct": result["rfm"],
        "measurements_cm": m,
        "warnings": result["warnings"],
    })
    return {
        "id": folder.name,
        "body_fat_pct": round(result["body_fat"], 1),
        "waist_cm": round(m["waist"], 1),
        "hip_cm": round(m["hip"], 1),
        "waist_in": round(m["waist"] / INCH_TO_CM, 1),
        "hip_in": round(m["hip"] / INCH_TO_CM, 1),
        "warnings": result["warnings"],
    }
