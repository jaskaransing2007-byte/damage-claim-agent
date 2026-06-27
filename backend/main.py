import os
import io
import PIL.Image
import json
import base64
import asyncio
import csv
from pathlib import Path
from typing import Optional
from datetime import datetime
import google.generativeai as genai
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

BACKEND_DIR = Path(__file__).resolve().parent
ENV_PATH = BACKEND_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

app = FastAPI(title="Damage Claim AI Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

BASE_DIR = BACKEND_DIR.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_FILE = BASE_DIR / "output.csv"
MODEL_PATH = BACKEND_DIR / "saved_validation_model"

try:
    gate_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    gate_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
except Exception:
    gate_tokenizer, gate_model = None, None

DATABASE_URL = f"sqlite:///{BACKEND_DIR}/claims.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class VerifiedClaim(Base):
    __tablename__ = "verified_claims"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), nullable=False, index=True)
    image_paths = Column(Text)
    user_claim = Column(Text, nullable=False)
    claim_object = Column(String(20), nullable=False)
    evidence_standard_met = Column(Boolean)
    evidence_standard_met_reason = Column(Text)
    risk_flags = Column(Text)
    issue_type = Column(String(50))
    object_part = Column(String(50))
    claim_status = Column(String(30))
    claim_status_justification = Column(Text)
    supporting_image_ids = Column(String(100))
    valid_image = Column(Boolean)
    severity = Column(String(20))
    verified_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def run_local_text_validation(text_claim: str) -> str:
    if not gate_model or not gate_tokenizer:
        return "Custom validation model unavailable"
    try:
        inputs = gate_tokenizer(text_claim, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = gate_model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        supported_score = probabilities[0][1].item()
        if supported_score > 0.70:
            return f"High probability of a valid claim layout ({supported_score*100:.1f}% confidence)"
        elif supported_score < 0.35:
            return f"High risk pattern detected by internal text gate ({ (1 - supported_score)*100:.1f}% risk score)"
        else:
            return f"Ambiguous text pattern structure ({supported_score*100:.1f}% validity confidence)"
    except Exception as e:
        return f"Error executing local validation gate: {str(e)}"

def save_to_database(user_id: str, image_paths: str, user_claim: str, claim_object: str, analysis: dict):
    db = SessionLocal()
    try:
        db_record = VerifiedClaim(
            user_id=user_id, image_paths=image_paths, user_claim=user_claim, claim_object=claim_object,
            evidence_standard_met=analysis.get("evidence_standard_met"),
            evidence_standard_met_reason=analysis.get("evidence_standard_met_reason"),
            risk_flags=analysis.get("risk_flags"), issue_type=analysis.get("issue_type"),
            object_part=analysis.get("object_part"), claim_status=analysis.get("claim_status"),
            claim_status_justification=analysis.get("claim_status_justification"),
            supporting_image_ids=analysis.get("supporting_image_ids"),
            valid_image=analysis.get("valid_image"), severity=analysis.get("severity")
        )
        db.add(db_record)
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()

def load_user_history(user_id: str) -> dict:
    try:
        df = pd.read_csv(DATASET_DIR / "user_history.csv")
        row = df[df["user_id"] == user_id]
        if row.empty:
            return {"past_claim_count": 0, "accept_claim": 0, "manual_review_claim": 0, "rejected_claim": 0, "last_90_days_claim_count": 0, "history_flags": "none", "history_summary": "No prior history found"}
        return row.iloc[0].to_dict()
    except Exception:
        return {"history_flags": "none", "history_summary": "History unavailable"}

def load_evidence_requirements(claim_object: str, issue_hint: str = "") -> list:
    try:
        df = pd.read_csv(DATASET_DIR / "evidence_requirements.csv")
        relevant = df[(df["claim_object"] == claim_object) | (df["claim_object"] == "all")]
        return relevant["minimum_image_evidence"].tolist()
    except Exception:
        return ["At least one clear image of the damaged area is required"]

def encode_image_to_base64(image_path: str) -> Optional[tuple]:
    full_path = BASE_DIR / image_path
    if not full_path.exists(): return None
    ext = full_path.suffix.lower()
    media_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    return base64.standard_b64encode(open(full_path, "rb").read()).decode("utf-8"), media_map.get(ext, "image/jpeg")

def encode_uploaded_image(image_bytes: bytes, content_type: str) -> tuple:
    return base64.standard_b64encode(image_bytes).decode("utf-8"), content_type

def build_analysis_prompt(user_claim: str, claim_object: str, user_history: dict, evidence_requirements: list, gatekeeper_assessment: str) -> str:
    req_text = "\n".join(f"- {r}" for r in evidence_requirements)
    history_str = json.dumps(user_history, indent=2)
    return f"""You are an expert damage claim verification AI. Analyze the submitted images carefully and evaluate the damage claim below.
- Object Type: {claim_object}
- User Claim: {user_claim}
- Internal Text Model Verdict: {gatekeeper_assessment}
{history_str}
{req_text}
Return JSON object:
{{
  "evidence_standard_met": true or false,
  "evidence_standard_met_reason": "string",
  "risk_flags": "string",
  "issue_type": "string",
  "object_part": "string",
  "claim_status": "supported/contradicted/not_enough_information",
  "claim_status_justification": "string",
  "supporting_image_ids": "string",
  "valid_image": true or false,
  "severity": "none/low/medium/high/unknown"
}}"""

async def analyze_claim_with_gemini(user_claim, claim_object, image_data_list, user_id):
    user_history = load_user_history(user_id)
    evidence_requirements = load_evidence_requirements(claim_object)
    gatekeeper_assessment = run_local_text_validation(user_claim)
    prompt = build_analysis_prompt(user_claim, claim_object, user_history, evidence_requirements, gatekeeper_assessment)
    model = genai.GenerativeModel("gemini-3.1-flash-lite")
    content = []
    for b64, media_type, img_id in image_data_list:
        img_bytes = base64.b64decode(b64)
        pil_img = PIL.Image.open(io.BytesIO(img_bytes))
        content.append(pil_img)
        content.append(f"[Image ID: {img_id}]")
    content.append(prompt)
    response = await asyncio.to_thread(model.generate_content, content)
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()
    result = json.loads(raw)
    user_history_data = load_user_history(user_id)
    rejected = int(user_history_data.get("rejected_claim", 0))
    recent = int(user_history_data.get("last_90_days_claim_count", 0))
    current_flags = result.get("risk_flags", "none")
    if "high risk pattern" in gatekeeper_assessment.lower() and "user_history_risk" not in current_flags:
        result["risk_flags"] = "user_history_risk" if current_flags == "none" else current_flags + ";user_history_risk"
    elif (rejected > 2 or recent > 3) and "user_history_risk" not in current_flags:
        result["risk_flags"] = "user_history_risk" if current_flags == "none" else current_flags + ";user_history_risk"
    return result

@app.get("/")
def root(): return {"status": "running"}

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/api/analyze-claim")
async def analyze_claim(user_id: str = Form(...), user_claim: str = Form(...), claim_object: str = Form(...), images: list[UploadFile] = File(...)):
    if claim_object not in ("car", "laptop", "package"): raise HTTPException(status_code=400)
    image_data_list, image_paths_list = [], []
    for i, img_file in enumerate(images):
        image_bytes = await img_file.read()
        b64, media_type = encode_uploaded_image(image_bytes, img_file.content_type or "image/jpeg")
        img_id = f"img_{i+1}"
        image_data_list.append((b64, media_type, img_id))
        image_paths_list.append(img_file.filename or img_id)
    image_paths_str = ";".join(image_paths_list)
    analysis = await analyze_claim_with_gemini(user_claim, claim_object, image_data_list, user_id)
    save_to_database(user_id, image_paths_str, user_claim, claim_object, analysis)
    return {"user_id": user_id, "image_paths": image_paths_str, "user_claim": user_claim, "claim_object": claim_object, **analysis}

@app.post("/api/process-csv")
async def process_csv_batch(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_batch_processing)
    return {"status": "started"}

batch_progress = {"status": "idle", "processed": 0, "total": 0, "errors": []}

@app.get("/api/batch-status")
def batch_status(): return batch_progress

async def run_batch_processing():
    global batch_progress
    batch_progress = {"status": "running", "processed": 0, "total": 0, "errors": []}
    try:
        claims_df = pd.read_csv(DATASET_DIR / "claims.csv")
        batch_progress["total"] = len(claims_df)
        output_rows, output_columns = [], ["user_id", "image_paths", "user_claim", "claim_object", "evidence_standard_met", "evidence_standard_met_reason", "risk_flags", "issue_type", "object_part", "claim_status", "claim_status_justification", "supporting_image_ids", "valid_image", "severity"]
        for _, row in claims_df.iterrows():
            user_id, image_paths_str, user_claim, claim_object = str(row["user_id"]), str(row["image_paths"]), str(row["user_claim"]), str(row["claim_object"])
            image_data_list = []
            img_paths = [p.strip() for p in image_paths_str.split(";")]
            for i, img_path in enumerate(img_paths):
                result = encode_image_to_base64(img_path)
                if result:
                    b64, media_type = result
                    image_data_list.append((b64, media_type, Path(img_path).stem))
            if not image_data_list:
                output_analysis = {"evidence_standard_met": False, "evidence_standard_met_reason": "No valid images found", "risk_flags": "damage_not_visible", "issue_type": "unknown", "object_part": "unknown", "claim_status": "not_enough_information", "claim_status_justification": "No images", "supporting_image_ids": "none", "valid_image": False, "severity": "unknown"}
                output_rows.append({"user_id": user_id, "image_paths": image_paths_str, "user_claim": user_claim, "claim_object": claim_object, **output_analysis})
                save_to_database(user_id, image_paths_str, user_claim, claim_object, output_analysis)
            else:
                analysis = await analyze_claim_with_gemini(user_claim, claim_object, image_data_list, user_id)
                output_rows.append({"user_id": user_id, "image_paths": image_paths_str, "user_claim": user_claim, "claim_object": claim_object, **analysis})
                save_to_database(user_id, image_paths_str, user_claim, claim_object, analysis)
            batch_progress["processed"] += 1
            await asyncio.sleep(1)
        pd.DataFrame(output_rows, columns=output_columns).to_csv(OUTPUT_FILE, index=False)
        batch_progress["status"] = "complete"
    except Exception as e:
        batch_progress["status"] = "error"

@app.get("/api/download-output")
def download_output():
    return FileResponse(OUTPUT_FILE, media_type="text/csv", filename="output.csv")

@app.get("/api/sample-claims")
def get_sample_claims():
    return pd.read_csv(DATASET_DIR / "sample_claims.csv").to_dict(orient="records")

@app.get("/api/user-history/{user_id}")
def get_user_history(user_id: str): return load_user_history(user_id)