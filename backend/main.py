import os
import io
import PIL.Image
import json
import base64
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# Modern Unified Google GenAI SDK Import
from google import genai
from google.genai import types

BACKEND_DIR = Path(__file__).resolve().parent
ENV_PATH = BACKEND_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

app = FastAPI(title="Damage Claim AI Agent Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Modern Google GenAI Client
ai_client = genai.Client()

BASE_DIR = BACKEND_DIR.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_FILE = BASE_DIR / "output.csv"
FRONTEND_DIR = BASE_DIR / "frontend"

# Lightweight fallback structure for the cloud tier since torch/transformers are omitted
gate_tokenizer, gate_model = None, None

# --- Dynamic Permanent Cloud Database Configuration ---
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    DATABASE_URL = f"sqlite:///{BACKEND_DIR}/claims.db"
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
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
    
    # 🔥 New Anti-Fraud Ownership Columns
    registered_identifier_input = Column(String(100), nullable=True)
    extracted_ownership_token = Column(String(100), nullable=True)
    ownership_verified = Column(Boolean, default=False)
    
    verified_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- Frontend Static Files Configuration ---
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

def run_local_text_validation(text_claim: str) -> str:
    return "Local validation model skipped on cloud deployment architecture."

def save_to_database(user_id: str, image_paths: str, user_claim: str, claim_object: str, registered_id: str, analysis: dict):
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
            valid_image=analysis.get("valid_image"), severity=analysis.get("severity"),
            # 🔥 Populating ownership parameters into persistent storage
            registered_identifier_input=registered_id,
            extracted_ownership_token=analysis.get("extracted_ownership_token"),
            ownership_verified=analysis.get("ownership_verified", False)
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

def build_analysis_prompt(user_claim: str, claim_object: str, user_history: dict, evidence_requirements: list, gatekeeper_assessment: str, registered_id: str) -> str:
    req_text = "\n".join(f"- {r}" for r in evidence_requirements)
    history_str = json.dumps(user_history, indent=2)
    
    # Dynamic prompt modification based on asset class target
    token_instruction = ""
    if claim_object == "car":
        token_instruction = f"Locate the car's license plate in the images. Extract the plate number text and cross-check if it perfectly matches the user's expected plate: '{registered_id}'."
    elif claim_object == "laptop":
        token_instruction = f"Locate the laptop's serial number barcode sticker or text panel. Extract the alphanumeric serial value and check if it matches: '{registered_id}'."
    elif claim_object == "package":
        token_instruction = f"Locate the shipping label/slip pasted on or inside the box. Extract the Tracking Number/Barcode ID string and verify against: '{registered_id}'."

    return f"""You are an expert damage claim verification and fraud detection AI. Analyze the submitted images carefully. 
One of the images contains an asset ownership token proof (License plate, serial sticker, or delivery slip). 

Your tasks:
1. Extract the asset identifier text value from that token image.
2. Cross-verify it directly against the User's Registered Identifier provided below.
3. Assess the structural damage claimed by the user.

- Object Type: {claim_object}
- User Claim: {user_claim}
- User's Registered Identifier Input: {registered_id}
- Internal Text Model Verdict: {gatekeeper_assessment}

Ownership Token Verification Directive:
{token_instruction}

{history_str}
{req_text}

Return a valid raw JSON object matching this structure perfectly:
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
  "severity": "none/low/medium/high/unknown",
  "extracted_ownership_token": "string containing extracted plate/serial/tracking text",
  "ownership_verified": true or false based on cross-check match
}}"""

async def analyze_claim_with_gemini(user_claim, claim_object, image_data_list, user_id, registered_id):
    user_history = load_user_history(user_id)
    evidence_requirements = load_evidence_requirements(claim_object)
    gatekeeper_assessment = run_local_text_validation(user_claim)
    prompt = build_analysis_prompt(user_claim, claim_object, user_history, evidence_requirements, gatekeeper_assessment, registered_id)
    
    content = []
    for b64, media_type, img_id in image_data_list:
        img_bytes = base64.b64decode(b64)
        pil_img = PIL.Image.open(io.BytesIO(img_bytes))
        content.append(pil_img)
        content.append(f"[Image ID: {img_id}]")
    content.append(prompt)
    
    response = await asyncio.to_thread(
        ai_client.models.generate_content,
        model="gemini-2.5-flash",
        contents=content
    )
    
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()
    result = json.loads(raw)
    
    # Internal Risk Flag Pipeline Calculations
    user_history_data = load_user_history(user_id)
    rejected = int(user_history_data.get("rejected_claim", 0))
    recent = int(user_history_data.get("last_90_days_claim_count", 0))
    current_flags = result.get("risk_flags", "none")
    
    # Force a risk flag append if ownership verification fails to match
    if not result.get("ownership_verified", False):
        current_flags = "ownership_mismatch" if current_flags == "none" else current_flags + ";ownership_mismatch"
        result["claim_status"] = "contradicted"
        result["risk_flags"] = current_flags

    if "high risk pattern" in gatekeeper_assessment.lower() and "user_history_risk" not in current_flags:
        result["risk_flags"] = "user_history_risk" if current_flags == "none" else current_flags + ";user_history_risk"
    elif (rejected > 2 or recent > 3) and "user_history_risk" not in current_flags:
        result["risk_flags"] = "user_history_risk" if current_flags == "none" else current_flags + ";user_history_risk"
    return result

@app.get("/")
def root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"status": "running", "message": "Frontend interface folder directory not detected."}

@app.get("/health")
def health(): return {"status": "ok"}
def inspect_image_metadata(image_bytes: bytes) -> dict:
    """
    Inspects image binary array for authentic camera hardware EXIF tags.
    Returns analysis flags for fraud checking.
    """
    from PIL import Image
    from PIL.ExifTags import TAGS
    
    analysis = {
        "hardware_signature_found": False,
        "camera_make": "Unknown",
        "camera_model": "Unknown",
        "potential_synthetic_or_screenshot": True
    }
    
    try:
        # Load the image array from raw stream bytes
        img = Image.open(io.BytesIO(image_bytes))
        exif_data = img._getexif()
        
        if exif_data:
            readable_exif = {TAGS.get(key, key): val for key, val in exif_data.items()}
            
            # Extract standard manufacturer attributes
            make = str(readable_exif.get("Make", "")).strip()
            model = str(readable_exif.get("Model", "")).strip()
            
            if make or model:
                analysis["hardware_signature_found"] = True
                analysis["camera_make"] = make if make else "Generic"
                analysis["camera_model"] = model if model else "Generic Sensor"
                analysis["potential_synthetic_or_screenshot"] = False
    except Exception:
        # Fallback if image type doesn't support EXIF (like raw PNG drops)
        pass
        
    return analysis
@app.post("/api/analyze-claim")
async def analyze_claim(
    user_id: str = Form(...), 
    user_claim: str = Form(...), 
    claim_object: str = Form(...), 
    registered_id: str = Form("unknown"),
    images: list[UploadFile] = File(...)
):
    if claim_object not in ("car", "laptop", "package"): raise HTTPException(status_code=400)
    
    image_data_list, image_paths_list = [], []
    metadata_fraud_detected = False
    
    for i, img_file in enumerate(images):
        image_bytes = await img_file.read()
        
        # 🔥 RUN THE NEW DIGITAL FORENSICS METADATA CHECK HERE
        meta_verdict = inspect_image_metadata(image_bytes)
        if meta_verdict["potential_synthetic_or_screenshot"]:
            metadata_fraud_detected = True
            
        b64, media_type = encode_uploaded_image(image_bytes, img_file.content_type or "image/jpeg")
        img_id = f"img_{i+1}"
        image_data_list.append((b64, media_type, img_id))
        image_paths_list.append(img_file.filename or img_id)
        
    image_paths_str = ";".join(image_paths_list)
    
    # Analyze through Gemini engine
    analysis = await analyze_claim_with_gemini(user_claim, claim_object, image_data_list, user_id, registered_id)
    
    # 🔥 INJECT THE RISK FLAG IF METADATA IS MISSING
    if metadata_fraud_detected:
        current_flags = analysis.get("risk_flags", "none")
        if current_flags == "none":
            analysis["risk_flags"] = "missing_hardware_metadata"
        elif "missing_hardware_metadata" not in current_flags:
            analysis["risk_flags"] = current_flags + ";missing_hardware_metadata"
            
        # Optional: Force review or downgrade status for lacking native camera source
        if analysis["claim_status"] == "supported":
            analysis["claim_status_justification"] = (
                "[ALERT: Image lacks native hardware metadata signature] " + 
                analysis["claim_status_justification"]
            )
    
    save_to_database(user_id, image_paths_str, user_claim, claim_object, registered_id, analysis)
    return {"user_id": user_id, "image_paths": image_paths_str, "user_claim": user_claim, "claim_object": claim_object, "registered_identifier_input": registered_id, **analysis}
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
        output_rows, output_columns = [], ["user_id", "image_paths", "user_claim", "claim_object", "evidence_standard_met", "evidence_standard_met_reason", "risk_flags", "issue_type", "object_part", "claim_status", "claim_status_justification", "supporting_image_ids", "valid_image", "severity", "extracted_ownership_token", "ownership_verified"]
        for _, row in claims_df.iterrows():
            user_id, image_paths_str, user_claim, claim_object = str(row["user_id"]), str(row["image_paths"]), str(row["user_claim"]), str(row["claim_object"])
            reg_id = str(row.get("registered_id", "unknown"))
            image_data_list = []
            img_paths = [p.strip() for p in image_paths_str.split(";")]
            for i, img_path in enumerate(img_paths):
                result = encode_image_to_base64(img_path)
                if result:
                    b64, media_type = result
                    image_data_list.append((b64, media_type, Path(img_path).stem))
            if not image_data_list:
                output_analysis = {"evidence_standard_met": False, "evidence_standard_met_reason": "No valid images found", "risk_flags": "damage_not_visible", "issue_type": "unknown", "object_part": "unknown", "claim_status": "not_enough_information", "claim_status_justification": "No images", "supporting_image_ids": "none", "valid_image": False, "severity": "unknown", "extracted_ownership_token": "none", "ownership_verified": False}
                output_rows.append({"user_id": user_id, "image_paths": image_paths_str, "user_claim": user_claim, "claim_object": claim_object, **output_analysis})
                save_to_database(user_id, image_paths_str, user_claim, claim_object, reg_id, output_analysis)
            else:
                analysis = await analyze_claim_with_gemini(user_claim, claim_object, image_data_list, user_id, reg_id)
                output_rows.append({"user_id": user_id, "image_paths": image_paths_str, "user_claim": user_claim, "claim_object": claim_object, **analysis})
                save_to_database(user_id, image_paths_str, user_claim, claim_object, reg_id, analysis)
            batch_progress["processed"] += 1
            await asyncio.sleep(1)
        pd.DataFrame(output_rows, columns=output_columns).to_csv(OUTPUT_FILE, index=False)
        batch_progress["status"] = "complete"
    except Exception as e:
        batch_progress["status"] = "error"

@app.get("/api/download-output")
def download_output():
    if not OUTPUT_FILE.exists():
        output_columns = ["user_id", "image_paths", "user_claim", "claim_object", "evidence_standard_met", "evidence_standard_met_reason", "risk_flags", "issue_type", "object_part", "claim_status", "claim_status_justification", "supporting_image_ids", "valid_image", "severity", "extracted_ownership_token", "ownership_verified"]
        pd.DataFrame(columns=output_columns).to_csv(OUTPUT_FILE, index=False)
        
    return FileResponse(OUTPUT_FILE, media_type="text/csv", filename="output.csv")

@app.get("/api/sample-claims")
def get_sample_claims():
    return pd.read_csv(DATASET_DIR / "sample_claims.csv").to_dict(orient="records")

@app.get("/api/user-history/{user_id}")
def get_user_history(user_id: str): return load_user_history(user_id)