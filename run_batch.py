import os
import json
import time
from pathlib import Path
import concurrent.futures
import pandas as pd
import PIL.Image
import google.generativeai as genai
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_FILE = BASE_DIR / "output.csv"

load_dotenv(BASE_DIR / "backend" / ".env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

DATABASE_URL = f"sqlite:///{BASE_DIR}/backend/claims.db"
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

USER_HISTORY_CACHE = {}
EVIDENCE_REQS_CACHE = {}

def warm_up_caches():
    global USER_HISTORY_CACHE, EVIDENCE_REQS_CACHE
    try:
        uh_path = DATASET_DIR / "user_history.csv"
        if uh_path.exists():
            uh_df = pd.read_csv(uh_path)
            for _, row in uh_df.iterrows():
                USER_HISTORY_CACHE[str(row["user_id"])] = row.to_dict()
    except Exception:
        pass

    try:
        er_path = DATASET_DIR / "evidence_requirements.csv"
        if er_path.exists():
            er_df = pd.read_csv(er_path)
            for _, row in er_df.iterrows():
                obj = str(row["claim_object"])
                if obj not in EVIDENCE_REQS_CACHE:
                    EVIDENCE_REQS_CACHE[obj] = []
                EVIDENCE_REQS_CACHE[obj].append(str(row["minimum_image_evidence"]))
    except Exception:
        pass

def get_cached_user_history(user_id):
    return USER_HISTORY_CACHE.get(str(user_id), {"history_flags": "none", "history_summary": "No prior history"})

def get_cached_evidence_requirements(claim_object):
    reqs = EVIDENCE_REQS_CACHE.get(str(claim_object), [])
    reqs.extend(EVIDENCE_REQS_CACHE.get("all", []))
    if not reqs:
        return ["At least one clear image of the damaged area is required"]
    return reqs

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

def load_image_as_pil(image_path):
    full_path = BASE_DIR / image_path
    img_id = Path(image_path).name
    if not full_path.exists():
        return None, f"{img_id} (Status: File Missing)"
    try:
        pil_img = PIL.Image.open(full_path)
        pil_img.verify()
        pil_img = PIL.Image.open(full_path).convert("RGB")
        pil_img.thumbnail((512, 512), PIL.Image.Resampling.LANCZOS)
        return pil_img, img_id
    except Exception:
        return None, f"{img_id} (Status: Corrupted/Unreadable Image File)"

def build_prompt(user_claim, claim_object, user_history, evidence_reqs):
    req_text = "\n".join("- " + r for r in evidence_reqs)
    history_str = json.dumps(user_history)
    prompt = "You are an expert damage claim verification AI.\n\n"
    prompt += "Claim Object: " + claim_object + "\n"
    prompt += "User Claim: " + user_claim + "\n"
    prompt += "User History: " + history_str + "\n"
    prompt += "Evidence Requirements:\n" + req_text + "\n\n"
    prompt += "Return ONLY a valid JSON object with exactly these keys:\n"
    prompt += "evidence_standard_met (true/false)\n"
    prompt += "evidence_standard_met_reason (string)\n"
    prompt += "risk_flags (none or semicolon-separated options from: blurry_image, cropped_or_obstructed, wrong_angle, wrong_object, damage_not_visible, claim_mismatch, possible_manipulation, text_instruction_present, user_history_risk, manual_review_required)\n"
    prompt += "issue_type (dent/scratch/crack/glass_shatter/broken_part/missing_part/torn_packaging/crushed_packaging/water_damage/stain/none/unknown)\n"
    prompt += "object_part (front_bumper/rear_bumper/door/hood/windshield/screen/keyboard/box/package_corner/etc or unknown)\n"
    prompt += "claim_status (supported/contradicted/not_enough_information)\n"
    prompt += "claim_status_justification (string maximum 2 sentences referencing explicit filenames like img_1.jpg)\n"
    prompt += "supporting_image_ids (semicolon-separated filenames like img_1.jpg;img_2.jpg or none)\n"
    prompt += "valid_image (true/false)\n"
    prompt += "severity (none/low/medium/high/unknown)\n\n"
    prompt += "No markdown blocks. Only pure valid JSON text output."
    return prompt

def analyze_claim(row):
    user_id = str(row["user_id"])
    image_paths_str = str(row["image_paths"])
    user_claim = str(row["user_claim"])
    claim_object = str(row["claim_object"])

    user_history = get_cached_user_history(user_id)
    evidence_reqs = get_cached_evidence_requirements(claim_object)
    prompt = build_prompt(user_claim, claim_object, user_history, evidence_reqs)

    img_paths = [p.strip() for p in image_paths_str.split(";")]
    content = []

    for img_path in img_paths:
        pil_img, img_id = load_image_as_pil(img_path)
        if pil_img is not None:
            content.append(pil_img)
            content.append(f"[Image Filename: {img_id}]")
        else:
            content.append(f"[Image Filename: {img_id} - ALERT: This image file is corrupted, unreadable, or missing on disk.]")

    if not any(isinstance(item, PIL.Image.Image) for item in content):
        content.append("[CRITICAL NOTICE: All submitted images for this case are corrupted or missing. Evaluate solely based on text claims and history flags.]")

    content.append(prompt)

    model = genai.GenerativeModel("gemini-3.1-flash-lite")
    response = model.generate_content(content)

    raw = response.text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()

    result = json.loads(raw)

    rejected = int(user_history.get("rejected_claim", 0))
    recent = int(user_history.get("last_90_days_claim_count", 0))
    current_flags = result.get("risk_flags", "none")
    if (rejected > 2 or recent > 3) and "user_history_risk" not in current_flags:
        result["risk_flags"] = "user_history_risk" if current_flags == "none" else current_flags + ";user_history_risk"

    return result

def worker_task(idx, row, total):
    row_dict = row.to_dict()
    user_id = row_dict["user_id"]
    try:
        analysis = analyze_claim(row_dict)
        save_to_database(str(user_id), str(row_dict["image_paths"]), str(row_dict["user_claim"]), str(row_dict["claim_object"]), analysis)
        return idx, {
            "user_id": user_id, "image_paths": row_dict["image_paths"],
            "user_claim": row_dict["user_claim"], "claim_object": row_dict["claim_object"],
            **analysis
        }, f"OK -> [{analysis.get('claim_status')}]"
    except Exception as e:
        error_analysis = {"evidence_standard_met": False, "evidence_standard_met_reason": "Execution failure", "risk_flags": "manual_review_required", "issue_type": "unknown", "object_part": "unknown", "claim_status": "not_enough_information", "claim_status_justification": f"Error: {e}", "supporting_image_ids": "none", "valid_image": False, "severity": "unknown"}
        return idx, {
            "user_id": user_id, "image_paths": row_dict["image_paths"],
            "user_claim": row_dict["user_claim"], "claim_object": row_dict["claim_object"],
            **error_analysis
        }, f"ERROR -> {e}"

def main():
    warm_up_caches()
    claims_df = pd.read_csv(DATASET_DIR / "claims.csv")
    total = len(claims_df)
    output_columns = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason",
        "risk_flags", "issue_type", "object_part", "claim_status",
        "claim_status_justification", "supporting_image_ids",
        "valid_image", "severity"
    ]
    results_map = [None] * total
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_row = {
            executor.submit(worker_task, idx, row, total): idx 
            for idx, row in claims_df.iterrows()
        }
        for future in concurrent.futures.as_completed(future_to_row):
            idx, row_result, log_msg = future.result()
            results_map[idx] = row_result
            time.sleep(0.2)

    results = [r for r in results_map if r is not None]
    output_df = pd.DataFrame(results, columns=output_columns)
    output_df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()