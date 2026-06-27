"""
Evaluation script for the Damage Claim AI Agent.
Runs predictions against sample_claims.csv and computes accuracy metrics.
"""

import os
import sys
import json
import base64
import time
import asyncio
from pathlib import Path

import pandas as pd
import anthropic
from dotenv import load_dotenv

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Metrics tracking
call_count = 0
total_input_tokens = 0
total_output_tokens = 0
images_processed = 0
start_time = None


def load_user_history(user_id: str) -> dict:
    try:
        df = pd.read_csv(DATASET_DIR / "user_history.csv")
        row = df[df["user_id"] == user_id]
        if row.empty:
            return {"history_flags": "none", "history_summary": "No prior history"}
        return row.iloc[0].to_dict()
    except Exception:
        return {"history_flags": "none", "history_summary": "History unavailable"}


def load_evidence_requirements(claim_object: str) -> list:
    try:
        df = pd.read_csv(DATASET_DIR / "evidence_requirements.csv")
        relevant = df[(df["claim_object"] == claim_object) | (df["claim_object"] == "all")]
        return relevant["minimum_image_evidence"].tolist()
    except Exception:
        return ["At least one clear image of the damaged area is required"]


def encode_image(image_path: str):
    full_path = BASE_DIR / image_path
    if not full_path.exists():
        return None
    ext = full_path.suffix.lower()
    media_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    media_type = media_map.get(ext, "image/jpeg")
    with open(full_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8"), media_type


def build_prompt(user_claim: str, claim_object: str, user_history: dict, evidence_requirements: list) -> str:
    req_text = "\n".join(f"- {r}" for r in evidence_requirements)
    return f"""You are an expert damage claim verification AI. Analyze the submitted images carefully.

## Claim
- Object Type: {claim_object}
- User Claim: {user_claim}

## User History
{json.dumps(user_history, indent=2)}

## Evidence Requirements
{req_text}

Return ONLY a JSON object with these fields:
{{
  "evidence_standard_met": true/false,
  "evidence_standard_met_reason": "reason",
  "risk_flags": "none or semicolon-separated flags",
  "issue_type": "dent/scratch/crack/glass_shatter/broken_part/missing_part/torn_packaging/crushed_packaging/water_damage/stain/none/unknown",
  "object_part": "relevant part name or unknown",
  "claim_status": "supported/contradicted/not_enough_information",
  "claim_status_justification": "image-grounded explanation with image IDs",
  "supporting_image_ids": "img_1;img_2 or none",
  "valid_image": true/false,
  "severity": "none/low/medium/high/unknown"
}}

No markdown, no extra text. Only valid JSON."""


def analyze_single_claim(row: dict) -> dict:
    global call_count, total_input_tokens, total_output_tokens, images_processed

    user_id = str(row["user_id"])
    image_paths_str = str(row["image_paths"])
    user_claim = str(row["user_claim"])
    claim_object = str(row["claim_object"])

    user_history = load_user_history(user_id)
    evidence_requirements = load_evidence_requirements(claim_object)
    prompt = build_prompt(user_claim, claim_object, user_history, evidence_requirements)

    img_paths = [p.strip() for p in image_paths_str.split(";")]
    content = []

    for i, img_path in enumerate(img_paths):
        result = encode_image(img_path)
        if result:
            b64, media_type = result
            img_id = Path(img_path).stem
            content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}})
            content.append({"type": "text", "text": f"[Image ID: {img_id}]"})
            images_processed += 1

    if not content:
        return {
            "evidence_standard_met": False,
            "evidence_standard_met_reason": "No valid images found",
            "risk_flags": "damage_not_visible",
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "not_enough_information",
            "claim_status_justification": "No images could be loaded for review",
            "supporting_image_ids": "none",
            "valid_image": False,
            "severity": "unknown"
        }

    content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": content}]
    )

    call_count += 1
    total_input_tokens += response.usage.input_tokens
    total_output_tokens += response.usage.output_tokens

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    result = json.loads(raw)

    # Apply history risk
    rejected = int(user_history.get("rejected_claim", 0))
    recent = int(user_history.get("last_90_days_claim_count", 0))
    current_flags = result.get("risk_flags", "none")
    if (rejected > 2 or recent > 3) and "user_history_risk" not in current_flags:
        result["risk_flags"] = "user_history_risk" if current_flags == "none" else current_flags + ";user_history_risk"

    return result


def compute_metrics(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> dict:
    """Compare predictions vs ground truth on key fields."""
    metrics = {}
    key_fields = ["claim_status", "issue_type", "object_part", "severity", "evidence_standard_met", "valid_image"]

    for field in key_fields:
        if field in predictions_df.columns and field in ground_truth_df.columns:
            pred = predictions_df[field].astype(str).str.strip().str.lower()
            truth = ground_truth_df[field].astype(str).str.strip().str.lower()
            matches = (pred == truth).sum()
            total = len(pred)
            metrics[field] = {
                "correct": int(matches),
                "total": total,
                "accuracy": round(matches / total * 100, 1) if total > 0 else 0
            }

    return metrics


def run_evaluation():
    global start_time
    start_time = time.time()

    print("=" * 60)
    print("DAMAGE CLAIM AI AGENT - EVALUATION")
    print("=" * 60)

    sample_df = pd.read_csv(DATASET_DIR / "sample_claims.csv")

    # Only use input columns
    input_cols = ["user_id", "image_paths", "user_claim", "claim_object"]
    input_df = sample_df[input_cols].copy()

    output_columns = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason",
        "risk_flags", "issue_type", "object_part", "claim_status",
        "claim_status_justification", "supporting_image_ids",
        "valid_image", "severity"
    ]

    predictions = []
    print(f"\nProcessing {len(input_df)} sample claims...\n")

    for idx, row in input_df.iterrows():
        print(f"  [{idx+1}/{len(input_df)}] Analyzing claim for user {row['user_id']}...")
        try:
            result = analyze_single_claim(row.to_dict())
            pred_row = {
                "user_id": row["user_id"],
                "image_paths": row["image_paths"],
                "user_claim": row["user_claim"],
                "claim_object": row["claim_object"],
                **result
            }
            predictions.append(pred_row)
            print(f"    → Status: {result.get('claim_status')} | Severity: {result.get('severity')}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            predictions.append({
                "user_id": row["user_id"],
                "image_paths": row["image_paths"],
                "user_claim": row["user_claim"],
                "claim_object": row["claim_object"],
                "evidence_standard_met": False,
                "evidence_standard_met_reason": "Error during evaluation",
                "risk_flags": "manual_review_required",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": str(e),
                "supporting_image_ids": "none",
                "valid_image": False,
                "severity": "unknown"
            })
        time.sleep(0.5)  # Rate limiting

    predictions_df = pd.DataFrame(predictions, columns=output_columns)
    predictions_df.to_csv(BASE_DIR / "evaluation" / "sample_predictions.csv", index=False)

    elapsed = time.time() - start_time

    # Compute metrics
    print("\n" + "=" * 60)
    print("METRICS (Predictions vs Ground Truth)")
    print("=" * 60)

    metrics = compute_metrics(predictions_df, sample_df)
    for field, vals in metrics.items():
        print(f"  {field}: {vals['correct']}/{vals['total']} correct ({vals['accuracy']}%)")

    # Write evaluation report
    report = f"""# Damage Claim AI Agent - Evaluation Report

## System Summary
Multi-modal AI agent using Claude claude-sonnet-4-6 (Vision) to verify damage claims from images.

## Processing Metrics (Sample Set - {len(input_df)} claims)

| Metric | Value |
|--------|-------|
| Total model calls | {call_count} |
| Images processed | {images_processed} |
| Total input tokens | {total_input_tokens:,} |
| Total output tokens | {total_output_tokens:,} |
| Elapsed time | {elapsed:.1f} seconds |
| Avg latency per call | {elapsed/max(call_count,1):.1f} seconds |

## Accuracy Metrics

| Field | Correct | Total | Accuracy |
|-------|---------|-------|----------|
{"".join(f"| {f} | {v['correct']} | {v['total']} | {v['accuracy']}% |" + chr(10) for f, v in metrics.items())}

## Cost Estimation (Full Test Set)

**Pricing Assumptions (claude-sonnet-4-6 as of 2025):**
- Input tokens: $3.00 per 1M tokens  
- Output tokens: $15.00 per 1M tokens
- Images: ~1,600 tokens per image (base), more for detailed images

**Sample Set Observed:**
- Avg input tokens/call: {total_input_tokens // max(call_count,1):,}
- Avg output tokens/call: {total_output_tokens // max(call_count,1):,}

**Projected for Full Test Set (assume 50 claims, avg 2 images each):**
- Estimated calls: ~50
- Estimated input tokens: ~{(total_input_tokens // max(call_count,1)) * 50:,}
- Estimated output tokens: ~{(total_output_tokens // max(call_count,1)) * 50:,}
- **Estimated cost: ~${((total_input_tokens // max(call_count,1)) * 50 / 1_000_000 * 3) + ((total_output_tokens // max(call_count,1)) * 50 / 1_000_000 * 15):.2f} USD**

## TPM / RPM Considerations

- **claude-sonnet-4-6 default limits**: ~4,000 RPM, 400K TPM (varies by tier)
- **Strategy used**: Sequential processing with 0.5s delay between calls
- **Batching**: One claim per call (images + prompt bundled together)
- **No caching applied** for images (images change per claim); prompt structure is stable
- **Retry strategy**: None implemented (add exponential backoff for production)
- **Recommendation for large test sets**: Use asyncio with semaphore limiting to 5 concurrent requests

## Operational Notes

1. All images encoded to base64 per call — no cross-call caching possible for unique images
2. User history and evidence requirements are loaded from CSV (fast, no extra API calls)
3. System prompt is stable across claims; token overhead is ~500 tokens per call
4. Main cost driver is image encoding: each image adds ~1,000–6,000 tokens depending on resolution
5. For production: resize images to max 1024px before encoding to reduce token usage by ~40%
"""

    eval_report_path = BASE_DIR / "evaluation" / "evaluation_report.md"
    with open(eval_report_path, "w") as f:
        f.write(report)

    print(f"\n✓ Sample predictions saved to evaluation/sample_predictions.csv")
    print(f"✓ Evaluation report saved to evaluation/evaluation_report.md")
    print(f"\nTotal API calls: {call_count}")
    print(f"Total images processed: {images_processed}")
    print(f"Total tokens used: {total_input_tokens + total_output_tokens:,}")
    print(f"Time elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    run_evaluation()
