# 🔍 ClaimVision — AI Damage Claim Verification Agent

A full-stack AI agent that verifies damage claims (car, laptop, package) using multi-modal vision analysis via Claude claude-sonnet-4-6. Accepts images + claim text, returns a structured verdict with issue type, severity, risk flags, and supporting evidence.

---

## 📁 Project Structure

```
damage-claim-agent/
├── backend/
│   ├── main.py              # FastAPI backend (REST API)
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # API key (you create this)
├── frontend/
│   └── index.html           # Single-page frontend (no build step)
├── evaluation/
│   ├── evaluate.py          # Runs sample_claims.csv evaluation
│   └── evaluation_report.md # Auto-generated metrics report
├── dataset/
│   ├── claims.csv           # Test claims (input)
│   ├── sample_claims.csv    # Labeled examples (for evaluation)
│   ├── user_history.csv     # User risk history
│   ├── evidence_requirements.csv
│   └── images/
│       ├── sample/          # Images for sample_claims.csv
│       └── test/            # Images for claims.csv
├── run_batch.py             # Standalone batch processor
├── output.csv               # Generated predictions (after batch run)
└── README.md
```

---

## ⚡ Quick Start (5 Steps)

### Step 1 — Get your Anthropic API Key
1. Go to https://console.anthropic.com
2. Create an account and add credits (minimum $5)
3. Navigate to **API Keys** and create a new key
4. Copy the key (starts with `sk-ant-...`)

### Step 2 — Set up the Backend

```bash
# Navigate to backend folder
cd damage-claim-agent/backend

# Create virtual environment
python -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API key
echo "ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE" > .env
```

### Step 3 — Start the Backend Server

```bash
# From the backend/ folder (with venv activated)
uvicorn main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Test it works: open http://localhost:8000 in your browser.
API docs are at: http://localhost:8000/docs

### Step 4 — Open the Frontend

Simply open `frontend/index.html` in your browser:

```bash
# Mac
open frontend/index.html

# Linux
xdg-open frontend/index.html

# Windows
start frontend/index.html
```

Or serve it with Python:
```bash
cd frontend
python -m http.server 3000
# Then visit http://localhost:3000
```

### Step 5 — Add Your Images and Run

**For single claims (via UI):**
1. Enter a User ID (e.g., `U101`)
2. Select object type
3. Describe the damage
4. Upload 1–10 images
5. Click "Analyze Claim"

**For batch processing (claims.csv → output.csv):**
1. Place your images in `dataset/images/test/` matching the paths in `claims.csv`
2. Either click "Start Batch" in the UI, or run:
   ```bash
   cd damage-claim-agent
   python run_batch.py
   ```
3. Results are saved to `output.csv`

---

## 🧪 Running Evaluation

```bash
cd damage-claim-agent
python evaluation/evaluate.py
```

This will:
- Process all rows in `dataset/sample_claims.csv`
- Compare predictions to ground truth labels
- Print accuracy metrics
- Save `evaluation/sample_predictions.csv`
- Update `evaluation/evaluation_report.md`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Health status |
| POST | `/api/analyze-claim` | Analyze single claim (multipart form) |
| POST | `/api/process-csv` | Start batch CSV processing |
| GET | `/api/batch-status` | Check batch progress |
| GET | `/api/download-output` | Download output.csv |
| GET | `/api/sample-claims` | View sample claims |
| GET | `/api/user-history/{user_id}` | Get user history |

---

## 🖼️ Image Folder Structure

```
dataset/images/
├── sample/
│   ├── case_001/
│   │   ├── img_1.jpg
│   │   └── img_2.jpg
│   ├── case_002/
│   │   └── img_1.jpg
│   └── ...
└── test/
    ├── case_001/
    │   ├── img_1.jpg
    │   └── img_2.jpg
    └── ...
```

Image paths in CSV files use relative paths like: `images/test/case_001/img_1.jpg`

---

## 📊 Output Schema

Each row in `output.csv` contains:

| Column | Description |
|--------|-------------|
| `user_id` | User identifier |
| `image_paths` | Semicolon-separated image paths |
| `user_claim` | Original claim text |
| `claim_object` | car / laptop / package |
| `evidence_standard_met` | true/false |
| `evidence_standard_met_reason` | Why evidence standard was/wasn't met |
| `risk_flags` | Semicolon-separated flags or "none" |
| `issue_type` | dent/scratch/crack/etc. |
| `object_part` | Specific part affected |
| `claim_status` | supported / contradicted / not_enough_information |
| `claim_status_justification` | Image-grounded explanation |
| `supporting_image_ids` | img_1;img_2 or none |
| `valid_image` | true/false |
| `severity` | none/low/medium/high/unknown |

---

## 🔧 Troubleshooting

**"CORS error" in browser:**
- Make sure the FastAPI server is running on port 8000
- Try opening the frontend via `python -m http.server 3000` instead of file://

**"ANTHROPIC_API_KEY not found":**
- Check your `backend/.env` file exists and has the correct key
- Make sure you activated the virtual environment before running uvicorn

**"Image not found" errors in batch mode:**
- Verify image paths in `claims.csv` match actual files in `dataset/images/`
- Paths are relative to the project root directory

**Rate limit errors (429):**
- The system adds 0.5s delay between calls by default
- Increase the delay in `run_batch.py` (change `time.sleep(0.5)` to `time.sleep(2)`)

---

## 💡 Tips for Better Accuracy

1. **Image quality**: Use clear, well-lit photos. Blurry images get flagged and may return `not_enough_information`
2. **Multiple angles**: Submit 2–3 photos from different angles for complex damage
3. **Specificity**: More detailed claim descriptions lead to better analysis
4. **Image size**: Images are processed at full resolution. Resize to max 1024px for faster/cheaper processing
