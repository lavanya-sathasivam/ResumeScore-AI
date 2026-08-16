# ResumeScore-AI

Simple Flask app that computes an ATS-style similarity score between a resume PDF and a job description using `sentence-transformers`.

## Requirements
- Python 3.11 or newer
- See `requirements.txt` for Python package dependencies

## Setup
1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# or cmd
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser, upload a resume PDF and paste a job description to get a similarity score.

## Notes
- Uploaded files are saved to the `uploads` folder created by the app.
- The first run may download model weights (internet required) and could take some time.
- If you run into GPU/CPU compatibility issues with `sentence-transformers`, ensure appropriate `torch` binaries are installed for your platform.
