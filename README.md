# ResumeScore-AI

ResumeScore-AI is a Flask-based resume analysis tool that evaluates how well a resume matches a given job description.

It generates an ATS-style similarity score using **TF-IDF and cosine similarity** and also performs **skill-gap analysis** to identify matched and missing skills.

## Features

- Upload a resume in PDF format
- Enter a target job description
- Calculate resume-job similarity score
- Extract skills from the resume and job description
- Identify matched skills
- Identify missing skills
- Calculate skill coverage percentage
- Simple web-based Flask interface

## Tech Stack

- Python 3.11+
- Flask
- PyMuPDF
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- HTML/CSS

## Project Structure

```text
ResumeScore-AI/
│
├── app.py
├── requirements.txt
├── Procfile
├── README.md
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
└── uploads/