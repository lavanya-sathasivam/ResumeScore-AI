import os
import fitz  # PyMuPDF

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


SKILL_CATALOG = {
    "Python": ["python"],
    "SQL": ["sql", "mysql", "postgresql", "sqlite"],
    "Excel": ["excel", "microsoft excel"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Scikit-learn": ["scikit-learn", "sklearn"],
    "Machine Learning": ["machine learning", "ml models"],
    "Deep Learning": ["deep learning", "neural networks"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Data Visualization": [
        "data visualization",
        "data visualisation"
    ],
    "Statistics": ["statistics", "statistical"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "Docker": ["docker"],
    "Git": ["git", "github", "gitlab"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "REST APIs": [
        "rest api",
        "restful api",
        "api development"
    ],
    "Spark": ["apache spark", "pyspark"],
    "Hadoop": ["hadoop"],
    "NLP": [
        "natural language processing",
        "nlp"
    ],
    "Computer Vision": ["computer vision"],
}


def extract_skills(text):
    text = text.lower()
    found_skills = []

    for skill, keywords in SKILL_CATALOG.items():
        if any(keyword in text for keyword in keywords):
            found_skills.append(skill)

    return set(found_skills)


def analyze_skill_gap(resume_text, job_description):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    matched_skills = sorted(
        resume_skills.intersection(job_skills)
    )

    missing_skills = sorted(
        job_skills.difference(resume_skills)
    )

    if job_skills:
        coverage = round(
            (len(matched_skills) / len(job_skills)) * 100
        )
    else:
        coverage = 0

    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_coverage": coverage
    }


def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()

    return text.strip()


def calculate_similarity(resume_text, job_description):
    vectorizer = TfidfVectorizer(stop_words="english")

    vectors = vectorizer.fit_transform(
        [resume_text, job_description]
    )

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )

    return round(similarity[0][0] * 100, 2)


@app.route("/", methods=["GET", "POST"])
def index():
    score = None
    skill_analysis = None
    job_description = ""
    filename = ""

    if request.method == "POST":
        file = request.files.get("resume")
        job_description = request.form.get(
            "job_description",
            ""
        )

        if (
            not file
            or file.filename == ""
            or job_description.strip() == ""
        ):
            return "Invalid input", 400

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        resume_text = extract_text_from_pdf(filepath)

        score = calculate_similarity(
            resume_text,
            job_description
        )

        skill_analysis = analyze_skill_gap(
            resume_text,
            job_description
        )

    return render_template(
        "index.html",
        score=score,
        job_description=job_description,
        skill_analysis=skill_analysis,
        filename=filename
    )


if __name__ == "__main__":
    port = int(
        os.environ.get("PORT", 10000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )