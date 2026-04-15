import os
import fitz  # PyMuPDF
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()


def calculate_similarity(resume_text, job_description):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return round(similarity[0][0] * 100, 2)


@app.route("/", methods=["GET", "POST"])
def index():
    score = None
    job_description = ""
    filename = ""

    if request.method == "POST":
        file = request.files.get("resume")
        job_description = request.form.get("job_description", "")

        if not file or file.filename == "" or job_description.strip() == "":
            return "Invalid input", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        resume_text = extract_text_from_pdf(filepath)
        score = calculate_similarity(resume_text, job_description)

    return render_template(
        "index.html",
        score=score,
        job_description=job_description,
        filename=filename
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
