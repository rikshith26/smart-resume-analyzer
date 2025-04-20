from flask import Flask, request, render_template
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def clean_text(text):
    return re.sub(r'\W+', ' ', text.lower())

def calculate_similarity(resume_text, jd_text):
    vectorizer = CountVectorizer().fit_transform([resume_text, jd_text])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0] * 100

@app.route("/", methods=["GET", "POST"])
def index():
    result = {}
    if request.method == "POST":
        resume = request.files["resume"]
        job_desc = request.form["job_desc"]
        resume_text = extract_text_from_pdf(resume)
        resume_text_clean = clean_text(resume_text)
        jd_text_clean = clean_text(job_desc)

        match_score = round(calculate_similarity(resume_text_clean, jd_text_clean), 2)
        result = {
            "resume_text": resume_text[:1000],
            "score": match_score
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    print("Flask is running! Visit http://127.0.0.1:5000")
    app.run(debug=True)
