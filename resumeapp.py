import os
import streamlit as st
import pickle
import re
import nltk
import pdfplumber  # Added for PDF text extraction

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load models once at the top
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Category Mapping
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

# Clean Resume Text
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s*', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', r' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

# Web App
def main():
    st.title("📝 Resume Screening App")

    # Debug info inside app
    st.write("📂 Current working directory:", os.getcwd())
    st.write("✅ Does clf.pkl exist?", os.path.exists('clf.pkl'))
    st.write("✅ Does tfidf.pkl exist?", os.path.exists('tfidf.pkl'))

    upload_file = st.file_uploader('Upload Resume (TXT or PDF)', type=['txt', 'pdf'])

    if upload_file is not None:
        resume_text = ""

        if upload_file.name.endswith('.pdf'):
            with pdfplumber.open(upload_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        resume_text += text + "\n"
        else:
            resume_bytes = upload_file.read()
            try:
                resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                resume_text = resume_bytes.decode('latin-1')

        st.subheader("📄 Extracted Resume Text:")
        if resume_text.strip():
            st.write(resume_text)
        else:
            st.error("❌ No extractable text found in this PDF.")

        if resume_text.strip():
            # Clean and transform
            cleaned_resume = cleanResume(resume_text)
            vectorized_text = tfidf.transform([cleaned_resume])

            # Prediction
            prediction_id = clf.predict(vectorized_text)[0]
            prediction_id = int(prediction_id)

            # Get category
            category_name = category_mapping.get(prediction_id, "Unknown")

            st.success(f"🎯 Predicted Category: **{category_name}**")

# Run App
if __name__ == "__main__":
    main()