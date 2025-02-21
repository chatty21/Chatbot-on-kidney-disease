from flask import Flask, request, jsonify, render_template
import requests
import spacy
import scispacy
from scispacy.linking import EntityLinker
from bs4 import BeautifulSoup
from functools import lru_cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from transformers import pipeline
import os

# Set Transformers to use PyTorch instead of TensorFlow (to avoid Keras issues)
os.environ["USE_TORCH"] = "1"

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Apply rate limiting to prevent abuse
limiter = Limiter(get_remote_address, app=app, default_limits=["10 per minute"])

# Load SciSpaCy model
MODEL_PATH = "/Users/chaitanya/Desktop/Chatbot/en_core_sci_sm-0.5.4"  # Update if needed
try:
    nlp = spacy.load(MODEL_PATH)  # Load from local path
except OSError:
    nlp = spacy.load("en_core_sci_sm")  # Load installed model

print("✅ SciSpaCy model loaded successfully!")

# Add EntityLinker from SciSpaCy
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True})
print("✅ EntityLinker successfully added to pipeline!")

# Load a better medical-focused QA model
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")


# **Expanded Medical Context for Accurate Responses**
medical_context = """
### Overview
Kidney disease, also known as renal disease, refers to conditions that impair kidney function. 
The kidneys are vital organs responsible for filtering waste from the blood, balancing body fluids, regulating blood pressure, and producing essential hormones.

### Causes of Kidney Disease
1. **Chronic Kidney Disease (CKD)**: Often caused by diabetes, high blood pressure, and glomerulonephritis.
2. **Acute Kidney Injury (AKI)**: Can result from severe infections, dehydration, heart failure, or toxic reactions to medications.
3. **Polycystic Kidney Disease (PKD)**: A genetic disorder that leads to cysts in the kidneys.
4. **Kidney Infections**: Such as pyelonephritis, often due to untreated urinary tract infections (UTIs).
5. **Kidney Stones**: Can block urine flow, leading to infections and kidney damage.

### Symptoms of Kidney Disease
- Swelling in the legs, ankles, feet, or face due to fluid retention
- Fatigue and weakness
- Nausea and vomiting
- Shortness of breath (caused by fluid buildup in the lungs)
- High blood pressure (hypertension)
- Frequent urination, especially at night
- Blood in urine (hematuria) or foamy urine (proteinuria)
- Loss of appetite and weight loss
- Confusion and difficulty concentrating

### Diagnosis
- **Blood Tests**: Measures creatinine and blood urea nitrogen (BUN) to evaluate kidney function.
- **Urine Tests**: Detects protein, blood, and waste levels.
- **Imaging Tests**: Ultrasound, CT scans, and MRI help detect structural abnormalities.
- **Biopsy**: Involves removing a small kidney tissue sample for analysis.

### Treatment Options
#### 1. **Lifestyle Modifications**
- **Dietary Changes**: Reducing sodium, potassium, and phosphorus intake to ease kidney workload.
- **Exercise**: Helps maintain blood pressure and control diabetes.
- **Fluid Management**: Adjusting fluid intake based on kidney function.

#### 2. **Medications**
- **ACE Inhibitors & ARBs**: Lower blood pressure and protect kidneys.
- **Diuretics**: Help remove excess fluid.
- **Phosphate Binders**: Reduce phosphorus buildup in the blood.
- **Erythropoiesis-Stimulating Agents (ESAs)**: Treat anemia caused by kidney disease.

#### 3. **Advanced Treatment**
- **Dialysis**: Used when kidney function drops below 10-15%. Includes:
  - **Hemodialysis**: Blood is filtered outside the body using a dialysis machine.
  - **Peritoneal Dialysis**: Uses the peritoneal membrane to filter waste inside the abdomen.
- **Kidney Transplant**: Replacing a failing kidney with a healthy donor kidney.

### Prevention
- Managing blood sugar levels (for diabetics).
- Controlling blood pressure through diet and medication.
- Staying hydrated and avoiding excessive use of painkillers like NSAIDs.
- Getting regular kidney function tests if at risk.
"""

# PubMed API URL (for additional medical research results)
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# Function to fetch related medical articles from PubMed
def search_pubmed(query):
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": 3  # Limit to top 3 results
    }
    response = requests.get(PUBMED_SEARCH_URL, params=params)
    if response.status_code == 200:
        return response.json()["esearchresult"]["idlist"]
    return []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
@limiter.limit("5 per minute")
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"response": "❌ Please enter a valid question."})

    # Use transformers model for simple QA
    response = qa_pipeline({"question": user_message, "context": medical_context})
    
    # Extract and format the answer
    bot_reply = response["answer"]

    # Search PubMed for related articles
    related_articles = search_pubmed(user_message)

    # Construct response
    pubmed_links = "\n".join([f"https://pubmed.ncbi.nlm.nih.gov/{article_id}" for article_id in related_articles])
    final_response = f"🩺 {bot_reply}\n\n🔍 Related Articles:\n{pubmed_links}" if related_articles else f"🩺 {bot_reply}"

    return jsonify({"response": final_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
