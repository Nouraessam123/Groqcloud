from flask import Flask, request, jsonify
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 👇 استخدمي مفتاح Groq من Environment Variables في Vercel
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 👇 دالة إرسال الطلب إلى Groq API
def call_groq_model(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "whisper-large-v3-turbo"  # أو llama3 لو حابة
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body)
    return response.json()["choices"][0]["message"]["content"]

analyzer = SentimentIntensityAnalyzer()

# ✅ توليد أسئلة مقابلة من Groq بدل Hugging Face
def generate_questions(role):
    prompt = f"Generate 3 behavioral and 2 technical interview questions for a {role} role. Just list the questions."
    result = call_groq_model(prompt)
    questions = [q.strip("- ").strip() for q in result.split("\n") if q.strip()]
    return questions

@app.route('/start-interview', methods=['POST'])
def start_interview():
    data = request.get_json()
    job_role = data.get('job_role', 'Data Scientist')
    questions = generate_questions(job_role)
    return jsonify({'questions': questions})

@app.route('/submit-answer', methods=['POST'])
def submit_answer():
    data = request.get_json()
    question = data.get('question')
    answer = data.get('answer')

    sentiment = analyzer.polarity_scores(answer)

    prompt = (
        f"Evaluate the following answer to this interview question:\n\n"
        f"Question: {question}\nAnswer: {answer}\n\n"
        f"Give detailed feedback and a score out of 10 like this:\nRating: X/10"
    )

    feedback = call_groq_model(prompt)

    # استخراج التقييم
    rating = None
    lines = feedback.split('\n')
    for line in lines:
        if "Rating:" in line:
            import re
            match = re.search(r'\d+', line)
            if match:
                rating = int(match.group())
                break

    feedback_text = "\n".join([line for line in lines if not line.strip().startswith("Rating:")])

    return jsonify({
        'feedback': feedback_text.strip(),
        'rating': rating,
        'sentiment': sentiment
    })

# 🟡 هذا السطر غير مطلوب في Vercel لكن كويس لو هتجربي محليًا
if __name__ == "__main__":
    print("Starting Flask app... 🚀")
    app.run(debug=True)
