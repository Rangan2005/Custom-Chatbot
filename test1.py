from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_store = FAISS.load_local("faiss_store", model, allow_dangerous_deserialization=True)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
courses = [
    {
        "title": "Chatbot Creators: Design a ChatGPT-like AI",
        "price": "$13",
        "lessons": "8 Lessons",
        "description": "Join us on a 7-day bootcamp to step into the world of AI and create your own chatbot like ChatGPT."
    },
    {
        "title": "Web Development from Scratch",
        "price": "$30",
        "lessons": "7 Lessons",
        "description": "Unlock the boundless potential of web development with our Web Development Essentials course."
    },
    {
        "title": "Summer Camp: Introduction to Python",
        "price": "$11",
        "lessons": "6 Lessons",
        "description": "Step into our 7-day Python Project Playground camp where coding meets creativity!"
    }
]

def find_relevant_courses(user_input):
    embeddings = model.encode(user_input)
    docs = faiss_store.similarity_search_by_vector(embeddings, k=3)
    response_from_faiss = "\n\n".join([doc.page_content for doc in docs])
    user_input = user_input.lower()
    relevant_courses = [
        course for course in courses 
        if any(keyword in course['title'].lower() or keyword in course['description'].lower() for keyword in user_input.split())
    ]

    return {
        "faiss_response": response_from_faiss,
        "relevant_courses": relevant_courses
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('query', '')
    if user_input.lower() in ['hi', 'hello', 'hey']:
        return jsonify({"response": "Hello! How can I help you today?"})
    elif user_input.lower() in ['how are you', 'how are you doing']:
        return jsonify({"response": "I'm just a program, but I'm here to assist you. How can I help?"})
    result = find_relevant_courses(user_input)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
