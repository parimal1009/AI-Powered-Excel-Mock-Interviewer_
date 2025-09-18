# 🚀 AI Excel Mastery Interviewer Pro

**AI Excel Mastery Interviewer Pro** is the world's most advanced **Excel skills assessment platform**, delivering a realistic, AI-driven **mock interview experience** for roles that demand advanced Excel proficiency.  

This enterprise-grade application simulates **real-world Excel interviews** with **dynamic adaptive questioning**, provides **comprehensive evaluation reports**, and empowers **recruiters, trainers, and candidates** with **deep analytics**.  

---

## 🖼️ Platform in Action  

### Candidate Profile & Details  
<img src="https://github.com/parimal1009/AI-Powered-Excel-Mock-Interviewer_/blob/main/images/user_details.png?raw=true" width="800" alt="User Details">

### Adaptive Assessment (Q&A + Evaluation)  
<img src="https://github.com/parimal1009/AI-Powered-Excel-Mock-Interviewer_/blob/main/images/assessment_qna.png?raw=true" width="800" alt="Assessment QnA">

### Candidate Report after Assessment  
<img src="https://github.com/parimal1009/AI-Powered-Excel-Mock-Interviewer_/blob/main/images/report.png?raw=true" width="800" alt="User Report">

### Analytics Dashboard  
<img src="https://github.com/parimal1009/AI-Powered-Excel-Mock-Interviewer_/blob/main/images/analytics.png?raw=true" width="800" alt="Analytics Dashboard">

---

## 🌟 Key Highlights  

- 🤖 **AI-Powered Mock Interviews** – Conducted in real-time by a powerful LLM.  
- 🧩 **Role-Based Assessments** – Tailored interview questions for roles like:  
  - Financial Analyst  
  - Data Analyst  
  - Operations Analyst  
  - Business Analyst  
  - Consultant  
- 📈 **Adaptive Difficulty** – Questions adjust dynamically based on performance.  
- 📊 **Detailed Evaluation** – Scores, strengths, weaknesses, and recommendations.  
- ⚡ **Real-time Interaction** – Seamless interview experience powered by WebSockets.  
- 📉 **Analytics Dashboard** – Track performance trends and insights.  
- 🎨 **Modern Frontend** – Clean, responsive UI built with HTML, CSS, and Vanilla JavaScript.  

---

## ⚙️ Tech Stack  

### 🔹 Backend  
- **Framework:** FastAPI  
- **WebSockets:** FastAPI WebSocket support  
- **Database:** SQLite  
- **AI/ML:**  
  - LangChain  
  - Groq (LLM)  
  - Hugging Face Sentence Transformers (embeddings)  
  - FAISS (vector search)  
- **Environment Management:** python-dotenv  

### 🔹 Frontend  
- HTML5  
- CSS3  
- Vanilla JavaScript  

---

## 🚀 Getting Started  

### Prerequisites  
- Python 3.8+  
- A **Groq API key**  

### Installation  

```bash
# Clone the repository
git clone https://github.com/your-username/ai-excel-interviewer.git
cd ai-excel-interviewer

# Create & activate virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate


Environment Setup

Create a .env file in the root directory and add your Groq API key:

GROQ_API_KEY=your_groq_api_key

Running the Application

Start the FastAPI server:

uvicorn main:app --reload


Open your browser:
Navigate to 👉 http://127.0.0.1:8000

📂 Project Structure
.env
README.md
main.py
requirements.txt
data/
└── excel_interviewer.db
static/
└── ...
templates/
└── index.html


main.py → Core FastAPI application (backend logic, WebSocket handling, API endpoints).

requirements.txt → Python dependencies.

.env → Stores environment variables (e.g., Groq API key).

data/ → SQLite database files.

templates/ → Frontend HTML templates.

static/ → Static assets (CSS, JavaScript, images).

🧑‍💻 Usage

Start the Interview:

Open the application in your browser.

Enter your name and email (optional).

Select your target job role from the dropdown.

Click “Start Excel Mastery Assessment”.

Answer Questions:

Read the question carefully.

Enter your answer in the text box.

Submit your response.

Review Feedback:

After each question, you’ll receive detailed feedback with scores.

Complete the Assessment:

At the end, receive a final report with your overall score and recommendations.

🗄️ Database Schema

The application uses a SQLite database with the following tables:

interview_sessions → Stores interview session metadata.

interview_responses → Stores questions, answers, and evaluations.

candidate_profiles → Tracks strengths, weaknesses, and performance.

interview_analytics → Stores analytics and performance data.

🤝 Contributing

Contributions are welcome! 🎉

Fork the repo

Create your feature branch (git checkout -b feature/your-feature)

Commit your changes (git commit -m 'Add new feature')

Push to the branch (git push origin feature/your-feature)

Open a Pull Request 🚀

📜 License

This project is licensed under the MIT License – feel free to use and modify.

💡 Inspiration

AI Excel Mastery Interviewer Pro was built to revolutionize Excel skill assessments, making them smarter, fairer, and more realistic for both candidates and recruiters.

# Install dependencies
pip install -r requirements.txt
