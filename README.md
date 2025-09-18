# 🚀 AI Excel Mastery Interviewer Pro

**AI Excel Mastery Interviewer Pro** is the world's most advanced Excel skills assessment platform — delivering a realistic, AI-driven mock interview experience for roles that demand advanced Excel proficiency.  

This enterprise-grade application simulates real-world interviews with **dynamic, adaptive questioning**, provides **comprehensive evaluation reports**, and empowers recruiters, trainers, and candidates alike with **deep analytics**.

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

- **AI-Powered Mock Interviews** – Conducted in real-time by a powerful LLM.  
- **Role-Based Assessments** – Tailored interview questions for roles like **Financial Analyst, Data Analyst, Operations Analyst, Business Analyst, and Consultant**.  
- **Adaptive Difficulty** – Questions adjust dynamically based on performance.  
- **Detailed Evaluation** – Get immediate feedback with scores, strengths, weaknesses, and final recommendations.  
- **Real-time Interaction** – WebSocket-driven seamless interview experience.  
- **Analytics Dashboard** – Gain insights with interactive analytics.  
- **Modern Frontend** – Clean, responsive UI with **HTML, CSS, and Vanilla JavaScript**.  

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
- JavaScript (Vanilla)  

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

# Install dependencies
pip install -r requirements.txt


4.  **Set up your environment variables:**

    Create a `.env` file in the root directory and add your Groq API key:

    ```
    GROQ_API_KEY=your_groq_api_key
    ```

### Running the Application

1.  **Start the FastAPI server:**

    ```bash
    uvicorn main:app --reload
    ```

2.  **Open your browser:**

    Navigate to `http://127.0.0.1:8000` to access the application.

## Project Structure

```
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
```

-   `main.py`: The main FastAPI application file containing the backend logic, WebSocket handling, and API endpoints.
-   `requirements.txt`: A list of all the Python dependencies for the project.
-   `.env`: The environment variables file (contains the Groq API key).
-   `data/`: The directory where the SQLite database is stored.
-   `templates/`: Contains the HTML templates for the frontend.
-   `static/`: Contains static assets like CSS, JavaScript, and images (if any).

## Usage

1.  **Start the Interview:**
    -   Open the application in your browser.
    -   Optionally, enter your name and email.
    -   Select your target job role from the dropdown menu.
    -   Click the "Start Excel Mastery Assessment" button.

2.  **Answer Questions:**
    -   Read the questions carefully.
    -   Type your answers in the response text area.
    -   Submit your answers.

3.  **Review Feedback:**
    -   After each question, you will receive a detailed evaluation of your answer.

4.  **Complete the Assessment:**
    -   At the end of the interview, you will receive a final report with an overall score and recommendation.

## Database Schema

The application uses a SQLite database with the following tables:

-   `interview_sessions`: Stores information about each interview session.
-   `interview_responses`: Stores each question, the candidate's answer, and the evaluation.
-   `candidate_profiles`: Stores the candidate's profile, including strengths, weaknesses, and performance trends.
-   `interview_analytics`: Stores analytics data about the interviews.

## Contributing


Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.
