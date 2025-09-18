# AI Excel Mastery Interviewer Pro

Welcome to the AI Excel Mastery Interviewer Pro, the world's most advanced Excel skills assessment platform. This enterprise-grade application provides a realistic, AI-driven mock interview experience for various job roles that require advanced Excel proficiency.

## Key Features

-   **AI-Powered Interviews:** Utilizes a powerful Large Language Model (LLM) to conduct dynamic and adaptive interviews.
-   **Role-Based Assessments:** Tailored interview questions for different job roles, including Financial Analyst, Data Analyst, Operations Analyst, Business Analyst, and Consultant.
-   **Adaptive Difficulty:** The difficulty of questions adjusts based on the candidate's performance.
-   **Comprehensive Evaluation:** Provides detailed feedback, including scores, strengths, weaknesses, and a final recommendation.
-   **Real-time Interaction:** Uses WebSockets for a seamless, real-time interview experience.
-   **Database Integration:** Stores session data, user responses, and analytics in a SQLite database.
-   **Analytics Dashboard:** Displays statistics about the interviews conducted.
-   **Modern Frontend:** A clean, responsive, and user-friendly interface built with HTML, CSS, and vanilla JavaScript.

## Tech Stack

### Backend

-   **Framework:** FastAPI
-   **WebSockets:** FastAPI's WebSocket support
-   **Database:** SQLite
-   **AI/ML:**
    -   LangChain
    -   Groq (for the LLM)
    -   Hugging Face Sentence Transformers (for embeddings)
    -   FAISS (for vector storage)
-   **Environment Management:** python-dotenv

### Frontend

-   HTML5
-   CSS3
-   JavaScript (Vanilla)

## Getting Started

### Prerequisites

-   Python 3.8+
-   A Groq API key

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/ai-excel-interviewer.git
    cd ai-excel-interviewer
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

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