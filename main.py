import os
import json
import uuid
import asyncio
import logging
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import defaultdict
import statistics
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("data", exist_ok=True)

class InterviewState(Enum):
    NOT_STARTED = "not_started"
    INTRODUCTION = "introduction"
    SKILL_ASSESSMENT = "skill_assessment"
    BASIC_CONCEPTS = "basic_concepts"
    FORMULAS_FUNCTIONS = "formulas_functions"
    DATA_ANALYSIS = "data_analysis"
    ADVANCED_FEATURES = "advanced_features"
    SCENARIO_BASED = "scenario_based"
    ROLE_SPECIFIC = "role_specific"
    FINAL_CHALLENGE = "final_challenge"
    COMPLETED = "completed"

class QuestionType(Enum):
    THEORETICAL = "theoretical"
    PRACTICAL = "practical"
    SCENARIO = "scenario"
    CODING = "coding"
    PROBLEM_SOLVING = "problem_solving"

class SkillLevel(Enum):
    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class JobRole(Enum):
    FINANCIAL_ANALYST = "financial_analyst"
    DATA_ANALYST = "data_analyst"
    OPERATIONS_ANALYST = "operations_analyst"
    BUSINESS_ANALYST = "business_analyst"
    CONSULTANT = "consultant"

class DatabaseManager:
    def __init__(self, db_path: str = "data/excel_interviewer.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS interview_sessions (
                    session_id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    state TEXT NOT NULL,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    total_score INTEGER DEFAULT 0,
                    max_score INTEGER DEFAULT 0,
                    skill_level TEXT DEFAULT 'beginner',
                    candidate_name TEXT,
                    candidate_email TEXT,
                    is_completed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS interview_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    question_number INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    max_possible INTEGER NOT NULL,
                    response_time REAL NOT NULL,
                    difficulty TEXT NOT NULL,
                    question_type TEXT NOT NULL,
                    evaluation_json TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES interview_sessions (session_id)
                );

                CREATE TABLE IF NOT EXISTS candidate_profiles (
                    session_id TEXT PRIMARY KEY,
                    strengths TEXT,
                    weaknesses TEXT,
                    confidence_scores TEXT,
                    response_times TEXT,
                    performance_trend TEXT,
                    learning_pace TEXT DEFAULT 'moderate',
                    final_recommendation TEXT,
                    FOREIGN KEY (session_id) REFERENCES interview_sessions (session_id)
                );

                CREATE TABLE IF NOT EXISTS interview_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    total_interviews INTEGER DEFAULT 0,
                    completed_interviews INTEGER DEFAULT 0,
                    avg_score REAL DEFAULT 0,
                    avg_duration_minutes REAL DEFAULT 0,
                    role_distribution TEXT,
                    skill_level_distribution TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_role ON interview_sessions(role);
                CREATE INDEX IF NOT EXISTS idx_sessions_state ON interview_sessions(state);
                CREATE INDEX IF NOT EXISTS idx_responses_session ON interview_responses(session_id);
                CREATE INDEX IF NOT EXISTS idx_analytics_date ON interview_analytics(date);
            """)
        logger.info(f"Database initialized at {self.db_path}")
    
    def create_session(self, session_id: str, role: str, candidate_name: str = None, candidate_email: str = None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO interview_sessions (session_id, role, state, candidate_name, candidate_email)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, role, InterviewState.NOT_STARTED.value, candidate_name, candidate_email))
            
            conn.execute("""
                INSERT INTO candidate_profiles (session_id, strengths, weaknesses, confidence_scores, 
                response_times, performance_trend) 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, "[]", "[]", "[]", "[]", "[]"))
    
    def update_session(self, session_id: str, **kwargs):
        if not kwargs:
            return
        
        set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
        values = list(kwargs.values()) + [session_id]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
                UPDATE interview_sessions 
                SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
            """, values)
    
    def save_response(self, session_id: str, response_data: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO interview_responses (
                    session_id, question_number, category, topic, question, answer,
                    score, max_possible, response_time, difficulty, question_type, evaluation_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                response_data['question_number'],
                response_data['category'],
                response_data['topic'],
                response_data['question'],
                response_data['answer'],
                response_data['score'],
                response_data['max_possible'],
                response_data['response_time'],
                response_data['difficulty'],
                response_data['question_type'],
                json.dumps(response_data['evaluation'])
            ))
    
    def get_session(self, session_id: str) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            result = conn.execute("""
                SELECT * FROM interview_sessions WHERE session_id = ?
            """, (session_id,)).fetchone()
            return dict(result) if result else None
    
    def get_session_responses(self, session_id: str) -> List[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            results = conn.execute("""
                SELECT * FROM interview_responses 
                WHERE session_id = ? 
                ORDER BY question_number
            """, (session_id,)).fetchall()
            return [dict(row) for row in results]
    
    def get_analytics_summary(self, days: int = 30) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            total_sessions = conn.execute("""
                SELECT COUNT(*) as count FROM interview_sessions 
                WHERE created_at >= ?
            """, (cutoff_date,)).fetchone()['count']
            
            completed_sessions = conn.execute("""
                SELECT COUNT(*) as count FROM interview_sessions 
                WHERE is_completed = TRUE AND created_at >= ?
            """, (cutoff_date,)).fetchone()['count']
            
            avg_score = conn.execute("""
                SELECT AVG(CAST(total_score AS REAL) / CAST(max_score AS REAL) * 100) as avg_score
                FROM interview_sessions 
                WHERE is_completed = TRUE AND max_score > 0 AND created_at >= ?
            """, (cutoff_date,)).fetchone()['avg_score'] or 0
            
            role_distribution = conn.execute("""
                SELECT role, COUNT(*) as count 
                FROM interview_sessions 
                WHERE created_at >= ?
                GROUP BY role
            """, (cutoff_date,)).fetchall()
            
            return {
                'total_sessions': total_sessions,
                'completed_sessions': completed_sessions,
                'completion_rate': (completed_sessions / total_sessions * 100) if total_sessions > 0 else 0,
                'avg_score': round(avg_score, 1),
                'role_distribution': {row['role']: row['count'] for row in role_distribution}
            }

EXCEL_KNOWLEDGE_BASE = {
    "skill_assessment": [
        {
            "topic": "Experience Assessment",
            "question": "Walk me through your Excel journey. What's the most sophisticated Excel project you've delivered, and what advanced features did you leverage?",
            "key_points": ["experience depth", "project complexity", "advanced features", "business impact", "problem-solving approach"],
            "type": QuestionType.THEORETICAL,
            "difficulty": "assessment",
            "max_score": 8,
            "role_specific": False,
            "follow_up_questions": [
                "What challenges did you face in that project?",
                "How did you ensure data accuracy and validation?",
                "What would you do differently if you rebuilt it today?"
            ]
        }
    ],
    "basic_concepts": [
        {
            "topic": "Advanced Cell References",
            "question": "You're building a dynamic financial model where formulas need to adapt based on user inputs. Explain how you'd use different reference types (A1, $A$1, $A1, A$1) strategically, and describe a scenario where mixed references become critical.",
            "key_points": ["relative references", "absolute references", "mixed references", "dynamic modeling", "formula efficiency", "error prevention"],
            "type": QuestionType.PRACTICAL,
            "difficulty": "intermediate",
            "max_score": 15,
            "role_specific": False
        },
        {
            "topic": "Data Architecture & Types",
            "question": "Design a data validation strategy for a multi-user financial dashboard. How would you handle different data types, prevent common input errors, and ensure data integrity across linked worksheets?",
            "key_points": ["data validation rules", "error handling", "data types", "cross-sheet references", "user experience", "audit trails"],
            "type": QuestionType.SCENARIO,
            "difficulty": "advanced",
            "max_score": 20,
            "role_specific": True,
            "roles": [JobRole.FINANCIAL_ANALYST, JobRole.DATA_ANALYST]
        }
    ],
    "formulas_functions": [
        {
            "topic": "Next-Gen Lookup Functions",
            "question": "Compare VLOOKUP, INDEX-MATCH, XLOOKUP, and the new FILTER/SORT dynamic arrays. Build a decision framework for when to use each, considering performance, maintainability, and future-proofing.",
            "key_points": ["lookup function comparison", "performance optimization", "error handling", "dynamic arrays", "maintainability", "Excel version compatibility"],
            "type": QuestionType.PRACTICAL,
            "difficulty": "advanced",
            "max_score": 22,
            "role_specific": False
        },
        {
            "topic": "Dynamic Array Mastery",
            "question": "Design a real-time executive dashboard using only dynamic array formulas (FILTER, SORT, UNIQUE, SEQUENCE). How would you create interactive drill-downs and handle large datasets efficiently?",
            "key_points": ["dynamic arrays", "spill ranges", "performance optimization", "interactive dashboards", "data relationships", "error handling"],
            "type": QuestionType.SCENARIO,
            "difficulty": "expert",
            "max_score": 28,
            "role_specific": True,
            "roles": [JobRole.DATA_ANALYST, JobRole.BUSINESS_ANALYST]
        }
    ],
    "data_analysis": [
        {
            "topic": "Advanced Pivot Intelligence",
            "question": "Create a multi-dimensional profitability analysis system using pivot tables with calculated fields, custom groupings, and automated refresh. How would you handle data from multiple sources and create executive-ready visualizations?",
            "key_points": ["pivot table architecture", "calculated measures", "data model design", "automation", "visualization", "performance optimization"],
            "type": QuestionType.SCENARIO,
            "difficulty": "advanced",
            "max_score": 25,
            "role_specific": False
        },
        {
            "topic": "Power Query Excellence",
            "question": "Build an automated ETL pipeline using Power Query to consolidate data from APIs, databases, and files. How would you handle schema changes, implement error recovery, and optimize for performance?",
            "key_points": ["ETL design", "data transformation", "error handling", "query optimization", "automation", "data governance"],
            "type": QuestionType.PRACTICAL,
            "difficulty": "expert",
            "max_score": 32,
            "role_specific": True,
            "roles": [JobRole.DATA_ANALYST, JobRole.OPERATIONS_ANALYST]
        }
    ],
    "advanced_features": [
        {
            "topic": "VBA vs Modern Excel",
            "question": "When should you choose VBA over Power Query, dynamic arrays, or Office Scripts? Design a decision matrix and provide examples where each technology excels.",
            "key_points": ["technology comparison", "automation strategies", "performance considerations", "maintenance", "security", "future-proofing"],
            "type": QuestionType.THEORETICAL,
            "difficulty": "expert",
            "max_score": 25,
            "role_specific": False
        },
        {
            "topic": "Enterprise Data Validation",
            "question": "Architect a comprehensive data governance system for a financial planning tool used by 50+ analysts. Include input validation, audit trails, version control, and user permission management.",
            "key_points": ["enterprise architecture", "data governance", "validation systems", "audit capabilities", "user management", "scalability"],
            "type": QuestionType.SCENARIO,
            "difficulty": "expert",
            "max_score": 30,
            "role_specific": True,
            "roles": [JobRole.FINANCIAL_ANALYST, JobRole.CONSULTANT]
        }
    ],
    "scenario_based": [
        {
            "topic": "C-Suite Dashboard Architecture",
            "question": "Design and implement a real-time executive dashboard that integrates financial, operational, and market data. Include drill-down capabilities, alert systems, and mobile optimization. Walk through your complete architecture.",
            "key_points": ["dashboard architecture", "data integration", "real-time updates", "user experience", "mobile design", "performance", "security"],
            "type": QuestionType.SCENARIO,
            "difficulty": "expert",
            "max_score": 35,
            "role_specific": False
        }
    ],
    "role_specific": {
        JobRole.FINANCIAL_ANALYST: [
            {
                "topic": "Advanced Financial Modeling",
                "question": "Build a comprehensive DCF model with Monte Carlo simulation for scenario analysis. How would you structure the model for audit-ability, implement sensitivity analysis, and create stress-testing capabilities?",
                "key_points": ["DCF modeling", "Monte Carlo simulation", "sensitivity analysis", "model validation", "stress testing", "audit trails"],
                "type": QuestionType.SCENARIO,
                "difficulty": "expert",
                "max_score": 40
            }
        ],
        JobRole.DATA_ANALYST: [
            {
                "topic": "Statistical Analysis & ML Integration",
                "question": "Implement advanced statistical analysis in Excel including regression, hypothesis testing, and correlation analysis. How would you integrate with Python/R and validate your statistical models?",
                "key_points": ["statistical functions", "regression analysis", "hypothesis testing", "model validation", "tool integration", "statistical significance"],
                "type": QuestionType.PRACTICAL,
                "difficulty": "expert",
                "max_score": 35
            }
        ],
        JobRole.OPERATIONS_ANALYST: [
            {
                "topic": "Optimization & Resource Planning",
                "question": "Create an optimization model for resource allocation across multiple projects with complex constraints. Use Solver add-in and explain your approach to handling multiple objectives.",
                "key_points": ["optimization modeling", "solver functionality", "constraint management", "multi-objective optimization", "sensitivity analysis", "scenario planning"],
                "type": QuestionType.SCENARIO,
                "difficulty": "expert",
                "max_score": 38
            }
        ]
    },
    "final_challenge": [
        {
            "topic": "Excel Mastery Integration",
            "question": "You have 72 hours to build a complete business intelligence solution in Excel for a Fortune 500 company. The solution must handle multiple data sources, provide real-time analytics, include predictive capabilities, and support 100+ concurrent users. Present your complete architecture and implementation strategy.",
            "key_points": ["enterprise architecture", "scalability design", "performance optimization", "security implementation", "user experience", "deployment strategy", "maintenance plan"],
            "type": QuestionType.SCENARIO,
            "difficulty": "expert",
            "max_score": 50,
            "role_specific": False
        }
    ]
}

class InterviewSession:
    def __init__(self, session_id: str, role: JobRole, db_manager: DatabaseManager):
        self.session_id = session_id
        self.role = role
        self.db_manager = db_manager
        self.state = InterviewState.NOT_STARTED
        self.current_category = 0
        self.current_question = 0
        self.responses = []
        self.score = 0
        self.max_score = 0
        self.start_time = datetime.now()
        self.conversation_history = []
        self.candidate_profile = {
            "strengths": [],
            "weaknesses": [],
            "skill_level": SkillLevel.BEGINNER,
            "confidence_scores": [],
            "response_times": [],
            "learning_pace": "moderate"
        }
        self.adaptive_parameters = {
            "current_difficulty": 1,
            "performance_trend": [],
            "question_weights": defaultdict(float)
        }
        
    def add_response(self, question: str, answer: str, evaluation: Dict, category: str, 
                    response_time: float, question_data: Dict):
        response_entry = {
            "question": question,
            "answer": answer,
            "category": category,
            "topic": question_data.get('topic', 'Unknown'),
            "score": evaluation.get('score', 0),
            "max_possible": question_data.get('max_score', 10),
            "evaluation": evaluation,
            "response_time": response_time,
            "difficulty": question_data.get('difficulty', 'intermediate'),
            "question_type": question_data.get('type', QuestionType.THEORETICAL).value,
            "question_number": len(self.responses) + 1
        }
        
        self.responses.append(response_entry)
        self.score += evaluation.get('score', 0)
        self.max_score += question_data.get('max_score', 10)
        
        self.db_manager.save_response(self.session_id, response_entry)
        self.db_manager.update_session(self.session_id, 
            total_score=self.score, 
            max_score=self.max_score,
            state=self.state.value
        )
        
        performance = evaluation.get('score', 0) / question_data.get('max_score', 10)
        self.adaptive_parameters['performance_trend'].append(performance)
        self.candidate_profile['confidence_scores'].append(evaluation.get('confidence', 0.5))
        self.candidate_profile['response_times'].append(response_time)
        
        self._update_skill_level()
        
    def _update_skill_level(self):
        if not self.responses:
            return
            
        recent_performance = self.adaptive_parameters['performance_trend'][-3:]
        avg_performance = statistics.mean(recent_performance)
        
        if avg_performance >= 0.9:
            self.candidate_profile['skill_level'] = SkillLevel.EXPERT
        elif avg_performance >= 0.75:
            self.candidate_profile['skill_level'] = SkillLevel.ADVANCED
        elif avg_performance >= 0.6:
            self.candidate_profile['skill_level'] = SkillLevel.INTERMEDIATE
        elif avg_performance >= 0.4:
            self.candidate_profile['skill_level'] = SkillLevel.BEGINNER
        else:
            self.candidate_profile['skill_level'] = SkillLevel.NOVICE
            
        self.db_manager.update_session(self.session_id, skill_level=self.candidate_profile['skill_level'].value)

class ExcelInterviewer:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.sessions: Dict[str, InterviewSession] = {}
        self.knowledge_base = EXCEL_KNOWLEDGE_BASE
        self.llm = ChatGroq(
            temperature=0.3,
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=3000
        )
        self.setup_evaluation_system()
        logger.info("Advanced Excel Interviewer initialized successfully")
    
    def setup_evaluation_system(self):
        documents = []
        for category, questions in self.knowledge_base.items():
            if category == "role_specific":
                for role, role_questions in questions.items():
                    for q in role_questions:
                        doc_text = f"Role: {role.value}\nTopic: {q['topic']}\nQuestion: {q['question']}\nKey Points: {', '.join(q['key_points'])}\nDifficulty: {q['difficulty']}\nMax Score: {q['max_score']}"
                        documents.append(Document(page_content=doc_text, metadata={
                            "category": category, 
                            "role": role.value, 
                            "topic": q['topic'],
                            "difficulty": q['difficulty'],
                            "max_score": q['max_score']
                        }))
            else:
                for q in questions:
                    doc_text = f"Category: {category}\nTopic: {q['topic']}\nQuestion: {q['question']}\nKey Points: {', '.join(q['key_points'])}\nDifficulty: {q['difficulty']}\nMax Score: {q['max_score']}"
                    documents.append(Document(page_content=doc_text, metadata={
                        "category": category, 
                        "topic": q['topic'],
                        "difficulty": q['difficulty'],
                        "max_score": q['max_score']
                    }))
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store = FAISS.from_documents(splits, embeddings)
        logger.info(f"Knowledge base initialized with {len(documents)} expert-level questions")
    
    def create_session(self, role: str = "financial_analyst", candidate_name: str = None, candidate_email: str = None) -> str:
        session_id = str(uuid.uuid4())
        job_role = JobRole(role) if role in [r.value for r in JobRole] else JobRole.FINANCIAL_ANALYST
        
        self.db_manager.create_session(session_id, role, candidate_name, candidate_email)
        self.sessions[session_id] = InterviewSession(session_id, job_role, self.db_manager)
        
        logger.info(f"Created advanced session {session_id} for {job_role.value}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[InterviewSession]:
        if session_id not in self.sessions:
            db_session = self.db_manager.get_session(session_id)
            if db_session:
                role = JobRole(db_session['role'])
                session = InterviewSession(session_id, role, self.db_manager)
                session.state = InterviewState(db_session['state'])
                session.score = db_session['total_score'] or 0
                session.max_score = db_session['max_score'] or 0
                
                responses = self.db_manager.get_session_responses(session_id)
                session.responses = responses
                session.current_question = len(responses)
                
                self.sessions[session_id] = session
        
        return self.sessions.get(session_id)
    
    async def start_interview(self, session_id: str) -> str:
        session = self.get_session(session_id)
        if not session:
            return "Session not found"
        
        session.state = InterviewState.INTRODUCTION
        self.db_manager.update_session(session_id, state=session.state.value)
        
        intro_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are Alexandra Chen, Principal Excel Architect at McKinsey & Company with 15+ years of experience. You're conducting the world's most comprehensive Excel skills assessment for a {session.role.value.replace('_', ' ').title()} position.

Your expertise spans:
- Advanced financial modeling and valuation
- Enterprise data architecture and automation  
- Business intelligence and executive dashboards
- Statistical analysis and predictive modeling
- VBA, Power Query, Power Pivot mastery
- Teaching Excel to Fortune 500 executives

Your personality:
- Brilliant yet approachable - you make complex concepts accessible
- Extremely thorough but efficient with time
- Passionate about Excel's potential to transform businesses
- Constructive and encouraging while maintaining high standards
- Known for discovering hidden talent and potential

Interview Approach:
- This is an adaptive, world-class technical assessment
- Questions will adjust based on demonstrated skill level
- Focus on practical business applications and real-world scenarios
- Evaluate not just knowledge but strategic thinking and problem-solving
- Look for innovation, efficiency, and best practices

Structure Overview (8-10 categories, 60-90 minutes):
1. Experience assessment & Excel philosophy
2. Advanced concepts & architectural thinking  
3. Formula mastery & function optimization
4. Data analysis & business intelligence
5. Automation & advanced features
6. Complex scenario problem-solving
7. Role-specific technical deep-dives
8. Final integration challenge

Create an engaging, professional introduction that:
- Establishes your credibility and the assessment's importance
- Sets high expectations while being encouraging
- Explains the adaptive nature and business focus
- Makes the candidate excited to showcase their skills

Keep it conversational but authoritative (3-4 paragraphs)."""),
            ("human", "Please begin this world-class Excel skills assessment.")
        ])
        
        response = await self.llm.ainvoke(intro_prompt.format_messages())
        session.conversation_history.append({"role": "assistant", "content": response.content})
        
        logger.info(f"Started premium interview for session {session_id}")
        return response.content
    
    async def get_next_question(self, session_id: str) -> Dict:
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        # Ensure session is in a valid state for questions
        if session.state == InterviewState.NOT_STARTED:
            session.state = InterviewState.INTRODUCTION
            self.db_manager.update_session(session_id, state=session.state.value)
        
        state_transitions = {
            InterviewState.INTRODUCTION: ("skill_assessment", InterviewState.SKILL_ASSESSMENT),
            InterviewState.SKILL_ASSESSMENT: ("basic_concepts", InterviewState.BASIC_CONCEPTS),
            InterviewState.BASIC_CONCEPTS: ("formulas_functions", InterviewState.FORMULAS_FUNCTIONS),
            InterviewState.FORMULAS_FUNCTIONS: ("data_analysis", InterviewState.DATA_ANALYSIS),
            InterviewState.DATA_ANALYSIS: ("advanced_features", InterviewState.ADVANCED_FEATURES),
            InterviewState.ADVANCED_FEATURES: ("scenario_based", InterviewState.SCENARIO_BASED),
            InterviewState.SCENARIO_BASED: ("role_specific", InterviewState.ROLE_SPECIFIC),
            InterviewState.ROLE_SPECIFIC: ("final_challenge", InterviewState.FINAL_CHALLENGE),
            InterviewState.FINAL_CHALLENGE: (None, InterviewState.COMPLETED)
        }
        
        current_category, next_state = state_transitions.get(session.state, (None, InterviewState.COMPLETED))
        
        if not current_category:
            session.state = InterviewState.COMPLETED
            self.db_manager.update_session(session_id, state=session.state.value, is_completed=True, end_time=datetime.now())
            return await self.generate_final_report(session_id)
        
        if current_category == "role_specific":
            questions = self.knowledge_base["role_specific"].get(session.role, [])
        else:
            questions = self.knowledge_base.get(current_category, [])
            questions = [q for q in questions if not q.get('role_specific', False) or 
                        session.role in q.get('roles', [])]
        
        if not questions:
            logger.warning(f"No questions found for category {current_category} and role {session.role}")
            session.state = next_state
            session.current_question = 0
            self.db_manager.update_session(session_id, state=session.state.value)
            return await self.get_next_question(session_id)
        
        if session.current_question >= len(questions):
            session.state = next_state
            session.current_question = 0
            self.db_manager.update_session(session_id, state=session.state.value)
            return await self.get_next_question(session_id)
        
        question_data = questions[session.current_question]
        
        if len(session.adaptive_parameters['performance_trend']) >= 2:
            recent_performance = statistics.mean(session.adaptive_parameters['performance_trend'][-2:])
            if recent_performance > 0.8 and session.current_question < len(questions) - 1:
                session.current_question = min(session.current_question + 1, len(questions) - 1)
                question_data = questions[session.current_question]
        
        try:
            question_prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are Alexandra Chen, continuing this world-class Excel assessment. Present the next question with your signature expertise and enthusiasm.

Current Assessment Context:
- Candidate Role: {session.role.value.replace('_', ' ').title()}
- Current Topic: {question_data['topic']}
- Difficulty Level: {question_data['difficulty']}
- Question Type: {question_data['type'].value}
- Demonstrated Skill Level: {session.candidate_profile['skill_level'].value}
- Performance Trend: {'Improving' if len(session.adaptive_parameters['performance_trend']) >= 2 and len(session.adaptive_parameters['performance_trend']) > 0 and session.adaptive_parameters['performance_trend'][-1] > session.adaptive_parameters['performance_trend'][0] else 'Consistent'}

Your Presentation Style:
- Make it conversational and intellectually engaging
- Connect to real business scenarios they'll face in their role  
- Show genuine excitement for advanced Excel capabilities
- Provide context for why this skill matters at senior levels
- For scenario questions, paint vivid business contexts
- Challenge them while remaining encouraging
- Demonstrate your deep expertise naturally

Current question to present: {question_data['question']}"""),
                MessagesPlaceholder(variable_name="chat_history")
            ])
            
            recent_history = session.conversation_history[-3:] if len(session.conversation_history) > 3 else session.conversation_history
            
            response = await self.llm.ainvoke(question_prompt.format_messages(chat_history=recent_history))
            session.conversation_history.append({"role": "assistant", "content": response.content})
            
            formatted_question = response.content if response.content else question_data['question']
            
        except Exception as e:
            logger.error(f"LLM error in get_next_question for session {session_id}: {e}")
            # Fallback to raw question if LLM fails
            formatted_question = question_data['question']
        
        total_questions = sum(
            len(qs) if isinstance(qs, list) else sum(len(role_qs) for role_qs in qs.values())
            for qs in self.knowledge_base.values()
        )
        
        # Increment current_question for next call
        session.current_question += 1
        
        result = {
            "question": formatted_question,
            "topic": question_data['topic'],
            "difficulty": question_data['difficulty'],
            "type": question_data['type'].value,
            "question_number": len(session.responses) + 1,
            "category": current_category,
            "max_score": question_data['max_score'],
            "total_questions": total_questions,
            "progress": (len(session.responses) / 25) * 100,
            "follow_up_questions": question_data.get('follow_up_questions', []),
            "session_state": session.state.value
        }
        
        logger.info(f"Generated question for session {session_id}: {result['question_number']} in category {current_category}")
        return result
    
    async def evaluate_answer(self, session_id: str, answer: str, question_context: Dict, 
                            response_time: float) -> Dict:
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        category = question_context.get('category', 'basic_concepts')
        if category == "role_specific":
            questions = self.knowledge_base["role_specific"].get(session.role, [])
        else:
            questions = self.knowledge_base.get(category, [])
            questions = [q for q in questions if not q.get('role_specific', False) or 
                        session.role in q.get('roles', [])]
        
        # Fix: Use current_question - 1 since get_next_question already incremented it
        # Add bounds checking to prevent index out of range errors
        question_index = session.current_question - 1
        if question_index < 0 or question_index >= len(questions):
            logger.error(f"Question index {question_index} out of bounds for {len(questions)} questions in category {category}")
            # Fallback question data for evaluation
            question_data = {
                'topic': 'Unknown Topic',
                'question': question_context.get('question', 'Unknown Question'),
                'key_points': ['general knowledge', 'problem solving'],
                'max_score': 10,
                'difficulty': 'intermediate',
                'type': QuestionType.THEORETICAL
            }
        else:
            question_data = questions[question_index]
        
        logger.info(f"Evaluating answer for session {session_id}, question index {question_index}/{len(questions)} in category {category}")
        
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are Alexandra Chen, expert Excel assessor. Provide a comprehensive, world-class evaluation of this candidate's response.

Assessment Context:
- Candidate Role: {session.role.value.replace('_', ' ').title()}
- Question Topic: {question_data['topic']}
- Original Question: {question_data['question']}
- Expected Key Points: {', '.join(question_data['key_points'])}
- Maximum Possible Score: {question_data['max_score']}
- Difficulty Level: {question_data['difficulty']}
- Question Type: {question_data['type'].value}
- Response Time: {response_time:.1f} seconds
- Current Skill Assessment: {session.candidate_profile['skill_level'].value}

Expert Evaluation Criteria:
1. Technical Mastery (35%): Deep Excel knowledge, best practices, advanced techniques
2. Business Application (25%): Real-world relevance, practical implementation
3. Strategic Thinking (20%): Architecture, scalability, enterprise considerations  
4. Communication Excellence (10%): Clarity, structure, professional presentation
5. Innovation & Efficiency (10%): Creative solutions, optimization, cutting-edge approaches

Scoring Standards (World-Class Assessment):
- 95-100%: Exceptional mastery, industry-leading expertise, innovative approaches
- 85-94%: Advanced proficiency, strong best practices, excellent business sense
- 70-84%: Solid competency, good fundamentals, practical understanding
- 55-69%: Basic knowledge, some gaps, requires development
- 40-54%: Limited understanding, significant training needed
- Below 40%: Insufficient knowledge for role requirements

Your evaluation should be:
- Precise and evidence-based
- Constructive and developmental
- Honest but encouraging
- Focused on business impact
- Forward-looking (growth potential)

Return JSON with fields:
- score (integer): Actual points earned out of {question_data['max_score']}
- feedback (string): 3-4 sentences of expert insight
- strengths (array): Specific demonstrated capabilities
- improvements (array): Targeted development areas
- confidence (float 0.0-1.0): Assessment confidence level
- skill_level (string): Demonstrated level for this response
- follow_up_needed (boolean): Whether clarification is needed
- business_impact_rating (integer 1-10): Potential business value
- innovation_score (integer 1-10): Creative/advanced approach rating"""),
            ("human", f"Candidate's Response: {answer}")
        ])
        
        parser = JsonOutputParser()
        chain = evaluation_prompt | self.llm | parser
        
        try:
            evaluation = await chain.ainvoke({})
            
            evaluation = {
                "score": min(question_data['max_score'], max(0, evaluation.get('score', 0))),
                "feedback": evaluation.get('feedback', 'Thank you for your comprehensive response.'),
                "strengths": evaluation.get('strengths', []),
                "improvements": evaluation.get('improvements', []),
                "confidence": max(0.0, min(1.0, evaluation.get('confidence', 0.5))),
                "skill_level": evaluation.get('skill_level', 'intermediate'),
                "follow_up_needed": evaluation.get('follow_up_needed', False),
                "business_impact_rating": max(1, min(10, evaluation.get('business_impact_rating', 5))),
                "innovation_score": max(1, min(10, evaluation.get('innovation_score', 5)))
            }
            
        except Exception as e:
            logger.error(f"Advanced evaluation failed: {e}")
            evaluation = await self.fallback_evaluation(answer, question_data, response_time)
        
        semantic_score = await self.calculate_semantic_similarity(answer, question_data['key_points'])
        
        quality_multiplier = 0.75 + (0.25 * semantic_score)
        if response_time < 45:
            quality_multiplier += 0.03
        elif response_time > 600:
            quality_multiplier -= 0.05
        
        final_score = int(evaluation['score'] * quality_multiplier)
        final_score = max(0, min(question_data['max_score'], final_score))
        evaluation['score'] = final_score
        
        session.add_response(
            question=question_data['question'],
            answer=answer,
            evaluation=evaluation,
            category=category,
            response_time=response_time,
            question_data=question_data
        )
        
        if evaluation.get('strengths'):
            strengths = evaluation['strengths'] if isinstance(evaluation['strengths'], list) else [evaluation['strengths']]
            session.candidate_profile['strengths'].extend(strengths)
        
        if evaluation.get('improvements'):
            improvements = evaluation['improvements'] if isinstance(evaluation['improvements'], list) else [evaluation['improvements']]
            session.candidate_profile['weaknesses'].extend(improvements)
        
        # Fix: Remove the duplicate increment - get_next_question already handles this
        # session.current_question += 1  # REMOVED - This was causing the double increment bug
        session.conversation_history.append({"role": "human", "content": answer})
        session.conversation_history.append({"role": "assistant", "content": evaluation['feedback']})
        
        logger.info(f"Expert evaluation completed for session {session_id}: {final_score}/{question_data['max_score']}")
        
        return {
            "score": final_score,
            "max_score": question_data['max_score'],
            "feedback": evaluation['feedback'],
            "strengths": evaluation.get('strengths', []),
            "improvements": evaluation.get('improvements', []),
            "confidence": evaluation.get('confidence', 0.5),
            "skill_level": evaluation.get('skill_level', 'intermediate'),
            "business_impact_rating": evaluation.get('business_impact_rating', 5),
            "innovation_score": evaluation.get('innovation_score', 5),
            "total_score": session.score,
            "total_max_score": session.max_score,
            "percentage": (session.score / session.max_score * 100) if session.max_score > 0 else 0,
            "follow_up_needed": evaluation.get('follow_up_needed', False),
            "performance_trend": session.adaptive_parameters['performance_trend'][-3:] if len(session.adaptive_parameters['performance_trend']) >= 3 else []
        }
    
    async def calculate_semantic_similarity(self, answer: str, key_points: List[str]) -> float:
        try:
            similar_docs = self.vector_store.similarity_search(answer, k=7)
            
            answer_lower = answer.lower()
            key_points_lower = [point.lower() for point in key_points]
            
            exact_matches = sum(1 for point in key_points_lower if point in answer_lower)
            partial_matches = sum(1 for point in key_points_lower 
                                if any(word in answer_lower for word in point.split() if len(word) > 3))
            
            context_bonus = 0.1 if len(answer) > 300 else 0
            technical_terms = ['pivot', 'vlookup', 'index', 'match', 'power', 'query', 'vba', 'macro', 
                             'formula', 'function', 'dashboard', 'model', 'validation', 'automation']
            technical_score = sum(0.05 for term in technical_terms if term in answer_lower)
            
            exact_weight = 0.6
            partial_weight = 0.3
            technical_weight = 0.1
            
            similarity = (exact_matches * exact_weight + 
                         partial_matches * partial_weight + 
                         technical_score * technical_weight + 
                         context_bonus) / len(key_points)
            
            return min(1.0, similarity)
            
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.6
    
    async def fallback_evaluation(self, answer: str, question_data: Dict, response_time: float) -> Dict:
        answer_lower = answer.lower()
        key_points = question_data['key_points']
        
        matches = 0
        total_keywords = 0
        
        for point in key_points:
            keywords = [w for w in point.lower().split() if len(w) > 3]
            point_matches = sum(1 for keyword in keywords if keyword in answer_lower)
            matches += point_matches
            total_keywords += len(keywords)
        
        keyword_score = matches / total_keywords if total_keywords > 0 else 0.2
        length_score = min(1.0, len(answer) / 250)
        time_score = 1.0 if response_time < 240 else max(0.6, 1.0 - (response_time - 240) / 480)
        
        combined_score = (keyword_score * 0.6 + length_score * 0.2 + time_score * 0.2)
        final_score = int(question_data['max_score'] * combined_score)
        
        skill_levels = ["novice", "beginner", "intermediate", "advanced", "expert"]
        skill_index = min(4, int(combined_score * 5))
        skill_level = skill_levels[skill_index]
        
        return {
            "score": final_score,
            "feedback": f"Your response demonstrates {skill_level} understanding. " + 
                       ("Consider providing more comprehensive examples and technical details." if combined_score < 0.7 
                        else "Strong grasp of the core concepts with good practical insight."),
            "strengths": ["Clear communication", "Good structure"] if len(answer) > 150 else ["Concise response"],
            "improvements": ["More technical depth needed", "Add specific examples", "Consider business impact"] if combined_score < 0.6 else ["Minor refinements possible"],
            "confidence": combined_score,
            "skill_level": skill_level,
            "follow_up_needed": combined_score < 0.4,
            "business_impact_rating": max(1, min(10, int(combined_score * 8))),
            "innovation_score": max(1, min(10, int(combined_score * 6) + 2))
        }
    
    async def generate_final_report(self, session_id: str) -> Dict:
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        duration = datetime.now() - session.start_time
        percentage_score = (session.score / session.max_score * 100) if session.max_score > 0 else 0
        
        category_scores = defaultdict(list)
        for response in session.responses:
            category_scores[response['category']].append(
                response['score'] / response['max_possible']
            )
        
        category_averages = {
            category: statistics.mean(scores) * 100 for category, scores in category_scores.items()
        }
        
        if percentage_score >= 92:
            overall_level = "Exceptional Expert"
            recommendation = "Strongly Recommend - Top Tier"
        elif percentage_score >= 85:
            overall_level = "Advanced Expert"
            recommendation = "Strongly Recommend"
        elif percentage_score >= 75:
            overall_level = "Senior Advanced"
            recommendation = "Recommend"
        elif percentage_score >= 65:
            overall_level = "Solid Intermediate"
            recommendation = "Consider"
        elif percentage_score >= 50:
            overall_level = "Developing Intermediate"
            recommendation = "Consider with Training"
        else:
            overall_level = "Entry Level"
            recommendation = "Extensive Training Required"
        
        avg_response_time = statistics.mean(session.candidate_profile['response_times']) if session.candidate_profile['response_times'] else 0
        confidence_trend = session.candidate_profile['confidence_scores']
        
        business_impact_scores = [r.get('business_impact_rating', 5) for r in [resp['evaluation'] for resp in session.responses]]
        innovation_scores = [r.get('innovation_score', 5) for r in [resp['evaluation'] for resp in session.responses]]
        
        avg_business_impact = statistics.mean(business_impact_scores) if business_impact_scores else 5
        avg_innovation = statistics.mean(innovation_scores) if innovation_scores else 5
        
        report_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Generate the definitive, world-class Excel skills assessment report for a Fortune 500 hiring decision.

COMPREHENSIVE PERFORMANCE DATA:
- Target Role: {session.role.value.replace('_', ' ').title()}
- Final Score: {session.score}/{session.max_score} ({percentage_score:.1f}%)
- Overall Assessment: {overall_level}
- Assessment Duration: {duration}
- Questions Mastered: {len(session.responses)}
- Average Response Time: {avg_response_time:.1f}s
- Confidence Progression: {confidence_trend}
- Business Impact Rating: {avg_business_impact:.1f}/10
- Innovation Score: {avg_innovation:.1f}/10

CATEGORY MASTERY BREAKDOWN:
{json.dumps(category_averages, indent=2)}

DEMONSTRATED STRENGTHS:
{chr(10).join(f"• {strength}" for strength in list(set(session.candidate_profile['strengths']))[:8]) if session.candidate_profile['strengths'] else "• Strong foundational knowledge"}

DEVELOPMENT OPPORTUNITIES  
{chr(10).join(f"• {weakness}" for weakness in list(set(session.candidate_profile['weaknesses']))[:6]) if session.candidate_profile['weaknesses'] else "• Minor refinements recommended"}

EXECUTIVE ASSESSMENT REPORT STRUCTURE:

## EXECUTIVE SUMMARY
- Clear recommendation with confidence level
- Overall Excel mastery assessment
- Role alignment and business readiness
- Key differentiating factors

## TECHNICAL EXCELLENCE ANALYSIS
**Category-by-Category Mastery Assessment:**
- Detailed breakdown of each technical area
- Standout capabilities and expertise depth
- Areas requiring attention or development
- Comparison to role requirements

## BUSINESS IMPACT POTENTIAL
- Practical application of skills
- Strategic thinking and architecture capability
- Innovation and efficiency mindset
- Problem-solving approach and methodology

## ROLE-SPECIFIC ASSESSMENT
- Critical alignment with {session.role.value.replace('_', ' ').title()} requirements
- Essential skills verification
- Advanced capabilities for role excellence
- Growth trajectory and learning potential

## COMPREHENSIVE RECOMMENDATIONS
- Hiring decision with detailed rationale
- Onboarding and integration suggestions
- Professional development roadmap
- Performance expectations and timeline

## STANDOUT MOMENTS & INSIGHTS
- Best responses and technical demonstrations
- Unique approaches or exceptional insights
- Areas where candidate exceeded expectations
- Notable patterns in problem-solving style

Create a comprehensive, authoritative report that executives and HR leaders can confidently use for critical hiring decisions. Focus on business impact, technical excellence, and strategic value."""),
            ("human", "Generate the definitive Excel mastery assessment report.")
        ])
        
        try:
            report_response = await self.llm.ainvoke(report_prompt.format_messages())
            detailed_report = report_response.content
        except Exception as e:
            logger.error(f"Premium report generation failed: {e}")
            detailed_report = self._generate_premium_fallback_report(session, percentage_score, overall_level, recommendation)
        
        session.state = InterviewState.COMPLETED
        self.db_manager.update_session(session_id, 
            state=session.state.value, 
            is_completed=True, 
            end_time=datetime.now()
        )
        
        response_breakdown = []
        for i, response in enumerate(session.responses, 1):
            eval_data = response.get('evaluation', {})
            response_breakdown.append({
                "question_number": i,
                "category": response['category'],
                "topic": response.get('topic', 'Unknown'),
                "difficulty": response.get('difficulty', 'intermediate'),
                "score": f"{response['score']}/{response['max_possible']}",
                "percentage": round((response['score'] / response['max_possible']) * 100, 1),
                "response_time": f"{response['response_time']:.1f}s",
                "business_impact": eval_data.get('business_impact_rating', 5),
                "innovation_score": eval_data.get('innovation_score', 5),
                "key_strengths": eval_data.get('strengths', [])[:3],
                "development_areas": eval_data.get('improvements', [])[:2]
            })
        
        performance_consistency = (
            round(statistics.stdev([r['score']/r['max_possible'] for r in session.responses]) * 100, 1) 
            if len(session.responses) > 1 else 0
        )
        
        logger.info(f"Generated world-class report for session {session_id}: {percentage_score:.1f}% - {recommendation}")
        
        return {
            "completed": True,
            "report": detailed_report,
            "executive_summary": {
                "recommendation": recommendation,
                "overall_score": f"{session.score}/{session.max_score}",
                "percentage": round(percentage_score, 1),
                "skill_level": overall_level,
                "role": session.role.value.replace('_', ' ').title(),
                "assessment_duration": str(duration).split('.')[0],
                "questions_completed": len(session.responses),
                "business_readiness": "High" if percentage_score >= 75 else "Moderate" if percentage_score >= 60 else "Developing"
            },
            "advanced_metrics": {
                "category_performance": category_averages,
                "avg_response_time": round(avg_response_time, 1),
                "confidence_progression": confidence_trend,
                "performance_consistency": performance_consistency,
                "business_impact_average": round(avg_business_impact, 1),
                "innovation_average": round(avg_innovation, 1),
                "learning_velocity": self._calculate_learning_velocity(session),
                "technical_depth_score": round(percentage_score * (avg_innovation / 10), 1)
            },
            "detailed_breakdown": response_breakdown,
            "strategic_insights": {
                "top_strengths": list(set(session.candidate_profile['strengths']))[:8],
                "priority_development": list(set(session.candidate_profile['weaknesses']))[:6],
                "learning_patterns": self._assess_learning_patterns(session),
                "role_alignment": self._assess_role_alignment(session),
                "growth_potential": self._assess_growth_potential(session, percentage_score),
                "business_impact_forecast": self._forecast_business_impact(session, avg_business_impact)
            }
        }
    
    def _generate_premium_fallback_report(self, session: InterviewSession, percentage_score: float, 
                                        overall_level: str, recommendation: str) -> str:
        return f"""
# Executive Excel Mastery Assessment Report

## EXECUTIVE SUMMARY
**Candidate Assessment**: {overall_level} Excel practitioner with {percentage_score:.1f}% mastery score
**Hiring Recommendation**: {recommendation} for {session.role.value.replace('_', ' ').title()} position
**Business Readiness**: {'High' if percentage_score >= 75 else 'Moderate' if percentage_score >= 60 else 'Developing'}

## TECHNICAL PERFORMANCE OVERVIEW
- **Final Score**: {session.score}/{session.max_score} ({percentage_score:.1f}%)
- **Questions Completed**: {len(session.responses)}
- **Assessment Duration**: {datetime.now() - session.start_time}
- **Skill Progression**: Demonstrated growth during assessment

## KEY STRENGTHS IDENTIFIED
{chr(10).join(f"• {strength}" for strength in list(set(session.candidate_profile['strengths']))[:8]) if session.candidate_profile['strengths'] else "• Strong foundational knowledge"}

## DEVELOPMENT OPPORTUNITIES  
{chr(10).join(f"• {weakness}" for weakness in list(set(session.candidate_profile['weaknesses']))[:6]) if session.candidate_profile['weaknesses'] else "• Minor refinements recommended"}

## STRATEGIC RECOMMENDATION
Based on comprehensive technical assessment, the candidate demonstrates {overall_level.lower()} Excel capabilities with strong potential for the {session.role.value.replace('_', ' ').title()} role. Recommended for {recommendation.lower()}.
"""
    
    def _calculate_learning_velocity(self, session: InterviewSession) -> str:
        if len(session.adaptive_parameters['performance_trend']) < 3:
            return "Standard"
        
        trend = session.adaptive_parameters['performance_trend']
        early_avg = statistics.mean(trend[:len(trend)//2])
        later_avg = statistics.mean(trend[len(trend)//2:])
        
        if later_avg > early_avg + 0.15:
            return "Rapid"
        elif later_avg > early_avg + 0.05:
            return "Accelerating"
        elif later_avg >= early_avg - 0.05:
            return "Consistent"
        else:
            return "Variable"
    
    def _assess_learning_patterns(self, session: InterviewSession) -> List[str]:
        patterns = []
        
        if len(session.adaptive_parameters['performance_trend']) >= 3:
            trend = session.adaptive_parameters['performance_trend']
            if trend[-1] > trend[0] + 0.1:
                patterns.append("Strong improvement trajectory")
            if statistics.mean(trend[-3:]) > statistics.mean(trend[:3]):
                patterns.append("Excellent learning agility")
            if max(trend) - min(trend) < 0.15:
                patterns.append("Highly consistent performance")
        
        if session.candidate_profile['response_times']:
            times = session.candidate_profile['response_times']
            if len(times) >= 3 and times[-1] < times[0] * 0.8:
                patterns.append("Increasing efficiency over time")
            avg_time = statistics.mean(times)
            if avg_time < 180:
                patterns.append("Quick analytical processing")
            elif avg_time > 300:
                patterns.append("Methodical, thorough approach")
        
        return patterns or ["Standard learning progression observed"]
    
    def _assess_role_alignment(self, session: InterviewSession) -> List[str]:
        role_alignments = {
            JobRole.FINANCIAL_ANALYST: [
                "Financial modeling expertise evaluated",
                "Data validation and accuracy emphasis",
                "Scenario analysis capabilities reviewed",
                "Audit-ready documentation approach"
            ],
            JobRole.DATA_ANALYST: [
                "Statistical analysis proficiency assessed",
                "Data transformation and ETL skills evaluated",
                "Visualization and reporting excellence reviewed",
                "Large dataset handling capabilities"
            ],
            JobRole.OPERATIONS_ANALYST: [
                "Process optimization mindset demonstrated",
                "Automation and efficiency focus confirmed",
                "Resource allocation modeling capability",
                "Performance metrics design skill"
            ],
            JobRole.BUSINESS_ANALYST: [
                "Requirements analysis approach evaluated",
                "Stakeholder communication skills assessed",
                "Solution architecture thinking reviewed",
                "Business process integration capability"
            ],
            JobRole.CONSULTANT: [
                "Client presentation readiness assessed",
                "Complex problem-solving methodology evaluated",
                "Best practice knowledge demonstrated",
                "Strategic thinking capability confirmed"
            ]
        }
        
        return role_alignments.get(session.role, ["General Excel proficiency comprehensively assessed"])
    
    def _assess_growth_potential(self, session: InterviewSession, percentage_score: float) -> str:
        learning_velocity = self._calculate_learning_velocity(session)
        
        if percentage_score >= 85 and learning_velocity in ["Rapid", "Accelerating"]:
            return "Exceptional - Leadership potential"
        elif percentage_score >= 75 and learning_velocity != "Variable":
            return "High - Senior contributor trajectory"
        elif percentage_score >= 60:
            return "Strong - Solid advancement potential"
        elif learning_velocity in ["Rapid", "Accelerating"]:
            return "Promising - High learning capacity"
        else:
            return "Standard - Steady development expected"
    
    def _forecast_business_impact(self, session: InterviewSession, avg_business_impact: float) -> str:
        if avg_business_impact >= 8.5:
            return "Transformational - Will drive significant efficiency gains"
        elif avg_business_impact >= 7.0:
            return "High - Will deliver measurable improvements"
        elif avg_business_impact >= 5.5:
            return "Positive - Will contribute to team productivity"
        else:
            return "Standard - Will meet basic role requirements"

interviewer = ExcelInterviewer()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.connection_metadata[session_id] = {
            "connected_at": datetime.now(),
            "last_activity": datetime.now(),
            "message_count": 0
        }
        logger.info(f"Premium WebSocket connected for session {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.connection_metadata:
            del self.connection_metadata[session_id]
        logger.info(f"WebSocket disconnected for session {session_id}")
    
    async def send_message(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(message)
                if session_id in self.connection_metadata:
                    self.connection_metadata[session_id]["last_activity"] = datetime.now()
                    self.connection_metadata[session_id]["message_count"] += 1
            except Exception as e:
                logger.error(f"Failed to send message to {session_id}: {e}")
                self.disconnect(session_id)

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 World's Premier Excel Mock Interviewer System Starting...")
    yield
    logger.info("✅ System Shutdown Complete")

app = FastAPI(
    title="AI Excel Mastery Interviewer Pro", 
    description="The World's Most Advanced Excel Skills Assessment Platform - Enterprise Grade",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    analytics = interviewer.db_manager.get_analytics_summary()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "analytics": analytics
    })

class StartRequest(BaseModel):
    role: str = "financial_analyst"
    candidate_name: Optional[str] = None
    candidate_email: Optional[str] = None

@app.post("/start-interview")
async def start_interview_endpoint(request: StartRequest, background_tasks: BackgroundTasks):
    try:
        session_id = interviewer.create_session(request.role, request.candidate_name, request.candidate_email)
        intro_message = await interviewer.start_interview(session_id)
        
        background_tasks.add_task(cleanup_session, session_id, delay=10800)  # 3 hours
        
        return {
            "session_id": session_id,
            "message": intro_message,
            "status": "started",
            "role": request.role,
            "candidate_name": request.candidate_name,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to start premium interview: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize interview system")

async def cleanup_session(session_id: str, delay: int):
    await asyncio.sleep(delay)
    if session_id in interviewer.sessions:
        del interviewer.sessions[session_id]
        logger.info(f"Cleaned up session {session_id}")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            timestamp = datetime.now()
            
            if action == "get_question":
                question_data = await interviewer.get_next_question(session_id)
                question_data["timestamp"] = timestamp.isoformat()
                await manager.send_message({
                    "type": "question",
                    "data": question_data
                }, session_id)
            
            elif action == "submit_answer":
                answer = data.get("answer", "").strip()
                question_context = data.get("context", {})
                response_time = data.get("response_time", 0)
                
                if not answer:
                    await manager.send_message({
                        "type": "error",
                        "message": "Please provide a comprehensive answer before submitting."
                    }, session_id)
                    continue
                
                if len(answer) < 20:
                    await manager.send_message({
                        "type": "warning",
                        "message": "Your response seems quite brief. Consider providing more detailed explanation for better assessment."
                    }, session_id)
                
                evaluation = await interviewer.evaluate_answer(
                    session_id, answer, question_context, response_time
                )
                evaluation["timestamp"] = timestamp.isoformat()
                
                await manager.send_message({
                    "type": "evaluation", 
                    "data": evaluation
                }, session_id)
                
                session = interviewer.get_session(session_id)
                if session and session.state == InterviewState.COMPLETED:
                    final_report = await interviewer.generate_final_report(session_id)
                    final_report["timestamp"] = timestamp.isoformat()
                    await manager.send_message({
                        "type": "final_report",
                        "data": final_report
                    }, session_id)
            
            elif action == "get_progress":
                session = interviewer.get_session(session_id)
                if session:
                    progress_data = {
                        "current_state": session.state.value,
                        "questions_completed": len(session.responses),
                        "current_score": session.score,
                        "max_possible_score": session.max_score,
                        "percentage": (session.score / session.max_score * 100) if session.max_score > 0 else 0,
                        "skill_level": session.candidate_profile['skill_level'].value,
                        "duration": str(datetime.now() - session.start_time).split('.')[0],
                        "performance_trend": session.adaptive_parameters['performance_trend'][-5:] if len(session.adaptive_parameters['performance_trend']) >= 5 else session.adaptive_parameters['performance_trend']
                    }
                    await manager.send_message({
                        "type": "progress",
                        "data": progress_data
                    }, session_id)
                    
            elif action == "get_hint":
                session = interviewer.get_session(session_id)
                if session and len(session.responses) > 0:
                    await manager.send_message({
                        "type": "hint",
                        "message": "Consider breaking down your answer into: 1) Technical approach, 2) Business context, 3) Implementation considerations, and 4) Potential challenges."
                    }, session_id)
                    
            elif action == "ping":
                await manager.send_message({
                    "type": "pong", 
                    "timestamp": timestamp.isoformat(),
                    "status": "connected"
                }, session_id)
                
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await manager.send_message({
            "type": "error",
            "message": "Connection error occurred. Please refresh the page."
        }, session_id)
        manager.disconnect(session_id)

@app.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    session = interviewer.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "state": session.state.value,
        "role": session.role.value,
        "current_question": session.current_question,
        "score": session.score,
        "max_score": session.max_score,
        "percentage": (session.score / session.max_score * 100) if session.max_score > 0 else 0,
        "responses_count": len(session.responses),
        "duration": str(datetime.now() - session.start_time).split('.')[0],
        "skill_level": session.candidate_profile['skill_level'].value,
        "is_completed": session.state == InterviewState.COMPLETED
    }

@app.get("/session/{session_id}/report")
async def get_session_report(session_id: str):
    session = interviewer.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.state != InterviewState.COMPLETED:
        raise HTTPException(status_code=400, detail="Interview not completed yet")
    
    return await interviewer.generate_final_report(session_id)

@app.get("/analytics")
async def get_analytics():
    try:
        analytics = interviewer.db_manager.get_analytics_summary(days=30)
        return {
            "status": "success",
            "data": analytics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Analytics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Analytics temporarily unavailable")

@app.get("/health")
async def health_check():
    try:
        db_status = "healthy"
        try:
            interviewer.db_manager.get_analytics_summary(days=1)
        except:
            db_status = "degraded"
            
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "active_sessions": len(interviewer.sessions),
            "active_connections": len(manager.active_connections),
            "database_status": db_status,
            "version": "3.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(status_code=503, content={
            "status": "degraded",
            "error": "System health check failed"
        })

@app.get("/roles")
async def get_available_roles():
    return {
        "roles": [
            {
                "value": role.value,
                "name": role.value.replace('_', ' ').title(),
                "description": {
                    "financial_analyst": "Advanced financial modeling, valuation, and analysis expertise",
                    "data_analyst": "Statistical analysis, data visualization, and business intelligence",
                    "operations_analyst": "Process optimization, resource planning, and performance metrics",
                    "business_analyst": "Requirements analysis, solution design, and stakeholder management",
                    "consultant": "Strategic problem-solving, client presentation, and best practices"
                }.get(role.value, "Excel expertise assessment")
            } for role in JobRole
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
