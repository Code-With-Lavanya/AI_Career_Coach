# AI Career Copilot 🚀

An AI-powered Career Assistant built using LangChain, ChromaDB, HuggingFace Embeddings, Mistral AI, and Streamlit.

The application helps users analyze resumes, evaluate ATS compatibility, generate interview questions, identify skill gaps, and create personalized learning roadmaps using Retrieval-Augmented Generation (RAG).

## Features

### Resume Analysis

* ATS Score Estimation
* Resume Strengths
* Resume Weaknesses
* Missing Skills Detection
* Resume Improvement Suggestions

### Interview Preparation

* HR Interview Questions
* Technical Interview Questions
* Scenario-Based Questions
* Project-Based Questions

### Personalized Roadmap Generation

* Skill Gap Analysis
* Weekly Learning Plan
* Industry-Relevant Skill Recommendations
* Project Suggestions

### Multi-Document RAG

Supports retrieval across multiple documents such as:

* Resume
* Job Description (JD)
* Company Information
* Interview Notes

### Source Attribution

Displays document sources used to generate responses.

---

## Tech Stack

### LLM

* Mistral AI

### Frameworks

* LangChain
* Streamlit

### Vector Database

* ChromaDB

### Embeddings

* HuggingFace Embeddings

### Document Processing

* PyMuPDFLoader
* RecursiveCharacterTextSplitter

### Retrieval Strategy

* Maximum Marginal Relevance (MMR)

---

## Project Architecture

User Query
↓
Retriever (ChromaDB)
↓
Relevant Chunks
↓
Prompt Routing
↓
ATS Chain / Interview Chain / Roadmap Chain / General RAG Chain
↓
Mistral AI
↓
Response + Sources

---

## Specialized Chains

### ATS Analysis Chain

Provides:

* ATS Score
* Strengths
* Weaknesses
* Missing Skills
* Improvement Suggestions

### Interview Generation Chain

Generates:

* HR Questions
* Technical Questions
* Scenario-Based Questions
* Project-Based Questions

### Roadmap Generation Chain

Provides:

* Skill Gap Analysis
* Weekly Learning Plan
* Project Recommendations

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ai-career-copilot.git
cd ai-career-copilot
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
MISTRAL_API_KEY=your_api_key_here
```

Run the application:

```bash
streamlit run app.py
```

---

## Example Queries

* Analyze my resume
* Generate interview questions
* Create a learning roadmap
* What skills are missing from my profile?
* Compare my resume with the job description
* What projects should I build for this role?

---

## Future Improvements

* FastAPI Backend
* User Authentication
* Chat History Persistence
* PDF Report Export
* Cloud Vector Database
* Resume vs JD Match Percentage
* Deployment Monitoring

---

## Author

Lavanya Singh

Aspiring AI/ML Engineer passionate about Generative AI, RAG Systems, Machine Learning, Deep Learning, and AI-powered applications.
