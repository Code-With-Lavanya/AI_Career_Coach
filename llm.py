from urllib import response

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

llm = ChatMistralAI(
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0
)

def pdf_system(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    return texts
embedding_model = HuggingFaceEmbeddings()

pdfs = [
    "resume.pdf",
    "jd.pdf",
    "company.pdf"
]

all_docs = []

for pdf in pdfs:
    chunks = pdf_system(pdf)

    for chunk in chunks:
        chunk.metadata["source"] = pdf

    all_docs.extend(chunks)

if os.path.exists("chroma_db"):
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )
else:
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )
retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k":10,
        "lambda_mult" :0.5
    }
)
ats_prompt = ChatPromptTemplate.from_template("""
You are an ATS evaluator.

Analyze the resume and provide:

1. ATS Score
2. Strengths
3. Weaknesses
4. Missing Skills
5. Improvements

Resume:

{context}
""")
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert AI Career Coach, Senior Recruiter, ATS Specialist, and Technical Interview Mentor.

Your responsibility is to help users improve their resumes, understand job descriptions, identify skill gaps, prepare for interviews, and generate professional career-related documents.

You must answer only using the provided context retrieved from the knowledge base. The context may include:

* Candidate Resume
* Job Description (JD)
* Company Information
* Interview Notes
* Career Resources

Rules:

1. Use the retrieved context as the primary source of truth.
2. If the answer is not present in the context, clearly say:
   "I could not find sufficient information in the provided documents."
3. Do not hallucinate facts.
4. Be professional, concise, and actionable.
5. When comparing Resume and JD, explicitly mention:

   * Matching Skills
   * Missing Skills
   * Recommended Improvements
6. When generating interview questions:

   * Include HR Questions
   * Technical Questions
   * Scenario-Based Questions
7. When generating ATS analysis:

   * Provide ATS Score (estimated)
   * Strengths
   * Weaknesses
   * Action Items
8. Use bullet points whenever possible.
9. Format output clearly using markdown.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

USER QUESTION:
{question}

ASSISTANT RESPONSE:
Think step-by-step using the provided context and generate a helpful response."""),
    ("user", "{question}")
])
interview_prompt = ChatPromptTemplate.from_template("""
You are a Senior Technical Interviewer and Hiring Manager.

Using the provided context, generate interview questions that are highly relevant to the candidate's profile.

CONTEXT:
{context}

Instructions:

1. Generate 10 HR Questions.
2. Generate 10 Technical Questions.
3. Generate 5 Scenario-Based Questions.
4. Generate 5 Project-Based Questions.
5. Questions should be tailored to the candidate's skills, projects, and experience.
6. Avoid generic questions.
7. Format the response in markdown.

Output Format:

# HR Questions
- Question 1
- Question 2

# Technical Questions
- Question 1
- Question 2

# Scenario-Based Questions
- Question 1
- Question 2

# Project-Based Questions
- Question 1
- Question 2
""")
roadmap_prompt = ChatPromptTemplate.from_template("""
You are an Expert Career Mentor and Industry Coach.

Analyze the candidate's profile and identify missing skills, technologies, and knowledge areas.

CONTEXT:
{context}

Instructions:

1. Identify important missing skills.
2. Explain why each skill matters.
3. Create a practical learning roadmap.
4. Divide roadmap into weekly milestones.
5. Prioritize skills based on industry demand.
6. Suggest projects for each phase.
7. Keep roadmap realistic and actionable.

Output Format:

# Missing Skills
- Skill 1
- Skill 2

# Why These Skills Matter
- Explanation

# Learning Roadmap

## Week 1
Topics:
Projects:

## Week 2
Topics:
Projects:

## Week 3
Topics:
Projects:

## Week 4
Topics:
Projects:

# Final Portfolio Projects
- Project 1
- Project 2
- Project 3
""")
ats_chain = ats_prompt | llm

interview_chain = interview_prompt | llm

roadmap_chain = roadmap_prompt | llm
chat_history = []
print("Rag system created ")

print("press 0 to exit ")

def run_query(
    query,
    retriever,
    llm,
    chat_history=[]
):

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    sources = list({
        doc.metadata.get(
            "source",
            "Unknown"
        )
        for doc in docs
    })

    if "analyze" in query.lower():

        response = ats_chain.invoke({
            "context": context
        })

    elif "interview" in query.lower():

        response = interview_chain.invoke({
            "context": context
        })

    elif "roadmap" in query.lower():

        response = roadmap_chain.invoke({
            "context": context
        })

    else:

        final_prompt = prompt.invoke({
            "context": context,
            "question": query,
            "chat_history": chat_history
        })

        response = llm.invoke(
            final_prompt
        )
    return (
        response.content,
        sources
    )
while True:
    query = input("You : ")
    if query == "0":
        break 
    answer, sources = run_query(
    query=query,
    retriever=retriever,
    llm=llm,
    chat_history=[]
)
    print("\nAI:", answer)
    print("\nSources:")
    for source in sources:
        print(source)
        