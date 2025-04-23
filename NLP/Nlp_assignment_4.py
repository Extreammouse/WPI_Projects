# Task 4: LangChain with Gemini

import os, logging
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableSequence

class QA(BaseModel):
    analysis: str
    answer: str

json_parser = PydanticOutputParser(pydantic_object=QA)
single_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Return ONLY valid JSON that matches this schema:\n{schema}"),
    ("human", "QUESTION: {question}")
]).partial(schema=json_parser.get_format_instructions())

llm_flash = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

custom_chain = single_prompt | llm_flash | json_parser
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

os.environ["GOOGLE_API_KEY"] = ""
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def list_available_models() -> list[str]:
    try:
        return [
            m.name.replace("models/", "")      # strip prefix if present
            for m in genai.list_models()
            if "gemini" in m.name.lower() and "1.0" not in m.name.lower()
        ]
    except Exception as exc:
        logging.error("Could not list models: %s", exc)
        return []

available = list_available_models()
model_name = next((m for m in available if m == "gemini-1.5-flash"),
                  available[0] if available else "gemini-1.5-flash")
print(f"💡 Using model → {model_name}")

llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.4)
print("Model test:", llm.invoke("Hi!"))

output_parser = StrOutputParser()

# ======================================================================
# SEQUENTIAL CHAIN 1  
# ======================================================================
problem_prompt  = ChatPromptTemplate.from_template(
    "Given the following user query, extract the core problem:\n{query}\n\nCore problem:")
context_prompt  = ChatPromptTemplate.from_template(
    "Problem:\n{problem}\n\nGenerate background information helpful to solve it:")
solution_prompt = ChatPromptTemplate.from_template(
    "Problem: {problem}\nContext: {context}\n\nProvide a comprehensive solution:")

problem_chain   = problem_prompt  | llm | output_parser
context_chain   = context_prompt  | llm | output_parser
solution_chain  = solution_prompt | llm | output_parser

def sequential_chain_1(inputs: Dict[str, Any]) -> Dict[str, str]:
    try:
        problem  = problem_chain.invoke({"query": inputs["query"]})
        context  = context_chain.invoke({"problem": problem})
        solution = solution_chain.invoke({"problem": problem, "context": context})
        return {"problem": problem, "context": context, "solution": solution}
    except Exception as exc:
        logging.error("Chain 1 error: %s", exc)
        return {"problem": f"Error: {exc}", "context": "", "solution": ""}

def test_chain_1():
    res = sequential_chain_1({"query": "How can I improve my ML model's accuracy?"})
    print("\n--- CHAIN 1 RESULTS ---")
    for k, v in res.items():
        print(f"\n{k.upper()}:\n{v}")

# ======================================================================
# SEQUENTIAL CHAIN 2  ---------------------------------------------------
#   Summary  ->  Clarifying questions  ->  Action plan
# ======================================================================
summary_prompt = ChatPromptTemplate.from_template(
    "Summarise the user's request in one sentence:\n{query}\n\nSummary:")
clarify_prompt = ChatPromptTemplate.from_template(
    "Summary: {summary}\n\nAsk three clarifying questions you would need answered:")
plan_prompt    = ChatPromptTemplate.from_template(
    "Summary: {summary}\nAnswers to clarifying questions: {answers}\n\n"
    "Draft a step‑by‑step action plan:")

summary_chain  = summary_prompt  | llm | output_parser
clarify_chain  = clarify_prompt  | llm | output_parser
plan_chain     = plan_prompt    | llm | output_parser

def sequential_chain_2(inputs: Dict[str, Any]) -> Dict[str, str]:
    try:
        summary  = summary_chain.invoke({"query": inputs["query"]})
        questions = clarify_chain.invoke({"summary": summary})
        stub_answers = "1. ___\n2. ___\n3. ___"
        plan = plan_chain.invoke({"summary": summary, "answers": stub_answers})
        return {"summary": summary, "clarifying_questions": questions, "action_plan": plan}
    except Exception as exc:
        logging.error("Chain 2 error: %s", exc)
        return {"summary": "", "clarifying_questions": "", "action_plan": f"Error: {exc}"}

def test_chain_2():
    res = sequential_chain_2({"query": "I need to deploy my Flask app on AWS with CI/CD; what should I do?"})
    print("\n--- CHAIN 2 RESULTS ---")
    for k, v in res.items():
        print(f"\n{k.upper()}:\n{v}")

# =====================================================================
# CUSTOM CHAIN 3  – single call, strict JSON, auto‑parsed
# =====================================================================
class QA(BaseModel):
    analysis: str = Field(description="Key challenges extracted from the question")
    answer: str = Field(description="Concise recommendation/answer")

json_parser = PydanticOutputParser(pydantic_object=QA)

llm_flash = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)

single_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Return ONLY valid JSON that matches this schema:\n{schema}"),
    ("human", "QUESTION: {question}")
]).partial(schema=json_parser.get_format_instructions())

custom_chain = single_prompt | llm_flash | json_parser

def test_custom_chain():
    try:
        res: QA = custom_chain.invoke({"question": "What are the trade‑offs between SQL and NoSQL databases?"})
        print("\n--- CUSTOM CHAIN 3 RESULTS ---")
        print("\nANALYSIS:\n", res.analysis)
        print("\nANSWER:\n",   res.answer)
    except Exception as exc:
        print(f"Chain 3 failed: {exc}")


def main():
    print("Testing Sequential Chain 1...")
    test_chain_1()

    print("\nTesting Sequential Chain 2...")
    test_chain_2()

    print("\nTesting Custom Chain 3...")
    test_custom_chain()

if __name__ == "__main__":
    main()