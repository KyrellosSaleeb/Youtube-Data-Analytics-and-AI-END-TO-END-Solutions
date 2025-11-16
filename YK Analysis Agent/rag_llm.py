# Install all required packages
# %pip install --upgrade pip
# %pip install pyngrok
# %pip install langchain langchain-community langchain-core langchain-text-splitters
# %pip install langchain-groq langchain-openai
# %pip install python-dotenv
# %pip install PyPDF2 pypdf
# %pip install sentence-transformers
# %pip install faiss-cpu
# %pip install streamlit
# %pip uninstall -y langchain langchain-core langchain-community langchain-groq langchain-openai
# %pip install "langchain==1.0.0" "langchain-community==1.0.0" "langchain-openai==1.0.0" "langchain-groq==1.0.0" faiss-cpu
# %pip install langchain_groq

import os
# import langchain
# from langchain_community.utilities import SQLDatabase
import urllib.parse
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from sqlalchemy import inspect, create_engine
import adodbapi
# import json

import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnableSequence
# from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import StreamingStdOutCallbackHandler

import pandas as pd
import streamlit as st

"""# Connecting to the LLM API"""

# Absolute or relative path to your .env file
dotenv_path = r"D:\Langchain Study\LangChain Projetcs\1-Analysis Generator RAG\MySecrets.env"

# Load environment variables from that file
load_dotenv(dotenv_path=dotenv_path)

# Access your secret key
secret_key = os.getenv("GROQ_KEY")

# Identify and connect to the Required reasoning LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    top_p=0.90,
    include_reasoning=True,
    groq_api_key = secret_key,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

"""# Data Loading

**Loading the data from PDF**
"""

def load_pdf(path):
    """
    Loads a PDF document using LangChain's PyPDFLoader.
    Returns a list of Document objects.
    """
    print(f"📄 Loading PDF: {path}")
    loader = PyPDFLoader(path)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} pages.")
    return docs

"""**Smart Paragraph Splitter**"""

def smart_paragraph_splitter(documents, max_chunk_size=1000):
    """
    Splits PDF documents into clean, paragraph-based chunks.
    Handles section headers (e.g., '2.') and very long paragraphs gracefully.
    Removes all '\n' and excessive whitespace for embedding readiness.
    """
    all_chunks = []

    for doc in documents:
        text = doc.page_content

        # Normalize all whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\r', '\n', text)

        # Split paragraphs on multiple newlines or section headers (e.g., "2.", "3.")
        paragraphs = re.split(r'\n{2,}|\n\s*\d+\.\s', text)

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue

            # 🧹 Clean paragraph (remove internal newlines, double spaces)
            p = re.sub(r'\s*\n\s*', ' ', p)
            p = re.sub(r'\s{2,}', ' ', p)

            # ✅ If paragraph is small enough → keep it whole
            if len(p) <= max_chunk_size:
                all_chunks.append(p)
            else:
                # ⚙️ If paragraph is too long → split by sentences
                sentences = re.split(r'(?<=[.!?]) +', p)
                temp_chunk = ""
                for s in sentences:
                    if len(temp_chunk) + len(s) < max_chunk_size:
                        temp_chunk += s + " "
                    else:
                        all_chunks.append(temp_chunk.strip())
                        temp_chunk = s + " "
                if temp_chunk.strip():
                    all_chunks.append(temp_chunk.strip())

    print(f"✅ Total paragraph-aware chunks: {len(all_chunks)}")
    return all_chunks

"""**Apply the data Splitter of the PDF to get the Chunks**"""

pdf_path = r"D:\Langchain Study\LangChain Projetcs\1-Analysis Generator RAG\DataFiles\YouTube_Trending_Video_Dataset_Full_Report.pdf"

docs = load_pdf(pdf_path)
paragraph_chunks = smart_paragraph_splitter(docs)

# # View sample
# print("\n📘 Example chunk:")
# print(paragraph_chunks[1])

"""# Chunks Embedding"""

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Convert your cleaned chunks to embeddings
texts = paragraph_chunks
# print("🔍 Creating embeddings for text chunks...")
embeddings = embedding_model.embed_documents(texts)
# print(f"✅ Created {len(embeddings)} embeddings.")

"""# Creating Vector Database"""

# # Create FAISS index from texts + embeddings
# vector_db = FAISS.from_texts(texts, embedding_model)

# # Save index to disk
# vector_db.save_local("youtube_trending_faiss")
# print("💾 Saved FAISS vector database: youtube_trending_faiss")

# Load saved FAISS index
db = FAISS.load_local("youtube_trending_faiss", embedding_model, allow_dangerous_deserialization=True)

"""# Build Retrieval chain

**Function To Retrieve data from the vector database**
"""

retriever = db.as_retriever(search_kwargs={"k": 3})

template = """
You are an intelligent AI assistant for YouTube Trending Data Analysis.
Use the context from documentation to answer accurately.

Context:
{context}

Question:
{question}

Answer:
"""

# Define prompt
prompt = ChatPromptTemplate.from_template(template = template)

rag_chain = (
    {
        "context": lambda x: "\n\n".join(
            d.page_content for d in retriever.invoke(x["question"])
        ),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

def DataRetriever(question, rag_chain=rag_chain):
    """
    Function to retrieve data using the RAG chain.
    """
    return rag_chain.invoke({"question": question})

"""**Function to automatically gets the model structure**"""

# -----------------------------
# 🧩 Get Model Structure (LLM-Enhanced)
# -----------------------------
def get_model_structure():
    """
    Retrieves and summarizes the data model structure (tables, columns, relationships)
    from the FAISS database using similarity search + LLM summarization.

    Returns:
        str: A clean, readable summary of the data warehouse structure.
    """

    structure_keywords = [
        "table", "fact", "dimension", "columns",
        "relationships", "joins", "schema", "data model", "structure"
    ]

    # 🔍 Retrieve relevant chunks from FAISS
    combined_context = []
    for keyword in structure_keywords:
        docs_and_scores = db.similarity_search_with_score(keyword, k=15)
        for doc, score in docs_and_scores:
            if score > 0.5:  # filter by similarity threshold
                combined_context.append(doc.page_content)

    if not combined_context:
        return "No structure-related data found in FAISS."

    # Combine top chunks into one context block
    context_text = "\n\n".join(combined_context[:10])

    # 🧠 Ask LLM to summarize into a readable data model structure
    prompt = ChatPromptTemplate.from_template(f"""
        You are an expert in data warehousing and semantic modeling.

        The following text describes pieces of the model metadata:
        {context_text}

        Your task:
        - Summarize the schema structure into a clean, readable format.
        - Clearly separate Fact and Dimension tables.
        - List each table’s key columns.
        - Mention relationships between them when available.
        - Return only the summarized structure, no explanations.
    """)

    formatted = prompt.format_messages()
    model_structure = llm.invoke(formatted).content.strip()

    return model_structure

"""**Functions To generate And run SQL Queries on the DWH**"""

sql_prompt = ChatPromptTemplate.from_template("""
You are a senior Data Engineer specialized in MS SQL Server.
Your task: generate valid and optimized T-SQL queries for a YouTube Trending Video Data Warehouse.
⚠️ You must return **only** the SQL query text — no explanations.

## General Rules
1. Always use correct T-SQL syntax.
2. NEVER execute the query — only generate it.
3. Use proper JOINs between fact and dimension tables.
4. Avoid `TOP` without an `ORDER BY`.
5. Prefer readable aliases like v (VideosFact), c (Country), cat (Category), ch (Channel), d (Date).

## Overflow Prevention Rules
When you perform any aggregation (SUM, COUNT, AVG, etc.):
- Estimate whether the column values (like Views, Likes, CommentsCount, etc.) can be large.
- If the result of any arithmetic or aggregation can exceed INT range (2,147,483,647):
  👉 Use `CAST(... AS BIGINT)` or `CONVERT(BIGINT, ...)`.
- Otherwise, keep it as-is (don’t overuse casting).
Examples:
- Use `SUM(CAST(v.ViewCounts AS BIGINT))` ✅
- Avoid `SUM(v.CountryID)` ❌ (no need for BIGINT)
- Use `COUNT_BIG(*)` if counting millions of rows.

## Data Warehouse Schema
{sql_schema}

Question: {question}

Return ONLY the SQL query text.
""")

sql_chain = (
    sql_prompt
    | llm
    | StrOutputParser()
)

def SQLGenerator(question, sql_chain=sql_chain):
    """
    Function to generate SQL query from user question.
    """
    sql_schema = get_model_structure()

    return sql_chain.invoke({"question": question, "sql_schema": sql_schema})

"""**Manage connection to the Server**"""

# -----------------------------
# 🔌 Connection Configuration
# -----------------------------
SERVER = "localhost"
DATABASE = "YoutubeDWH"

odbc_str = (
    r"DRIVER={ODBC Driver 17 for SQL Server};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    "Trusted_Connection=yes;"
)

odbc_conn_str = urllib.parse.quote_plus(odbc_str)
connection_uri = f"mssql+pyodbc:///?odbc_connect={odbc_conn_str}"

# Create a global SQLAlchemy engine
engine = create_engine(connection_uri)

# -----------------------------
# 🧠 Function to Run SQL Queries
# -----------------------------
def run_sql_query(sql_query: str) -> pd.DataFrame:
    """
    Executes a SQL query on the connected SQL Server database and returns a DataFrame.

    Parameters:
        sql_query (str): The SQL query to execute.

    Returns:
        pd.DataFrame: The result of the query.
    """
    try:
        with engine.connect() as connection:
            df = pd.read_sql_query(sql_query, connection)
            return df
    except Exception as e:
        print(f"❌ SQL query failed:\n{e}")
        return pd.DataFrame()

"""**Generate and Return Dax Measures**"""

# -----------------------------
# 🔧 Configuration
# -----------------------------
FAISS_PATH = "youtube_trending_faiss"
SSAS_CONN = "Provider=MSOLAP;Data Source=localhost;Initial Catalog=YoutubeAnalysis;"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------
# 🧩 Load FAISS DB
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# -----------------------------
# 🧱 Execute DAX Query
# -----------------------------
def execute_dax_query_adodbapi(connection_string, dax_input):
    """
    Executes a DAX query, measure name, or measure definition against SSAS using ADODBAPI.
    """

    dax_clean = dax_input.strip()
    query_type = None
    dax_query = None

    # 🩺 Case 1: Full DAX query (starts with EVALUATE)
    if re.match(r"(?i)^EVALUATE\s", dax_clean):
        dax_query = dax_clean
        query_type = "full_query"

    # 🩺 Case 2: True measure definition (e.g., 'Total Sales = CALCULATE(...)')
    elif re.match(r"^[A-Za-z_][A-Za-z0-9_ ]*\s*=", dax_clean):
        name_part, expr_part = dax_clean.split("=", 1)
        measure_name = name_part.strip()
        expression = expr_part.strip()
        dax_query = f"""
        EVALUATE
        ROW("{measure_name}", {expression})
        """
        query_type = "measure_definition"

    # 🩺 Case 3: Table-returning expressions
    elif any(keyword in dax_clean.upper() for keyword in ["SUMMARIZE", "ADDCOLUMNS", "SELECTCOLUMNS", "FILTER", "CROSSJOIN", "VALUES"]):
        dax_query = f"EVALUATE {dax_clean}"
        query_type = "table_expression"

    # 🩺 Case 4: Scalar DAX expressions (non-table)
    elif any(keyword in dax_clean.upper() for keyword in ["SUM(", "COUNT", "CALCULATE", "VAR ", "DIVIDE", "AVERAGE"]):
        dax_query = f"""
        EVALUATE
        ROW("Result", {dax_clean})
        """
        query_type = "scalar_expression"

    # 🩺 Case 5: Existing measure
    else:
        dax_query = f"""
        EVALUATE
        ROW("{dax_clean}", [{dax_clean}])
        """
        query_type = "measure"

    # ----------------------------------
    # 🔌 Execute the query
    # ----------------------------------
    try:
        with adodbapi.connect(connection_string) as conn:
            cur = conn.cursor()
            cur.execute(dax_query)
            data = cur.fetchall()
            columns = [col[0] for col in cur.description] if cur.description else []
            df = pd.DataFrame(data, columns=columns)

            # Flatten single-value results
            if query_type in ["scalar_expression", "measure", "measure_definition"] and df.shape == (1, 1):
                return pd.DataFrame([{columns[0]: df.iloc[0, 0]}])

            return df

    except Exception as e:
        print(f"\n❌ DAX query failed for '{dax_input}': {str(e)}")
        print("Command:\n", dax_query)
        return pd.DataFrame()

# -----------------------------
# 🧠 Find Measure in FAISS (Optimized)
# -----------------------------
def find_measure_in_db(query):
    """
    Searches for a relevant predefined DAX measure in FAISS.
    If found and validated by the LLM, returns the full valid DAX query
    (ready to be executed directly in SSAS).
    Otherwise, returns None.
    """
    docs_and_scores = db.similarity_search_with_score(query, k=1)

    if not docs_and_scores:
        return None

    doc, score = docs_and_scores[0]
    if score <= 0.45:
        return None

    found_measure = doc.page_content  # includes measure details

    # 🔍 Step 1: Build verification prompt
    verification_prompt = ChatPromptTemplate.from_template("""
    You are an expert in DAX and data modeling.

    The user asked:
    "{query}"

    A predefined measure was found:
    {found_measure}

    Task:
    - Determine if this measure fully and correctly answers the user's question.
    - If yes, return ONLY the exact measure name (no extra text).
    - If not, return "None".
    """)

    formatted_prompt = verification_prompt.format_messages(
        query=query,
        found_measure=found_measure
    )

    # 🔧 Step 2: Ask the LLM to verify
    response = llm.invoke(formatted_prompt).content.strip()

    # 🧠 Step 3: Build executable DAX if confirmed suitable
    if response.lower() != "none" and len(response) > 0:
        measure_name = response
        # ✅ Return a full valid DAX command
        full_dax = f"""
        EVALUATE
        ROW("{measure_name}", [{measure_name}])
        """
        return full_dax.strip()

    return None

# -----------------------------
# 🧩 Generate Executable DAX Using LLM
# -----------------------------
def generate_dax_measure(question):
    """
    Generates a ready-to-execute DAX query string using the LLM.
    Ensures the output is compatible with the execute_dax_query_adodbapi() function.
    """

    schema_context = get_model_structure()
    prompt = ChatPromptTemplate.from_template(f"""
        You are a senior Power BI and DAX expert.

        The following describes the semantic model structure:
        {schema_context}

        The user asked: "{{question}}"

        Your task:
        - Generate a valid DAX formula that correctly answers the user's question.
        - The formula can be:
            1. A full DAX query (starts with EVALUATE)
            2. A measure definition (Name = Expression)
            3. A table expression (SUMMARIZECOLUMNS, ADDCOLUMNS, etc.)
            4. A scalar expression (SUM, COUNT, CALCULATE, AVERAGE, etc.)
            5. Or an existing measure name if applicable.
        - Do NOT wrap the formula in EVALUATE unless it is a full query.
        - Return only the pure DAX formula — no explanations, comments, or Markdown.
        - Use fully titled country names (e.g., "United States of America") when filtering by Country.
        """)

    formatted = prompt.format_messages(question=question)
    raw_output = llm.invoke(formatted).content.strip()

    # 🧹 Clean Markdown artifacts
    clean_output = (
        raw_output.replace("```DAX", "")
                  .replace("```dax", "")
                  .replace("```", "")
                  .strip()
    )

    # ✅ Normalize spacing
    dax_expression = " ".join(clean_output.split())
    dax_upper = dax_expression.upper()

    # -----------------------------
    # 🧠 Auto-format for Execution
    # -----------------------------
    if dax_upper.startswith("EVALUATE"):
        final_dax = dax_expression  # already a full query

    elif re.match(r"^[A-Za-z_][A-Za-z0-9_ ]*\s*=", dax_expression):
        # Measure definition (e.g., Total Sales = SUM(...))
        final_dax = dax_expression

    elif any(keyword in dax_upper for keyword in ["SUMMARIZE", "SUMMARIZECOLUMNS", "ADDCOLUMNS", "SELECTCOLUMNS", "FILTER", "CROSSJOIN", "VALUES"]):
        # Table expression
        final_dax = f"EVALUATE {dax_expression}"

    elif any(keyword in dax_upper for keyword in ["SUM(", "COUNT", "CALCULATE", "DIVIDE", "AVERAGE", "VAR ", "MAX(", "MIN("]):
        # Scalar expression
        final_dax = f"EVALUATE ROW('Result', {dax_expression})"

    else:
        # Assume it’s an existing measure
        final_dax = f"EVALUATE ROW('{dax_expression}', [{dax_expression}])"

    return final_dax

# -----------------------------
# 🚀 Main Query Flow
# -----------------------------
def query_semantic_model(question: str):
    print(f"🔍 Question: {question}")

    # Step 1: Try to find an existing measure
    found_measure = find_measure_in_db(question)

    if found_measure:
        # print(f"✅ Found cached measure: {found_measure}")
        result = execute_dax_query_adodbapi(SSAS_CONN,found_measure)
        return result
    else:
        print("⚠️ Measure not found. Generating a new DAX measure...")
        new_dax = generate_dax_measure(question)
        # print(f"🧮 Suggested DAX Measure:\n{new_dax}")
        result = execute_dax_query_adodbapi(SSAS_CONN,new_dax)
        return result

# -----------------------------
# 💡 Example usage
# -----------------------------
if __name__ == "__main__":
    user_question = "Get the total views by country for the United States of America as a Measure."
    result = query_semantic_model(user_question)
    # print("\n📊 Response:\n", response)
    print("\n📊 Result:\n", result)

# -----------------------------
# 💡 Example usage
# -----------------------------
if __name__ == "__main__":
    user_question = "what are Top 3 Countries by Total Views?."
    result = query_semantic_model(user_question)
    # print("\n📊 Response:\n", response)
    print("\n📊 Result:\n", result)

"""# Router Every Thing together"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

# -----------------------------
# 🧩 1. Intent detection using LLM
# -----------------------------
intent_prompt = ChatPromptTemplate.from_template("""
You are an AI classifier. Given a user question about YouTube trending data,
decide which processing type it needs.

Options:
"SQL" → if the question asks for aggregated data, counts, totals, or tabular results.
"DAX" → if the question mentions Power BI, measures, SSAS, or DAX formulas.
"RAG" → if the question is conceptual, about explanations, metadata, or documentation.

Return only one word: SQL, DAX, or RAG.

Question: {question}
""")

intent_chain = intent_prompt | llm | StrOutputParser()


# -----------------------------
# 🧠 2. Router logic based on intent
# -----------------------------
def format_with_context(raw_output: str, question: str):
    """
    Streaming version — yields tokens instead of returning one block.
    """

    formatting_prompt = f"""
You are an advanced analytical AI assistant.

Your task is to turn raw analytical results into a high-quality conversational response.

Follow this exact structure:

1. Friendly Introduction
2. Markdown Table / Clean Data
3. Clear Explanation
4. Short Recap
5. Ask a helpful follow-up question

### User Question:
{question}

### Raw Analytical Output:
{raw_output}

Now produce the polished final answer:
"""

    # Streaming response (Groq)
    for chunk in llm.stream(formatting_prompt):
        if chunk.content:
            yield chunk.content



# Define the normalization prompt
normalize_prompt = ChatPromptTemplate.from_template("""
You are an expert data assistant helping to prepare user queries for an analytical model.

The database uses full country names such as "United States of America", "Brazil", "Canada", "United Kingdom", etc.

Rewrite the following question by replacing any short or informal country names
(e.g., "USA", "US", "UK", "BR") with their correct full official country names.
Do NOT answer the question — just rewrite it.

Question: {question}

Rewritten:
""")


def ask_intelligent(question: str, context=None):
    print(f"\n🧭 Query received: {question}\n")

    # Step 1: Normalize abbreviations like USA → United States of America
    try:
        normalized_prompt = normalize_prompt.format(question=question)
        normalized_question = llm.invoke(normalized_prompt).content.strip()
        print(f"🌍 Normalized Question: {normalized_question}")
    except Exception as e:
        print(f"⚠️ Normalization failed ({e}), using original question.")
        normalized_question = question

    # Step 2: Build context if available
    context_text = ""
    if context:
        context_pairs = []
        for msg in context:
            role = "User" if msg.type == "human" else "Assistant"
            context_pairs.append(f"{role}: {msg.content}")
        context_text = "\n".join(context_pairs)
    full_prompt = f"{context_text}\n\nUser: {normalized_question}" if context_text else normalized_question

    # Step 3: Process intent + query
    try:
        intent = intent_chain.invoke({"question": full_prompt}).strip().upper()
        print(f"🤖 Detected Intent: {intent}")

        if intent == "SQL":
            raw_output = run_sql_query(SQLGenerator(full_prompt))
        elif intent == "DAX":
            raw_output = query_semantic_model(full_prompt)
        else:
            raw_output = DataRetriever(full_prompt)

        # Step 4: STREAM the formatted output
        for chunk in format_with_context(raw_output, normalized_question):
            yield chunk   # <-- 🔥 STREAMING HAPPENS HERE

    except Exception as e:
        yield f"❌ Error while processing query: {e}"
