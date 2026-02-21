##################### CHAIN ARCHETICTURE ###############################

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import re
import os
from dotenv import load_dotenv


load_dotenv()
groq_api = os.getenv("groq_api")
langchain_api = os.getenv("langchain_api")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = langchain_api
os.environ["LANGCHAIN_PROJECT"] = "testing"

# ------------------- CONFIG -------------------

DB_URL = "postgresql://postgres:OacjgdqDKllQBXpxLedIPtQRFoPVLuoy@mainline.proxy.rlwy.net:15211/railway"

st.set_page_config(page_title="SQL Chatbot", page_icon=":bar_chart:", layout="wide")
st.title("Chat with Postgres DB :bar_chart:")

# ------------------- DATABASE -------------------
@st.cache_resource
def get_db_engine():
    return create_engine(DB_URL)

@st.cache_data
def get_schema():
    engine = get_db_engine()
    inspector_query = text("""
        SELECT table_name, column_name 
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """)

    schema_string = ""
    with engine.connect() as conn:
        result = conn.execute(inspector_query)
        current_table = None
        for row in result:
            table_name, column_name = row
            if table_name != current_table:
                if current_table is not None:
                    schema_string += "\n"
                schema_string += f"Table: {table_name}\n"
                current_table = table_name
            schema_string += f"  - {column_name}\n"

    return schema_string

# ------------------- LLM -------------------
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        api_key=groq_api,
        temperature = 0
    )

@st.cache_resource
def get_sql_chain(llm):
    sql_prompt = PromptTemplate.from_template("""
You are an expert PostgreSQL data analyst.

Database schema:
{schema}

User Question:
{question}

Write ONLY the SQL query.
- Use double quotes around table and column names exactly as in the schema.
- Return only valid SELECT SQL.
- If any column storing dates is TEXT, cast it to DATE first using "::DATE".
""")
    return sql_prompt | llm

@st.cache_resource
def get_answer_chain(llm):
    answer_prompt = PromptTemplate.from_template("""
User Question:
{question}

SQL Result:
{sql_result}

Answer the question strictly based on the SQL result.
If the result is empty, say:
"The data does not provide a clear answer to the question."
""")
    return answer_prompt | llm

# ------------------- CLEAN SQL -------------------
def clean_sql(sql_text: str) -> str:
    sql_text = re.sub(r"```sql", "", sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r"```", "", sql_text)
    return sql_text.strip()

# ------------------- STREAMLIT APP -------------------
if __name__ == "__main__":

    schema = get_schema()
    if not schema:
        st.stop()

    llm = get_llm()
    sql_chain = get_sql_chain(llm)
    answer_chain = get_answer_chain(llm)

    # Session state caching
    if "sql_cache" not in st.session_state:
        st.session_state.sql_cache = {}
    if "answer_cache" not in st.session_state:
        st.session_state.answer_cache = {}

    user_question = st.text_input("Ask a question about the database:")

    if st.button("Get Answer") and user_question:

        # ================= CHAIN 1 =================
        if user_question in st.session_state.sql_cache:
            sql_query, result_df = st.session_state.sql_cache[user_question]
        else:
            sql_response = sql_chain.invoke({
                "schema": schema,
                "question": user_question
            })
            sql_query = clean_sql(sql_response.content)

            # Execute SQL
            if sql_query.lower().startswith("select"):
                try:
                    engine = get_db_engine()
                    with engine.connect() as conn:
                        result_df = pd.read_sql(sql_query, conn)
                except Exception as e:
                    st.error(f"SQL Execution Error: {e}")
                    result_df = pd.DataFrame()
            else:
                st.warning("Generated query is not a SELECT statement.")
                result_df = pd.DataFrame()

            # Cache the SQL and result
            st.session_state.sql_cache[user_question] = (sql_query, result_df)

        # Display SQL & results
        st.code(sql_query, language="sql")
        st.dataframe(result_df)

        # ================= CHAIN 2 =================
        if not result_df.empty:
            if user_question in st.session_state.answer_cache:
                final_answer = st.session_state.answer_cache[user_question]
            else:
                final_answer_response = answer_chain.invoke({
                    "question": user_question,
                    "sql_result": result_df.to_string()
                })
                final_answer = final_answer_response.content
                st.session_state.answer_cache[user_question] = final_answer

            st.markdown(f"**Answer:** {final_answer}")
        else:
            st.markdown("**Answer:** The data does not provide a clear answer to the question.")
