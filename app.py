import streamlit as st
import pandas as pd
import pdfplumber
import re
from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import create_engine, text


# ─────────────────────────────────────────────
# DATABASE CONNECTION
# ─────────────────────────────────────────────

def connectDatabase(username, port, host, password, database):
    mysql_uri = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}"
    st.session_state.db = SQLDatabase.from_uri(mysql_uri)
    st.session_state.engine = create_engine(mysql_uri)
    st.session_state.mysql_uri = mysql_uri


def runQuery(query):
    try:
        return st.session_state.db.run(query) if "db" in st.session_state else "Please connect to database"
    except Exception as e:
        return f"Query Error: {str(e)}"


def getDatabaseSchema():
    return st.session_state.db.get_table_info() if "db" in st.session_state else "Please connect to database"


# ─────────────────────────────────────────────
# LLM SETUP
# ─────────────────────────────────────────────

llm = ChatOllama(model="llama3")


def getQueryFromLLM(question):
    template = """below is the schema of MYSQL database, read the schema carefully about the table and column names. Also take care of table or column name case sensitivity.
    Finally answer user's question in the form of SQL query.

    {schema}

    IMPORTANT RULES:
    - For PDF tables, the columns are: id, page_number, content
    - To search content inside PDF, use LIKE operator: SELECT content FROM table WHERE content LIKE '%keyword%'
    - To get a specific page: SELECT content FROM table WHERE page_number = N
    - please only provide the SQL query and nothing else, no explanation

    for example:
    question: what is on page 1
    SQL query: SELECT content FROM generative_ai_cfg WHERE page_number = 1
    question: find information about neural networks
    SQL query: SELECT page_number, content FROM generative_ai_cfg WHERE content LIKE '%neural networks%'
    question: how many pages in the document
    SQL query: SELECT COUNT(*) FROM generative_ai_cfg

    your turn :
    question: {question}
    SQL query :
    please only provide the SQL query and nothing else
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    response = chain.invoke({"question": question, "schema": getDatabaseSchema()})
    return response.content.strip()


def getResponseForQueryResult(question, query, result):
    template2 = """You are a helpful assistant. Based on the question, SQL query and result from the database,
    write a clear and helpful response in natural language.

    If the result contains PDF page content, summarize it nicely.
    If result is empty, say no matching content was found.

    Database Schema:
    {schema}

    Question: {question}
    SQL Query: {query}
    Result: {result}

    Response (be helpful and clear):
    """
    prompt2 = ChatPromptTemplate.from_template(template2)
    chain2 = prompt2 | llm
    response = chain2.invoke({
        "question": question,
        "schema": getDatabaseSchema(),
        "query": query,
        "result": result
    })
    return response.content


# ─────────────────────────────────────────────
# FILE UPLOAD HELPERS
# ─────────────────────────────────────────────

def sanitize_name(name):
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip())
    if name[0].isdigit():
        name = "_" + name
    return name.lower()


def infer_mysql_type(series):
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    elif pd.api.types.is_float_dtype(series):
        return "DOUBLE"
    else:
        max_len = series.astype(str).str.len().max()
        length = min(max(int(max_len) * 2, 64), 65535)
        return f"VARCHAR({length})"


def upload_csv_to_mysql(df, table_name):
    engine = st.session_state.engine
    table_name = sanitize_name(table_name)
    df.columns = [sanitize_name(c) for c in df.columns]

    col_defs = ", ".join(
        f"`{col}` {infer_mysql_type(df[col])}" for col in df.columns
    )
    create_sql = f"CREATE TABLE IF NOT EXISTS `{table_name}` (id INT AUTO_INCREMENT PRIMARY KEY, {col_defs});"

    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`;"))
        conn.execute(text(create_sql))
        conn.commit()

    df.to_sql(table_name, con=engine, if_exists="append", index=False)
    st.session_state.db = SQLDatabase.from_uri(st.session_state.mysql_uri)
    return table_name


def upload_pdf_to_mysql(pdf_file, table_name):
    engine = st.session_state.engine
    table_name = sanitize_name(table_name)

    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`;"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS `{table_name}` (
                id INT AUTO_INCREMENT PRIMARY KEY,
                page_number INT,
                content LONGTEXT
            );
        """))
        conn.commit()

    rows = []
    with pdfplumber.open(pdf_file) as pdf:
        total_pages = len(pdf.pages)
        progress = st.progress(0, text="Extracting pages...")
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            rows.append({"page_number": i + 1, "content": page_text})
            progress.progress((i + 1) / total_pages, text=f"Extracting page {i+1} of {total_pages}...")

    if rows:
        df = pd.DataFrame(rows)
        df.to_sql(table_name, con=engine, if_exists="append", index=False)

    st.session_state.db = SQLDatabase.from_uri(st.session_state.mysql_uri)
    st.session_state.last_pdf_table = table_name
    return table_name, len(rows)


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

st.set_page_config(
    page_icon="🤖",
    page_title="Chat with MySQL DB",
    layout="centered"
)

st.title("🤖 Chat with MySQL Database")

# ── Status Bar ──
col1, col2 = st.columns(2)
with col1:
    if "db" in st.session_state:
        st.success("🟢 Database Connected")
    else:
        st.warning("🔴 Database Not Connected")
with col2:
    if "last_pdf_table" in st.session_state:
        st.info(f"📄 PDF: `{st.session_state.last_pdf_table}`")
    elif "last_csv_table" in st.session_state:
        st.info(f"📊 CSV: `{st.session_state.last_csv_table}`")

# ── Quick Question Buttons (show after PDF upload) ──
if "last_pdf_table" in st.session_state:
    tbl = st.session_state.last_pdf_table
    st.markdown("#### 💡 Quick Questions — Click to Ask:")
    qcol1, qcol2, qcol3 = st.columns(3)
    quick_q = None
    with qcol1:
        if st.button("📄 What is on page 1?"):
            quick_q = f"What is on page 1 of {tbl}?"
    with qcol2:
        if st.button("📊 How many pages?"):
            quick_q = f"How many pages are in {tbl}?"
    with qcol3:
        if st.button("🔍 Show first 3 pages"):
            quick_q = f"Show me content from first 3 pages of {tbl}"

    if quick_q:
        if "chat" not in st.session_state:
            st.session_state.chat = []
        st.session_state.chat.append({"role": "user", "content": quick_q})
        with st.spinner("Thinking..."):
            query = getQueryFromLLM(quick_q)
            result = runQuery(query)
            response = getResponseForQueryResult(quick_q, query, result)
        st.session_state.chat.append({"role": "assistant", "content": response})
        st.rerun()

st.divider()

# ── Chat History ──
if "chat" not in st.session_state:
    st.session_state.chat = []

for chat in st.session_state.chat:
    st.chat_message(chat["role"]).markdown(chat["content"])

# ── Chat Input ──
question = st.chat_input("Ask anything about your database or uploaded file...")

if question:
    if "db" not in st.session_state:
        st.error("❌ Please connect to the database first (use the sidebar).")
    else:
        st.session_state.chat.append({"role": "user", "content": question})
        st.chat_message("user").markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Generating SQL and fetching answer..."):
                query = getQueryFromLLM(question)
                st.caption(f"🔍 Generated SQL: `{query}`")
                result = runQuery(query)
                response = getResponseForQueryResult(question, query, result)
            st.markdown(response)

        st.session_state.chat.append({"role": "assistant", "content": response})


# ─── SIDEBAR ───────────────────────────────────
with st.sidebar:

    # ── 1. DB Connection ──
    st.title("🔌 Connect to Database")
    st.text_input(label="Host",     key="host",     value="localhost")
    st.text_input(label="Port",     key="port",     value="3306")
    st.text_input(label="Username", key="username", value="root")
    st.text_input(label="Password", key="password", value="", type="password")
    st.text_input(label="Database", key="database", value="rag_test")
    connectBtn = st.button("Connect", use_container_width=True)

    if connectBtn:
        try:
            connectDatabase(
                username=st.session_state.username,
                port=st.session_state.port,
                host=st.session_state.host,
                password=st.session_state.password,
                database=st.session_state.database,
            )
            st.success("✅ Database connected!")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")

    st.divider()

    # ── 2. File Upload ──
    st.title("📂 Upload File to Database")

    file_type = st.radio("Select File Type", ["PDF", "CSV"], horizontal=True)

    uploaded_file = st.file_uploader(
        f"Upload {file_type} file",
        type=["pdf"] if file_type == "PDF" else ["csv"],
    )

    if uploaded_file:
        default_table = sanitize_name(uploaded_file.name.rsplit(".", 1)[0])
        table_name_input = st.text_input(
            "Table name in MySQL",
            value=default_table,
            key="table_name_input"
        )

        file_size = round(uploaded_file.size / (1024 * 1024), 2)
        st.caption(f"📁 {uploaded_file.name} ({file_size} MB)")

        if file_type == "PDF":
            st.info("⏳ Large PDFs (6MB+) may take 1-2 minutes.")

        upload_btn = st.button("⬆️ Upload & Create Table", use_container_width=True)

        if upload_btn:
            if "engine" not in st.session_state:
                st.error("❌ Please connect to the database first.")
            else:
                try:
                    if file_type == "CSV":
                        with st.spinner("Reading CSV and creating table..."):
                            df = pd.read_csv(uploaded_file)
                            created_table = upload_csv_to_mysql(df, table_name_input)
                            st.session_state.last_csv_table = created_table
                        st.success(f"✅ Table `{created_table}` created!\n{len(df)} rows uploaded.")
                        st.info(f"📊 Columns: {', '.join(df.columns.tolist())}")

                    elif file_type == "PDF":
                        created_table, num_pages = upload_pdf_to_mysql(uploaded_file, table_name_input)
                        st.success(f"✅ Table `{created_table}` created!\n{num_pages} pages uploaded.")
                        st.info("💬 Quick question buttons are now available in the chat area!")

                except Exception as e:
                    st.error(f"❌ Upload Error: {str(e)}")

    st.divider()

    # ── 3. Clear Chat ──
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

    st.divider()

    # ── 4. Tips ──
    st.markdown("### 💡 How to Use")
    st.markdown("""
1. **Connect** your MySQL database
2. **Upload** PDF or CSV file
3. Click **Upload & Create Table**
4. **Ask questions** in the chat!

**PDF Examples:**
- *"What is on page 1?"*
- *"Find content about AI"*
- *"How many pages?"*

**CSV Examples:**
- *"How many rows?"*
- *"Show top 5 records"*
    """)
