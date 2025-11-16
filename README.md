# YouTube Trending Video Analytics – End-to-End BI & AI Data Engineering Solution

## 📌 Project Description  
This project is an end-to-end **on-premise Data Engineering pipeline** built using Microsoft technologies. It uses the **YouTube Trending Video Dataset** from Kaggle, focusing on data from **Canada, the United States, and Brazil**.  

The goal is to build a **data warehouse architecture** that ingests raw files, transforms and integrates them, and finally delivers insights through an **interactive Power BI dashboard** and an **Intelligent AI Agent**.  
This project was completed as part of my training in **Global Brand Solutions Academy**.  

## 🔎 Project Overview  
1. **Getting the Data (SSIS)** → Extracted CSV files for (Canada, US, and Brazil).  
2. **ODS Layer (SSIS)** → Loaded raw data into the ODS as the initial layer.  
3. **STG Layer (SSIS)** → Performed cleansing, data conversion, and standardization.  
4. **DWH Layer (SSIS)** → Built star schema, added surrogate keys, and created lookups between Fact & Dimensions.  
5. **Semantic Model (SSAS)** → Created relationships and DAX measures for KPIs (views, likes, dislikes, comments).  
6. **Dashboard (Power BI)** → Designed interactive visuals to analyze trends, regional comparisons, and channel performance.
7. **AI Agent (LangChain/Groq)** → Implemented a RAG-based Large Language Model to allow natural language querying of the database and documentation.


## 🗂 Data Extraction Challenges  

While extracting the dataset into **SSIS**, we faced an issue with the **Connection Manager**. Some columns were incorrectly detected as **object** instead of their actual data type.  

🔎 After analyzing the problem, we found that the **`title`** column contained text values with embedded commas. These commas confused the CSV parser, causing SSIS to misinterpret column structures.  

✅ **Solution:** We used **Python preprocessing** to fix the issue before loading into SSIS. Specifically, we added quotes (`""`) around any text field containing at least one comma. This ensured that SSIS correctly treated the values as a single column.  

👉 Check the detailed implementation here: [![Open Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](fixing-the-youtube-title-feature.ipynb)  

## 🔹 Step 1: Getting the Data & Creating the ODS Layer  

After fixing the **title column issue**, we successfully created the **Connection Manager** for the CSV files. Using the advanced options in the CSV connection manager, we defined the most suitable data types for each column based on the dataset description.  

Next, we created a new database called **`YoutubeODS`** with a table named **`Videos`** using a SQL script.  

👉 Check the script here: [![Open Script](https://img.shields.io/badge/SQL-Script-blue?logo=databricks)](SQL%20Queries/ODS_Creation.sql)  

After preparing the dataset, the CSV files for **Canada, US, and Brazil** were added into the **Data Flow** in the package called **ODS**. The files were then **unioned** and loaded into the **`YoutubeODS.Videos`** table as the destination.  

<p align="center">  
  <img src="Full Flow ScreenShots/1) ETL Pipeline/ODS.png" alt="ODS Data Flow" width="500"/>  
</p>  

## 🔹 Step 2: Creating the STG Layer  

After loading the data into the **`YoutubeODS.Videos`** table, the next step was to design the **STG database structure** to prepare the data for the Data Warehouse.  

We started by creating both a **Logical Model** and a **Conceptual Model** to define the dimensions and fact table.  

<p align="center">  
  <img src="Data Models/Logical Model.png" alt="Logical Model" width="400" height="400"/>  
  <img src="Data Models/Conceptual model.png" alt="Conceptual Model" width="400"/>  
</p>  

Based on these models, we created a new database called **`YoutubeSTG`**. This database contains the **Dimension** and **Fact** tables as defined by the modeling step.  

👉 Check the SQL script here: [![Open Script](https://img.shields.io/badge/SQL-Script-blue?logo=databricks)](SQL%20Queries/DimsFactCreation.sql)  

After creating the structure, we filled the **Dimensions** and **Fact** tables in the STG database using SSIS, as shown below:  

<p align="center">  
  <img src="Full Flow ScreenShots/1) ETL Pipeline/STG Full View.png" alt="STG Data Flow" width="500"/>  
</p>  

## 🔹 Step 3: Creating the DWH Layer  

After finalizing the **Dimensions** and **Fact** structure in the STG layer, the next step was to build the **Data Warehouse (DWH)**.  

We created a new database called **`YoutubeDWH`** containing all the **Dimension** and **Fact** tables, ready to host the data.  

👉 Check the SQL script here: [![Open Script](https://img.shields.io/badge/SQL-Script-blue?logo=databricks)](SQL%20Queries/DWH_Creation.sql)  

Once the schema was in place, we transferred the data from the **STG dimensions** into the **DWH dimensions**.  
Using SSIS, we performed **lookups** between the Fact and Dimensions so the Fact table could reference the correct **surrogate keys** (IDs).  

<p align="center">  
  <img src="Full Flow ScreenShots/1) ETL Pipeline/STG Full View.png" alt="Full DWH View" width="400"/>  
  <img src="Full Flow ScreenShots/1) ETL Pipeline/DWH Fact Lookup.png" alt="DWH Lookup" width="400"/>  
</p>  

## 🔹 Step 4: Building the Semantic Model (SSAS)  

After creating the **DWH**, we loaded the **Fact** and **Dimension** tables into an **SSAS Tabular Project** to build the **Semantic Model**.  
This step involved:  
- Creating **DAX measures** (e.g., total views, likes, dislikes, engagement).  
- Defining **relationships** between Fact and Dimensions.  

<p align="center">  
  <img src="Full Flow ScreenShots/2) Semantic Model (SSAS)/Semantic Layer Model View.png" alt="SSAS Semantic Model View" width="500"/>  
</p>  

Once the measures and relationships were defined, the final **Semantic Model** was deployed to SSAS.  
This allowed the model to be cached and updated for efficient querying and analysis.  

<p align="center">  
  <img src="Full Flow ScreenShots/2) Semantic Model (SSAS)/Deployment Operation.png" alt="SSAS Deployment Operation" width="500"/>  
</p>  

## 🔹 Step 5: Power BI Interactive Dashboard  

Finally, we connected **Power BI** to the deployed **SSAS Semantic Model** to build an interactive dashboard.  
The dashboard was designed with **two pages**, providing insights into:  
- Video performance metrics (views, likes, dislikes, comments).  
- Regional comparisons between **Canada, US, and Brazil**.  
- Trending patterns and engagement rates.  

<p align="center">  
  <img src="Full Flow ScreenShots/3) Dashboard Screenshots/Data Overview.png" alt="Power BI Dashboard - Page 1" width="400"/>  
  <img src="Full Flow ScreenShots/3) Dashboard Screenshots/Channels Analysis.png" alt="Power BI Dashboard - Page 2" width="400"/>  
</p>  

---

# 🤖 Phase 2: Intelligent RAG Agent  

<p align="center">
  <a href="YK Analysis Agent/rag_llm.py">
    <img src="https://img.shields.io/badge/Python-Agent_Code-blue?logo=python" alt="Python Agent Code">
  </a>
  <a href="YK Analysis Agent/RAG Documentation/YouTube_Trending_Video_Dataset_Full_Report.pdf">
    <img src="https://img.shields.io/badge/RAG-Source_PDF-red?logo=adobeacrobatreader" alt="RAG Source PDF">
  </a>
  <a href="YK Analysis Agent">
    <img src="https://img.shields.io/badge/Project-Agent_Folder-grey?logo=folder" alt="Agent Project Folder">
  </a>
</p>

To enhance the accessibility of the analytics, we developed an **AI-powered Agent** using **Python, LangChain, and Groq**. This agent, run via a Streamlit interface, allows users to interact with the entire data ecosystem using natural language, abstracting the complexities of SQL, DAX, and project documentation.

### 🧠 How the Agent Works: A Detailed Flow

The agent's logic is orchestrated by the `ask_intelligent` function, which follows a precise, multi-step process for every query.

#### Step 1: Query Normalization
Before any processing, the user's question is passed to a normalization chain.
* **Purpose:** To standardize informal inputs (like "USA" or "UK") into their full, official names (e.g., "United States of America") that match the database entries.
* **Implementation:** Uses a `ChatPromptTemplate` (`normalize_prompt`) that instructs the LLM to *only* rewrite the question and not answer it.

#### Step 2: Intent Routing
The normalized question is then sent to the `intent_chain`.
* **Purpose:** To classify the user's goal and determine the correct data source to query.
* **Implementation:** The `intent_prompt` forces the LLM to output *only one word*:
    * **`SQL`**: For requests about aggregated data, counts, or tabular results from the DWH.
    * **`DAX`**: For questions mentioning Power BI, measures, SSAS, or specific KPI calculations.
    * **`RAG`**: For conceptual questions about the project, metadata, or documentation.

#### Step 3: Execution via Specialized Chains
Based on the detected intent, the agent routes the query to one of three specialized handlers.

##### 🔵 **Handler 1: `RAG` (Documentation Query)**
* **Function:** `DataRetriever`.
* **Process:** This is a standard RAG chain. The agent retrieves the most relevant text chunks (`k=3`) from the pre-loaded FAISS vector database (which contains the project documentation). These chunks are injected as `context` into a prompt, allowing the LLM to generate an answer based *only* on the provided documents.

##### 🟡 **Handler 2: `SQL` (Data Warehouse Query)**
This is a two-part process:
1.  **Dynamic Schema Retrieval:** The agent first calls `get_model_structure`. This function is a RAG chain itself—it queries the FAISS database for keywords like "table", "fact", "dimension", and "schema". It then passes the retrieved text to the LLM to summarize the DWH schema in a clean, readable format.
2.  **SQL Generation & Execution:** The generated schema is passed to the `SQLGenerator` function.
    * The `sql_prompt` instructs the LLM to act as a senior data engineer, using the provided schema to write optimized T-SQL.
    * **Crucial Safeguard:** The prompt strictly enforces "Overflow Prevention Rules," forcing the LLM to use `CAST(SUM(v.ViewCounts) AS BIGINT)` for large number aggregations to prevent SQL errors.
    * The final, generated T-SQL string is executed by the `run_sql_query` function using SQLAlchemy, returning a pandas DataFrame.

##### 🟢 **Handler 3: `DAX` (Semantic Model Query)**
This is the most complex flow, managed by `query_semantic_model`:
1.  **Find Existing Measure:** The agent first calls `find_measure_in_db`. This RAG function searches FAISS for a pre-defined DAX measure that might answer the query. It uses the LLM to verify if the found measure is a suitable match. If a match is found (e.g., `[Total Views]`), it formats it as an executable DAX query (`EVALUATE ROW(...)`) and skips the next step.
2.  **Generate New Measure:** If no existing measure is found, the agent calls `generate_dax_measure`. This chain also uses `get_model_structure` for schema context and prompts the LLM to write a *new* DAX formula from scratch.
3.  **Execute DAX:** The resulting DAX (either found or generated) is sent to `execute_dax_query_adodbapi`. This function intelligently handles various DAX formats (full queries, single expressions, table functions) by wrapping them in the correct `EVALUATE` syntax needed by the `adodbapi` driver to query the SSAS cube. The result is returned as a pandas DataFrame.

#### Step 4: Final Response Formatting
The raw output from Step 3 (whether it's text from RAG or a DataFrame from SQL/DAX) is passed to the `format_with_context` function.
* **Purpose:** To transform the raw data into a polished, conversational response.
* **Implementation:** This final prompt instructs the LLM to follow a strict structure: "1. Friendly Introduction, 2. Markdown Table / Clean Data, 3. Clear Explanation, 4. Short Recap, 5. Ask a helpful follow-up question.".
* The `ask_intelligent` function streams this final, formatted answer back to the user in the Streamlit app.

---

## Agent Sample Response

<p align="center">
  
</p>
<p align="center">
  
</p>
<p align="center">
  
</p>
