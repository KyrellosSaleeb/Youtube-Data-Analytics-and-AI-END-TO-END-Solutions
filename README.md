# YouTube Trending Video Analytics – End-to-End BI & AI Data Engineering Solution

## 📌 Project Description  
This project is an end-to-end **on-premise Data Engineering pipeline** built using Microsoft technologies. It uses the **YouTube Trending Video Dataset** from Kaggle, focusing on data from **Canada, the United States, and Brazil**.  

The goal is to build a **data warehouse architecture** that ingests raw files, transforms and integrates them, and finally delivers insights through an **interactive Power BI dashboard**.  
This project was completed as part of my training in **Global Brand Solutions Academy**.  

## 🔎 Project Overview  
1. **Getting the Data (SSIS)** → Extracted CSV files for (Canada, US, and Brazil).  
2. **ODS Layer (SSIS)** → Loaded raw data into the ODS as the initial layer.  
3. **STG Layer (SSIS)** → Performed cleansing, data conversion, and standardization.  
4. **DWH Layer (SSIS)** → Built star schema, added surrogate keys, and created lookups between Fact & Dimensions.  
5. **Semantic Model (SSAS)** → Created relationships and DAX measures for KPIs (views, likes, dislikes, comments).  
6. **Dashboard (Power BI)** → Designed interactive visuals to analyze trends, regional comparisons, and channel performance.  


## 🗂 Data Extraction Challenges  

While extracting the dataset into **SSIS**, we faced an issue with the **Connection Manager**. Some columns were incorrectly detected as **object** instead of their actual data type.  

🔎 After analyzing the problem, we found that the **`title`** column contained text values with embedded commas. These commas confused the CSV parser, causing SSIS to misinterpret column structures.  

✅ **Solution:**  
We used **Python preprocessing** to fix the issue before loading into SSIS. Specifically, we added quotes (`""`) around any text field containing at least one comma. This ensured that SSIS correctly treated the values as a single column.  

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


