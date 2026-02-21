

# HR Attrition and Training Prediction API

This project provides a robust, production-ready solution for Human Resources analytics, leveraging Machine Learning (ML) and Generative AI (LLM) to predict employee flight risk and recommend targeted interventions. The API is optimized for handling both large-scale dashboard data retrieval and detailed single-employee analysis.

---
## Intended Use
This API is designed for integration within the **Konecta ERP HR Analytics Dashboard**.  
It supports automated dashboards (via bulk endpoints) and on-demand employee insights (via single-record queries).

---
## Live API Base URL

The entire API service is accessible via the following base URL:

https://robahesham25-hr-atrrition-and-training-prediction.hf.space

---

##  Technical Architecture

| Component | Technology | Purpose |
| :---- | :---- | :---- |
| **Framework** | **FastAPI / Uvicorn** | High-performance API server for handling HTTP requests. |
| **ML Models** | **XGBoost** | **Primary model for Attrition Risk prediction.** |
| **ML Models** | **Scikit-learn Logistic Regression** | **Model for Training Needs prediction.** |
| **Generative AI** | **Google Gemini-flash-2.5** | Providing root cause analysis and actionable recommendations. |
| **Deployment** | **Hugging Face Spaces** | Hosting the live API service. |
| **Data Processing** | **Pandas / Joblib** | Efficient data handling, scaling, and feature alignment. |

---

##  API Endpoints

The API supports two core access patterns: **Bulk Retrieval** (for dashboards) and **Single-Record Query** (for user interaction). All smart lookup and analysis routes accept only the EmployeeID in the URL path.

| Endpoint | Method | Functionality | Input Example | Output |
| :---- | :---- | :---- | :---- | :---- |
| **/get\_high\_risk\_list** | **GET** | **Macro Attrition View:** Reads internal data, runs bulk **XGBoost** prediction, and returns all employees flagged as high attrition risk ($\\text{Prob} \\ge 50\\%$). | *Base URL*/get\_high\_risk\_list | List of {ID, FullName, Probability} |
| **/get\_training\_needs\_list** | **GET** | **Macro Training View:** Reads internal data, runs bulk **Logistic Regression** prediction, and returns all employees flagged as needing training. | *Base URL*/get\_training\_needs\_list | List of {ID, FullName, Probability} |
| **/analyze\_employee/{id}** | **GET** | **Deep Dive LLM Analysis:** Retrieves full employee features by ID and runs the Generative AI analysis for root cause and intervention. | *Base URL*/analyze\_employee/{ID} | Detailed Text Analysis |
| **/predict\_attrition/{id}** | **GET** | **Smart Attrition Lookup:** Retrieves full features by ID and returns the attrition prediction. | *Base URL*/predict\_attrition/{ID} | Attrition Risk/Probability |
| **/predict\_training/{id}** | **GET** | **Smart Training Lookup:** Retrieves full features by ID and returns the training needs prediction. | *Base URL*/predict\_training/{ID} | Training Need Prediction/Probability |

---

##  Project Contents

| File Name | Purpose |
| :---- | :---- |
| **app.py** | The main **FastAPI application code**, including all API routes, Pydantic models, and prediction logic. |
| **requirements.txt** | List of all Python dependencies (e.g., fastapi, uvicorn, google-generativeai). |
| **combined.csv** | The **dataset** used by the server's GET endpoints for internal data lookup and bulk processing (relative path is required). |
| **\*.joblib files** | Serialized ML artifacts: models, scaler, and feature name lists. |

---

## Security and Setup

## Google AI Key

The Generative AI endpoint (/analyze\_employee/{id}) requires a key for the Gemini model.

* **Variable Name:** GOOGLE\_API\_KEY  
* **Configuration:** This must be set as a **Secret** in the hosting environment. The server reads this using python-dotenv locally and from environment secrets in production.

### **Data Source**

All internal GET routes rely on the final, cleaned dataset being accessible.

* **Requirement:** The **combined.csv** file must be uploaded to the root directory of the application environment alongside app.py to be accessible via the relative path.
