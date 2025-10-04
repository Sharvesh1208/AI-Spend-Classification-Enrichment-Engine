# AI-Based Spend Classification and Enrichment Engine 💰

## Presented by: Team CogniSynth
**Team Members:** 1) Sharvesh K  |  2) Keerthi Vardhan M  |  3) Prathibha M

---

## 🎯 Problem Statement: AI-Based Spend Classification and Enrichment Engine

This project delivers a sophisticated solution to automate, standardize, and enrich raw procurement spend data, transforming messy transaction descriptions into clean, actionable intelligence.

### The Business Challenge
Companies lose significant time and money due to:

1.  **Inconsistent Data:** Spend descriptions are often non-standard, ambiguous text across various source systems (e.g., "10 laptops lenovo" vs. "ten units of portable computers").
2.  **Lack of Granularity:** Critical fields like precise **UNSPSC categories** and standardized vendor names are frequently missing, crippling accurate analysis.
3.  **Manual Overheads:** Manual data cleaning and classification is slow, costly, and prone to human error, hindering strategic decision-making.

### Value Proposition
Our engine provides immediate business value by:
* **Driving Cost Savings:** Enabling strategic sourcing decisions and identifying maverick spend by providing accurate, aggregated data.
* **Ensuring Compliance:** Maintaining a secure, verifiable history of all classification processes for auditing purposes.
* **Boosting Efficiency:** Slashing the time spent on manual data preparation, allowing finance and procurement teams to focus on strategy.

---

## 🛠️ Technical Architecture & Tech Stack

Our solution is built on a high-performance, scalable stack combining modern web technology with advanced Large Language Models (LLMs).

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | **React** with **Vite.js** (JavaScript) | Provides a fast, modern, and interactive user interface for data upload and visualization (Analytics/Data Table). |
| **Backend** | **Python** with **FastAPI** | High-performance API integration layer, connecting the frontend to the AI models and handling data processing logic. |
| **Classification Model** | **Mistral LLM** | Core intelligence for **classification and prediction** (e.g., determining Vendor and Procurement Category). |
| **Enrichment Model** | **T5 LLM** | Generates standardized, **enriched descriptions** from vague input text, improving data clarity. |
| **Data Storage** | **Firebase (Firestore)** | Securely stores all transactional data and maintains the **Classification History** for audit and re-loading purposes. |

---

## ⚙️ Core Functionality: Step-by-Step Process

The platform guides the user through a simple yet powerful data processing workflow:

1.  **Data Ingestion (Upload Screen):** User uploads a CSV or pastes text, initiating the FastAPI backend.
2.  **Data Processing and AI Engine:** Python executes preprocessing (normalization/missing data handling), and the **Mistral** and **T5** models run in parallel for prediction and enrichment.
3.  **Data Table (Classification Results):** Processed data is displayed, including the **Original Text**, **Enriched Description**, predicted **Vendor/Category** (with confidence scores), and status indicators.
4.  **Analytics and Insights:** A visual dashboard provides **key metrics** (Total Spend, Avg. Confidence) and charts (Spend Distribution) for instant business intelligence.
5.  **Classification History (Audit & Governance):** Every session is stored in **Firebase**. Users can access the history to **Load** previous results or **Delete** old entries, maintaining an audit trail.

---

## 📺 Demo and Documentation

Please find the comprehensive demonstration video and detailed project report in the Google Drive folder below:

**[Project Demo Video & Report](https://drive.google.com/drive/folders/1p0wkTPUaxhsbzYMX_gcL3PDZm7Oj5-dL?usp=drive_link)**

---

## ▶️ Getting Started

Instructions for setting up and running the project locally will be provided here:

1.  Clone the repository: `git clone https://github.com/Sharvesh1208/AI-Spend-Classification-Enrichment-Engine`
2.  Setup Frontend (React/Vite):
    * `cd frontend`
    * `npm install`
    * `npm run dev`
3.  Setup Backend (Python/FastAPI):
    * `cd backend`
    * `pip install -r requirements.txt`
    * `uvicorn main:app --reload`
    
*(Ensure your Firebase and LLM API keys are configured in the environment variables.)*
