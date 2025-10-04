# AI-Spend-Classification-Enrichment-Engine

Presented by: Team CogniSynth
Team Leader: Sharvesh K

🎯 Problem Statement: AI-Based Spend Classification and Enrichment Engine
This project delivers a sophisticated solution to automate, standardize, and enrich raw procurement spend data, transforming messy transaction descriptions into clean, actionable intelligence.

The Business Challenge
Companies lose significant time and money due to:

Inconsistent Data: Spend descriptions are often non-standard, ambiguous text across various source systems (e.g., "10 laptops lenovo" vs. "ten units of portable computers").

Lack of Granularity: Critical fields like precise UNSPSC categories and standardized vendor names are frequently missing, crippling accurate analysis.

Manual Overheads: Manual data cleaning and classification is slow, costly, and prone to human error, hindering strategic decision-making.

Value Proposition
Our engine provides immediate business value by:

Driving Cost Savings: Enabling strategic sourcing decisions and identifying maverick spend by providing accurate, aggregated data.

Ensuring Compliance: Maintaining a secure, verifiable history of all classification processes for auditing purposes.

Boosting Efficiency: Slashing the time spent on manual data preparation, allowing finance and procurement teams to focus on strategy.

🛠️ Technical Architecture & Tech Stack
Our solution is built on a high-performance, scalable stack combining modern web technology with advanced Large Language Models (LLMs).

Component	Technology	Purpose
Frontend	React with Vite.js (JavaScript)	Provides a fast, modern, and interactive user interface for data upload and visualization (Analytics/Data Table).
Backend	Python with FastAPI	High-performance API integration layer, connecting the frontend to the AI models and handling data processing logic.
Classification Model	Mistral LLM	Core intelligence for classification and prediction (e.g., determining Vendor and Procurement Category).
Enrichment Model	T5 LLM	Generates standardized, enriched descriptions from vague input text, improving data clarity.
Data Storage	Firebase (Firestore)	Securely stores all transactional data and maintains the Classification History for audit and re-loading purposes.

Export to Sheets
⚙️ Core Functionality: Step-by-Step Process
The platform guides the user through a simple yet powerful data processing workflow:

1. Data Ingestion (Upload Screen)
User uploads a CSV file or pastes text input into the interface.

Clicking 'Upload & Classify' initiates the backend process via FastAPI.

2. Data Processing and AI Engine
Preprocessing: The Python engine first addresses data quality, performing normalization and handling missing data.

AI Execution: The Mistral and T5 models run in parallel:

Mistral predicts the most probable Vendor and Category.

T5 generates a clean, enriched description.

3. Data Table (Classification Results)
Processed data is displayed in a table format, showing the Original Text, the Enriched Description, predicted Vendor/Category (with confidence scores), and status indicators (✓ normalization / △ missing-data).

4. Analytics and Insights
The system provides a visual dashboard with key metrics (Total Spend, Avg. Confidence) and charts (Spend Distribution, Confidence Distribution) for immediate business insights.

5. Classification History (Audit & Governance)
Every processing session is stored in Firebase. Users can access the History to view past classification jobs, Load previous results for re-examination, or Delete old entries, maintaining a clear audit trail.

📺 Demo and Documentation
Please find the comprehensive demonstration video and detailed project report in the Google Drive folder below:

Project Demo Video & Report
