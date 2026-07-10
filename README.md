# Predictive Maintenance System

> **Manufacturing Machine Failure Prediction** — A data-driven project that analyzes industrial sensor data, builds machine learning models to detect potential equipment failures, and delivers interactive dashboards and automated alerts to support proactive maintenance.

[![GitHub Repo](https://img.shields.io/badge/GitHub-predictive--maintenance--system-blue)](https://github.com/Yota-khaled/predictive-maintenance-system)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Key Features](#key-features)
- [Team Members](#team-members)
- [Instructor](#instructor)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Machine Learning Results](#machine-learning-results)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Running the Apps](#running-the-apps)
- [Automation Workflow (n8n)](#automation-workflow-n8n)
- [Tableau Dashboard](#tableau-dashboard)
- [KPIs & Success Metrics](#kpis--success-metrics)
- [Project Deliverables](#project-deliverables)
- [Additional Resources](#additional-resources)

---

## Project Overview

Unexpected machine failures disrupt manufacturing processes, cause costly downtime, and increase maintenance expenses. Traditional maintenance is either **reactive** (fix after failure) or **scheduled** (regardless of actual machine condition).

This project builds an end-to-end **Predictive Maintenance System** that:

1. Cleans and analyzes the **AI4I 2020 Predictive Maintenance Dataset** using SQL and Python
2. Performs **EDA and KPI analysis** in Jupyter notebooks and an interactive Plotly dashboard
3. Trains and compares multiple classification models to predict **Machine Failure**
4. Deploys the best model in **Streamlit prediction apps** (light and dark themes)
5. Connects predictions to an **n8n automation workflow** for logging and email alerts
6. Presents operational insights through a **Tableau dashboard**

---

## Business Problem

Manufacturing lines rely on continuous equipment uptime. When a machine fails without warning, production stops, repair costs spike, and delivery schedules slip.

**Goal:** Detect machines at risk of failure *before* breakdown occurs, enabling maintenance teams to act proactively — reducing downtime, lowering costs, and improving operational efficiency.

---

## Key Features

| Component | Description |
|-----------|-------------|
| **SQL Data Pipeline** | SQL Server scripts for import, validation, outlier detection, and cleaning |
| **EDA & KPI Notebook** | Business questions, KPI metrics, and Plotly charts in `Analysis_and_KPI_Plotly.ipynb` |
| **ML Notebook** | Full exploratory analysis, feature engineering, and model comparison |
| **Plotly KPI Dashboard** | Interactive Streamlit app for failure rates, tool wear risk, and business insights |
| **Best Model** | Tuned Random Forest classifier saved as `best_machine_failure_model.pkl` |
| **Prediction Apps** | Streamlit dashboards with sensor sliders, confidence gauge, and verdict display |
| **Automation** | n8n workflow logs predictions to Google Sheets and sends Gmail alerts on failure |
| **Tableau Dashboard** | Interactive BI dashboard for machine health monitoring |
| **Presentation** | Final project slides in `Presentation/Presentation.pptx` |

---

## Team Members

1. Aya Khaled
2. Nouran Hassan
3. Malak Amr
4. Mariam Mahmoud
5. Philopater Emeel
6. Mina Saad

All team members collaboratively contributed to data preprocessing, EDA, KPI analysis, model building, dashboard design, app development, automation setup, and presentation preparation.

---

## Instructor

**Dina Ezzat**

---

## Project Structure

```
Manufacturing_Project/
├── App/
│   ├── Dark App/
│   │   └── app.py                          # ML prediction app — dark theme
│   └── Light App/
│       ├── app.py                          # ML prediction app — light theme
│       └── config.toml                     # Streamlit theme configuration
├── Automation/
│   └── Machine Failure.json                # n8n workflow (webhook → Sheets → Gmail)
├── Best_Model/
│   └── best_machine_failure_model.pkl      # Saved Random Forest model
├── Dashboard/
│   ├── Final_dashboard.twbx                # Tableau workbook
│   ├── Background.png
│   ├── Logo.png
│   ├── Logo sidebar.png
│   └── icon.png
├── Data/
│   ├── raw/
│   │   └── ai4i2020.csv                    # Original dataset (10,000 records)
│   └── processed/
│       └── Predictive_Maintenance_Cleaned.csv
├── Notebooks/
│   ├── Analysis & KPI Notebook/
│   │   └── Analysis_and_KPI_Plotly.ipynb   # KPI analysis & business insights
│   └── Analysis & ML Notebook/
│       └── final-machine-failure.ipynb     # EDA, modeling, evaluation
├── Presentation/
│   └── Presentation.pptx                   # Final presentation slides
├── SQL/
│   └── Cleaning.sql                          # SQL Server data cleaning pipeline
├── Visulization With Plotly/
│   ├── app.py                              # Interactive KPI & EDA Streamlit dashboard
│   └── data.csv                            # Cleaned data for the dashboard
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

> **Note:** The `Demo/` folder (project walkthrough video) is excluded from GitHub due to file size limits. Watch the demo on [Google Drive](https://drive.google.com/drive/u/0/folders/12Wy7fURDho-qV7tB4ZjT1J4Wl4GLqppB).

---

## Dataset

**Source:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) (UCI / Kaggle)

| Property | Value |
|----------|-------|
| Total records | 10,000 |
| Failure cases | 339 (3.39%) |
| Non-failure cases | 9,661 (96.61%) |
| Machine types | L (Low) — 6,000 · M (Medium) — 2,997 · H (High) — 1,003 |

### Features Used for Prediction

| Feature | Description | Unit |
|---------|-------------|------|
| `Type` | Product / machine quality variant (L, M, H) | — |
| `Air_Temperature` | Ambient air temperature | K |
| `Process_Temperature` | Process operating temperature | K |
| `Rotational_Speed` | Rotational speed of the machine | rpm |
| `Torque` | Torque applied | Nm |
| `Tool_Wear` | Tool wear duration | min |

### Target Variable

| Column | Description |
|--------|-------------|
| `Machine_Failure` | Binary label — `1` = failure, `0` = no failure |

### Failure Mode Columns (Analysis Only)

| Column | Full Name |
|--------|-----------|
| `TWF` / `Tool_Wear_Failure` | Tool Wear Failure |
| `HDF` / `Heat_Dissipation_Failure` | Heat Dissipation Failure |
| `PWF` / `Power_Failure` | Power Failure |
| `OSF` / `Overstrain_Failure` | Overstrain Failure |
| `RNF` / `Random_Failure` | Random Failure |

---

## Methodology

### 1. Data Cleaning (SQL Server)

The `SQL/Cleaning.sql` script performs:

- Database and table creation (`Machine_Failure_Project`)
- Bulk import of the raw CSV
- Missing value and duplicate checks
- IQR-based outlier detection and removal across all numeric sensor columns
- Column renaming and removal of non-predictive fields (`UDI`, `Product_ID`)
- Export of the cleaned dataset to `Data/processed/Predictive_Maintenance_Cleaned.csv`

### 2. Exploratory Data Analysis & KPIs

Performed in two places:

**Notebook** — `Notebooks/Analysis & KPI Notebook/Analysis_and_KPI_Plotly.ipynb`:
- Data loading and quality checks
- KPI metrics (total operations, failure rate, duplicates)
- Business questions on product type, tool wear, operating conditions, and failure modes
- Business insights and recommendations

**Interactive Dashboard** — `Visulization With Plotly/app.py`:
- Filterable KPI dashboard (product type, tool wear range)
- Failure distribution and failure rate by machine type
- Tool wear vs. failure risk (violin plots, wear-group trends)
- Operating condition comparisons for failed vs. non-failed machines
- Failure type frequency and mechanical stress analysis
- Business insights and recommendations sections

### 3. Machine Learning Pipeline

Performed in `Notebooks/Analysis & ML Notebook/final-machine-failure.ipynb`:

- Distribution analysis and failure mode breakdown
- **Split:** 80% train / 20% test (stratified)
- **Cross-validation:** Stratified K-Fold during hyperparameter tuning
- **Scoring metric:** ROC-AUC (to handle class imbalance)
- **Class imbalance handling:** `class_weight='balanced'` / `scale_pos_weight` where applicable
- **Label encoding:** Machine type (`L → 0`, `M → 1`, `H → 2`)

---

## Machine Learning Results

Six models were trained, tuned with `GridSearchCV`, and evaluated on the held-out test set (2,000 records).

| Model | ROC-AUC | F1-Score | Accuracy | Precision | Recall |
|-------|---------|----------|----------|-----------|--------|
| **Random Forest** ✅ | **0.9764** | **0.6974** | **0.9770** | **0.6310** | **0.7794** |
| XGBoost | 0.9761 | 0.6061 | 0.9610 | 0.4615 | 0.8824 |
| CatBoost | 0.9667 | 0.5405 | 0.9490 | 0.3896 | 0.8824 |
| SVM | 0.9644 | 0.4321 | 0.9185 | 0.2831 | 0.9118 |
| Logistic Regression | 0.9062 | 0.2415 | 0.8210 | 0.1411 | 0.8382 |
| KNN | 0.8662 | 0.2963 | 0.9715 | 0.9231 | 0.1765 |

### Best Model — Random Forest

```
Best hyperparameters:
  n_estimators      = 300
  max_depth         = 15
  min_samples_split = 10
  min_samples_leaf  = 4
  class_weight      = balanced
```

The trained model is saved at `Best_Model/best_machine_failure_model.pkl` and loaded by both prediction apps at runtime.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10+ |
| Data Analysis | pandas, NumPy, SciPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | scikit-learn, XGBoost, CatBoost |
| Web Apps | Streamlit |
| Database | Microsoft SQL Server |
| BI Dashboard | Tableau |
| Automation | n8n (webhook, Google Sheets, Gmail) |
| Model Serialization | joblib |

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip
- (Optional) Microsoft SQL Server — for running the SQL cleaning script
- (Optional) n8n — for the automation workflow
- (Optional) Tableau Desktop — for opening the dashboard workbook

### Installation

```bash
# Clone the repository
git clone https://github.com/Yota-khaled/predictive-maintenance-system.git
cd predictive-maintenance-system

# Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install all dependencies
pip install -r requirements.txt
```

---

## Running the Apps

This project includes **three Streamlit applications**:

### 1. ML Prediction App — Light Theme

```bash
cd "App/Light App"
streamlit run app.py
```

Opens at `http://localhost:8501`.

### 2. ML Prediction App — Dark Theme

```bash
cd "App/Dark App"
streamlit run app.py
```

### 3. KPI & EDA Dashboard (Plotly)

```bash
cd "Visulization With Plotly"
streamlit run app.py
```

> Keep `data.csv` in the same folder as `app.py` — the dashboard reads it automatically.

### How to Use the Prediction Apps

1. Enter your **email address** in the sidebar
2. Select the **machine type** (L / M / H)
3. Adjust **sensor readings** using the sliders:
   - Air Temperature (295.3 – 304.5 K)
   - Process Temperature (305.7 – 313.8 K)
   - Rotational Speed (1,168 – 2,886 rpm)
   - Torque (3.8 – 76.6 Nm)
   - Tool Wear (0 – 253 min)
4. Click **Run Prediction**
5. View the result:
   - **Verdict banner** — Failure or No Failure
   - **Confidence gauge** — probability percentage
   - **Probability bars** — breakdown of failure vs. no-failure likelihood

The prediction apps automatically load the model from `Best_Model/best_machine_failure_model.pkl`.

### How to Use the KPI Dashboard

1. Use the **sidebar filters** to select product type and tool wear range
2. Review the **KPI cards** — total operations, failure rate, non-failure rate
3. Explore interactive charts for failure distribution, product type risk, tool wear trends, and failure modes
4. Read the **Business Insights** and **Business Recommendations** sections at the bottom

---

## Automation Workflow (n8n)

The file `Automation/Machine Failure.json` defines an n8n workflow:

```
Streamlit App  →  Webhook  →  Google Sheets  →  IF (Failure?)  →  Gmail Alert
```

| Step | Action |
|------|--------|
| 1 | Streamlit sends a POST request to `http://localhost:5678/webhook/machine-failure` |
| 2 | n8n logs the prediction (timestamp, email, sensor values, result, confidence) to **Google Sheets** |
| 3 | If the prediction is **Failure**, an automated **Gmail alert** is sent with sensor readings and recommended actions |

### Webhook Payload

```json
{
  "email": "user@example.com",
  "product_type": "M",
  "air_temperature": 300.0,
  "process_temperature": 310.0,
  "rotational_speed": 1539,
  "torque": 40.0,
  "tool_wear": 108,
  "prediction": "Failure",
  "confidence": 87.5
}
```

### Setup

1. Install and start [n8n](https://n8n.io/)
2. Import `Automation/Machine Failure.json`
3. Configure Gmail and Google Sheets credentials in n8n
4. Activate the workflow
5. Run a prediction app — results will be logged and alerts sent automatically

---

## Tableau Dashboard

The interactive dashboard is located at `Dashboard/Final_dashboard.twbx`.

It provides visual monitoring of:

- Machine failure rates and trends
- Sensor reading distributions
- Failure mode breakdowns
- Machine type performance comparisons

Supporting assets (`Background.png`, `Logo.png`, `icon.png`) are included in the `Dashboard/` folder.

Open the `.twbx` file with **Tableau Desktop** or publish it to **Tableau Public / Server**.

---

## KPIs & Success Metrics

| KPI | Target / Result |
|-----|-----------------|
| Model ROC-AUC | **0.9764** (Random Forest) |
| Model Recall (failure detection) | **77.9%** |
| Model F1-Score | **0.6974** |
| Failure rate identified in EDA | **~3.39%** of operations |
| Dashboard clarity & usability | Tableau + 3 Streamlit apps |
| Actionable insights | KPI analysis + failure mode breakdown + automated alerts |
| Downtime reduction potential | Proactive maintenance before breakdown |

---

## Project Deliverables

| Deliverable | Location |
|-------------|----------|
| Cleaned dataset | `Data/processed/Predictive_Maintenance_Cleaned.csv` |
| SQL cleaning script | `SQL/Cleaning.sql` |
| KPI & EDA notebook | `Notebooks/Analysis & KPI Notebook/Analysis_and_KPI_Plotly.ipynb` |
| ML notebook | `Notebooks/Analysis & ML Notebook/final-machine-failure.ipynb` |
| Plotly KPI dashboard | `Visulization With Plotly/app.py` |
| Trained model | `Best_Model/best_machine_failure_model.pkl` |
| Prediction app (Light) | `App/Light App/app.py` |
| Prediction app (Dark) | `App/Dark App/app.py` |
| n8n automation workflow | `Automation/Machine Failure.json` |
| Tableau dashboard | `Dashboard/Final_dashboard.twbx` |
| Presentation slides | `Presentation/Presentation.pptx` |
| Demo video | [Demo on Google Drive](https://drive.google.com/drive/u/0/folders/12Wy7fURDho-qV7tB4ZjT1J4Wl4GLqppB) (`Demo.mp4`) |

---

## License

See [LICENSE](LICENSE) for details.
