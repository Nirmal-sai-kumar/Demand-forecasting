# Retail Demand Forecasting

## Setup Instructions

Create and activate virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Train the model:

```powershell
python ml_model_xgb.py
```

Run the web app:

```powershell
python app.py
```

## Project Structure

```
Retail_Demand_Forecasting/
│
├── app.py
├── ml_model_xgb.py
├── README.md
├── requirements.txt
├── supabase_client.py
├── users.csv
├── __pycache__/
├── dataset/
├── downloads/
├── graphs/
├── model/
│   └── xgb_model.joblib
└── templates/
    ├── index.html
    ├── login.html
    ├── register.html
    ├── result.html
    └── upload.html
```
