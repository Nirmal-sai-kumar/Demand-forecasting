```powershell
# 1) Go to project folder
# 2) Create + activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate

# 3) Install all dependencies (CORRECT)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4) Train + save model (creates: model\xgb_model.joblib and graphs\xgb_graph.png)
python ml_model_xgb.py

# 5) Run the web app
python app.py
```
