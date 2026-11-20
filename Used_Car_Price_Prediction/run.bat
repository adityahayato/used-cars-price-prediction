@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>&1 || (echo Python was not found. Install Python 3.10+ and enable Add Python to PATH.& pause & exit /b 1)
python -c "import flask,pandas,sklearn,joblib" >nul 2>&1 || (echo Dependencies are missing. Run install.bat first.& pause & exit /b 1)
if not exist models\best_model.pkl echo Model missing. Train it with: python train_model.py
echo Starting Used Car Price Prediction System at http://127.0.0.1:5000
python app.py
pause
