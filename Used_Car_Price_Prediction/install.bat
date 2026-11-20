@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>&1 || (echo Python was not found. Install Python 3.10+ and enable Add Python to PATH.& pause & exit /b 1)
python -m pip install --upgrade pip
if errorlevel 1 (echo Could not access pip. & pause & exit /b 1)
python -m pip install -r requirements.txt
if errorlevel 1 (echo Dependency installation failed. & pause & exit /b 1)
echo Dependencies installed successfully.
pause
