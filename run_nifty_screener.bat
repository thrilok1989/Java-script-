@echo off
REM 🎯 NIFTY Option Screener v7.0 - Startup Script (Windows)
REM This script launches the standalone NIFTY Option Screener app

echo 🎯 Starting NIFTY Option Screener v7.0...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Streamlit not found. Installing dependencies...
    pip install -r requirements.txt
)

REM Check if secrets file exists
if not exist ".streamlit\secrets.toml" (
    echo ⚠️  Warning: .streamlit\secrets.toml not found
    echo.
    echo Please create .streamlit\secrets.toml with your credentials:
    echo.
    echo [DHAN]
    echo CLIENT_ID = "your_client_id"
    echo ACCESS_TOKEN = "your_access_token"
    echo.
    echo [TELEGRAM]
    echo BOT_TOKEN = "your_bot_token"
    echo CHAT_ID = "your_chat_id"
    echo.
    echo Press any key to continue anyway, or close this window to exit...
    pause >nul
)

echo ✅ Launching NIFTY Option Screener...
echo 📊 The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the app
echo.

REM Run the Streamlit app
streamlit run nifty_screener_app.py
