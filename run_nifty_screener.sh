#!/bin/bash

# 🎯 NIFTY Option Screener v7.0 - Startup Script
# This script launches the standalone NIFTY Option Screener app

echo "🎯 Starting NIFTY Option Screener v7.0..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "⚠️  Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if secrets file exists
if [ ! -f ".streamlit/secrets.toml" ]; then
    echo "⚠️  Warning: .streamlit/secrets.toml not found"
    echo ""
    echo "Please create .streamlit/secrets.toml with your credentials:"
    echo ""
    echo "[DHAN]"
    echo "CLIENT_ID = \"your_client_id\""
    echo "ACCESS_TOKEN = \"your_access_token\""
    echo ""
    echo "[TELEGRAM]"
    echo "BOT_TOKEN = \"your_bot_token\""
    echo "CHAT_ID = \"your_chat_id\""
    echo ""
    echo "Press Enter to continue anyway, or Ctrl+C to exit and configure..."
    read -r
fi

echo "✅ Launching NIFTY Option Screener..."
echo "📊 The app will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""

# Run the Streamlit app
streamlit run nifty_screener_app.py
