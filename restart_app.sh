#!/bin/bash

echo "🔄 Stopping Streamlit app..."
pkill -f "streamlit run"
sleep 2

echo "🧹 Clearing Python cache..."
find /home/user/Java-script- -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find /home/user/Java-script- -name "*.pyc" -delete 2>/dev/null

echo "🗑️ Clearing Streamlit cache..."
rm -rf /home/user/Java-script-/.streamlit/cache 2>/dev/null

echo "✅ Starting fresh Streamlit app..."
cd /home/user/Java-script-
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

echo "🎉 App restarted successfully!"
echo "📍 Access at: http://localhost:8501"
