# 24/7 Background Service Setup Guide

## Problem Statement

**Streamlit apps STOP when browser closes** because they are web applications, not background services.

**What you need:** A separate background script that runs 24/7 independently.

---

## Solution: Deploy Background Service

The `telegram_background_service.py` script runs continuously and sends telegram messages even when the Streamlit app is closed.

---

## Option 1: Run Locally (Simple)

### **1. Setup Environment Variables**

Create `.env` file in project root:

```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
DHAN_CLIENT_ID=your_dhan_client_id
DHAN_ACCESS_TOKEN=your_dhan_access_token
```

### **2. Install Dependencies**

```bash
pip install python-dotenv
```

### **3. Run Background Service**

**Keep running forever (even after closing terminal):**

```bash
# Linux/Mac:
nohup python telegram_background_service.py > service.log 2>&1 &

# Windows:
pythonw telegram_background_service.py
```

**Check if running:**

```bash
# Linux/Mac:
ps aux | grep telegram_background_service

# View logs:
tail -f service.log
```

**Stop service:**

```bash
# Find process ID:
ps aux | grep telegram_background_service

# Kill process:
kill <PID>
```

---

## Option 2: Deploy to Cloud (Recommended for 24/7)

### **A. Heroku (Free Tier Available)**

1. Create `Procfile`:
   ```
   worker: python telegram_background_service.py
   ```

2. Deploy:
   ```bash
   heroku create your-app-name
   heroku config:set TELEGRAM_BOT_TOKEN=xxx
   heroku config:set TELEGRAM_CHAT_ID=xxx
   heroku config:set DHAN_CLIENT_ID=xxx
   heroku config:set DHAN_ACCESS_TOKEN=xxx
   git push heroku main
   ```

3. Scale worker:
   ```bash
   heroku ps:scale worker=1
   ```

### **B. Railway.app (Easiest)**

1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub"
3. Select your repo
4. Add environment variables in Settings
5. Change start command to: `python telegram_background_service.py`
6. Done! It runs 24/7 automatically

### **C. AWS Lambda (Serverless)**

1. Package script with dependencies
2. Create Lambda function
3. Add CloudWatch trigger (every 10 seconds during market hours)
4. Set environment variables
5. Lambda runs on schedule automatically

### **D. Google Cloud Run (Serverless)**

1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.9
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   CMD ["python", "telegram_background_service.py"]
   ```

2. Deploy:
   ```bash
   gcloud run deploy signal-service \
     --source . \
     --platform managed \
     --region us-central1 \
     --set-env-vars TELEGRAM_BOT_TOKEN=xxx,...
   ```

---

## Option 3: VPS/Server (Full Control)

### **Using systemd (Linux)**

1. Create service file: `/etc/systemd/system/telegram-signals.service`

```ini
[Unit]
Description=NIFTY Option Telegram Signal Service
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/your/project
Environment="TELEGRAM_BOT_TOKEN=xxx"
Environment="TELEGRAM_CHAT_ID=xxx"
Environment="DHAN_CLIENT_ID=xxx"
Environment="DHAN_ACCESS_TOKEN=xxx"
ExecStart=/usr/bin/python3 /path/to/telegram_background_service.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

2. Enable and start:

```bash
sudo systemctl enable telegram-signals.service
sudo systemctl start telegram-signals.service
sudo systemctl status telegram-signals.service
```

3. View logs:

```bash
sudo journalctl -u telegram-signals.service -f
```

---

## Option 4: Process Manager (PM2)

### **Install PM2:**

```bash
npm install -g pm2
```

### **Start service:**

```bash
pm2 start telegram_background_service.py --name "nifty-signals" --interpreter python3
```

### **Make it run on boot:**

```bash
pm2 startup
pm2 save
```

### **Monitor:**

```bash
pm2 status
pm2 logs nifty-signals
pm2 monit
```

### **Stop/Restart:**

```bash
pm2 stop nifty-signals
pm2 restart nifty-signals
pm2 delete nifty-signals  # Remove
```

---

## How It Works

### **Background Service:**

```
┌──────────────────────────────────────┐
│  Background Service (24/7)           │
│                                      │
│  1. Runs continuously                │
│  2. Checks market every 10s          │
│  3. Analyzes option data             │
│  4. Detects trading signals          │
│  5. Sends Telegram messages          │
│                                      │
│  NO BROWSER REQUIRED! ✅             │
└──────────────────────────────────────┘
```

### **Streamlit App (Separate):**

```
┌──────────────────────────────────────┐
│  Streamlit App (Web UI)              │
│                                      │
│  1. Opens in browser                 │
│  2. Shows live charts & analysis     │
│  3. Manual signal viewing            │
│  4. Interactive controls             │
│                                      │
│  REQUIRES OPEN BROWSER ⚠️            │
└──────────────────────────────────────┘
```

**BOTH can run simultaneously!**

---

## Monitoring & Logs

### **Check Logs:**

The service logs to `telegram_service.log`:

```bash
tail -f telegram_service.log
```

### **Log Format:**

```
2026-01-20 14:30:00 - INFO - 🚀 Starting 24/7 Telegram Signal Service
2026-01-20 14:30:00 - INFO - ✅ All modules imported successfully
2026-01-20 14:30:00 - INFO - ✅ All credentials loaded
2026-01-20 14:30:05 - INFO - [1] 📈 Market hours - checking for signals...
2026-01-20 14:30:05 - INFO - ✅ NIFTY Spot: ₹25,371.00, Expiry: 2026-01-20
2026-01-20 14:30:08 - INFO - 🎯 NEW SIGNAL DETECTED!
2026-01-20 14:30:09 - INFO - ✅ Signal sent to Telegram successfully
```

---

## Comparison: Streamlit App vs Background Service

| Feature | Streamlit App | Background Service |
|---------|---------------|-------------------|
| **Runs when browser closed** | ❌ NO | ✅ YES |
| **Sends telegram 24/7** | ❌ NO | ✅ YES |
| **Visual interface** | ✅ YES | ❌ NO |
| **Interactive controls** | ✅ YES | ❌ NO |
| **Real-time charts** | ✅ YES | ❌ NO |
| **Requires user action** | ✅ YES | ❌ NO |
| **Auto-refresh reliability** | ⚠️ Medium | ✅ High |
| **Mobile friendly** | ✅ YES | N/A |
| **Resource usage** | Medium | Low |

**RECOMMENDATION:** Run BOTH!
- Use Streamlit app for visual monitoring & manual trading
- Use background service for 24/7 automated signals

---

## Troubleshooting

### **Service not starting:**

1. Check Python version: `python3 --version` (need 3.8+)
2. Check dependencies: `pip install -r requirements.txt`
3. Check environment variables: `echo $TELEGRAM_BOT_TOKEN`
4. Check logs: `cat telegram_service.log`

### **No signals being sent:**

1. Check if service is running: `ps aux | grep telegram`
2. Check logs for errors: `tail -f telegram_service.log`
3. Verify credentials are correct
4. Check if market is open: Service sleeps when market closed

### **Service stops randomly:**

1. Use process manager (PM2 or systemd) for auto-restart
2. Check server resources (CPU, memory, disk)
3. Check for Python crashes in logs
4. Ensure stable internet connection

---

## Next Steps

1. ✅ **Fix Streamlit auto-refresh** (already done in app.py)
2. ✅ **Deploy background service** (choose option above)
3. ✅ **Test telegram messages** (run service and verify)
4. ✅ **Monitor logs** (ensure service is working)
5. ✅ **Set up alerts** (get notified if service stops)

**For maximum reliability:** Deploy to Railway.app or Heroku worker - easiest cloud deployment with auto-restart on failure!
