#!/bin/bash
# AWS EC2 Ubuntu 22.04 — GenAI Environment Setup Script
# Run as ubuntu user on a fresh EC2 instance

set -e
echo "=== GenAI Training Environment Setup ==="

# ── System Updates ─────────────────────────────────────────────────────────
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install -y git curl wget unzip python3.11 python3.11-venv python3-pip     build-essential libssl-dev libffi-dev docker.io docker-compose

# ── Python Virtual Environment ────────────────────────────────────────────
python3.11 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip

# ── Clone Repository ──────────────────────────────────────────────────────
git clone https://github.com/YOUR_USERNAME/GenAI_Training_Codebase.git ~/genai-app
cd ~/genai-app

# ── Install Dependencies ──────────────────────────────────────────────────
pip install -r requirements.txt

# ── Environment File ──────────────────────────────────────────────────────
cp .env.example .env
echo ">> Edit .env and add your API keys: nano .env"

# ── Redis ─────────────────────────────────────────────────────────────────
sudo apt-get install -y redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
echo "Redis running: $(redis-cli ping)"

# ── Docker Group ──────────────────────────────────────────────────────────
sudo usermod -aG docker $USER
echo "Log out and back in for Docker group to take effect"

# ── Systemd Service ───────────────────────────────────────────────────────
cat << "SERVICE" | sudo tee /etc/systemd/system/genai-app.service
[Unit]
Description=GenAI Training Application
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/genai-app
Environment="PATH=/home/ubuntu/venv/bin"
ExecStart=/home/ubuntu/venv/bin/uvicorn 05_Guardrails_Security.jwt_auth_fastapi.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable genai-app

echo ""
echo "=== Setup Complete ==="
echo "1. Edit .env with your API keys: nano ~/genai-app/.env"
echo "2. Start the app: sudo systemctl start genai-app"
echo "3. Check status: sudo systemctl status genai-app"
