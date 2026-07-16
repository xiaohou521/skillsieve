#!/bin/bash
# SkillSieve Orange Pi AIpro Setup Script
# Run as root on a fresh Orange Pi AIpro (Ubuntu 22.04, ARM64)
set -e

echo "=== SkillSieve Orange Pi Setup ==="

# 1. System packages
echo "[1/6] Installing system packages..."
apt-get update
apt-get install -y git python3.11 python3.11-venv python3.11-dev python3-pip curl

# 2. Set Python 3.11 as default
echo "[2/6] Configuring Python 3.11..."
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
python3 --version

# 3. Clone SkillSieve
echo "[3/6] Cloning SkillSieve..."
cd /root
if [ ! -d "skillsieve" ]; then
    git clone https://github.com/xiaohou521/skillsieve.git
fi
cd skillsieve

# 4. Install dependencies
echo "[4/6] Installing Python dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
python3 -m pip install httpx

# 5. Run tests to verify
echo "[5/6] Running tests..."
python3 -m pytest tests/ -x -q

# 6. Create .env template
echo "[6/6] Creating .env template..."
cat > /root/skillsieve/.env << 'ENVEOF'
# SkillSieve API Keys
# Fill in your actual keys below

# Kimi 2.5 (Moonshot) - Layer 2 default + Layer 3 Juror 1
KIMI_API_KEY=sk-kimi-REPLACE_ME

# MiniMax M2.7 - Layer 3 Juror 2
MINIMAX_API_KEY=sk-cp-REPLACE_ME

# Baidu Qianfan (DeepSeek-V3 via Baidu) - Layer 3 Juror 3
BAIDU_API_KEY=bce-v3/REPLACE_ME
ENVEOF

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Edit /root/skillsieve/.env with your API keys"
echo "  2. Transfer openclaw/skills data (from Mac via scp)"
echo "  3. Run: cd /root/skillsieve && source .env && python3 experiments/real_scan.py"
