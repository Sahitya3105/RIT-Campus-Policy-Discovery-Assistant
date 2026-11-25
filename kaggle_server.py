"""
MSRIT AI Policy Assistant — Kaggle GPU Server
==============================================
Runs qwen3.8:27b (Q4_K_M, 17.7GB) on 2×Tesla T4 via Ollama,
exposed publicly via ngrok with --host-header=rewrite (fixes Ollama's
allowedHostsMiddleware 403 on non-localhost Host headers).

Setup:
  1. Kaggle sidebar → Accelerator = GPU T4 x2, Internet = On
  2. Paste your ngrok authtoken (https://dashboard.ngrok.com/get-started/your-authtoken)
  3. Run All — takes ~5 min first time (model pull), ~1 min on restart
  4. Copy the printed URL → paste into .env on your local machine:
       LLM_BASE_URL = https://<ngrok-domain>/v1
"""

import os
import re
import subprocess
import sys
import time

# ── CONFIG ─────────────────────────────────────────────────────────────────
# Retrieve ngrok token from environment or Kaggle Secrets, with fallback placeholder
NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN", "YOUR_NGROK_AUTHTOKEN_HERE")

# If you have a reserved ngrok domain, paste it here (keeps URL stable).
# Otherwise leave as "" and a random free URL is generated each session.
NGROK_DOMAIN = os.environ.get("NGROK_DOMAIN", "")  # e.g. "cornstalk-prescribe-propeller.ngrok-free.dev"

MODEL = "qwen3.8:27b"   # Q4_K_M, 17.7GB — fits across 2×T4 (29.1GB VRAM total)

# ── OLLAMA ENV (must be set before ollama serve) ────────────────────────────
os.environ["OLLAMA_FLASH_ATTENTION"] = "1"      # qwen3 is on Ollama's allowlist
os.environ["OLLAMA_KV_CACHE_TYPE"]   = "q8_0"  # saves ~1.5GB VRAM vs f16
os.environ["OLLAMA_HOST"]            = "0.0.0.0:11434"
os.environ["OLLAMA_ORIGINS"]         = "*"      # belt-and-suspenders alongside ngrok rewrite
os.environ["CUDA_VISIBLE_DEVICES"]   = "0,1"   # explicit 2-GPU spread

# ── 1. Install system deps ──────────────────────────────────────────────────
print("[1/7] Installing zstd + pciutils (Ollama GPU detection)...")
subprocess.run(
    "apt-get update -qq && apt-get install -y -qq zstd pciutils",
    shell=True, check=True
)

# ── 2. Install Ollama ───────────────────────────────────────────────────────
print("[2/7] Installing Ollama...")
subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)

print("      GPU visible to OS:")
subprocess.run(
    "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo 'nvidia-smi FAILED'",
    shell=True
)

# ── 3. Start Ollama server ──────────────────────────────────────────────────
# Write to a real file (not PIPE) to avoid the 64KB OS pipe-buffer deadlock that
# wedges the Go logger mutex and eventually hangs all request handlers.
print("[3/7] Starting ollama serve...")
ollama_log = open("/kaggle/working/ollama.log", "w")
ollama_proc = subprocess.Popen(
    ["ollama", "serve"],
    stdout=ollama_log,
    stderr=subprocess.STDOUT,
    env=os.environ.copy()
)
time.sleep(5)
print(f"      pid={ollama_proc.pid}  (tail: /kaggle/working/ollama.log)")

# ── 4. Pull model ───────────────────────────────────────────────────────────
print(f"[4/7] Pulling {MODEL} (~17.7GB, first run only)...")
subprocess.run(["ollama", "pull", MODEL], check=True)

# ── 5. Local sanity check ───────────────────────────────────────────────────
# Use /api/chat with think:False — avoids the qwen3 thinking-bleed bug where
# the full token budget is consumed by hidden <think> blocks, leaving content empty.
# /no_think token is a belt-and-suspenders soft-switch on top of that.
import requests

print("[5/7] Local sanity check (/api/chat, think=False)...")
print(f"      ⏳ First call loads {MODEL} (~17.7GB) into 2×T4 VRAM — allow up to 5 min...")
resp = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model":    MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: LLM ONLINE /no_think"}],
        "stream":   False,
        "think":    False,
    },
    timeout=600,   # 27B cold-start load can take 3-5 min on first inference
)
resp.raise_for_status()
reply = resp.json()["message"]["content"]
# Strip any leaked <think> block defensively
reply_clean = re.sub(r"^\s*(?:<think>)?.*?</think>\s*", "", reply, count=1, flags=re.DOTALL).strip()
if not reply_clean:
    print("      ⚠️  WARNING: empty content — thinking bleed detected, check model tag")
else:
    print(f"      ✅ Model replied: {reply_clean!r}")

print("      ollama ps (PROCESSOR column should say GPU):")
subprocess.run(["ollama", "ps"])

# ── 6. Install ngrok CLI + start tunnel ────────────────────────────────────
# --host-header=rewrite is THE critical flag: it rewrites the Host header back to
# localhost:11434 before the request reaches Ollama, bypassing allowedHostsMiddleware.
# Without it, Ollama sees Host: <ngrok-domain> and returns 403 "not allowed by policy".
# (Cloudflare tunnels have no equivalent of this flag — that's why cloudflared fails.)
print("[6/7] Installing ngrok CLI...")
subprocess.run(
    'curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc '
    '| tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null '
    '&& echo "deb https://ngrok-agent.s3.amazonaws.com buster main" '
    '| tee /etc/apt/sources.list.d/ngrok.list '
    '&& apt-get update -qq && apt-get install -y -qq ngrok',
    shell=True, check=True,
)
subprocess.run(["ngrok", "config", "add-authtoken", NGROK_AUTHTOKEN], check=True)

ngrok_cmd = ["ngrok", "http", "--host-header=rewrite", "11434"]
if NGROK_DOMAIN:
    ngrok_cmd.append(f"--domain={NGROK_DOMAIN}")

ngrok_log = open("/kaggle/working/ngrok.log", "w")
ngrok_proc = subprocess.Popen(ngrok_cmd, stdout=ngrok_log, stderr=subprocess.STDOUT)
time.sleep(6)
print(f"      ngrok pid={ngrok_proc.pid}  (tail: /kaggle/working/ngrok.log)")

# ── 7. Print the public URL ─────────────────────────────────────────────────
print("[7/7] Getting public tunnel URL...")
try:
    tunnels = requests.get("http://localhost:4040/api/tunnels", timeout=10).json()
    public_url = tunnels["tunnels"][0]["public_url"]
except Exception:
    public_url = f"https://{NGROK_DOMAIN}" if NGROK_DOMAIN else "check /kaggle/working/ngrok.log"

print()
print("=" * 60)
print(f"  🚀  LIVE ENDPOINT: {public_url}/v1")
print("=" * 60)
print()
print("→ Paste into c:\\Users\\sahit\\Desktop\\Dell\\.env:")
print(f"    LLM_BASE_URL   = {public_url}/v1")
print(f"    LLM_MODEL_NAME = {MODEL}")
print()

# ── Heartbeat loop — keeps session alive + logs GPU health ─────────────────
print("Heartbeat running (Ctrl+C / interrupt cell to stop)...")
try:
    while True:
        time.sleep(60)
        print(f"[heartbeat] up — {public_url}/v1")
        subprocess.run(["ollama", "ps"])
        subprocess.run(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
            "--format=csv,noheader",
            shell=True,
        )
except KeyboardInterrupt:
    print("Stopped.")
