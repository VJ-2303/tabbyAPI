@echo off
title TabbyAPI Server - Swift-1.5-Qwen3.8-27B
cd /d "%~dp0"
echo ======================================================================
echo Starting TabbyAPI on http://127.0.0.1:5000 (0.0.0.0:5000)
echo Model: Swift-1.5-Qwen3.8-27B-EXL3-3.0bpw
echo Hardware: NVIDIA RTX 2000 Ada (16GB VRAM)
echo API Key: webpcqwencreo
echo ======================================================================

"%~dp0venv\Scripts\python.exe" main.py %*

