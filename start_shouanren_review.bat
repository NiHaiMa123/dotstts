@echo off
title Shouanren Review Server - close this window to stop
cd /d "%~dp0"
echo Serving: http://127.0.0.1:8787/outputs/shouanren_review/
echo Close this window = stop server.
start "" microsoft-edge:http://127.0.0.1:8787/outputs/shouanren_review/
.venv\Scripts\python.exe -m http.server 8787 --bind 127.0.0.1
pause
