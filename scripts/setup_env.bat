@echo off
REM One-time setup for TOF Analysis (Windows — double-click or run in cmd).
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_env.ps1"
if errorlevel 1 pause
