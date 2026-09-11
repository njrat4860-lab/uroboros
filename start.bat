@echo off
title Уроборус
chcp 65001 >nul
cd /d %~dp0
where python >nul 2>&1
if %errorlevel% neq 0 (
echo Python не найден. Установи Python 3.12 с python.org, при установке отметь Add to PATH, потом запусти снова.
pause
exit /b 1
)
if not exist .venv\Scripts\python.exe (
echo Первый запуск: создаю окружение...
python -m venv .venv
)
.venv\Scripts\python -m pip install --upgrade pip >nul
.venv\Scripts\python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
echo Не получилось поставить пакеты. Проверь интернет и запусти снова.
pause
exit /b 1
)
.venv\Scripts\python src\main.py
pause
