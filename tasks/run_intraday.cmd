@echo off
cd /d C:\Users\User\OneDrive\Desktop\stock
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe scripts\monitor_intraday.py >> logs\intraday.log 2>&1
