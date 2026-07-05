@echo off
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe scripts\preview_scan.py >> logs\preview.log 2>&1
