@echo off
cd /d %~dp0..
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe scripts\fetch_chips.py --datasets inst,margin --update --sleep 6.5 >> logs\chips_update.log 2>&1
