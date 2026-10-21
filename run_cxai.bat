@echo off
chcp 65001 > nul

echo ====================================
echo   Concept XAI - Setup & Execution
echo ====================================
echo Installing package in editable mode...
pip install -e .
if errorlevel 1 goto ERROR

echo.
echo Starting Concept XAI CLI...
call cxai

if errorlevel 1 goto ERROR

echo.
echo ------------------------------------
echo Process completed.
echo ------------------------------------
pause
exit /b 0

:ERROR
echo.
echo ====================================
echo [ERROR] An error occurred.
echo ====================================
pause
exit /b 1