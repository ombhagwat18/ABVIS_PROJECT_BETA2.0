@echo off
rem NexaInspect - system report: measures this computer, the cameras, the PLC link, the saved ladder, the models
rem and the settings, and compares them with the project specification (SPEC in tools\system_report.py).
rem Read-only: nothing is written to the PLC, settings.json, labels or models.
cd /d "%~dp0"
title NexaInspect - System report

where python >nul 2>nul
if errorlevel 1 (
    echo   Python was not found on PATH. Run run.bat once first.
    pause
    exit /b 1
)

echo.
echo   NexaInspect system report
echo   -------------------------
echo   Close NexaInspect, ISPSoft and COMMGR first: a camera or a COM port
echo   can be used by only one program at a time.
echo.
echo   The PLC test only READS (link replies, latency, RUN, X/M/Y/T/C).
echo   It never writes M0/M1 or any output.
echo.
choice /C YN /M "  Also test the PLC link (read-only)"
if errorlevel 2 goto noplc
set "PLCARG=--plc"
goto go
:noplc
set "PLCARG="
:go
echo.
python -m tools.system_report %PLCARG% %*
echo.
echo   The full report (Markdown + JSON + one frame per camera) is in reports\system\
pause
