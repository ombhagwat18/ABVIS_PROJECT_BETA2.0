@echo off
rem NexaInspect - timing check: are the PLC ladder, the settings, the cameras and the AI timed so that the REJECT
rem hits the right bottle at the right moment?  Menu: plan (no hardware), apply, watch the PLC, stopwatch travel.
rem Plan = arithmetic on settings.json + the ladder file.  Watch / travel only READ the PLC (never M0 / M1 / Y).
cd /d "%~dp0"
title NexaInspect - Timing check

where python >nul 2>nul
if errorlevel 1 (
    echo   Python was not found on PATH. Run run.bat once first.
    pause
    exit /b 1
)

:menu
echo.
echo   NexaInspect timing check   (report is saved in reports\timing\)
echo   ---------------------------------------------------------------
echo   1  PLAN     check the saved timing on paper (no hardware, safe any time)
echo   2  APPLY    PLAN + write the ladder values to settings.json (close NexaInspect first)
echo   3  WATCH    watch the real PLC for 2 minutes and show each bottle's measured timeline
echo   4  TRAVEL   stopwatch: measure photo-eye to reject cylinder (8 bottles) and save it
echo   5  QUIT
echo.
choice /C 12345 /N /M "  Choose 1-5: "
if errorlevel 5 goto end
if errorlevel 4 goto travel
if errorlevel 3 goto watch
if errorlevel 2 goto apply
python -m tools.timing_check %*
goto again

:apply
python -m tools.timing_check --apply %*
goto again

:watch
echo.
echo   Close NexaInspect, ISPSoft and COMMGR first (one program per COM port).
echo   Run bottles through the machine now; nothing is written to the PLC.
pause
python -m tools.timing_check --watch 120
goto again

:travel
echo.
echo   Close NexaInspect, ISPSoft and COMMGR first (one program per COM port).
echo   Put a bottle on the belt; press ENTER when it is in front of the reject cylinder.
pause
python -m tools.timing_check --travel 8 --apply
goto again

:again
echo.
pause
goto menu

:end
