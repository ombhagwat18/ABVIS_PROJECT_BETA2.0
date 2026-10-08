@echo off
rem Bottle Inspection - double-click to set up (once) and start the dashboard.
cd /d "%~dp0"
title Bottle Inspection
rem Delayed expansion: %VAR% inside a parenthesised block is expanded when the
rem block is parsed, not when it runs, so TORCH_INDEX below would read empty.
setlocal EnableDelayedExpansion

where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo   Python was not found on PATH.
    echo   Install Python 3.10+ from python.org and tick "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

rem ---------------------------------------------------------------- deps
rem Only install when something is actually missing, so a normal launch does
rem not wait on pip resolving a dozen packages that are already there.
python -c "import torch, torchvision, cv2, customtkinter, PIL" 2>nul
if errorlevel 1 (
    echo.
    echo   Installing dependencies. The first run downloads ~2.5 GB and takes
    echo   a few minutes. Later launches skip this entirely.
    echo.

    rem CUDA wheels are worth 2.5 GB only if there is an NVIDIA card to use them.
    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
    powershell -NoProfile -Command "if (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' }) { exit 0 } else { exit 1 }" >nul 2>nul
    if not errorlevel 1 set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
    echo   Using PyTorch build: !TORCH_INDEX!
    echo.

    python -m pip install --upgrade pip
    python -m pip install torch torchvision --index-url !TORCH_INDEX!
    python -m pip install -r requirements.txt

    python -c "import torch, torchvision, cv2, customtkinter, PIL" 2>nul
    if errorlevel 1 (
        echo.
        echo   Install did not complete. Scroll up for the pip error.
        echo.
        pause
        exit /b 1
    )
    echo.
    echo   Dependencies installed.
    echo.
)

python -c "import torch;print('   Torch',torch.__version__,'| CUDA',('yes - '+torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'no (CPU only, training will be slow)')"

rem ------------------------------------------- migrate the old single-project layout
if exist "All Datasets" (
    echo.
    echo   Moving the old layout into projects\ ...
    python -m tools.migrate "OM Bottle" --run
    echo.
)

rem ------------------------------------------------------------ first run
rem The crop region must be measured before anything can train. Which project is
rem active is Python's business, so ask it rather than guessing a path here.
python -c "import dataset as D,sys; sys.exit(0 if D.CONFIG_JSON.exists() or not D.scan_images() else 1)"
if errorlevel 1 (
    echo.
    echo   First run - measuring the crop region from your images...
    python -m vision.calibrate
    echo.
)

rem ---------------------------------------------------------------- launch
rem python.exe, not pythonw.exe: the window keeps the training log and any
rem traceback visible. Close the app window to stop.
echo.
echo   Starting the dashboard...
python gui.py

echo.
echo   Closed.
pause
