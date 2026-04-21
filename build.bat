@echo off
REM FanFu - Build package (Windows)
cd /d "%~dp0"

set PYTHON=python

echo === FanFu Build ===

echo [1/3] Cleaning old builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist *.egg-info rmdir /s /q *.egg-info
if exist fanfu.egg-info rmdir /s /q fanfu.egg-info

echo [2/3] Installing build tools...
%PYTHON% -m pip install --upgrade build -q

echo [3/3] Building package...
%PYTHON% -m build
%PYTHON% -m twine check dist\*

echo === Build Complete! ===
echo Files in dist/:
dir /b dist\
pause
