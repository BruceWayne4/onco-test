@echo off
REM Script to recreate virtual environment after reinstalling Python
echo Removing old virtual environment...
rmdir /s /q .venv

echo Creating new virtual environment...
python -m venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install -e .

echo Done! Virtual environment recreated.
echo You can now run: python -m oncocontext --help
pause
