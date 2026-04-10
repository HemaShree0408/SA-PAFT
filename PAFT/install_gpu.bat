@echo off
echo [SA-PAFT] Starting Installation for GPU Machine...
echo.

echo 1. Installing Core Dependencies (Torch GPU first)...
pip install torch>=2.3.0 --extra-index-url https://download.pytorch.org/whl/cu121

echo.
echo 2. Installing specialized research libraries...
pip install -r requirements.txt --no-warn-conflicts

echo.
echo 3. Installing Llama-Factory in editable mode...
cd src
pip install -e .
cd ..

echo.
echo [SUCCESS] Dependencies installed. 
echo You can now run: llamafactory-cli train reproduction_paper.yaml
pause
