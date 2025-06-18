@echo off
echo Starting Bitcoin Prediction Dashboard...
echo.

REM Change to the bitcoin-timeseries-predictor directory
cd /d "%~dp0.."

REM Check if we're in the right directory
if not exist "models\enhanced_predictions.csv" (
    echo Error: Model files not found in models/ directory
    echo Please ensure you're running this from the bitcoin-timeseries-predictor folder
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo Found model files. Starting dashboard...
echo.

REM Run the Streamlit dashboard
streamlit run deployment/dashboard.py

pause 