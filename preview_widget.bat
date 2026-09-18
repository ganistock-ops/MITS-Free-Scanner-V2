@echo off
setlocal
title MITS Free Scanner V2 - Local Preview Launcher

echo =====================================================================
echo           MITS Free Scanner V2 - Local Preview Launcher
echo =====================================================================
echo.

where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [INFO] Python detected on system.
    echo [INFO] Starting local HTTP server on port 8088 to verify async JSON fetch...
    echo [INFO] Opening preview in default browser: http://127.0.0.1:8088/free_scanner_widget.html
    echo.
    echo Press Ctrl+C in this window to stop the preview server.
    echo =====================================================================
    start http://127.0.0.1:8088/free_scanner_widget.html
    python -m http.server 8088
) else (
    echo [INFO] Python not detected in PATH.
    echo [INFO] Opening free_scanner_widget.html directly in your default browser...
    echo [INFO] The widget will seamlessly render with its high-fidelity inline dataset!
    echo =====================================================================
    start free_scanner_widget.html
    pause
)
