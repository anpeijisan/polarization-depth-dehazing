@echo off
setlocal
set "TEMP=D:\tmp"
set "TMP=D:\tmp"
set "TMPDIR=D:\tmp"
set "PYTHONDONTWRITEBYTECODE=1"
set "YOLO_AUTOINSTALL=false"
set "PATH=E:\CONDA\envs\onnx;E:\CONDA\envs\onnx\Scripts;E:\CONDA\envs\onnx\Library\bin;%PATH%"
cd /d "%~dp0"
"E:\CONDA\envs\onnx\python.exe" "%~dp0main_v2.py"
if errorlevel 1 pause

