@echo off
chcp 65001 >nul
title 清新小助手 - 滇东AQI智能预测系统

echo ============================================
echo   清新小助手 - 滇东AQI智能预测系统
echo ============================================
echo.

:: 检查 .env 文件
if not exist .env (
    echo [警告] 未找到 .env 配置文件
    echo 请复制 .env.example 为 .env 并填入真实配置
    echo.
    copy .env.example .env >nul 2>&1
    echo 已自动创建 .env，请编辑后再运行
    echo.
    pause
    exit /b 1
)

:: 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.9+
    pause
    exit /b 1
)

:: 安装依赖（如需要）
echo [1/3] 检查依赖...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)
echo       依赖检查完成

:: 加载环境变量
echo [2/3] 加载配置...
for /f "tokens=*" %%a in (.env) do (
    echo %%a | findstr /v /c:"#" >nul 2>&1
    if not errorlevel 1 set %%a
)
echo       配置加载完成

:: 启动应用
echo [3/3] 启动 Web 服务...
echo.
echo ============================================
echo   访问地址: http://localhost:5000
echo   按 Ctrl+C 停止服务
echo ============================================
echo.

python app.py

pause
