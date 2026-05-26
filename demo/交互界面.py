from flask import Flask, render_template, request, jsonify
import os
import sys
from datetime import datetime
import json

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

# 导入对话助手模块
from ds输出 import chat_with_assistant

app = Flask(__name__)

# 全局变量存储模型
models = {}

# 加载模型
def init_models():
    """初始化并加载模型"""
    global models
    from ds输出 import load_models
    print("正在加载空气质量预测模型...")
    models = load_models()
    return models

# 启动时加载模型
models = init_models()

# ==================== 路由定义 ====================

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API接口：接收用户输入并返回预测结果"""
    try:
        data = request.json
        user_input = data.get('message', '').strip()

        if not user_input:
            return jsonify({
                'success': False,
                'error': '请输入您的问题'
            }), 400

        # 调用对话助手
        response = chat_with_assistant(user_input)

        return jsonify({
            'success': True,
            'response': response,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/quick-query', methods=['POST'])
def api_quick_query():
    """API接口：快捷查询"""
    try:
        data = request.json
        location = data.get('location', '')
        site_name = data.get('site_name', '')

        if site_name:
            query = f"{site_name}的空气质量如何？"
        elif location:
            query = f"{location}的空气质量怎么样？"
        else:
            query = "曲靖今天空气如何？"

        response = chat_with_assistant(query)

        return jsonify({
            'success': True,
            'response': response,
            'query': query
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/status', methods=['GET'])
def api_status():
    """API接口：获取系统状态"""
    return jsonify({
        'success': True,
        'models_loaded': len(models),
        'available_sites': list(models.keys()),
        'status': 'running'
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🌿 清新小助手 - Web交互界面")
    print("=" * 60)
    print(f"✓ 已加载 {len(models)} 个站点模型")
    print("\n 服务启动中...")
    print(" 访问地址: http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
