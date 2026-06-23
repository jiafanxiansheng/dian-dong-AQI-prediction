from flask import Flask, render_template, request, jsonify
import os
import sys
from datetime import datetime
import traceback

sys.path.insert(0, os.path.dirname(__file__))

from ds输出 import chat_with_assistant

app = Flask(__name__)

models = {}

def init_models():
    global models
    from ds输出 import load_models
    print("正在加载空气质量预测模型...")
    models = load_models()
    return models

models = init_models()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        user_input = data.get('message', '').strip()

        if not user_input:
            return jsonify({'success': False, 'error': '请输入您的问题'}), 400

        result = chat_with_assistant(user_input)

        return jsonify({
            'success': True,
            'response': result['response'],
            'data': result.get('data'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': '服务暂时不可用，请稍后重试',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500


@app.route('/api/quick-query', methods=['POST'])
def api_quick_query():
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

        result = chat_with_assistant(query)

        return jsonify({
            'success': True,
            'response': result['response'],
            'data': result.get('data'),
            'query': query,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': '服务暂时不可用，请稍后重试',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500


@app.route('/api/status', methods=['GET'])
def api_status():
    loaded_count = 0
    available_sites = []
    for site, fh_models in models.items():
        if fh_models:
            loaded_count += 1
            available_sites.append(site)

    return jsonify({
        'success': True,
        'models_loaded': loaded_count,
        'available_sites': available_sites,
        'forecast_hours': [6, 12],
        'status': 'running'
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🌿 清新小助手 - Web交互界面")
    print("=" * 60)
    site_count = sum(1 for m in models.values() if m)
    print(f"✓ 已加载 {site_count} 个站点模型")
    print("\n 服务启动中...")
    print(" 访问地址: http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
