import os
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

from 实时数据获取 import get_or_fetch_data

DEEPSEEK_API_KEY = "REDACTED_API_KEY"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'air_data',
    'charset': 'utf8mb4'
}

MODEL_DIR = r'C:\Users\28927\dazuoye\pythonProject3\模型'
FORECAST_HOURS = 3
LAG_HOURS = 23

SITE_DETAILS = {
    '1916A': {'name': '环境监测站', 'location': '曲靖'},
    '1917A': {'name': '烟厂办公区', 'location': '曲靖'},
    '2596A': {'name': '监测站', 'location': '昭通'},
    '2597A': {'name': '环保局', 'location': '昭通'},
    '2610A': {'name': '州水务局', 'location': '文山州'},
    '2611A': {'name': '市便民服务中心', 'location': '文山州'},
    '3376A': {'name': '南苑二区', 'location': '曲靖'},
    '3377A': {'name': '曲靖师范学院', 'location': '曲靖'}
}

AQI_LEVELS = {
    (0, 50): {'level': '优', 'color': '绿色', 'emoji': '😊', 'desc': '空气清新，非常适合户外活动'},
    (51, 100): {'level': '良', 'color': '黄色', 'emoji': '🙂', 'desc': '空气质量可以接受，敏感人群应减少长时间户外高强度运动'},
    (101, 150): {'level': '轻度污染', 'color': '橙色', 'emoji': '😐', 'desc': '敏感人群应减少户外活动，一般人群适量减少户外运动'},
    (151, 200): {'level': '中度污染', 'color': '红色', 'emoji': '😷', 'desc': '建议佩戴口罩，减少户外活动，特别是老人和儿童'},
    (201, 300): {'level': '重度污染', 'color': '紫色', 'emoji': '😨', 'desc': '请避免户外活动，关闭门窗，使用空气净化器'},
    (301, 500): {'level': '严重污染', 'color': '褐红色', 'emoji': '', 'desc': '健康警报！请留在室内，必须外出时请佩戴防护口罩'}
}


class ProphetModelWrapper:
    """Prophet模型包装器"""
    
    def __init__(self, site_name, regressors=None):
        self.site_name = site_name
        self.regressors = regressors or []
        self.model = None
        self.scaler_y = None
    
    def predict(self, df_future):
        """预测"""
        if self.model is None:
            raise ValueError("模型未加载")
        
        prophet_df = df_future[['ds'] + self.regressors].copy()
        forecast = self.model.predict(prophet_df)
        
        predictions_scaled = forecast['yhat'].values
        predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        
        return predictions
    
    @staticmethod
    def load(filepath):
        """加载模型"""
        data = joblib.load(filepath)
        wrapper = ProphetModelWrapper(data['site_name'], data['regressors'])
        wrapper.model = data['model']
        wrapper.scaler_y = data['scaler_y']
        return wrapper


def load_models():
    """加载所有站点的Prophet模型"""
    models = {}
    for site in SITE_DETAILS.keys():
        model_path = os.path.join(MODEL_DIR, f'aqi_prophet_model_{site}_future{FORECAST_HOURS}h.pkl')
        if os.path.exists(model_path):
            models[site] = ProphetModelWrapper.load(model_path)
            print(f"✓ 已加载站点 {site}（{SITE_DETAILS[site]['name']}）的Prophet模型")
        else:
            print(f"✗ 未找到站点 {site} 的模型文件")
    return models


engine = create_engine(
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?"
    f"charset={DB_CONFIG['charset']}"
)


def predict_aqi(site_name, forecast_hours=FORECAST_HOURS):
    """使用Prophet预测指定站点未来N小时的AQI"""
    
    if site_name not in models:
        print(f"⚠️ 站点 {site_name} 的模型未加载")
        return None
    
    model = models[site_name]
    location = SITE_DETAILS[site_name]['location']
    
    realtime_df = get_or_fetch_data(site_name, location, force_fetch=False)
    
    table_name = f'air_quality_site_{site_name.lower()}'
    query = f"SELECT * FROM `{table_name}` ORDER BY `datetime` DESC LIMIT 100"
    
    try:
        df = pd.read_sql(query, engine)
        
        if len(df) < 30:
            print(f"️ 站点 {site_name} 数据不足（{len(df)}条），尝试实时获取...")
            realtime_df = get_or_fetch_data(site_name, location, force_fetch=True)
            if realtime_df is not None:
                df = pd.read_sql(query, engine)
        
        if len(df) < 30:
            print(f"✗ 数据仍然不足，无法预测")
            return None
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        
        latest_time = df.index[-1]
        
        df_feat = df.copy()
        df_feat['hour'] = df_feat.index.hour
        df_feat['day_of_week'] = df_feat.index.dayofweek
        df_feat['month'] = df_feat.index.month
        df_feat['is_weekend'] = df_feat['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
        key_lags = [1, 2, 3, 6, 12, 23]
        for pollutant in pollutants:
            if pollutant in df_feat.columns:
                for lag in key_lags:
                    if len(df_feat) >= lag:
                        df_feat[f'{pollutant}_lag{lag}'] = df_feat[pollutant].shift(lag)
                
                for window in [3]:
                    if len(df_feat) >= window:
                        df_feat[f'{pollutant}_mean_{window}h'] = df_feat[pollutant].iloc[-window:].mean()
                        df_feat[f'{pollutant}_std_{window}h'] = df_feat[pollutant].iloc[-window:].std()
        
        if len(df_feat) >= 3:
            df_feat['AQI_mean_3h'] = df_feat['AQI'].iloc[-3:].mean()
            df_feat['AQI_std_3h'] = df_feat['AQI'].iloc[-3:].std()
        if len(df_feat) >= 2:
            df_feat['AQI_diff_1h'] = df_feat['AQI'].iloc[-1] - df_feat['AQI'].iloc[-2]
        
        target_col = f'AQI_future_{forecast_hours}h'
        df_feat[target_col] = df_feat['AQI'].shift(-forecast_hours)
        
        df_feat = df_feat.reset_index()
        df_feat = df_feat.rename(columns={'datetime': 'ds', 'AQI': 'y'})
        
        future_row = df_feat.iloc[[-1]].copy()
        future_ds = latest_time + timedelta(hours=forecast_hours)
        future_row['ds'] = future_ds
        
        # ✅ 修复：检查并填充回归变量中的NaN值
        regressor_cols = model.regressors
        for col in regressor_cols:
            if col in future_row.columns:
                if future_row[col].isna().any():
                    # 使用该列的历史中位数填充
                    historical_median = df_feat[col].median()
                    future_row[col] = future_row[col].fillna(historical_median)
        
        # ✅ 再次检查是否还有NaN
        missing_regressors = [col for col in regressor_cols if col in future_row.columns and future_row[col].isna().any()]
        if missing_regressors:
            print(f"⚠️ 仍有缺失值: {missing_regressors}，使用0填充")
            for col in missing_regressors:
                future_row[col] = future_row[col].fillna(0)
        
        predicted_aqi = model.predict(future_row)[0]
        
        predicted_aqi = max(0, min(500, predicted_aqi))
        
        prediction_time = future_ds
        
        return {
            'aqi': round(float(predicted_aqi), 1),
            'time': prediction_time,
            'current_time': latest_time
        }
        
    except Exception as e:
        print(f"❌ 预测出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_location_aqi(location, forecast_hours=FORECAST_HOURS):
    """综合预测某个地区所有站点的AQI"""
    location_sites = [site for site, info in SITE_DETAILS.items() 
                     if info['location'] == location]
    
    if not location_sites:
        print(f"⚠️ 未找到地区 {location} 的站点")
        return None
    
    print(f"📍 地区 {location} 共有 {len(location_sites)} 个监测站点")
    
    predictions = []
    for site in location_sites:
        print(f"  正在预测站点 {site}（{SITE_DETAILS[site]['name']}）...")
        pred = predict_aqi(site, forecast_hours)
        if pred is not None:
            pred['site_code'] = site
            pred['site_name'] = SITE_DETAILS[site]['name']
            predictions.append(pred)
    
    if not predictions:
        print(f"✗ 地区 {location} 所有站点预测失败")
        return None
    
    avg_aqi = np.mean([p['aqi'] for p in predictions])
    
    latest_time = max([p['time'] for p in predictions])
    
    return {
        'avg_aqi': round(float(avg_aqi), 1),
        'time': latest_time,
        'site_count': len(predictions),
        'sites': predictions
    }


def get_aqi_info(aqi_value):
    """根据AQI值获取等级信息"""
    for (min_val, max_val), info in AQI_LEVELS.items():
        if min_val <= aqi_value <= max_val:
            return info
    return AQI_LEVELS[(0, 50)]


def call_deepseek_api(user_query, aqi_info, location, predicted_aqi, site_name=None):
    """调用DeepSeek API生成有情感的回复"""
    
    aqi_level = aqi_info['level']
    emoji = aqi_info['emoji']
    desc = aqi_info['desc']
    
    try:
        time_str = aqi_info['time'].strftime('%Y年%m月%d日 %H:%M')
    except:
        time_str = "未来3小时"
    
    site_info = ""
    if site_name:
        site_info = f"- 监测站点：{site_name}\n"
    
    if predicted_aqi <= 50:
        style_guide = """
回复风格建议：
- 可以活泼欢快一些，表达对好天气的喜悦
- 鼓励用户多出门活动，享受美好时光
- 可以加入一些生活化的场景描述（如散步、跑步、郊游等）
- 语气轻松愉快，像好朋友分享好消息
"""
    elif predicted_aqi <= 100:
        style_guide = """
回复风格建议：
- 语气温和友好，带点轻松的提醒
- 可以提到适合的活动，但也给出小贴士
- 像朋友之间的日常聊天，自然随意
- 可以适当加入一些关怀的语气
"""
    elif predicted_aqi <= 150:
        style_guide = """
回复风格建议：
- 语气带点关切，但不要过于紧张
- 给出实用的建议，但不要太严肃
- 可以提到一些室内活动的建议
- 像朋友之间的善意提醒
"""
    elif predicted_aqi <= 200:
        style_guide = """
回复风格建议：
- 语气认真但不恐慌，表达真诚的关心
- 给出具体的防护建议
- 可以提到一些室内活动或替代方案
- 像家人一样的叮嘱和关怀
"""
    else:
        style_guide = """
回复风格建议：
- 语气紧急但不制造恐慌，表达深切的关心
- 强调防护措施的重要性
- 给出明确的行动建议
- 像好朋友在关键时刻的提醒
"""
    
    system_prompt = f"""你是一个贴心的空气质量助手，名叫"清新小助手"。你的任务是根据AQI预测数据，用温暖、有感情的语言回答用户的问题。

当前信息：
- 地点：{location}
{site_info}- 预测时间：{time_str}
- 预测AQI：{predicted_aqi:.0f}
- 空气质量等级：{aqi_level} {emoji}
- 健康建议：{desc}

{style_guide}

你可以自由发挥，不一定要严格按照模板。可以：
- 使用不同的表达方式和句式
- 适当变化emoji的使用
- 根据用户的提问方式调整回复风格
- 如果用户问得比较随意，可以更轻松地回应
- 如果用户问得很正式，可以稍微正式一些
- 可以加入一些生活化的比喻或场景
- 不必每次都提到所有信息，重点突出最关键的内容

用户问题：{user_query}

请用温暖、自然的语言回答："""
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            "temperature": 0.9,
            "max_tokens": 300,
            "top_p": 0.95,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.3
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except Exception as e:
        print(f"⚠️ API调用失败: {e}，使用备用回复")
        return generate_fallback_response(user_query, aqi_info, location, predicted_aqi, site_name)


def generate_fallback_response(user_query, aqi_info, location, predicted_aqi, site_name=None):
    """当API调用失败时，生成备用回复"""
    
    aqi_level = aqi_info['level']
    emoji = aqi_info['emoji']
    desc = aqi_info['desc']
    
    site_text = f"（{site_name}）" if site_name else ""
    
    if predicted_aqi <= 50:
        templates = [
            f"{emoji} {location}{site_text}的空气太棒啦！AQI只有{predicted_aqi:.0f}，{aqi_level}级别～{desc} 这么好的天气，出去走走吧！🌿✨",
            f"{emoji} 哇！{location}{site_text}未来3小时空气质量超赞！AQI约{predicted_aqi:.0f}，{aqi_level}！{desc} 快去呼吸新鲜空气～🌈",
            f"{emoji} {location}{site_text}的空气质量满分！AQI {predicted_aqi:.0f}，{aqi_level}级别哦～{desc} 今天适合户外活动！🎉"
        ]
    elif predicted_aqi <= 100:
        templates = [
            f"{emoji} {location}{site_text}空气质量还不错～AQI约{predicted_aqi:.0f}，{aqi_level}。{desc} 大部分活动都没问题！☀️",
            f"{emoji} {location}{site_text}未来3小时空气还可以，AQI {predicted_aqi:.0f}，{aqi_level}级别～{desc} 出门记得适度就好！🌤️",
            f"{emoji} 告诉您个消息，{location}{site_text}的AQI是{predicted_aqi:.0f}，{aqi_level}。{desc} 适合日常活动～"
        ]
    elif predicted_aqi <= 150:
        templates = [
            f"{emoji} {location}{site_text}的空气一般般呢...AQI约{predicted_aqi:.0f}，{aqi_level}。{desc} 敏感人群要注意哦～💛",
            f"{emoji} {location}{site_text}未来3小时AQI {predicted_aqi:.0f}，{aqi_level}级别。{desc} 建议减少户外活动～😐",
            f"{emoji} 提醒您一下，{location}{site_text}的空气质量是{aqi_level}，AQI {predicted_aqi:.0f}。{desc} 注意保护自己哦～"
        ]
    elif predicted_aqi <= 200:
        templates = [
            f"{emoji} {location}{site_text}的空气不太好😟 AQI约{predicted_aqi:.0f}，{aqi_level}。{desc} 出门一定要戴口罩呀！😷",
            f"{emoji} 注意啦！{location}{site_text}未来3小时AQI {predicted_aqi:.0f}，{aqi_level}。{desc} 尽量减少外出～🏠",
            f"{emoji} {location}{site_text}的空气质量是{aqi_level}，AQI {predicted_aqi:.0f}。{desc} 请保护好自己！️"
        ]
    else:
        templates = [
            f"{emoji} {location}{site_text}的空气很差⚠️ AQI约{predicted_aqi:.0f}，{aqi_level}。{desc} 请尽量待在室内！🏠",
            f"{emoji} 紧急提醒！{location}{site_text}未来3小时AQI {predicted_aqi:.0f}，{aqi_level}！{desc} 务必做好防护！",
            f"{emoji} {location}{site_text}的空气质量是{aqi_level}，AQI {predicted_aqi:.0f}。{desc} 请一定保护好自己！💪"
        ]
    
    import random
    return random.choice(templates)


def parse_user_intent(user_input):
    """解析用户输入，提取地点/站点和意图"""
    
    user_input_lower = user_input.lower()
    
    for site_code, site_info in SITE_DETAILS.items():
        site_name = site_info['name']
        if site_name in user_input_lower or site_name in user_input:
            print(f"🎯 识别到具体站点：{site_name}（{site_code}）")
            return 'site', site_code, site_name
    
    for site_code, site_info in SITE_DETAILS.items():
        location = site_info['location']
        if location in user_input_lower or location in user_input:
            print(f" 识别到地区：{location}")
            return 'location', location, location
    
    print("️ 未识别到具体地点，默认查询曲靖地区")
    return 'location', '曲靖', '曲靖'


def chat_with_assistant(user_input):
    """与空气质量助手对话"""
    
    predict_type, target_code, target_name = parse_user_intent(user_input)
    
    if predict_type == 'site':
        print(f"\n🔍 正在查询站点 {target_name}（{target_code}）的空气质量...")
        
        prediction = predict_aqi(target_code)
        
        if prediction is None:
            return f"抱歉，暂时无法获取{target_name}站点的空气质量数据，请稍后再试～"
        
        if 'aqi' not in prediction or 'time' not in prediction:
            return f"抱歉，{target_name}站点的预测数据不完整，请稍后再试～"
        
        predicted_aqi = prediction['aqi']
        aqi_info = get_aqi_info(predicted_aqi)
        
        print(f"📊 预测结果: AQI={predicted_aqi:.0f}, 等级={aqi_info['level']}")
        print(f"⏰ 预测时间: {prediction['time']}")
        
        response = call_deepseek_api(user_input, aqi_info, SITE_DETAILS[target_code]['location'], 
                                    predicted_aqi, site_name=target_name)
        
        return response
    
    else:
        print(f"\n🔍 正在查询 {target_name} 地区的空气质量...")
        
        prediction = predict_location_aqi(target_name)
        
        if prediction is None:
            return f"抱歉，暂时无法获取{target_name}地区的空气质量数据，请稍后再试～"
        
        avg_aqi = prediction['avg_aqi']
        aqi_info = get_aqi_info(avg_aqi)
        
        print(f"\n📊 综合预测结果:")
        print(f"  平均AQI: {avg_aqi:.0f}")
        print(f"  空气质量等级: {aqi_info['level']}")
        print(f"  监测站点数: {prediction['site_count']}")
        print(f"\n  各站点详情:")
        for site_pred in prediction['sites']:
            site_aqi_info = get_aqi_info(site_pred['aqi'])
            print(f"    - {site_pred['site_name']}: AQI={site_pred['aqi']:.0f} ({site_aqi_info['level']})")
        print(f"  ⏰ 预测时间: {prediction['time']}")
        
        response = call_deepseek_api(
            user_input, 
            aqi_info, 
            target_name, 
            avg_aqi,
            site_name=f"综合{prediction['site_count']}个站点"
        )
        
        return response


print("=" * 60)
print("🌿 欢迎使用清新小助手 - 滇东空气质量智能预测系统（Prophet版）")
print("=" * 60)
print("\n正在加载Prophet模型...")
models = load_models()
print(f"\n✓ 成功加载 {len(models)} 个站点Prophet模型")

print("\n💡 您可以这样问我：")
print("  📍 精确站点：")
print("    - '曲靖师范学院空气如何？'")
print("    - '烟厂办公区空气质量怎么样？'")
print("  🌍 地区综合：")
print("    - '曲靖今天空气如何？'")
print("    - '文山的空气质量怎么样？'")
print("    - '昭通未来空气好吗？'")
print("\n输入 'quit' 或 '退出' 结束对话\n")

if __name__ == "__main__":
    while True:
        try:
            user_input = input("👤 您：").strip()
            
            if user_input.lower() in ['quit', '退出', 'exit']:
                print("\n 再见！祝您呼吸愉快～🌿")
                break
            
            if not user_input:
                continue
            
            response = chat_with_assistant(user_input)
            print(f"\n🤖 清新小助手：{response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！祝您呼吸愉快～🌿")
            break
        except Exception as e:
            print(f"\n❌ 出错了：{e}")
            import traceback
            traceback.print_exc()
            print("请重试\n")
