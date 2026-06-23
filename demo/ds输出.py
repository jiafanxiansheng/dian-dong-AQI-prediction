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

MODEL_DIR = r'C:\Users\28927\dazuoye\\pythonProject3\模型'
FORECAST_HOURS_LIST = [6, 12]
PRIMARY_FORECAST_HOURS = 12
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

LOCATION_CITY_MAP = {
    '曲靖': '曲靖',
    '昭通': '昭通',
    '文山州': '文山'
}

WEATHER_ZH_MAP = {
    'Sunny': '晴', 'Clear': '晴',
    'Partly cloudy': '多云', 'Partly Cloudy': '多云',
    'Cloudy': '阴', 'Overcast': '阴',
    'Mist': '薄雾', 'Fog': '雾', 'Freezing fog': '冻雾',
    'Patchy rain nearby': '局部小雨', 'Patchy rain possible': '可能有零星小雨',
    'Light rain': '小雨', 'Light rain shower': '小阵雨',
    'Moderate rain': '中雨', 'Moderate rain at times': '时有中雨',
    'Heavy rain': '大雨', 'Heavy rain at times': '时有大雨',
    'Light drizzle': '小毛毛雨', 'Patchy light drizzle': '局部小毛毛雨',
    'Torrential rain': '暴雨',
    'Patchy snow nearby': '局部小雪', 'Light snow': '小雪',
    'Moderate snow': '中雪', 'Heavy snow': '大雪',
    'Patchy light rain': '局部小雨', 'Light rain shower': '小阵雨',
    'Thundery outbreaks possible': '可能有雷阵雨',
    'Thundery outbreaks in nearby': '附近有雷阵雨',
    'Lightning': '闪电', 'Light rain with thunderstorm': '雷阵雨',
    'Patchy light snow': '局部小雪', 'Blowing snow': '吹雪',
    'Blizzard': '暴风雪',
    'Patchy freezing drizzle nearby': '局部冻毛毛雨',
    'Light freezing rain': '小冻雨', 'Moderate or heavy freezing rain': '中到大冻雨',
    'Light sleet': '小冰雹', 'Moderate or heavy sleet': '中到大冰雹',
    'Patchy sleet nearby': '局部冰雹',
    'Light shower snow': '小阵雪', 'Patchy moderate snow': '局部中雪',
    'Patchy heavy snow': '局部大雪',
    'Windy': '大风', 'Calm': '平静',
    'Light drizzle': '小毛毛雨', 'Heavy freezing drizzle': '大冻毛毛雨',
}


def get_weather_info(location):
    """通过wttr.in获取实时天气信息（免费，无需API Key）"""
    city = LOCATION_CITY_MAP.get(location, location)
    try:
        url = f"https://wttr.in/{city}?format=j1&lang=zh"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        current = data['current_condition'][0]
        today = data['weather'][0]
        
        temp = current['temp_C']
        feels_like = current['FeelsLikeC']
        humidity = current['humidity']
        
        weather_en = current['weatherDesc'][0]['value']
        weather_zh = current.get('lang_zh', [{}])[0].get('value', '')
        if not weather_zh or weather_zh == weather_en:
            weather_zh = WEATHER_ZH_MAP.get(weather_en, weather_en)
        
        wind_speed = current['windspeedKmph']
        wind_dir = current['winddir16Point']
        max_temp = today['maxtempC']
        min_temp = today['mintempC']
        
        return {
            'temp': temp,
            'feels_like': feels_like,
            'humidity': humidity,
            'weather': weather_zh,
            'wind_speed': wind_speed,
            'wind_dir': wind_dir,
            'max_temp': max_temp,
            'min_temp': min_temp
        }
    except Exception as e:
        print(f"⚠️ 获取{city}天气失败: {e}")
        return None


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
    """加载所有站点的Prophet模型（多时间跨度）"""
    models = {}
    for site in SITE_DETAILS.keys():
        models[site] = {}
        for fh in FORECAST_HOURS_LIST:
            model_path = os.path.join(MODEL_DIR, f'aqi_prophet_model_{site}_future{fh}h.pkl')
            if os.path.exists(model_path):
                models[site][fh] = ProphetModelWrapper.load(model_path)
                print(f"✓ 已加载站点 {site}（{SITE_DETAILS[site]['name']}）的{fh}h Prophet模型")
            else:
                print(f"✗ 未找到站点 {site} 的{fh}h模型文件")
    return models


engine = create_engine(
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?"
    f"charset={DB_CONFIG['charset']}"
)


def predict_aqi(site_name):
    """使用多个Prophet模型分步预测指定站点未来不同时间的AQI"""

    if site_name not in models:
        print(f"⚠️ 站点 {site_name} 的模型未加载")
        return None

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

        df_feat = df_feat.reset_index()
        df_feat = df_feat.rename(columns={'datetime': 'ds', 'AQI': 'y'})

        multi_preds = {}
        for fh in FORECAST_HOURS_LIST:
            if fh not in models[site_name]:
                print(f"⚠️ 站点 {site_name} 的{fh}h模型未加载，跳过")
                continue

            model = models[site_name][fh]

            future_row = df_feat.iloc[[-1]].copy()
            future_ds = latest_time + timedelta(hours=fh)
            future_row['ds'] = future_ds

            regressor_cols = model.regressors
            for col in regressor_cols:
                if col in future_row.columns:
                    if future_row[col].isna().any():
                        historical_median = df_feat[col].median()
                        future_row[col] = future_row[col].fillna(historical_median)

            missing_regressors = [col for col in regressor_cols if col in future_row.columns and future_row[col].isna().any()]
            if missing_regressors:
                print(f"⚠️ {fh}h预测仍有缺失值: {missing_regressors}，使用0填充")
                for col in missing_regressors:
                    future_row[col] = future_row[col].fillna(0)

            predicted_aqi = model.predict(future_row)[0]
            predicted_aqi = max(0, min(500, predicted_aqi))

            multi_preds[fh] = {
                'aqi': round(float(predicted_aqi), 1),
                'time': future_ds
            }

        if not multi_preds:
            return None

        return {
            'multi_preds': multi_preds,
            'current_time': latest_time
        }

    except Exception as e:
        print(f"❌ 预测出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_location_aqi(location):
    """综合预测某个地区所有站点的AQI（多时间跨度）"""
    location_sites = [site for site, info in SITE_DETAILS.items() 
                     if info['location'] == location]
    
    if not location_sites:
        print(f"⚠️ 未找到地区 {location} 的站点")
        return None
    
    print(f"📍 地区 {location} 共有 {len(location_sites)} 个监测站点")
    
    predictions = []
    for site in location_sites:
        print(f"  正在预测站点 {site}（{SITE_DETAILS[site]['name']}）...")
        pred = predict_aqi(site)
        if pred is not None:
            pred['site_code'] = site
            pred['site_name'] = SITE_DETAILS[site]['name']
            predictions.append(pred)
    
    if not predictions:
        print(f"✗ 地区 {location} 所有站点预测失败")
        return None
    
    latest_time = max([p['current_time'] for p in predictions])
    
    return {
        'site_count': len(predictions),
        'current_time': latest_time,
        'sites': predictions
    }


def get_aqi_info(aqi_value):
    """根据AQI值获取等级信息"""
    for (min_val, max_val), info in AQI_LEVELS.items():
        if min_val <= aqi_value <= max_val:
            return info
    return AQI_LEVELS[(0, 50)]


def call_deepseek_api(user_query, aqi_info, location, predicted_aqi, site_name=None, weather_info=None, multi_preds=None, is_unsupported=False):
    """调用DeepSeek API生成有情感的回复"""
    
    aqi_level = aqi_info['level']
    emoji = aqi_info['emoji']
    desc = aqi_info['desc']
    
    try:
        time_str = aqi_info['time'].strftime('%Y年%m月%d日 %H:%M')
    except:
        time_str = f"未来{PRIMARY_FORECAST_HOURS}小时"
    
    if is_unsupported:
        forecast_desc = f"该地区（{location}）不在系统覆盖范围内，无法提供AQI预测"
    else:
        forecast_desc = f"从当前时间起，预测未来{PRIMARY_FORECAST_HOURS}小时后的空气质量（目标时间：{time_str}）"
    
    site_info = ""
    if site_name:
        site_info = f"- 监测站点：{site_name}\n"
    
    weather_text = ""
    if weather_info:
        weather_text = (
            f"- 当前天气：{weather_info['weather']}，气温{weather_info['temp']}°C"
            f"（体感{weather_info['feels_like']}°C）\n"
            f"- 今日温度范围：{weather_info['min_temp']}°C ~ {weather_info['max_temp']}°C\n"
            f"- 湿度：{weather_info['humidity']}% | 风速：{weather_info['wind_speed']}km/h {weather_info['wind_dir']}\n"
        )
    else:
        weather_text = "- 当前天气：暂无天气数据\n"
    
    multi_step_text = ""
    if multi_preds and not is_unsupported:
        steps = []
        for fh in sorted(multi_preds.keys()):
            pred = multi_preds[fh]
            step_aqi_info = get_aqi_info(pred['aqi'])
            step_time_str = pred['time'].strftime('%H:%M')
            steps.append(f"  - 未来{fh}小时（{step_time_str}）：AQI {pred['aqi']:.0f}（{step_aqi_info['level']}）")
        multi_step_text = "- 分步预测详情：\n" + "\n".join(steps) + "\n"
    
    if is_unsupported:
        system_prompt = f"""你是一个贴心的空气质量助手，名叫"清新小助手"。

当前情况：
- 用户询问了关于{location}的空气质量
- 但{location}不在我们系统的预测覆盖范围内
- 我们目前仅支持滇东地区：曲靖、昭通、文山州
{weather_text}
回复要求：
- 友好地告知用户该地区暂不在预测覆盖范围内
- 说明系统目前支持滇东地区（曲靖、昭通、文山州）的空气质量预测
- 如果有天气数据，可以分享当前天气信息作为参考
- 给出一些通用的空气质量关注建议
- 引导用户尝试查询支持的地区
- 语气温暖友好，不要让用户感到失望

用户问题：{user_query}

请用温暖、自然的语言回答："""
    else:
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
        
        system_prompt = f"""你是一个贴心的空气质量助手，名叫"清新小助手"。你的任务是根据AQI预测数据和天气信息，用温暖、有感情的语言回答用户的问题。

当前信息：
- 地点：{location}
{site_info}- 预测说明：{forecast_desc}
- 预测AQI：{predicted_aqi:.0f}
- 空气质量等级：{aqi_level} {emoji}
- 健康建议：{desc}
{weather_text}{multi_step_text}
{style_guide}

重要要求：
- 回复中必须明确提到预测的目标时间（{time_str}），让用户知道这是对未来哪个时间点的预测
- 必须提到这是未来{PRIMARY_FORECAST_HOURS}小时的预测结果
- 如果有分步预测数据，请简要提及空气质量的变化趋势（如逐渐好转或恶化）
- 请结合当前天气和气温情况给出综合建议（如穿衣、出行等）

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
        print(f"⚠️ API调用失败: {e}")
        return f"抱歉，API调用失败，请稍后再试。"


SUPPORTED_LOCATIONS = set(info['location'] for info in SITE_DETAILS.values())


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
            print(f"🎯 识别到地区：{location}")
            return 'location', location, location
    
    unsupported_keywords = ['昆明', '大理', '丽江', '玉溪', '楚雄', '红河', 
                            '西双版纳', '保山', '德宏', '怒江', '迪庆', '临沧',
                            '普洱', '曲靖', '昭通', '文山', '贵州', '四川',
                            '北京', '上海', '广州', '深圳', '成都', '重庆',
                            '贵阳', '南宁', '长沙', '武汉']
    
    for keyword in unsupported_keywords:
        if keyword in user_input:
            if keyword in SUPPORTED_LOCATIONS:
                print(f"🎯 识别到地区：{keyword}")
                return 'location', keyword, keyword
            print(f"⚠️ 识别到不支持的地区：{keyword}")
            return 'unsupported', keyword, keyword
    
    print("⚠️ 未识别到具体地点，默认查询曲靖地区")
    return 'location', '曲靖', '曲靖'


def chat_with_assistant(user_input):
    """与空气质量助手对话"""
    
    predict_type, target_code, target_name = parse_user_intent(user_input)
    
    if predict_type == 'unsupported':
        print(f"\n🔍 查询地区 {target_name}（不在覆盖范围内）...")
        
        weather_info = get_weather_info(target_name)
        
        if weather_info:
            print(f"  🌤️ {target_name}天气: {weather_info['weather']} {weather_info['temp']}°C")
        
        aqi_info = {
            'level': '未知',
            'emoji': '🌍',
            'desc': '暂无该地区空气质量预测数据',
            'time': datetime.now()
        }
        
        response = call_deepseek_api(
            user_input,
            aqi_info,
            target_name,
            0,
            weather_info=weather_info,
            is_unsupported=True
        )
        
        return {
            'response': response,
            'data': {
                'aqi': None,
                'level': '未知',
                'emoji': '🌍',
                'location': target_name,
                'site_name': None,
                'steps': [],
                'weather': weather_info
            }
        }
    
    if predict_type == 'site':
        print(f"\n🔍 正在查询站点 {target_name}（{target_code}）的空气质量...")
        
        prediction = predict_aqi(target_code)
        
        if prediction is None:
            return {
                'response': f"抱歉，暂时无法获取{target_name}站点的空气质量数据，请稍后再试～",
                'data': None
            }
        
        multi_preds = prediction['multi_preds']
        
        if PRIMARY_FORECAST_HOURS not in multi_preds:
            return {
                'response': f"抱歉，{target_name}站点的{PRIMARY_FORECAST_HOURS}h预测模型未加载，请稍后再试～",
                'data': None
            }
        
        primary_pred = multi_preds[PRIMARY_FORECAST_HOURS]
        predicted_aqi = primary_pred['aqi']
        aqi_info = get_aqi_info(predicted_aqi)
        aqi_info['time'] = primary_pred['time']
        
        location = SITE_DETAILS[target_code]['location']
        weather_info = get_weather_info(location)
        
        print(f"\n📊 分步预测结果:")
        for fh in sorted(multi_preds.keys()):
            pred = multi_preds[fh]
            step_info = get_aqi_info(pred['aqi'])
            print(f"  - 未来{fh}小时（{pred['time'].strftime('%H:%M')}）: AQI={pred['aqi']:.0f}（{step_info['level']}）")
        print(f"  ⏰ 预测目标时间跨度: {', '.join([f'{fh}h' for fh in sorted(multi_preds.keys())])}")
        if weather_info:
            print(f"  🌤️ 当前天气: {weather_info['weather']} {weather_info['temp']}°C")
        
        steps = []
        for fh in sorted(multi_preds.keys()):
            pred = multi_preds[fh]
            step_info = get_aqi_info(pred['aqi'])
            steps.append({
                'hours': fh,
                'time': pred['time'].strftime('%H:%M'),
                'aqi': pred['aqi'],
                'level': step_info['level'],
                'emoji': step_info['emoji']
            })
        
        response = call_deepseek_api(user_input, aqi_info, location, 
                                    predicted_aqi, site_name=target_name,
                                    weather_info=weather_info,
                                    multi_preds=multi_preds)
        
        return {
            'response': response,
            'data': {
                'aqi': predicted_aqi,
                'level': aqi_info['level'],
                'emoji': aqi_info['emoji'],
                'location': location,
                'site_name': target_name,
                'steps': steps,
                'weather': weather_info
            }
        }
    
    else:
        print(f"\n🔍 正在查询 {target_name} 地区的空气质量...")
        
        prediction = predict_location_aqi(target_name)
        
        if prediction is None:
            return {
                'response': f"抱歉，暂时无法获取{target_name}地区的空气质量数据，请稍后再试～",
                'data': None
            }
        
        weather_info = get_weather_info(target_name)
        
        print(f"\n📊 综合分步预测结果（{prediction['site_count']}个站点）:")
        for site_pred in prediction['sites']:
            print(f"  📍 {site_pred['site_name']}:")
            for fh in sorted(site_pred['multi_preds'].keys()):
                pred = site_pred['multi_preds'][fh]
                step_info = get_aqi_info(pred['aqi'])
                print(f"    - 未来{fh}h（{pred['time'].strftime('%H:%M')}）: AQI={pred['aqi']:.0f}（{step_info['level']}）")
        if weather_info:
            print(f"  🌤️ 当前天气: {weather_info['weather']} {weather_info['temp']}°C（{weather_info['min_temp']}~{weather_info['max_temp']}°C）")
        
        first_site_preds = prediction['sites'][0]['multi_preds']
        if PRIMARY_FORECAST_HOURS in first_site_preds:
            avg_aqi = np.mean([s['multi_preds'][PRIMARY_FORECAST_HOURS]['aqi'] 
                              for s in prediction['sites'] if PRIMARY_FORECAST_HOURS in s['multi_preds']])
            aqi_info = get_aqi_info(avg_aqi)
            aqi_info['time'] = first_site_preds[PRIMARY_FORECAST_HOURS]['time']
        else:
            fh_available = sorted(first_site_preds.keys())[0]
            avg_aqi = np.mean([s['multi_preds'][fh_available]['aqi'] 
                              for s in prediction['sites'] if fh_available in s['multi_preds']])
            aqi_info = get_aqi_info(avg_aqi)
            aqi_info['time'] = first_site_preds[fh_available]['time']
        
        avg_steps = []
        for fh in FORECAST_HOURS_LIST:
            if any(fh in s['multi_preds'] for s in prediction['sites']):
                avg_step_aqi = round(np.mean([s['multi_preds'][fh]['aqi'] 
                                    for s in prediction['sites'] if fh in s['multi_preds']]), 1)
                step_info = get_aqi_info(avg_step_aqi)
                avg_steps.append({
                    'hours': fh,
                    'time': first_site_preds[fh]['time'].strftime('%H:%M') if fh in first_site_preds else '',
                    'aqi': avg_step_aqi,
                    'level': step_info['level'],
                    'emoji': step_info['emoji']
                })
        
        site_details = []
        for site_pred in prediction['sites']:
            site_steps = []
            for fh in sorted(site_pred['multi_preds'].keys()):
                pred = site_pred['multi_preds'][fh]
                si = get_aqi_info(pred['aqi'])
                site_steps.append({'hours': fh, 'aqi': pred['aqi'], 'level': si['level']})
            site_details.append({
                'name': site_pred['site_name'],
                'steps': site_steps
            })
        
        response = call_deepseek_api(
            user_input, 
            aqi_info, 
            target_name, 
            avg_aqi,
            site_name=f"综合{prediction['site_count']}个站点",
            weather_info=weather_info,
            multi_preds={fh: {
                'aqi': np.mean([s['multi_preds'][fh]['aqi'] for s in prediction['sites'] if fh in s['multi_preds']]),
                'time': first_site_preds.get(fh, {}).get('time', None)
            } for fh in FORECAST_HOURS_LIST if any(fh in s['multi_preds'] for s in prediction['sites'])}
        )
        
        return {
            'response': response,
            'data': {
                'aqi': round(float(avg_aqi), 1),
                'level': aqi_info['level'],
                'emoji': aqi_info['emoji'],
                'location': target_name,
                'site_name': f"综合{prediction['site_count']}个站点",
                'steps': avg_steps,
                'sites': site_details,
                'weather': weather_info
            }
        }


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
            
            result = chat_with_assistant(user_input)
            print(f"\n🤖 清新小助手：{result['response']}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！祝您呼吸愉快～🌿")
            break
        except Exception as e:
            print(f"\n❌ 出错了，请重试\n")
