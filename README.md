# 🌿 滇东 AQI 智能预测系统 — "清新小助手"

基于 **Prophet + CatBoost 双模型引擎**的空气质量智能预测系统，覆盖滇东地区（曲靖、昭通、文山州）7 个国控监测站，支持自然语言对话查询，集成 DeepSeek 大模型提供健康建议。

> 🎓 课程设计项目 | 昆明理工大学 / 数据科学与大数据技术

## ✨ 功能特性

- **🔮 双模型预测** — Prophet 时间序列 + CatBoost 梯度提升，自动选择最优模型，支持 6h / 12h 未来 AQI 预测
- **💬 智能对话** — 集成 DeepSeek Chat API，支持自然语言查询（如"曲靖明天空气怎么样？"）
- **📊 可视化面板** — ECharts 5.5 趋势图表，AQI 等级参考线，多站点横向对比
- **🔄 实时数据** — 对接 air.cnemc.cn 官方 API，0.1 秒响应，aqicn.org 备用降级
- **⏰ 自动刷新** — 数据新鲜度检测（2 小时阈值），过期自动触发爬取
- **🌤️ 天气集成** — wttr.in 免费天气 API，温湿度风况一目了然
- **📱 响应式设计** — 适配桌面/平板/手机，8 个快捷查询按钮

## 🏗️ 技术栈

| 层级 | 技术 |
|------|------|
| **Web 框架** | Flask + Jinja2 |
| **预测模型** | Prophet（时间序列）、CatBoost（梯度提升） |
| **大语言模型** | DeepSeek Chat API（对话 + 意图解析） |
| **数据获取** | air.cnemc.cn 官方 API、aqicn.org 备用源 |
| **数据库** | MySQL 8.0（7 个站点独立表） |
| **前端** | HTML5 + CSS3 + JavaScript + ECharts 5.5 |
| **天气** | wttr.in 免费 API |
| **Python 库** | pandas, numpy, scikit-learn, joblib, SQLAlchemy |

## 📂 项目结构

```
pythonProject3/
├── app.py                      # Flask 主入口（路由、API）
├── config.py                   # 统一配置中心
├── run.bat                     # Windows 一键启动
├── requirements.txt            # Python 依赖
│
├── services/                   # 核心服务层
│   ├── predictor.py            # 双模型预测引擎
│   ├── data_fetcher.py         # 实时数据爬取（HTTP API）
│   ├── llm_client.py           # DeepSeek LLM 对话
│   ├── intent_parser.py        # 用户意图识别
│   └── weather.py              # 天气查询服务
│
├── scripts/                    # 离线脚本
│   ├── train_prophet.py        # Prophet 模型训练
│   ├── train_catboost.py       # CatBoost 模型训练
│   ├── import_data.py          # CSV → MySQL 数据导入
│   ├── compare_models.py       # 模型对比实验
│   └── refresh_data.py         # 一键刷新所有站点数据
│
├── templates/
│   └── index.html              # 前端聊天界面
│
├── 模型/                       # 训练好的模型文件（需运行训练脚本生成）
├── data/                       # 原始 CSV 数据（不入库）
│   └── 20140101-20251231/      # 按年份组织的历史数据
│
└── .env                        # 环境变量（不入库，需自行创建）
```

## 🚀 快速开始

### 环境要求

- Python 3.11+
- MySQL 8.0+（已创建 `air_data` 数据库）
- Windows / Linux / macOS

### 1. 克隆项目

```bash
git clone https://github.com/jiafanxiansheng/dian-dong-AQI-prediction.git
cd dian-dong-AQI-prediction
```

### 2. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 3. 配置环境变量

创建 `.env` 文件：

```env
# 数据库
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=air_data

# DeepSeek API（必填 — 从 https://platform.deepseek.com 获取）
DEEPSEEK_API_KEY=sk-your-key-here
```

### 4. 准备数据 & 训练模型

```bash
# 导入历史数据到 MySQL
python scripts/import_data.py

# 训练 Prophet 模型（所有站点）
python scripts/train_prophet.py --all

# （可选）训练 CatBoost 模型
python scripts/train_catboost.py --all
```

### 5. 启动应用

```bash
python app.py
# 或双击 run.bat（Windows）

# 访问 http://localhost:5000
```

## 📡 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 聊天界面 |
| `POST` | `/api/chat` | 自然语言对话查询 |
| `POST` | `/api/quick-query` | 快速站点查询 |
| `GET` | `/api/sites` | 获取所有站点信息 |
| `GET` | `/api/aqi-levels` | 获取 AQI 等级定义 |
| `GET` | `/api/status` | 系统状态（模型、数据时效） |

## 🗺️ 监测站点

| 编码 | 名称 | 城市 | 模型类型 |
|------|------|------|----------|
| 1916A | 环境监测站 | 曲靖 | Prophet |
| 1917A | 烟厂办公区 | 曲靖 | Prophet |
| 3376A | 南苑二区 | 曲靖 | Prophet |
| 3377A | 曲靖师范学院 | 曲靖 | Prophet |
| 2596A | 监测站 | 昭通 | Prophet |
| 2610A | 州水务局 | 文山州 | Prophet |
| 2611A | 市便民服务中心 | 文山州 | Prophet |

## 🔄 数据刷新

```bash
# 一键刷新所有站点（通过 air.cnemc.cn API）
python scripts/refresh_data.py

# 数据源优先级：
# 1. air.cnemc.cn 官方 API（0.1s 响应，24 条/站）
# 2. aqicn.org 公开 API（备用降级）
```

系统会自动检测数据新鲜度（默认 2 小时阈值），过期数据将在下次预测时自动刷新。

## 🖼️ 界面预览

- 🎨 渐变 AQI 等级徽章（绿→黄→橙→红→紫→褐红）
- 📈 ECharts 交互式趋势折线图（含 50/100/150/200 等级参考线）
- 🏷️ 趋势标签（"趋于好转" / "趋于恶化" + 百分比）
- 📋 污染物详情卡片（PM2.5 / PM10 / SO₂ / NO₂ / CO / O₃）
- 🌤️ 实时天气信息行
- 📱 响应式布局 + 8 个快捷查询按钮

## ⚠️ 注意事项

- **API Key**：`.env` 中的 DeepSeek Key 请妥善保管，勿提交到版本控制
- **模型文件**：`模型/` 目录未入库（文件过大），需运行训练脚本生成
- **数据目录**：`data/` 目录未入库（6.9GB），可从原始数据源重新下载
- **数据库**：确保 MySQL 已创建 `air_data` 数据库，字符集使用 `utf8mb4`

## 📄 License

MIT License — 仅供学习交流使用。
