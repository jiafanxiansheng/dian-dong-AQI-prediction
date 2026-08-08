# 🌿 滇东 AQI 智能预测系统

基于 **Prophet 时间序列模型**的空气质量预测系统，覆盖滇东地区（曲靖、昭通、文山州）7 个国控监测站，集成 DeepSeek 大语言模型提供自然语言对话查询与健康建议。

> 🎓 课程设计项目 | 曲靖师范学院 / 数据科学与大数据技术

## 🧱 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户交互层 (Frontend)                  │
│          HTML5 + CSS3 + JavaScript + ECharts 5.5         │
│         自然语言输入 · 趋势图表 · 多站点对比卡片           │
└──────────────────────┬──────────────────────────────────┘
                       │  HTTP POST
┌──────────────────────▼──────────────────────────────────┐
│                   Web 服务层 (Flask)                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │ /api/chat│ │/api/quick│ │/api/sites│ │/api/status │ │
│  │ 自然对话  │ │ -query  │ │ 站点列表  │ │ 系统状态    │ │
│  └────┬─────┘ └────┬─────┘ └──────────┘ └────────────┘ │
└───────┼────────────┼────────────────────────────────────┘
        │            │
┌───────▼────────────▼────────────────────────────────────┐
│                    业务逻辑层 (Services)                   │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ llm_client   │  │ intent_parser│  │   weather     │  │
│  │ DeepSeek API │  │ 意图识别      │  │  wttr.in 天气 │  │
│  │ 对话生成      │  │ 实体提取      │  │  实时查询      │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │               predictor.py (预测引擎)              │   │
│  │         Prophet 时间序列 · 特征工程                │   │
│  │   滞后特征 + 滚动统计 → 多步预测(6h/12h) → 约束     │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │            data_fetcher.py (数据管道)              │   │
│  │  air.cnemc.cn API (主) ──→ aqicn.org (备)        │   │
│  │  0.1s 响应 · 24条/站 · 自动去重 · 新鲜度检测       │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│                    数据存储层                              │
│          MySQL 8.0 · 7 站点独立表 · utf8mb4              │
│          air_quality_site_{1916A...3377A}                │
└──────────────────────────────────────────────────────────┘
```

### 预测流水线

```
历史数据 → 66维特征工程（滞后 + 滚动统计 + 差分 + 周期编码）
          → 多模型对比（RandomForest R²=0.45 vs Prophet R²=0.907）
          → 5折时序交叉验证 → 选定 Prophet
          → 6h / 12h 多步预测 → AQI 约束[0, 500] → 输出
```

## 🗺️ 监测站点

| 编码 | 名称 | 城市 |
|------|------|------|
| 1916A | 环境监测站 | 曲靖 |
| 1917A | 烟厂办公区 | 曲靖 |
| 3376A | 南苑二区 | 曲靖 |
| 3377A | 曲靖师范学院 | 曲靖 |
| 2596A | 监测站 | 昭通 |
| 2610A | 州水务局 | 文山州 |
| 2611A | 市便民服务中心 | 文山州 |

## 🚀 快速开始

### 环境要求

- Python 3.11+
- MySQL 8.0+
- 需创建数据库：`CREATE DATABASE air_data CHARACTER SET utf8mb4;`

### 1. 安装

```bash
git clone https://github.com/jiafanxiansheng/dian-dong-AQI-prediction.git
cd dian-dong-AQI-prediction
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### 2. 配置

创建 `.env` 文件（参考 `.env.example`）：

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=air_data

# 从 https://platform.deepseek.com 获取
DEEPSEEK_API_KEY=sk-your-key-here
```

### 3. 获取历史数据

历史空气质量数据可从以下源获取：

> **https://quotsoft.net/air/#archive** — 提供 2014 年至今的全国站点逐时 CSV 数据

下载后放入 `data/` 目录（按年份子目录组织），然后导入数据库：

```bash
python scripts/import_data.py
```

### 4. 训练模型

```bash
python scripts/train_prophet.py --all
```

模型文件将保存至 `模型/` 目录，每个站点生成 6h 和 12h 两个预测模型。

```bash
# （可选）运行多模型对比实验
python scripts/compare_models.py
```

### 5. 启动

```bash
python app.py
# 访问 http://localhost:5000
```

## 📡 API 参考

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 聊天界面 |
| `POST` | `/api/chat` | 自然语言对话（`{"message":"曲靖明天空气好吗？"}`） |
| `POST` | `/api/quick-query` | 快速站点查询（`{"site_name":"曲靖师范学院"}`） |
| `GET` | `/api/sites` | 所有站点信息 |
| `GET` | `/api/aqi-levels` | AQI 等级定义（6 级） |
| `GET` | `/api/status` | 系统状态（数据时效、上线时间） |

## 🔄 数据刷新机制

```
用户查询 → 检查 DB 最新数据时间
              ├─ < 2 小时 → 直接预测
              └─ > 2 小时 → 触发爬虫
                              ├─ air.cnemc.cn API (POST, 0.1s)
                              │   返回最近24h逐时数据
                              ├─ aqicn.org API (降级, 0.5s)
                              │   返回当前单点数据
                              └─ 失败 → 使用旧数据 + 警告
              → 写入 DB → 预测
```

```bash
# 手动批量刷新
python scripts/refresh_data.py
```

## 📂 项目结构

```
├── app.py                    # Flask 入口：路由注册 + 日志配置
├── config.py                 # 配置中心：DB/API/站点/模型路径
├── requirements.txt          # Python 依赖清单
├── run.bat                   # Windows 一键启动
│
├── services/                 # 核心服务（业务逻辑）
│   ├── predictor.py          # 预测引擎：模型加载 · 特征工程 · 多步预测
│   ├── data_fetcher.py       # 数据管道：多源降级 · 去重 · 新鲜度检测
│   ├── llm_client.py         # LLM 对话：DeepSeek API · 上下文组装
│   ├── intent_parser.py      # 意图识别：站点匹配 · 地区映射
│   └── weather.py            # 天气服务：wttr.in API · 中文翻译
│
├── scripts/                  # 离线工具
│   ├── train_prophet.py      # Prophet 模型训练
│   ├── import_data.py        # CSV → MySQL 批量导入
│   ├── compare_models.py     # 多模型对比评估
│   └── refresh_data.py       # 一键刷新实时数据
│
├── templates/
│   └── index.html            # SPA 前端（ECharts + 聊天 UI）
│
├── 模型/                     # 模型文件 (*.pkl)，训练后生成
├── data/                     # 原始 CSV 历史数据，不入库
└── .env                      # 环境变量（敏感信息，不入库）
```

## 🛠 技术选型理由

| 决策 | 选择 | 理由 |
|------|------|------|
| 预测算法 | Prophet | 经 RandomForest / Prophet 多模型 5 折时序交叉验证对比，Prophet R²=0.907 表现最优 |
| LLM 集成 | DeepSeek Chat | 中文能力强，API 兼容 OpenAI 格式，调用成本低 |
| 数据获取 | HTTP API (非 Selenium) | 0.1s vs 25s，无浏览器依赖，稳定可靠 |
| 前端图表 | ECharts 5.5 | 成熟的时间序列可视化库，交互丰富 |
| 特征工程 | 66维时序特征 | 6污染物×6阶滞后 + 3窗口滚动统计 + 4阶差分 + 周期编码，捕捉时间依赖 |

### 模型对比结果

在站点 **1916A**（曲靖环境监测站，93,410 条数据，6h 预测任务）上，基于统一 66 维特征集进行 5 折时序交叉验证，对比四类模型：

| 排名 | 模型 | R² 均值 | R² 标准差 | RMSE | 类型 | 结论 |
|------|------|---------|-----------|------|------|------|
| 1 | **Prophet** | **0.907** | — | — | 时间序列分解 | ✅ 选为主力 |
| 2 | CatBoost | 0.456 | ±0.095 | 12.87 | 梯度提升·Boosting | 略优于 RF，仍远逊于 Prophet |
| 3 | RandomForest | 0.450 | ±0.074 | 12.96 | 集成学习·Bagging | 非线性能力弱于时序模型 |
| 4 | LSTM | — | — | — | 深度学习·循环网络 | 训练成本高，小数据量易过拟合 |

> Prophet 最终胜出的关键原因：对缺失值鲁棒、自动捕捉季节性和节假日效应、训练高效。CatBoost 和 RandomForest 表现接近（R²≈0.45），说明纯树模型难以捕捉逐时 AQI 的时间依赖结构。LSTM 因训练成本与收益不成正比而未被采用。

## ⚠️ 注意事项

- `.env` 包含 API Key，已在 `.gitignore` 中排除，请勿提交
- `模型/` 和 `data/` 目录未入库（体积过大），需按文档步骤重新下载/训练
- 实时数据接口有频率限制，批量刷新时内置 0.5s 间隔
- 应用启动时自动加载全部模型到内存，首次启动需 30~60s

## 📄 License

MIT License — 仅供学习交流使用。
