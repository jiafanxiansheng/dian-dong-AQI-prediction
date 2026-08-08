"""
简历 PDF 生成器 — 使用 fpdf2 生成三个风格版本
"""
from fpdf import FPDF
import os

OUTPUT_DIR = r"c:\Users\28927\Desktop\2023145109戴浩天"


class ResumePDF(FPDF):
    """简历 PDF 基类，含通用的排版方法"""

    def __init__(self, style="general"):
        super().__init__("P", "mm", "A4")
        self.set_auto_page_break(True, 15)
        # 注册中文字体
        font_dir = r"C:\Windows\Fonts"
        self.add_font("YaHei", "", os.path.join(font_dir, "msyh.ttc"))
        self.add_font("YaHei", "B", os.path.join(font_dir, "msyhbd.ttc"))
        self.add_font("SimSun", "", os.path.join(font_dir, "simsun.ttc"))
        self.style = style
        self._setup_colors()

    def _setup_colors(self):
        if self.style == "tech":
            self.c_primary = (137, 180, 250)
            self.c_heading = (166, 227, 161)
            self.c_accent = (201, 166, 247)
            self.c_bg_dark = (30, 30, 46)
            self.c_text = (205, 214, 244)
            self.c_light = (49, 50, 68)
        elif self.style == "minimal":
            self.c_primary = (30, 30, 30)
            self.c_heading = (30, 30, 30)
            self.c_accent = (80, 80, 80)
            self.c_bg_dark = (255, 255, 255)
            self.c_text = (40, 40, 40)
            self.c_light = (240, 240, 240)
        else:  # general
            self.c_primary = (26, 58, 92)
            self.c_heading = (41, 128, 185)
            self.c_accent = (52, 73, 94)
            self.c_bg_dark = (255, 255, 255)
            self.c_text = (50, 50, 50)
            self.c_light = (234, 242, 248)

    def section_title(self, text):
        self.ln(5)
        self.set_font("YaHei", "B", 13)
        if self.style == "tech":
            self.set_text_color(*self.c_heading)
            self.cell(0, 7, "  " + text, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(*self.c_heading)
            self.line(self.l_margin, self.get_y(), self.l_margin + 190, self.get_y())
        elif self.style == "minimal":
            self.set_text_color(*self.c_primary)
            self.cell(0, 6, text.upper(), new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(*self.c_primary)
            self.line(self.l_margin, self.get_y(), self.l_margin + 190, self.get_y())
        else:
            self.set_text_color(*self.c_heading)
            self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(*self.c_heading)
            self.line(self.l_margin, self.get_y(), self.l_margin + 190, self.get_y())
        self.ln(3)

    def body_text(self, text, size=10, bold=False, color=None):
        c = color or self.c_text
        self.set_text_color(*c)
        style = "B" if bold else ""
        self.set_font("YaHei", style, size)
        self.set_x(self.l_margin)
        self.multi_cell(self.w - self.l_margin - self.r_margin, 5.5, text, align="L")
        if not bold:
            self.ln(1)

    def bullet(self, text, indent=5):
        self.set_text_color(*self.c_text)
        self.set_font("YaHei", "", 10)
        self.set_x(self.l_margin + indent)
        bullet_char = ">" if self.style == "tech" else "-"
        self.cell(6, 5.5, bullet_char)
        self.multi_cell(self.w - self.l_margin - self.r_margin - indent - 6, 5.5, text, align="L")

    def tag(self, text):
        if self.style == "tech":
            self.set_fill_color(*self.c_light)
            self.set_text_color(*self.c_primary)
        else:
            self.set_fill_color(*self.c_light)
            self.set_text_color(*self.c_accent)
        self.set_font("YaHei", "", 8)
        w = self.get_string_width(text) + 5
        self.cell(w, 5, text, fill=True)
        self.cell(2, 5, "")


# ═══════════════════════════════════════════════════════════
#  版本1：通用版
# ═══════════════════════════════════════════════════════════

def build_general(pdf):
    pdf.add_page()

    # 头部
    pdf.set_fill_color(*pdf.c_light)
    pdf.rect(0, 0, 210, 32, "F")
    pdf.set_y(6)
    pdf.set_font("YaHei", "B", 26)
    pdf.set_text_color(*pdf.c_primary)
    pdf.cell(0, 10, "个人简历", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("YaHei", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "2892775022@qq.com  |  18058854039  |  曲靖  |  数据处理 / Python后端 / 数据库", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 教育背景
    pdf.section_title("教育背景")
    pdf.body_text("曲靖师范学院  |  数据科学与大数据技术  |  本科（2023.09 - 2027.06）", bold=True)
    pdf.body_text("核心课程：数据库原理及应用、Python程序设计、机器学习、应用统计学与建模、时间序列分析、"
                  "数据采集技术、云计算与大数据、Linux应用实践、计算机网络、操作系统", size=9)

    # 项目经历
    pdf.section_title("项目经历")
    pdf.body_text("滇东（曲靖/昭通/文山）多站点AQI智能预测系统  |  全栈独立开发", bold=True, color=pdf.c_primary)
    pdf.set_font("YaHei", "", 8)
    tags = ["Python", "MySQL", "Flask", "Prophet", "Pandas", "Scikit-learn", "ECharts", "DeepSeek API"]
    for t in tags:
        pdf.tag(t)
    pdf.ln(4)
    pdf.body_text("GitHub: github.com/jiafanxiansheng/dian-dong-AQI-prediction", size=8, color=(100, 100, 150))
    pdf.ln(2)
    pdf.body_text("独立完成的空气质量时序预测全栈系统，覆盖数据采集、特征工程、模型训练、Web可视化全流程。"
                  "支持自然语言对话查询与 6h/12h 双步长 AQI 预测。", size=10)

    bullets_general = [
        "设计随机森林、Prophet 等多模型公平对照实验，基于 5 折时序交叉验证量化精度，Prophet 综合 R² 达 0.907，确定为主预测框架",
        "构建 66 维时序特征体系（6污染物×6阶滞后 + 3窗口滚动统计 + 4阶差分 + 周期编码），大幅提升模型区分度",
        "为 7 个国控监测站分别训练 6h/12h 独立 Prophet 模型，实现分步预测输出",
        "对接 air.cnemc.cn 官方数据 API（0.1s 响应），设计 aqicn.org 多源降级容错机制，替代传统 Selenium 爬虫",
        "设计 MySQL 数据表结构，规范化存储历史与实时数据，支撑后续特征工程与模型训练",
        "搭建 Web 交互页面，集成 ECharts 趋势可视化、实时天气和 DeepSeek 大模型智能问答",
    ]
    for b in bullets_general:
        pdf.bullet(b)

    # 个人简介
    pdf.section_title("个人简介")
    pdf.body_text(
        "准大四数据科学与大数据技术本科生，持有软考中级数据库系统工程师证书。两年蓝桥杯算法竞赛实战，"
        "斩获 Python 国赛三等奖、C 语言省二等奖。独立完成空气质量时序预测全栈系统，熟练掌握 Python、MySQL、"
        "Flask、时序建模与数据治理，具备数据库设计优化、接口开发、多源数据采集分析全链路落地能力。", size=10
    )

    # 荣誉
    pdf.section_title("荣誉奖项")
    pdf.bullet("软考中级 · 数据库系统工程师 — 国家人力资源和社会保障部")
    pdf.bullet("蓝桥杯 Python B 组 · 全国三等奖 — 工业和信息化部人才交流中心")
    pdf.bullet("蓝桥杯 C 语言 B 组 · 省级二等奖 — 工业和信息化部人才交流中心")

    # 技能
    pdf.section_title("专业技能")
    skills_general = [
        "编程语言：熟练掌握 Python，代码规范，具备独立脚本开发、算法实现、数据处理能力",
        "数据库：精通 MySQL，掌握表结构设计、复杂 SQL 编写、查询优化、备份恢复等实操技能",
        "Web 开发：熟悉 Flask 框架，具备 RESTful API 设计、前后端联调、应用部署经验",
        "数据工程：掌握多源数据采集、清洗、标准化、时序分析全流程；了解 Hive 数仓建模与 FineBI",
        "AI 应用：熟悉大模型 API 集成、Prompt 工程，对智能问答、文本处理等场景有实践经验",
        "工具链：熟练 Linux 基础命令、Git 版本控制，具备数据报表制作与运维实操能力",
    ]
    for s in skills_general:
        pdf.bullet(s)


# ═══════════════════════════════════════════════════════════
#  版本2：技术版（深色主题）
# ═══════════════════════════════════════════════════════════

def build_tech(pdf):
    pdf.add_page()

    # 深色背景头部
    pdf.set_fill_color(*pdf.c_bg_dark)
    pdf.rect(0, 0, 210, 297, "F")

    # 头部
    pdf.set_y(12)
    pdf.set_font("YaHei", "B", 24)
    pdf.set_text_color(*pdf.c_primary)
    pdf.cell(0, 9, "戴 浩 天", align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("YaHei", "", 9)
    pdf.set_text_color(166, 173, 200)
    pdf.cell(0, 6, "2892775022@qq.com  |  18058854039  |  曲靖  |  Python / 数据工程 / 后端", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*pdf.c_light)
    pdf.line(pdf.get_x(), pdf.get_y() + 2, pdf.get_x() + 190, pdf.get_y() + 2)
    pdf.ln(6)

    # 技术能力矩阵
    pdf.section_title("技术能力矩阵")
    rows = [
        ("语言", "Python（主力）· C（竞赛）· SQL（日常）"),
        ("数据", "Pandas · NumPy · Scikit-learn · 特征工程 · 时序分析"),
        ("模型", "Prophet（时间序列）· 交叉验证 · 超参调优"),
        ("后端", "Flask · RESTful API · MySQL（表设计 / 优化）"),
        ("前端", "HTML/CSS/JS · ECharts 可视化"),
        ("工具链", "Git · Linux · DeepSeek API · wttr.in API"),
    ]
    for label, content in rows:
        pdf.set_text_color(*pdf.c_heading)
        pdf.set_font("YaHei", "B", 9)
        pdf.cell(22, 5.5, label)
        pdf.set_text_color(*pdf.c_text)
        pdf.set_font("YaHei", "", 9)
        pdf.cell(0, 5.5, content, new_x="LMARGIN", new_y="NEXT")

    # 项目
    pdf.section_title("核心项目")
    pdf.set_text_color(*pdf.c_accent)
    pdf.set_font("YaHei", "B", 12)
    pdf.cell(0, 6, "滇东 AQI 智能预测系统  |  全栈独立开发", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_text_color(*pdf.c_text)
    pdf.set_font("YaHei", "B", 10)
    pdf.cell(0, 5.5, "架构设计", new_x="LMARGIN", new_y="NEXT")
    for b in [
        "分层架构：Flask Web 层 / Service 业务层 / MySQL 数据层，模块解耦",
        "Wapper 模式封装 Prophet，统一 predict() 接口，新模型零侵入扩展",
        "RESTful API 设计：6 个端点（chat / quick-query / sites / status / predict）",
    ]:
        pdf.bullet(b)

    pdf.set_font("YaHei", "B", 10)
    pdf.cell(0, 5.5, "数据管道", new_x="LMARGIN", new_y="NEXT")
    for b in [
        "多源降级：air.cnemc.cn API（0.1s 主源）→ aqicn.org（备用）→ 本地 DB 缓存",
        "2h 新鲜度自动检测 + 增量去重写入，保证预测输入始终最新",
    ]:
        pdf.bullet(b)

    pdf.set_font("YaHei", "B", 10)
    pdf.cell(0, 5.5, "模型对比 & 训练", new_x="LMARGIN", new_y="NEXT")
    for b in [
        "多模型对照实验：RandomForest / Prophet 公平对比，5折时序交叉验证",
        "Prophet R² = 0.907 胜出，确定为主预测框架",
        "66 维特征：6污染物×6滞后 + 3窗口滚动统计 + 4阶差分 + 4时间周期特征",
        "7站×2时次（6h/12h）= 14个独立模型，输出约束至 [0, 500]",
    ]:
        pdf.bullet(b)

    pdf.set_font("YaHei", "B", 10)
    pdf.cell(0, 5.5, "前端交互", new_x="LMARGIN", new_y="NEXT")
    for b in [
        "ECharts 5.5 趋势图（含 AQI 等级参考线）+ DeepSeek LLM 自然语言对话",
    ]:
        pdf.bullet(b)

    # 教育
    pdf.section_title("教育背景")
    pdf.set_text_color(*pdf.c_text)
    pdf.body_text("曲靖师范学院 · 数据科学与大数据技术 · 本科 · 2023.09 - 2027.06", bold=True)
    pdf.body_text("数据库原理、Python程序设计、机器学习、时间序列分析、数据采集技术、云计算与大数据", size=9)

    # 荣誉
    pdf.section_title("荣誉 & 证书")
    for b in [
        "软考中级 · 数据库系统工程师 — 人社部认证",
        "蓝桥杯 Python B 组 · 全国三等奖",
        "蓝桥杯 C 语言 B 组 · 省级二等奖",
    ]:
        pdf.bullet(b)


# ═══════════════════════════════════════════════════════════
#  版本3：简洁版（单页极简）
# ═══════════════════════════════════════════════════════════

def build_minimal(pdf):
    pdf.add_page()

    # 头部 — 居中对齐
    pdf.set_font("YaHei", "B", 22)
    pdf.set_text_color(*pdf.c_primary)
    pdf.cell(0, 9, "戴 浩 天", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("YaHei", "", 9)
    pdf.set_text_color(*pdf.c_accent)
    pdf.cell(0, 5, "2892775022@qq.com  |  18058854039  |  曲靖  |  数据处理 / Python后端 / 数据库", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_draw_color(*pdf.c_primary)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
    pdf.ln(6)

    # 教育
    pdf.section_title("教育背景")
    pdf.body_text("曲靖师范学院  ·  数据科学与大数据技术  ·  本科  ·  2023.09 – 2027.06", bold=True, size=10)

    # 项目
    pdf.section_title("项目经历")
    pdf.body_text("滇东多站点 AQI 智能预测系统（全栈独立开发）", bold=True, size=10)
    pdf.body_text("GitHub: github.com/jiafanxiansheng/dian-dong-AQI-prediction", size=7, color=(100, 100, 150))
    for b in [
        "基于 Flask + Prophet + MySQL，覆盖 7 个国控监测站，支持 6h/12h 未来 AQI 预测",
        "多模型对比实验（RandomForest / Prophet），5折交叉验证，Prophet R² = 0.907",
        "66 维时序特征工程 + HTTP API 多源降级数据管道（air.cnemc.cn → aqicn.org）",
        "集成 ECharts 可视化 + DeepSeek 大模型智能问答，前端全自主搭建",
    ]:
        pdf.bullet(b, indent=3)

    # 荣誉
    pdf.section_title("荣誉 & 证书")
    pdf.set_font("YaHei", "", 10)
    awards = [
        ("软考中级 · 数据库系统工程师", "2025"),
        ("蓝桥杯 Python B 组 · 全国三等奖", "2025"),
        ("蓝桥杯 C 语言 B 组 · 省级二等奖", "2024"),
    ]
    for name, year in awards:
        pdf.set_text_color(*pdf.c_text)
        pdf.cell(8, 5.5, "")
        pdf.cell(100, 5.5, name)
        pdf.set_text_color(*pdf.c_accent)
        pdf.cell(0, 5.5, year, new_x="LMARGIN", new_y="NEXT")

    # 技能
    pdf.section_title("专业技能")
    pdf.set_text_color(*pdf.c_text)
    pdf.set_font("YaHei", "", 9)
    pdf.set_x(pdf.l_margin + 5)
    pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 5, 5.5,
        "语言：Python（熟练）、C（竞赛）、SQL（日常）  |  "
        "数据库：MySQL 表设计、查询优化、数据治理  |  "
        "框架：Flask、RESTful API  |  "
        "数据：Pandas、Scikit-learn、时序分析、特征工程  |  "
        "工具：Git、Linux、ECharts、DeepSeek API"
    )


# ═══════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("生成简历 PDF...\n")

    configs = [
        ("general", build_general, "个人简历_通用版.pdf"),
        ("tech", build_tech, "个人简历_技术版.pdf"),
        ("minimal", build_minimal, "个人简历_简洁版.pdf"),
    ]

    for style, builder, filename in configs:
        pdf = ResumePDF(style=style)
        builder(pdf)
        path = os.path.join(OUTPUT_DIR, filename)
        pdf.output(path)
        print(f"  [OK] {filename}")

    print(f"\n全部完成！输出目录：{OUTPUT_DIR}")
