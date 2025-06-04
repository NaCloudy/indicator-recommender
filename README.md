# indicator-recommender

## 项目简介

这是一个电商指标推荐系统项目，基于用户输入自动推荐相关指标，方便数据分析和决策支持。

## 功能

- 读取指标数据文件（CSV 格式）
- 使用预训练模型计算文本嵌入
- 根据输入推荐相关指标
- 支持简单的交互式命令行操作

## 配置

使用`Python 3.9.7`.

1. 克隆项目代码：

```bash
git clone https://github.com/xxx/indicator-recommender.git
cd indicator-recommender
```

2. 创建并激活 Python 虚拟环境

```python
python -m venv venv
# Windows
.\venv\Scripts\activate
```

3. 安装依赖

```py
pip install -r requirements.txt
```

4. 运行

```python
$env:HTTP_PROXY = "http://xx.xx.xx.xx:xxxx" # 配置代理，确保hugging face模型可下载
$env:HTTPS_PROXY = "http://xx.xx.xx.xx:xxxx"
python app\main.py
```

## 示例演示

```python
=== E-commerce Indicator Recommender ===
Type 'exit' to quit at any time.

Enter your indicator demand (e.g., 'daily active users'):
```

> 输入：`number of users that active each day`

```python
Recommended Indicators:
1. Daily Active Users
   Definition: Number of users who logged in and used the platform on a given day
   Dimensions: date,region
   Similarity Score: 0.844

2. Monthly Conversion Rate
   Definition: Ratio of users who placed an order to those who visited the site in a month
   Dimensions: date,product
   Similarity Score: 0.371

3. Customer Retention Rate
   Definition: Percentage of users who made repeat purchases over a period
   Dimensions: date,customer_segment
   Similarity Score: 0.333

4. Bounce Rate
   Definition: Percentage of users who left the site after viewing only one page
   Dimensions: date,traffic_source
   Similarity Score: 0.270

5. Average Order Value
   Definition: Average value of orders placed by users in a given period
   Dimensions: date,product
   Similarity Score: 0.259

Enter your indicator demand (e.g., 'daily active users'): exit
Exiting. Goodbye!
```

## 技术栈说明

- 文本嵌入：sentence-transformers/all-MiniLM-L6-v2
- 相似度计算：dot product（正则化 embeddings 后，余弦相似度可以直接 dot prod）

## TODO 列表

- 增加下游大模型，根据 top3 结果完成进一步筛选推荐（计划使用 gemini）
- 增加命中结果记录，便于后续离线重训练（对于 bert 模型）
- 部署到网页端（便于用户使用）
- 将指标数据部署到数据库中（目前以 csv 形式存储）
- 预计算指标 embedding 并仅调用 embedding 文件，避免重复计算（？）
- 增加数据量（目前 indicators.csv 只有 10 条数据，计划增加到 1k+）
