### 文件层次梳理

#### 训练数据
fact_checks.csv - 包含fact-check数据，多语言
posts.csv - 包含post数据，多语言
pairs.csv - 包含train pair数据

#### 预测任务
tasks.json - JSON 文件，提供了fact-check ID，train post ID和dev post ID，多语言

#### 提交文件
monolingual_predictions.json - 包含了dev post ID列和空提交列，单语言
crosslingual_predictions.json - 包含了dev post ID列和空提交列，跨语言
