# 工具层 (src/utils/)

设备管理、日志记录、性能指标、性能分析、数据库管理。

## 目录结构

```
src/utils/
├── device.py          # get_device / DeviceManager — 设备检测与优化
├── logging.py         # Logger / ProgressTracker / TensorBoardLogger
├── metrics.py         # compute_perplexity / MetricsTracker — 指标计算
├── profiling.py       # OptimizationProfiler — 性能分析
├── database.py        # DatabaseManager — SQLite 管理器
├── database_schema.py # 数据库表结构定义
└── repository.py      # Repository 模式（数据访问层）
```

## 核心组件

### device.py — 设备管理

| 函数/类 | 说明 |
|---------|------|
| `get_device()` | 获取最佳设备（GPU/CPU） |
| `get_device_info()` | 获取设备信息（GPU型号、显存等） |
| `to_device(data, device)` | 将数据移动到设备 |
| `set_seed(seed)` | 设置随机种子 |
| `get_memory_info()` | 获取显存/内存信息 |
| `optimize_for_inference()` | 推理优化 |
| `enable_tf32()` | 启用 TF32（ Ampere+ GPU） |
| `set_num_threads(n)` | 设置 CPU 线程数 |
| `get_optimal_num_threads()` | 获取最优 CPU 线程数 |
| `DeviceManager` | 设备管理器类 |
| `print_device_info()` | 打印设备信息 |

**使用示例：**

```python
from src.utils import get_device, set_seed, print_device_info

# 自动检测最佳设备
device = get_device()
print(f"Using device: {device}")

# 打印详细信息
print_device_info()

# 设置随机种子
set_seed(42)

# 获取内存信息
from src.utils import get_memory_info
memory = get_memory_info()
print(f"GPU Memory: {memory['gpu_total_gb']:.1f} GB")
```

### logging.py — 日志记录

| 类/函数 | 说明 |
|---------|------|
| `Logger` | 统一日志记录器 |
| `ProgressTracker` | 训练进度跟踪器（tqdm 封装） |
| `TensorBoardLogger` | TensorBoard 日志记录 |
| `setup_logger()` | 配置日志系统 |

**使用示例：**

```python
from src.utils import Logger, ProgressTracker, setup_logger

# 配置日志
setup_logger("./logs", level="INFO")

# 进度跟踪器
tracker = ProgressTracker(total=1000, desc="Training")
for i in range(1000):
    tracker.update()
    if i % 100 == 0:
        tracker.log(f"Step {i}, loss={loss:.4f}")

# TensorBoard 日志
tb_logger = TensorBoardLogger("./logs/tensorboard")
tb_logger.log_scalar("train/loss", loss, step)
```

### metrics.py — 指标计算

**损失函数：**

| 函数 | 说明 |
|------|------|
| `compute_perplexity(loss)` | 从 loss 计算困惑度 |
| `compute_accuracy(logits, labels)` | 计算准确率 |
| `compute_token_accuracy(logits, labels)` | Token 级准确率 |
| `compute_bleu_score(pred, ref)` | BLEU 分数 |
| `compute_f1_score(pred, label)` | F1 分数 |

**跟踪器：**

| 类 | 说明 |
|-----|------|
| `MetricsTracker` | 指标跟踪器（均值、标准差） |
| `PerplexityCalculator` | 困惑度计算器 |
| `compute_generation_metrics()` | 生成指标计算 |

**使用示例：**

```python
from src.utils import compute_perplexity, MetricsTracker

# 计算困惑度
ppl = compute_perplexity(loss=2.5)
print(f"Perplexity: {ppl:.2f}")

# 跟踪指标
tracker = MetricsTracker(window_size=100)
tracker.update("loss", 2.5)
tracker.update("loss", 2.3)
print(f"Average loss: {tracker.get('loss')}")
```

### profiling.py — 性能分析

| 类 | 说明 |
|-----|------|
| `OptimizationProfiler` | 性能分析器，追踪各阶段耗时 |
| `GQAMetrics` | GQA 相关指标（KV 缓存减少比例） |
| `StreamingMetrics` | StreamingLLM 指标 |
| `SpeculativeMetrics` | 推测解码指标 |
| `FlashAttentionMetrics` | Flash Attention 指标 |
| `ModelOptimizationConfig` | 优化配置类 |
| `format_memory_size()` | 内存大小格式化 |

**性能阶段：**

| 阶段 | 说明 |
|------|------|
| `data_loading` | 数据加载耗时 |
| `forward` | 前向传播耗时 |
| `backward` | 反向传播耗时 |
| `optimizer_step` | 优化器步进耗时 |

**使用示例：**

```python
from src.utils import OptimizationProfiler

profiler = OptimizationProfiler()
profiler.start("data_loading")
# 数据加载...
profiler.stop("data_loading")

profiler.start("forward")
# 前向传播...
profiler.stop("forward")

# 打印报告
profiler.print_report()
# Data Loading:  0.123s (12.3%)
# Forward:       0.456s (45.6%)
# Backward:     0.321s (32.1%)
# Optimizer:    0.100s (10.0%)
```

### database.py — 数据库管理

| 函数/类 | 说明 |
|---------|------|
| `DatabaseManager` | SQLite 数据库管理器 |
| `get_default_db_path()` | 获取默认数据库路径 |
| `init_database()` | 初始化数据库 |
| `get_connection()` | 获取数据库连接 |
| `backup_database()` | 备份数据库 |

**使用示例：**

```python
from src.utils import DatabaseManager, init_database

# 初始化
init_database("mydb.db")

# 使用管理器
db = DatabaseManager("mydb.db")
db.execute("CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, name TEXT)")
db.execute("INSERT INTO items VALUES (?, ?)", (1, "item1"))
db.commit()

# 查询
results = db.execute("SELECT * FROM items").fetchall()

# 备份
from src.utils import backup_database
backup_database("mydb.db", "mydb_backup.db")
```

### database_schema.py — 数据库表结构

| 函数 | 说明 |
|------|------|
| `init_all_tables()` | 初始化所有表结构 |
| `drop_all_tables()` | 删除所有表 |
| `get_all_schemas()` | 获取所有表结构定义 |

### repository.py — Repository 模式

数据访问层封装：

| 类 | 说明 |
|-----|------|
| `CrawlerPageRepository` | 爬取页面数据访问 |
| `CrawlStatsRepository` | 爬取统计数据访问 |
| `CleanedDocumentRepository` | 清洗文档数据访问 |
| `CleaningRunRepository` | 清洗运行记录数据访问 |
| `get_crawler_page_repo()` | 获取爬取页面 Repository |
| `get_crawl_stats_repo()` | 获取爬取统计 Repository |
| `get_cleaned_document_repo()` | 获取清洗文档 Repository |
| `get_cleaning_run_repo()` | 获取清洗运行记录 Repository |

**使用示例：**

```python
from src.utils import get_crawler_page_repo

repo = get_crawler_page_repo()
repo.insert_page(url, title, content, domain)

pages = repo.get_pages_by_domain("example.com")
stats = repo.get_recent_stats(days=7)
```
