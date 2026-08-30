# BEA-Judge GitHub 公共发布说明

## 发布目标

GitHub 仓库用于保存可审阅、可复现的 BEA-Judge 源码和实验协议。模型权重、运行环境和大规模实验产物继续留在本地，或在确认许可证和访问方式后单独发布。

## 建议提交内容

首次提交建议包含：

- `src/`、`scripts/`、`configs/` 和 `tests/`
- `README.md`、`Makefile`、`requirements.txt` 和 `REPRODUCIBILITY_MANIFEST.json`
- `artifacts/` 中体积较小且不含原始样本的摘要文件
- 仓库内已有的轻量级配置、模式定义和审计说明

## 保持本地的内容

以下内容已经写入 `.gitignore`：

- `models/`、`judge/`、`_deps/` 和所有虚拟环境或下载缓存
- `datasets/raw/`、`datasets/processed/`、`datasets/sft/`、`datasets/judge_outputs/`、`datasets/model_outputs/`
- `datasets/方案2/raw_cache/`、原始 QA 文件和切分结果
- `logs/`、`archive/`、Word/PDF 论文文件以及 Python 缓存

这些文件没有被删除。若要公开数据或模型，请单独完成许可证、隐私/安全审查、文件校验和下载说明。

## 首次提交前检查

在仓库根目录运行：

```bash
git status --short
git add .
git diff --cached --stat
git diff --cached --check
```

确认暂存区只包含源码、配置、测试、文档和明确选定的小型摘要。不要用 `git add -f` 绕过忽略规则，也不要把访问令牌、个人路径或本地运行日志提交到仓库。

## 复现说明

最小检查仍按主 README 执行：

```bash
make check-env
make compile
make test
```

完整训练和推理需要本地下载基础模型、安装额外依赖，并准备被忽略的数据输入；GitHub 仓库本身不承诺包含这些大型运行时资源。
