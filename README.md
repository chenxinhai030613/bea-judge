# BEA-Judge：实验与复现仓库

这是 BEA-Judge 四模块框架的实验与写作仓库，包含数据处理、SFT、QLoRA 微调、对比试验、四模块消融、tie 敏感审计、SCI 表格与论文/Word 导出脚本。

## GitHub public release boundary

本仓库按“源码优先、产物外置”整理：

- 提交源码、配置、测试、复现说明、轻量级元数据和 `artifacts/` 中的小型摘要。
- 不提交模型权重、QLoRA checkpoint、虚拟环境、依赖缓存、原始/处理中间数据、完整模型输出、训练日志、历史归档和生成的 Word/PDF 文件；这些内容已由 `.gitignore` 排除，并保留在本地供后续通过合适的数据或模型存储发布。
- `datasets/model_outputs/` 中的完整实验产物不属于初始 GitHub 提交；正式结果的路径和门禁说明仍保留在本文档与 `REPRODUCIBILITY_MANIFEST.json` 中。
- 首次提交前请检查 `git status --short`、`git diff --cached --stat` 和 `git diff --cached --check`，不要使用 `git add -f` 强行加入被排除的大文件。

所有命令均从仓库根目录运行。Linux/macOS 示例：

```bash
cd /path/to/bea-judge
```

Windows PowerShell 可使用：

```powershell
Set-Location <repo-root>
```

## 1. 先看正式结果

正式可引用的结果目录和文件如下：

- `datasets/model_outputs/sci_tables_v2_20260521_110114/`
- `datasets/model_outputs/qlora_3seed_epoch1_1024_summary/three_seed_summary.json`：三种子保守汇总，不是最终提交包
- `datasets/model_outputs/qlora_3seed_epoch1_1024_summary/qlora_submission_ready_results.json`
- `datasets/model_outputs/qlora_3seed_epoch1_1024_tie_sensitive_dev_summary/three_seed_summary.json`
- `datasets/model_outputs/external_3b_baseline_comparison/`

不要把下面这些诊断/探测结果和主结果混为一谈：

- `datasets/judge_outputs/latest_summary.json`
- `datasets/model_outputs/accuracy_constrained_tie_rescue_global_strict_dev_summary/accuracy_constrained_tie_rescue_audit.json`

其中，`qlora_3seed_epoch1_1024_summary/three_seed_summary.json` 只是三种子保守汇总，当前 `gate.all_passed=false`，不能当作最终提交包；真正的提交包是 `qlora_submission_ready_results.json`，它把 accuracy-oriented operating point 和 tie-sensitive operating point 分开保存。

## 2. 环境准备

推荐环境是仓库内的 `.venv_qlora`。如果要完整跑训练/推理，还需要这些包已经可用：

- 基础分析和文档：`numpy`、`pandas`、`matplotlib`、`python-docx`、`pytest`
- QLoRA/推理栈：`torch`、`transformers`、`datasets`、`peft`、`bitsandbytes`、`accelerate`、`safetensors`

先做最小检查：

```bash
make check-env
make compile
make test
```

说明：

- `make check-env` 只检查 `pytest`、`matplotlib`、`docx`
- `make compile` 和 `make test` 会把字节码缓存写到 `/tmp/bea-judge-pycache`
- 如果你只是在只读挂载环境里做检查，`make compile` 仍然可以跑，因为缓存会落到 `/tmp`

## 3. 目录与产物

你最常会用到的路径是：

- `datasets/processed/bea_judge_cleaned_10000.json`：正式输入数据
- `datasets/sft/m_prometheus_pairwise/`：pairwise SFT 训练数据
- `models/M-Prometheus-3B/`：基础 Judge 模型
- `models/m_prometheus_3b_qlora_pairwise_seed*/`：QLoRA adapter
- `datasets/judge_outputs/m_prometheus_3b_qlora_pairwise_seed*/`：原始 base scores
- `datasets/model_outputs/bea_judge_qlora_pairwise_seed*/`：融合与校准后的正式输出
- `datasets/model_outputs/sci_tables_v2_20260521_110114/`：SCI 表格总目录
- `论文撰写/BEA-Judge中文论文_20260530/`：中文论文稿和 Word 输出

仓库里还有一些历史/中间目录，例如 `latest_*`、`smoke`、`probe`、`legacy`，它们只用于调试，不要直接当正式结论。

## 4. 推荐复现顺序

如果你想从头按顺序复现，建议依次做这几步：

1. 检查环境：`make check-env`、`make test`
2. 准备或确认正式数据：`datasets/processed/bea_judge_cleaned_10000.json`
3. 构建 pairwise SFT 数据：`scripts/build_judge_sft_dataset.py`
4. 跑 QLoRA 单种子和三种子主实验：`scripts/run_qlora_epoch1_seed.sh`、`scripts/run_qlora_3seed_epoch1.sh`
5. 跑外部 3B baseline：`scripts/run_external_3b_baselines.py`
6. 跑 epoch ablation 和 SFT size ablation：`scripts/run_qlora_ablation_grid.sh`
7. 跑 tie 敏感验证和 accuracy-constrained audit
8. 生成 SCI 表格、图和 Word 论文

## 5. 数据与 SFT

正式实验直接依赖已经冻结的输入：

```bash
datasets/processed/bea_judge_cleaned_10000.json
```

如果你只想重跑实验，不需要重新从原始来源重建数据。若你确实要从源头重建，仓库里还有这些数据脚本：

- `scripts/dataset_expansion_builder.py`
- `scripts/prepare_bea_judge_cleaned_dataset.py`
- `scripts/canonical_dataset_cleaner.py`
- `scripts/dataset_builder.py`

这些脚本适合做数据重建和清洗，不是主实验的必需步骤。

### 5.1 构建 pairwise SFT 数据

```bash
python scripts/build_judge_sft_dataset.py \
  --config configs/qlora_judge_sft_24gb_epoch1_1024.json
```

默认会输出：

- `datasets/sft/m_prometheus_pairwise/train.jsonl`
- `datasets/sft/m_prometheus_pairwise/dev.jsonl`
- `datasets/sft/m_prometheus_pairwise/metadata.json`

### 5.2 构建 SFT size 子集

```bash
python scripts/build_qlora_sft_subsets.py \
  --source-dir datasets/sft/m_prometheus_pairwise \
  --output-root datasets/sft \
  --sample-sizes 1202 2403 \
  --seed 42
```

这会生成 25% 和 50% 的确定性子集，用于 `sft_size` 消融。

## 6. QLoRA 微调与主实验

主实验的固定协议是 `configs/qlora_judge_sft_24gb_epoch1_1024.json`，推荐始终保留 `RUN_SUFFIX=_1024`，这样和正式结果完全对齐。

### 6.1 单种子微调

```bash
SEED=13 RUN_SUFFIX=_1024 CONFIG=configs/qlora_judge_sft_24gb_epoch1_1024.json \
bash scripts/run_qlora_epoch1_seed.sh
```

这个脚本会依次做：

- QLoRA adapter 训练
- train/dev/test base scores 生成
- base scores 合并
- BEA-Judge 融合/校准
- 与 frozen BEA-Judge 做对比

### 6.2 三种子主实验

```bash
SEEDS="13 42 2026" RUN_SUFFIX=_1024 RUN_TIE_SENSITIVE=1 \
CONFIG=configs/qlora_judge_sft_24gb_epoch1_1024.json \
bash scripts/run_qlora_3seed_epoch1.sh
```

这个脚本是主结果的一键入口，会自动生成：

- `datasets/model_outputs/qlora_3seed_epoch1_1024_summary/three_seed_summary.json`
- `datasets/model_outputs/qlora_3seed_epoch1_1024_tie_sensitive_dev_summary/three_seed_summary.json`
- `datasets/model_outputs/qlora_3seed_epoch1_1024_summary/qlora_submission_ready_results.json`
- `datasets/model_outputs/qlora_3seed_epoch1_1024_summary/qlora_submission_ready_results.md`

然后自动做提交包验证：

```bash
python scripts/validate_qlora_submission_package.py \
  --submission-summary datasets/model_outputs/qlora_3seed_epoch1_1024_summary/qlora_submission_ready_results.json
```

### 6.3 只重建提交包

如果三种子保守汇总和 tie-sensitive 汇总已经存在，但你想只重建提交包：

```bash
python scripts/build_qlora_submission_summary.py \
  --conservative-summary datasets/model_outputs/qlora_3seed_epoch1_1024_summary/three_seed_summary.json \
  --tie-sensitive-summary datasets/model_outputs/qlora_3seed_epoch1_1024_tie_sensitive_dev_summary/three_seed_summary.json \
  --output-dir datasets/model_outputs/qlora_3seed_epoch1_1024_summary
```

### 6.4 你也可以直接看单个结果文件

- `datasets/model_outputs/qlora_comparison_seed13_epoch1_1024/claim_gate_report.json`
- `datasets/model_outputs/qlora_comparison_seed42_epoch1_1024/claim_gate_report.json`
- `datasets/model_outputs/qlora_comparison_seed2026_epoch1_1024/claim_gate_report.json`

## 7. 对比试验

### 7.1 frozen baseline vs QLoRA baseline

每个 seed 先跑：

```bash
SEED=13 RUN_SUFFIX=_1024 CONFIG=configs/qlora_judge_sft_24gb_epoch1_1024.json \
bash scripts/run_qlora_epoch1_seed.sh
```

然后 `scripts/compare_qlora_experiments.py` 会自动产出：

- `qlora_comparison_report.json`
- `claim_gate_report.json`
- `main_comparison_table.md`

### 7.2 外部 3B baselines

同一个 `output-dir` 下顺序跑四次：

```bash
python scripts/run_external_3b_baselines.py \
  --dataset datasets/processed/bea_judge_cleaned_10000.json \
  --model-kind grm_reward \
  --model-name Ray2333/GRM-Llama3.2-3B-rewardmodel-ft \
  --model-path models/external_baselines/GRM-Llama3.2-3B-rewardmodel-ft \
  --output-dir datasets/model_outputs/external_3b_baseline_comparison

python scripts/run_external_3b_baselines.py \
  --dataset datasets/processed/bea_judge_cleaned_10000.json \
  --model-kind qwen_instruct \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --model-path models/external_baselines/Qwen2.5-3B-Instruct \
  --output-dir datasets/model_outputs/external_3b_baseline_comparison

python scripts/run_external_3b_baselines.py \
  --dataset datasets/processed/bea_judge_cleaned_10000.json \
  --model-kind prometheus2_pairwise \
  --model-name prometheus-eval/prometheus-7b-v2.0 \
  --model-path models/external_baselines/prometheus-7b-v2.0 \
  --output-dir datasets/model_outputs/external_3b_baseline_comparison

python scripts/run_external_3b_baselines.py \
  --dataset datasets/processed/bea_judge_cleaned_10000.json \
  --model-kind glider_evaluator \
  --model-name PatronusAI/glider \
  --model-path models/external_baselines/glider \
  --output-dir datasets/model_outputs/external_3b_baseline_comparison
```

如果显存紧张，可以额外加 `--load-in-4bit`。

### 7.3 外部基线整合比较

```bash
python scripts/build_external_3b_full_comparison.py \
  --epoch-summary datasets/model_outputs/qlora_epoch_ablation_3seed_1024_summary/epoch_ablation_summary.json \
  --external-report datasets/model_outputs/external_3b_baseline_comparison/external_3b_baseline_comparison_report.json \
  --tie-rescue-audit datasets/model_outputs/accuracy_constrained_tie_rescue_global_strict_dev_summary/accuracy_constrained_tie_rescue_audit.json \
  --setting epoch2_1024 \
  --output-dir datasets/model_outputs/external_3b_baseline_comparison
```

## 8. 消融实验

### 8.1 epoch ablation

```bash
GROUP=epoch SEEDS="13 42 2026" \
CONFIG=configs/qlora_judge_sft_24gb_epoch1_1024.json \
bash scripts/run_qlora_ablation_grid.sh
```

后处理汇总：

```bash
python scripts/summarize_qlora_ablation_grid.py \
  --settings epoch0p5_1024 epoch1_1024 epoch2_1024 \
  --seeds 13 42 2026 \
  --report-template datasets/model_outputs/qlora_comparison_seed{seed}_{setting}/qlora_comparison_report.json \
  --output-json datasets/model_outputs/qlora_epoch_ablation_3seed_1024_summary/epoch_ablation_summary.json \
  --output-md datasets/model_outputs/qlora_epoch_ablation_3seed_1024_summary/epoch_ablation_summary.md \
  --title "QLoRA Epoch Ablation Summary"
```

### 8.2 SFT size ablation

```bash
GROUP=sft_size SEEDS="13 42 2026" \
CONFIG=configs/qlora_judge_sft_24gb_epoch1_1024.json \
bash scripts/run_qlora_ablation_grid.sh
```

后处理汇总：

```bash
python scripts/summarize_qlora_ablation_grid.py \
  --settings sft25_epoch1_1024 sft50_epoch1_1024 sft100_epoch1_1024 \
  --seeds 13 42 2026 \
  --report-template datasets/model_outputs/qlora_comparison_seed{seed}_{setting}/qlora_comparison_report.json \
  --output-json datasets/model_outputs/qlora_sft_size_ablation_3seed_1024_summary/sft_size_ablation_summary.json \
  --output-md datasets/model_outputs/qlora_sft_size_ablation_3seed_1024_summary/sft_size_ablation_summary.md \
  --title "QLoRA SFT Size Ablation Summary"
```

### 8.3 四模块消融

```bash
python scripts/summarize_qlora_ablation_3seed.py \
  --seeds 13 42 2026 \
  --report-template datasets/model_outputs/qlora_ablation_seed{seed}_epoch1_1024/ablation_report.json \
  --output-dir datasets/model_outputs/qlora_ablation_3seed_epoch1_1024_summary
```

如果要重新跑某个 seed：

```bash
SEED=13 EXPERIMENT_TAG=epoch1_1024 \
CONFIG=configs/qlora_judge_sft_24gb_epoch1_1024.json \
bash scripts/run_qlora_ablation_seed.sh
```

## 9. tie 敏感与 accuracy-constrained 审计

### 9.1 dev-selected tie-sensitive validation

```bash
python scripts/build_tie_sensitive_validation_reports.py \
  --seeds 13 42 2026 \
  --frozen-report datasets/model_outputs/bea_judge_20260521_110114/validation_report.json \
  --validation-template datasets/model_outputs/bea_judge_qlora_pairwise_seed{seed}_epoch1_1024/validation_report.json \
  --calibrated-template datasets/model_outputs/bea_judge_qlora_pairwise_seed{seed}_epoch1_1024/calibrated_results.json \
  --output-template datasets/model_outputs/bea_judge_qlora_pairwise_seed{seed}_epoch1_1024_tie_sensitive_dev
```

它会写出每个 seed 的：

- `validation_report.json`
- `calibrated_results.json`
- `tie_sensitive_policy.json`

然后用：

```bash
python scripts/summarize_qlora_3seed.py \
  --seeds 13 42 2026 \
  --comparison-dirs \
  datasets/model_outputs/qlora_comparison_seed13_epoch1_1024_tie_sensitive_dev \
  datasets/model_outputs/qlora_comparison_seed42_epoch1_1024_tie_sensitive_dev \
  datasets/model_outputs/qlora_comparison_seed2026_epoch1_1024_tie_sensitive_dev \
  --output-dir datasets/model_outputs/qlora_3seed_epoch1_1024_tie_sensitive_dev_summary
```

### 9.2 accuracy-constrained tie rescue audit

这个审计是辅助诊断，不是主提交包。当前仓库里已经有一个全局严格 dev 版本：

- `datasets/model_outputs/accuracy_constrained_tie_rescue_global_strict_dev_summary/accuracy_constrained_tie_rescue_audit.json`

你可以重跑：

```bash
python scripts/accuracy_constrained_tie_rescue_audit.py \
  --settings epoch1_1024 epoch2_1024 \
  --selection-mode global \
  --calibrated-template datasets/model_outputs/bea_judge_qlora_pairwise_seed{seed}_{setting}/calibrated_results.json \
  --guardrail-validation-template datasets/model_outputs/bea_judge_qlora_pairwise_seed{seed}_epoch1_1024/validation_report.json \
  --output-dir datasets/model_outputs/accuracy_constrained_tie_rescue_global_strict_dev_summary
```

当前结果里，`epoch1_1024` 没有通过 accuracy gate，但 `epoch2_1024` 通过；这也是为什么它只作为诊断附件出现，而不是主结果。

## 10. 偏差、证据与 order-swap 诊断

这些不是主结果，但会进入 SCI 表格和讨论部分。

### 10.1 偏差审计

```bash
python scripts/bias_awareness_audit.py \
  --datasets datasets \
  --calibrated-results datasets/model_outputs/bea_judge_20260521_110114/calibrated_results.json
```

输出：

- `datasets/bias_awareness_report.json`
- `datasets/bias_awareness_report.md`
- `datasets/bias_profiles.json`

### 10.2 证据事实审计

```bash
python scripts/evidence_fact_audit.py --datasets datasets
```

输出：

- `datasets/evidence_fact_report.json`
- `datasets/evidence_fact_report.md`
- `datasets/evidence_profiles.json`

### 10.3 order-swap probe

```bash
python scripts/order_swap_probe.py \
  --experiment-config configs/experiment.json \
  --output-dir datasets/judge_outputs/order_swap_probe
```

这个探测主要服务于偏差和一致性分析，不要把它写成主实验。

## 11. SCI 表格、图和论文/Word 导出

### 11.1 生成 SCI 表格

正式表格目录是：

```bash
datasets/model_outputs/sci_tables_v2_20260521_110114/
```

建议显式冻结输入后再生成：

```bash
python scripts/generate_sci_results_tables.py \
  --base-scores datasets/judge_outputs/m_prometheus_3b_bea10k_v2/base_scores.repaired.json \
  --repair-report datasets/judge_outputs/m_prometheus_3b_bea10k_v2/base_scores_repair_report.json \
  --validation-report datasets/model_outputs/bea_judge_20260521_110114/validation_report.json \
  --calibrated-results datasets/model_outputs/bea_judge_20260521_110114/calibrated_results.json \
  --ablation-report datasets/model_outputs/latest_ablation_report.json \
  --bias-report datasets/bias_awareness_report.json \
  --bias-profiles datasets/bias_profiles.json \
  --evidence-report datasets/evidence_fact_report.json \
  --evidence-profiles datasets/evidence_profiles.json \
  --swap-report datasets/judge_outputs/order_swap_probe/swap_probe_report.json \
  --expansion-report datasets/expansion_v2_report.json \
  --manifest-v2 datasets/data_manifest_v2.json \
  --experiment-config configs/experiment.json \
  --output-dir datasets/model_outputs/sci_tables_v2_20260521_110114
```

这个目录里会有：

- `main_results_table.csv/.md`
- `metric_confidence_interval_table.csv/.md`
- `ablation_table.csv/.md`
- `ablation_significance_table.csv/.md`
- `evidence_feature_group_ablation_table.csv/.md`
- `tie_recall_table.csv/.md`
- `per_dataset_table.csv/.md`
- `ragtruth_results_table.csv/.md`
- `base_diagnostics_table.csv/.md`
- `bias_subgroup_table.csv/.md`
- `evidence_subtype_table.csv/.md`
- `swap_consistency_table.csv/.md`
- `source_provenance_table.csv/.md`
- `license_audit_table.csv/.md`
- `v2_distribution_table.csv/.md`
- `data_scaling_table.csv/.md`
- `method_summary.md`

### 11.2 生成 SCI 图

```bash
python scripts/generate_bea_judge_sci_figures.py
```

输出目录：

- `论文撰写/figures_bea_judge_10k/`

主输出是可编辑矢量图 `SVG`，同时会写 `PDF`、`PNG`、`TIFF` 和 `figure_manifest.csv`。

### 11.3 生成中文论文 Word

最稳妥的 Linux 路线是这两个脚本：

```bash
python scripts/build_bea_judge_docx.py
python scripts/generate_bea_judge_sci_cn_docx.py
```

它们分别生成：

- `论文撰写/BEA-Judge中文论文_20260530/BEA-Judge中文论文_20260530.docx`
- `论文撰写/BEA-Judge-10K-v2_SCI中文论文初稿_公式图表文献增强版.docx`

如果你只需要最小可编辑 Word，优先用 `build_bea_judge_docx.py`。这个脚本不依赖 `pandoc`，更适合当前 Linux 环境。

### 11.4 IEEE/备选 Word

```bash
python scripts/build_ieee_bea_judge_word.py
```

这是 best-effort 路线，依赖 `pandoc`，而且脚本里还带有 Windows 风格的 CSL 路径配置。当前 Linux 服务器上通常不是首选，只适合作为备份方案。

## 12. 验证命令

主流程验证建议按这个顺序跑：

```bash
make check-env
make compile
make test
make summarize-qlora
python scripts/validate_qlora_submission_package.py \
  --submission-summary datasets/model_outputs/qlora_3seed_epoch1_1024_summary/qlora_submission_ready_results.json
```

如果你要看三种子正式提交包是否已经锁定，直接看：

- `datasets/model_outputs/qlora_3seed_epoch1_1024_summary/qlora_submission_ready_results.json`
- `datasets/model_outputs/qlora_3seed_epoch1_1024_summary/qlora_submission_ready_results.md`

## 13. 已知限制

- `datasets/judge_outputs/latest_summary.json` 只是 probe 摘要，不能写进论文主结论
- `datasets/model_outputs/qlora_3seed_epoch1_1024_summary/three_seed_summary.json` 不是最终提交包，因为保守门禁没有全过
- `accuracy_constrained_tie_rescue_global_strict_dev_summary/accuracy_constrained_tie_rescue_audit.json` 里，`epoch1_1024` 没过 accuracy gate，`epoch2_1024` 过了
- `.ps1` 脚本是 Windows 备份入口，当前 Linux 服务器不作为主复现路径
- `generate_sci_results_tables.py` 的默认路径里有 `latest_*`，如果这些文件被你后续探测覆盖，请改成显式冻结路径再重跑
- `build_ieee_bea_judge_word.py` 需要 `pandoc`，不是当前最稳的 Word 路线

如果你只想复现论文里的主结果，优先跑第 6 节和第 11 节；如果你要补审稿人常问的扩展结果，再补第 7 到第 10 节。
