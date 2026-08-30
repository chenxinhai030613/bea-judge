from __future__ import annotations

import csv
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper_ieee_bea_judge"
FIG_DIR = OUT / "figures"
FORMAL = ROOT / "datasets" / "model_outputs" / "sci_tables_v2_20260521_110114"
EXTENDED = ROOT / "datasets" / "model_outputs" / "sci_tables_extended_20260522"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value: str, digits: int = 4) -> str:
    if value in {"", None}:  # type: ignore[comparison-overlap]
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return str(value)


def pct(value: str, digits: int = 1) -> str:
    if value in {"", None}:  # type: ignore[comparison-overlap]
        return "--"
    return f"{float(value) * 100:.{digits}f}"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = Path(r"C:\Windows\Fonts\simhei.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    return plt


def save_vector(fig, name: str) -> None:
    pdf = FIG_DIR / f"{name}.pdf"
    svg = FIG_DIR / f"{name}.svg"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")


def make_figures() -> None:
    plt = setup_matplotlib()
    import matplotlib.patches as patches
    import numpy as np

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Fig. 1: pipeline.
    fig, ax = plt.subplots(figsize=(8.8, 2.7))
    ax.axis("off")
    boxes = [
        ("数据构建\nBEA-Judge-10K v2", "多源偏好、偏差与事实性样本"),
        ("基座评判器\nM-Prometheus-3B", "真实成对打分与标签解析"),
        ("偏差与证据特征", "位置、长度、格式、证据支持与幻觉风险"),
        ("融合校准", "Softmax头、温度缩放、Tie策略、复核阈值"),
    ]
    colors = ["#eaf3ff", "#eef7e9", "#fff4e6", "#f3ecff"]
    for i, ((title, subtitle), color) in enumerate(zip(boxes, colors)):
        x = 0.03 + i * 0.24
        rect = patches.FancyBboxPatch(
            (x, 0.28),
            0.2,
            0.44,
            boxstyle="round,pad=0.02,rounding_size=0.015",
            linewidth=1.1,
            edgecolor="#3b3b3b",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + 0.1, 0.57, title, ha="center", va="center", fontsize=11, weight="bold")
        ax.text(x + 0.1, 0.39, subtitle, ha="center", va="center", fontsize=8.6)
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(x + 0.235, 0.50),
                xytext=(x + 0.205, 0.50),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"),
            )
    ax.text(0.5, 0.12, "输出：成对偏好标签、事实性标签、置信度、风险分数与人工复核标记", ha="center", fontsize=9.5)
    save_vector(fig, "fig1_pipeline")
    plt.close(fig)

    # Fig. 2: dataset distribution.
    dist = read_csv(FORMAL / "v2_distribution_table.csv")
    dataset_counts = [(r["value"], int(r["count"])) for r in dist if r["dimension"] == "by_dataset"]
    dataset_counts = sorted(dataset_counts, key=lambda item: item[1], reverse=True)
    fig, ax = plt.subplots(figsize=(8.7, 4.0))
    labels = [x[0] for x in dataset_counts]
    values = [x[1] for x in dataset_counts]
    ax.bar(labels, values, color="#3a6ea5", edgecolor="white")
    ax.set_ylabel("样本数")
    ax.set_title("BEA-Judge-10K v2 的数据来源分布")
    ax.tick_params(axis="x", rotation=40, labelsize=8.5)
    ax.grid(axis="y", alpha=0.28)
    save_vector(fig, "fig2_dataset_distribution")
    plt.close(fig)

    # Fig. 3: main results.
    main = [r for r in read_csv(FORMAL / "main_results_table.csv") if r["split"] == "test"]
    metrics = ["accuracy", "macro_f1", "ece", "brier"]
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    for i, row in enumerate(main):
        values = [float(row[m]) for m in metrics]
        label = "成对偏好头" if row["head"] == "pairwise" else "事实性头"
        ax.bar(x + (i - 0.5) * width, values, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Macro-F1", "ECE", "Brier"])
    ax.set_ylim(0, 0.86)
    ax.set_ylabel("指标值")
    ax.set_title("测试集主结果")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.28)
    save_vector(fig, "fig3_main_results")
    plt.close(fig)

    # Fig. 4: ablation.
    abl = [r for r in read_csv(FORMAL / "ablation_table.csv") if r["split"] == "test"]
    variants = ["Full BEA-Judge", "w/o Evidence Module", "w/o Calibration", "w/o Bias Module"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), sharey=True)
    for ax, head, title in zip(axes, ["pairwise", "factuality"], ["成对偏好头", "事实性头"]):
        rows = {r["variant"]: r for r in abl if r["head"] == head}
        xs = np.arange(len(variants))
        vals = [float(rows[v]["macro_f1"]) if v in rows else np.nan for v in variants]
        bars = ax.bar(xs, vals, color=["#2f6f9f", "#c94c4c", "#f0a33a", "#6d9f42"], edgecolor="white")
        ax.set_xticks(xs)
        ax.set_xticklabels(["Full", "无证据", "无校准", "无偏差"], rotation=25)
        ax.set_title(title)
        ax.set_ylim(0.58, 0.78)
        ax.grid(axis="y", alpha=0.25)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.004, f"{val:.3f}", ha="center", fontsize=8)
    axes[0].set_ylabel("Macro-F1")
    fig.suptitle("关键模块消融对比")
    fig.tight_layout()
    save_vector(fig, "fig4_ablation")
    plt.close(fig)

    # Fig. 5: risk coverage.
    risk = read_csv(EXTENDED / "risk_coverage_table.csv")
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    for head, marker in [("pairwise", "o"), ("factuality", "s")]:
        rows = [r for r in risk if r["head"] == head and r["split"] == "test"]
        xs = [float(r["review_rate"]) for r in rows]
        ys = [float(r["error_capture_rate"]) for r in rows]
        label = "成对偏好头" if head == "pairwise" else "事实性头"
        ax.plot(xs, ys, marker=marker, linewidth=1.8, label=label)
    ax.set_xlabel("人工复核比例")
    ax.set_ylabel("错误捕获率")
    ax.set_title("风险阈值复核曲线")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.28)
    ax.legend(frameon=False)
    save_vector(fig, "fig5_risk_coverage")
    plt.close(fig)

    # Fig. 6: evidence subtype diagnostics.
    evidence = [r for r in read_csv(FORMAL / "evidence_subtype_table.csv") if r["head"] == "factuality" and r["evidence_subtype"] != "none"]
    evidence = sorted(evidence, key=lambda r: int(r["error_count"]), reverse=True)[:8]
    labels = [r["evidence_subtype"].replace("_", "\n") for r in evidence]
    errors = [float(r["error_rate"]) for r in evidence]
    capture = [float(r["review_capture_rate"]) if r["review_capture_rate"] else 0.0 for r in evidence]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.bar(x - 0.18, errors, width=0.36, label="错误率", color="#c94c4c", edgecolor="white")
    ax.bar(x + 0.18, capture, width=0.36, label="复核捕获率", color="#3a6ea5", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("比例")
    ax.set_title("事实性证据错误子类型诊断")
    ax.grid(axis="y", alpha=0.28)
    ax.legend(frameon=False)
    save_vector(fig, "fig6_evidence_subtypes")
    plt.close(fig)


def make_bib() -> str:
    return dedent(
        r"""
        @inproceedings{zheng2023judging,
          title={Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena},
          author={Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and Zhuang, Siyuan and Wu, Zhanghao and Zhuang, Yonghao and Lin, Zi and Li, Zhuohan and Li, Dacheng and Xing, Eric P. and Zhang, Hao and Gonzalez, Joseph E. and Stoica, Ion},
          booktitle={Advances in Neural Information Processing Systems, Datasets and Benchmarks Track},
          year={2023}
        }

        @inproceedings{kim2024prometheus,
          title={Prometheus: Inducing Fine-grained Evaluation Capability in Language Models},
          author={Kim, Seungone and Shin, Jamin and Cho, Yejin and Jang, Joel and Longpre, Shayne and Lee, Hwaran and Yun, Sangdoo and Shin, Seongjin and Kim, Sungdong and Thorne, James and Seo, Minjoon},
          booktitle={International Conference on Learning Representations},
          year={2024}
        }

        @inproceedings{kim2024prometheus2,
          title={Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models},
          author={Kim, Seungone and Suk, Juyoung and Longpre, Shayne and Lin, Bill Yuchen and Shin, Jamin and Welleck, Sean and Neubig, Graham and Lee, Moontae and Lee, Kyungjae and Seo, Minjoon},
          booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
          year={2024}
        }

        @misc{pombal2025mprometheus,
          title={M-Prometheus: A Suite of Open Multilingual LLM Judges},
          author={Pombal, Jos{\'e} and Yoon, Dongkeun and Fernandes, Patrick and Wu, Ian and Kim, Seungone and Rei, Ricardo and Neubig, Graham and Martins, Andr{\'e} F. T.},
          year={2025},
          eprint={2504.04953},
          archivePrefix={arXiv},
          primaryClass={cs.CL}
        }

        @misc{niu2024ragtruth,
          title={RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models},
          author={Niu, Cheng and Wu, Yuanhao and Zhu, Juno and Xu, Siliang and Shum, Kashun and Zhong, Randy and Song, Juntong and Zhang, Tong},
          year={2024},
          eprint={2401.00396},
          archivePrefix={arXiv},
          primaryClass={cs.CL}
        }

        @inproceedings{wang2024helpsteer2,
          title={HelpSteer2: Open-source Dataset for Training Top-performing Reward Models},
          author={Wang, Zhilin and Dong, Yi and Delalleau, Olivier and Zeng, Jiaqi and Shen, Gerald and Egert, Daniel and Zhang, Jimmy J. and Sreedhar, Makesh Narsimhan and Kuchaiev, Oleksii},
          booktitle={Advances in Neural Information Processing Systems, Datasets and Benchmarks Track},
          year={2024}
        }

        @inproceedings{kopf2023openassistant,
          title={OpenAssistant Conversations: Democratizing Large Language Model Alignment},
          author={K{\"o}pf, Andreas and Kilcher, Yannic and von R{\"u}tte, Dimitri and Anagnostidis, Sotiris and Tam, Zhi-Rui and Stevens, Keith and Barhoum, Abdullah and Duc, Nguyen Minh and Stanley, Oliver and Nagyfi, Rich{\'a}rd and others},
          booktitle={Advances in Neural Information Processing Systems, Datasets and Benchmarks Track},
          year={2023}
        }

        @misc{park2024offsetbias,
          title={OffsetBias: Leveraging Debiased Data for Tuning Evaluators},
          author={Park, Junsoo and Jwa, Seungyeon and Ren, Meiying and Kim, Daeyoung and Choi, Sanghyuk},
          year={2024},
          eprint={2407.06551},
          archivePrefix={arXiv},
          primaryClass={cs.CL}
        }

        @inproceedings{guo2017calibration,
          title={On Calibration of Modern Neural Networks},
          author={Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
          booktitle={Proceedings of the 34th International Conference on Machine Learning},
          pages={1321--1330},
          year={2017}
        }

        @incollection{platt1999probabilistic,
          title={Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods},
          author={Platt, John C.},
          booktitle={Advances in Large Margin Classifiers},
          pages={61--74},
          publisher={MIT Press},
          year={1999}
        }
        """
    ).strip() + "\n"


def latex_table_main() -> str:
    rows = [r for r in read_csv(FORMAL / "main_results_table.csv") if r["split"] == "test"]
    body = []
    for r in rows:
        head = "成对偏好" if r["head"] == "pairwise" else "事实性"
        tie = fnum(r["tie_recall"]) if r["tie_recall"] else "--"
        body.append(
            f"{head} & {r['n']} & {fnum(r['accuracy'])} & {fnum(r['macro_f1'])} & "
            f"{fnum(r['ece'])} & {fnum(r['brier'])} & {tie} & {fnum(r['review_rate'])} \\\\"
        )
    return "\n".join(body)


def latex_table_sources() -> str:
    rows = read_csv(FORMAL / "source_provenance_table.csv")
    names = {
        "helpsteer2": "HelpSteer2",
        "oasst1": "OASST1",
        "offsetbias": "OffsetBias",
        "ragtruth": "RAGTruth",
        "rewardbench": "RewardBench",
    }
    body = []
    for r in rows:
        body.append(
            f"{names.get(r['source'], r['source'])} & {r['license']} & {r['accepted_records']} & "
            f"{'是' if r['redistribution_allowed'] == 'True' else '否'} & "
            f"{'是' if r['admission_allowed'] == 'True' else '否'} \\\\"
        )
    return "\n".join(body)


def latex_table_ablation() -> str:
    rows = [r for r in read_csv(FORMAL / "ablation_table.csv") if r["split"] == "test"]
    order = ["Full BEA-Judge", "w/o Bias Module", "w/o Evidence Module", "w/o Calibration"]
    body = []
    for name in order:
        for head in ["pairwise", "factuality"]:
            match = next((r for r in rows if r["variant"] == name and r["head"] == head), None)
            if not match:
                continue
            zh_variant = {
                "Full BEA-Judge": "完整模型",
                "w/o Bias Module": "去除偏差模块",
                "w/o Evidence Module": "去除证据模块",
                "w/o Calibration": "去除校准策略",
            }[name]
            zh_head = "成对偏好" if head == "pairwise" else "事实性"
            tie = fnum(match["tie_recall"]) if match["tie_recall"] else "--"
            body.append(
                f"{zh_variant} & {zh_head} & {fnum(match['accuracy'])} & {fnum(match['macro_f1'])} & "
                f"{fnum(match['ece'])} & {fnum(match['brier'])} & {tie} \\\\"
            )
    return "\n".join(body)


def latex_table_calibration() -> str:
    rows = [r for r in read_csv(EXTENDED / "calibration_methods_table.csv") if r["split"] == "test"]
    names = {
        "temperature": "Temperature Scaling",
        "platt": "Platt Scaling",
        "isotonic": "Isotonic Regression",
        "vector_scaling": "Vector Scaling",
        "conformal": "Conformal Prediction",
    }
    body = []
    for r in rows:
        body.append(
            f"{names.get(r['method'], r['method'])} & {fnum(r['accuracy'])} & {fnum(r['ece'])} & "
            f"{fnum(r['mce'])} & {fnum(r['brier'])} & {fnum(r['nll'])} & {fnum(r['coverage'])} \\\\"
        )
    return "\n".join(body)


def latex_table_significance() -> str:
    rows = read_csv(FORMAL / "ablation_significance_table.csv")
    keep = [
        r for r in rows
        if (r["variant"], r["head"]) in {
            ("w/o Evidence Module", "factuality"),
            ("w/o Calibration", "pairwise"),
            ("w/o Bias Module", "pairwise"),
        }
    ]
    names = {
        "w/o Evidence Module": "去除证据模块",
        "w/o Calibration": "去除校准策略",
        "w/o Bias Module": "去除偏差模块",
    }
    body = []
    for r in keep:
        head = "成对偏好" if r["head"] == "pairwise" else "事实性"
        body.append(
            f"{names[r['variant']]} & {head} & {r['paired_n']} & "
            f"{fnum(r['delta_accuracy_full_minus_variant'])} & "
            f"[{fnum(r['delta_accuracy_ci95_low'])}, {fnum(r['delta_accuracy_ci95_high'])}] & "
            f"{fnum(r['mcnemar_p'], 6)} \\\\"
        )
    return "\n".join(body)


def make_tex() -> str:
    return dedent(
        rf"""
        \documentclass[journal]{{IEEEtran}}
        \usepackage[UTF8,heading=false,fontset=windows]{{ctex}}
        \usepackage{{amsmath,amssymb,bm}}
        \usepackage{{graphicx}}
        \usepackage{{booktabs}}
        \usepackage{{array}}
        \usepackage{{multirow}}
        \usepackage{{url}}
        \usepackage[hidelinks]{{hyperref}}
        \graphicspath{{{{figures/}}}}

        \newcommand{{\model}}{{BEA-Judge}}
        \newcommand{{\dataset}}{{BEA-Judge-10K v2}}
        \newcommand{{\softmax}}{{\operatorname{{softmax}}}}
        \newcommand{{\ECE}}{{\operatorname{{ECE}}}}

        \begin{{document}}

        \title{{BEA-Judge：面向生成式人工智能内容评估的偏差感知证据增强评判校准框架}}

        \author{{作者信息待补充%
        \thanks{{本文为基于当前项目实验产物撰写的中文 IEEE Transactions 风格论文草稿。作者、单位、基金和通信作者信息需在投稿前补全。}}}}

        \markboth{{IEEE Transactions 风格中文草稿，2026年5月}}{{BEA-Judge}}

        \maketitle

        \begin{{abstract}}
        生成式人工智能内容评估正在从静态基准测试转向可解释、可校准且可复核的评判系统。现有 LLM-as-a-Judge 方法能够降低人工评估成本，但在成对偏好排序、事实性判断和偏差控制上仍面临三类问题：评判器对位置、长度和格式等表层因素敏感；事实性错误常以局部证据缺失、数值或实体错配等细粒度形式出现；单次模型输出缺乏面向生产评审的置信度校准与复核机制。本文提出 \model，一个建立在真实 M-Prometheus-3B 输出之上的偏差感知证据增强评判校准框架。该框架不训练新的大语言模型，而是将基座评判器分数、文本差异、偏差风险、证据支持特征和数据源元信息输入任务特定 softmax 头，并在开发集上进行温度缩放、Tie 策略选择和风险阈值选择。我们构建 \dataset，共 10,200 条样本，覆盖开放问答、成对偏差评估和 RAG 事实性判断。冻结测试结果显示，成对偏好头达到 0.7512 accuracy、0.6730 macro-F1 和 0.0558 ECE；事实性头达到 0.7649 accuracy、0.7405 macro-F1 和 0.0377 ECE。消融实验表明，证据增强是事实性可靠性的主要贡献，去除证据模块使事实性 macro-F1 从 0.7405 降至 0.6542；校准策略显著提升成对任务的 macro-F1 与 Tie 召回。实验同时显示，偏差模块更适合作为风险识别与复核优先级机制，而非简单追求总体准确率提升。本文给出可复现实验门禁、数据许可审计和风险覆盖分析，为生成式内容评估中的可控自动评判提供了一种工程可落地方案。
        \end{{abstract}}

        \begin{{IEEEkeywords}}
        生成式人工智能评估，LLM-as-a-Judge，校准，事实性验证，偏差感知，风险复核，RAG 幻觉检测
        \end{{IEEEkeywords}}

        \section{{引言}}
        大语言模型输出的开放性使传统基于精确答案匹配的评估范式难以覆盖真实应用中的质量、偏好和事实性要求。LLM-as-a-Judge 通过将强模型或专用评判模型作为评审器，为开放式对话和响应排序提供了可扩展方案 \cite{{zheng2023judging,kim2024prometheus,kim2024prometheus2}}。然而，近期研究也表明，评判器容易受到位置偏差、长度偏差、格式偏差和自信度错配的影响；这些偏差会降低评估结论的可解释性，尤其会影响需要人工复核或质量门禁的场景 \cite{{zheng2023judging,park2024offsetbias}}。

        本项目聚焦生成式内容评估中的两个高频任务。第一类是成对偏好评估，即给定提示、上下文或评分准则，判断候选回答 A、B 的相对质量或是否应判为 Tie。第二类是事实性评估，即判断回答是否被上下文或参考证据支持。成对任务要求模型处理偏好边界、Tie 标签和候选顺序敏感性；事实性任务则要求系统识别实体、数值、日期、否定和比较关系等局部证据错配。单独依赖基座评判器输出，会把这些异质误差压缩成一个未校准标签，不利于解释和复核。

        针对上述问题，本文提出 \model。图~\ref{{fig:pipeline}} 展示了整体流程。与训练新的评判大模型不同，\model 将 M-Prometheus-3B 的真实成对评判输出作为基座信号，再加入文本结构、偏差风险、证据支持和数据源特征，训练轻量级任务特定 softmax 头。随后，系统在开发集上选择温度缩放、数据集级温度策略、Tie 决策策略和低置信复核阈值。该设计将自动评判拆分为“基础判断、风险诊断、概率校准和复核控制”四个可审计环节，便于在科研报告和工程部署中追踪误差来源。

        本文贡献如下：
        \begin{{itemize}}
          \item 提出一种面向生成式内容评估的偏差感知证据增强校准框架，将基座 LLM 评判输出、偏差风险特征和证据事实性特征统一到任务特定概率模型中。
          \item 构建并审计 \dataset，覆盖开放问答、成对偏差和 RAG 事实性三类任务，并明确训练、开发和测试划分及数据许可状态。
          \item 在冻结测试集上报告主结果、消融实验、显著性检验、校准方法比较、风险覆盖曲线和证据错误子类型分析，给出严格可复现的实验门禁。
          \item 明确方法边界：\model 不是新训练的大语言模型，也不声称解决原子级事实核验；其主要作用是提升评判输出的校准性、证据敏感性和复核可控性。
        \end{{itemize}}

        \begin{{figure*}}[t]
          \centering
          \includegraphics[width=0.98\textwidth]{{fig1_pipeline.pdf}}
          \caption{{\model 框架流程。系统以 \dataset 为输入，使用 M-Prometheus-3B 产生真实基座评判输出，再融合偏差风险与证据支持特征，最终通过任务特定校准头输出标签、置信度、风险分数和复核标记。}}
          \label{{fig:pipeline}}
        \end{{figure*}}

        \section{{相关工作}}
        \subsection{{LLM-as-a-Judge 与开放评判模型}}
        MT-Bench 和 Chatbot Arena 推动了使用 LLM 近似人类偏好评估的研究，证明强模型在一定条件下可作为可扩展评审器 \cite{{zheng2023judging}}。Prometheus 系列进一步强调开放评判模型、评分准则和参考材料的重要性 \cite{{kim2024prometheus,kim2024prometheus2}}。M-Prometheus 将评判能力扩展到多语言直接评分和成对比较场景 \cite{{pombal2025mprometheus}}，因此适合作为本文中文与英文混合实验的基座评判器。与这些工作不同，本文不试图训练或替代评判大模型，而是在其输出之上建立可校准、可诊断的轻量融合层。

        \subsection{{偏差鲁棒性与事实性评估}}
        LLM 评判器已知存在位置、长度、格式、准则敏感性等偏差 \cite{{zheng2023judging,park2024offsetbias}}。OffsetBias 从数据层面构建去偏训练样本，而本文将偏差作为风险诊断信号，用于校准和复核优先级。事实性方面，RAGTruth 提供了检索增强生成中幻觉和证据不一致的细粒度语料 \cite{{niu2024ragtruth}}。本文借鉴其证据敏感问题设置，但采用确定性证据特征和任务特定概率头，而非直接进行原子声明级判定。

        \subsection{{概率校准}}
        现代神经模型经常出现置信度与正确率不匹配的问题，温度缩放是常用后处理方法 \cite{{guo2017calibration}}。Platt Scaling 及其后续校准方法为概率输出校正提供了经典基线 \cite{{platt1999probabilistic}}。本文使用开发集选择温度缩放、数据集级温度和复核阈值，并额外报告多种校准方法在成对偏好头上的对比。

        \section{{数据集构建}}
        \subsection{{任务定义与数据来源}}
        \dataset 由 10,200 条样本组成，包含三类任务：开放问答 4,000 条、成对偏差评估 2,700 条、RAG 事实性评估 3,500 条。数据来源包括 HelpSteer2 \cite{{wang2024helpsteer2}}、OpenAssistant Conversations \cite{{kopf2023openassistant}}、OffsetBias \cite{{park2024offsetbias}}、RAGTruth \cite{{niu2024ragtruth}} 以及项目内中文专业标注数据。PandaLM、MT-Bench、JudgeBench 和 WikiEval 等来源用于形成开放问答或辅助评估切片 \cite{{zheng2023judging}}。

        表~\ref{{tab:sources}} 给出项目数据源许可审计结果。训练、开发和测试划分分别为 7,084、1,578 和 1,538 条；语言分布为英文 9,148 条、中文 1,052 条。成对标签包括 A>B、B>A 和 Tie；事实性标签包括 supported 与 unsupported，ambiguous 在当前冻结训练和测试划分中未作为有效训练类别参与正式指标计算。

        \begin{{table}}[t]
        \caption{{数据来源、许可与纳入状态。RewardBench 因混合子集许可限制仅保留为外部评估候选，未纳入正式训练。}}
        \label{{tab:sources}}
        \centering
        \scriptsize
        \resizebox{{\columnwidth}}{{!}}{{%
        \begin{{tabular}}{{lcccc}}
        \toprule
        来源 & 许可 & 纳入样本 & 可再分发 & 训练纳入 \\
        \midrule
        {latex_table_sources()}
        \bottomrule
        \end{{tabular}}}}
        \end{{table}}

        \begin{{figure}}[t]
          \centering
          \includegraphics[width=\columnwidth]{{fig2_dataset_distribution.pdf}}
          \caption{{\dataset 的数据来源分布。分布体现了偏好评估、偏差评估和事实性评估之间的多源异质性。}}
          \label{{fig:dataset_distribution}}
        \end{{figure}}

        \subsection{{质量控制与复现实验门禁}}
        数据构建执行了样本数、字段类型、枚举值、重复 ID、重复内容和跨 split 泄漏检查。正式结果门禁要求样本数位于 9,500 到 10,200 区间，成对基座评判覆盖率为 6,946/6,946，启发式 fallback 行数为 0，未解析失败为 0。实验索引显示所有正式门禁均通过，包括基座分数覆盖、修复后解析覆盖、无启发式正式结果、消融变体存在、偏差预测覆盖率为 1.0、证据 profile 数量完整以及事实性 ECE 不超过 0.04。

        \section{{方法}}
        \subsection{{问题形式化}}
        对样本 $x_i$，成对偏好头的标签空间为 $\mathcal{{Y}}_p=\{{A>B,B>A,Tie\}}$，事实性头的有效标签空间为 $\mathcal{{Y}}_f=\{{supported,unsupported\}}$。给定基座评判器输出、文本特征、偏差风险和证据特征，\model 学习任务特定概率分布：
        \begin{{equation}}
        p_\theta(y\mid x_i,h)=\softmax\left(\mathbf{{W}}_h \mathbf{{z}}_i+\mathbf{{b}}_h\right)_y,\quad y\in\mathcal{{Y}}_h ,
        \label{{eq:softmax}}
        \end{{equation}}
        其中 $h\in\{{p,f\}}$ 表示任务头，$\mathbf{{z}}_i$ 是标准化后的融合特征向量。

        特征标准化仅使用训练集统计量：
        \begin{{equation}}
        z_{{ij}}=\frac{{\phi_j(x_i)-\mu_j}}{{\sigma_j+\epsilon}},
        \label{{eq:standardization}}
        \end{{equation}}
        其中 $\phi_j$ 为第 $j$ 个原始特征，$\mu_j$ 和 $\sigma_j$ 分别为训练集均值与标准差。

        \subsection{{成对偏好特征}}
        成对头使用四类特征。第一类是 M-Prometheus-3B 输出，包括回答 A、B 的分数、分差、绝对 margin、基座预测标签及顺序置换诊断。第二类是文本结构特征，包括回答长度、句子数、列表项数、数值项数以及 prompt、context、reference 与回答之间的 token overlap。第三类是偏差标志和偏差风险。第四类是数据集、任务类型和评分系统 one-hot 特征。

        对基座分差 $\Delta s=s_A-s_B$，基座 margin 记为
        \begin{{equation}}
        m_i = |\Delta s_i|.
        \label{{eq:margin}}
        \end{{equation}}
        该 margin 与基座预测标签共同用于表达基座评判器的相对确定性。

        \subsection{{偏差风险建模}}
        偏差模块检测位置、长度、格式、rubric sensitivity 和数据源风险。每个风险项被裁剪到 $[0,1]$，总体偏差风险定义为最大风险：
        \begin{{equation}}
        r_i^b=\max\{{r_i^{{pos}},r_i^{{len}},r_i^{{fmt}},r_i^{{rub}},r_i^{{src}}\}}.
        \label{{eq:biasrisk}}
        \end{{equation}}
        采用最大值而非加权和，是为了让任一高风险偏差均可触发复核优先级，而不被其他低风险项稀释。消融结果显示，偏差模块不应被解释为总准确率提升模块，而应解释为风险识别和复核排序模块。

        \subsection{{事实性证据特征}}
        事实性模块计算回答与上下文、参考答案之间的证据支持。令 $a$ 为回答，$c$ 为上下文，$r$ 为参考答案，$\operatorname{{cov}}(a,e)$ 表示回答 token 被证据 $e$ 覆盖的比例，$\operatorname{{num}}(a,e)$ 表示数值项支持率，则回答支持度定义为
        \begin{{equation}}
        \begin{{split}}
        S(a,c,r)=\max\big(&S(a,c\oplus r),\\
        &0.65S(a,c)+0.35S(a,r)\big),
        \end{{split}}
        \label{{eq:support}}
        \end{{equation}}
        其中 $\oplus$ 表示证据拼接，$S(\cdot)$ 由 token 覆盖、长 token 覆盖和数值支持加权得到。证据风险由数值缺口、日期缺口、实体缺口、否定错配、比较错配和低支持句子比例共同决定：
        \begin{{equation}}
        r_i^e=\max_k g_k(x_i),
        \label{{eq:evidencerisk}}
        \end{{equation}}
        其中 $g_k$ 表示第 $k$ 个证据错误信号。该定义与工程实现保持一致，优先捕获局部严重风险。

        \subsection{{训练目标与校准}}
        每个任务头使用交叉熵与 $L_2$ 正则训练：
        \begin{{equation}}
        \mathcal{{L}}_h=-\frac{{1}}{{N_h}}\sum_{{i=1}}^{{N_h}}\log p_\theta(y_i\mid x_i,h)+\frac{{\lambda}}{{2}}\|\mathbf{{W}}_h\|_2^2 .
        \label{{eq:loss}}
        \end{{equation}}
        学习率、batch size、$L_2$ 和 epoch 数在开发集上搜索，选择目标为
        \begin{{equation}}
        J=\operatorname{{MacroF1}}+0.25\operatorname{{Accuracy}}-0.05\ECE .
        \label{{eq:selection}}
        \end{{equation}}
        训练后使用开发集选择温度 $T$：
        \begin{{equation}}
        \hat{{p}}_T(y\mid x)=\softmax\left(\frac{{\log p_\theta(y\mid x)}}{{T}}\right),
        \label{{eq:temperature}}
        \end{{equation}}
        选择指标为 dev NLL + dev ECE + $0.25$ dev Brier。

        ECE 采用等宽置信度分桶：
        \begin{{equation}}
        \ECE=\sum_{{b=1}}^B \frac{{|I_b|}}{{n}}\left|\operatorname{{acc}}(I_b)-\operatorname{{conf}}(I_b)\right|.
        \label{{eq:ece}}
        \end{{equation}}
        对任一输出，复核风险定义为
        \begin{{equation}}
        r_i=1-\max_y \hat{{p}}_T(y\mid x_i),
        \label{{eq:reviewrisk}}
        \end{{equation}}
        若 $r_i\ge\tau_h$，则样本进入人工复核队列。阈值 $\tau_h$ 在开发集上选择，目标是在可行时捕获至少 80\% 的开发集错误。

        \section{{实验设置}}
        \subsection{{训练与评估协议}}
        所有超参数、温度、Tie 策略和复核阈值仅使用训练集和开发集选择，测试集仅用于最终报告。成对头训练样本数为 4,806、开发样本数为 1,087、测试样本数为 1,053。事实性头训练样本数为 2,278、开发样本数为 491、测试样本数为 485。基座评判器为 Unbabel/M-Prometheus-3B，温度为 0.0，最大新 token 数为 256，正式实验不允许启发式 fallback。

        \subsection{{评价指标}}
        本文报告 Accuracy、Macro-F1、ECE、Brier score、Tie recall 和 review rate。成对任务同时报告 Tie recall，因为 Tie 是高不确定度样本的重要错误来源。事实性任务额外报告 RAGTruth 子集、证据错误子类型和风险覆盖曲线。

        \subsection{{对比与消融设置}}
        消融实验包括去除偏差模块、去除证据模块和去除校准策略。扩展基线包括 Raw M-Prometheus-3B only、Text/metadata-only、Base + fusion calibration only、去除基座分数、去除 Tie 策略和去除复核阈值。校准对比包括 temperature scaling、Platt scaling、isotonic regression、vector scaling 与 split conformal prediction。

        \section{{实验结果}}
        \subsection{{主结果}}
        表~\ref{{tab:main}} 与图~\ref{{fig:main_results}} 给出冻结测试集主结果。成对偏好头在 1,053 条测试样本上达到 0.7512 accuracy 和 0.6730 macro-F1，ECE 为 0.0558，Tie recall 为 0.5231。事实性头在 485 条测试样本上达到 0.7649 accuracy 和 0.7405 macro-F1，ECE 为 0.0377。两类任务的 ECE 均处于较低区间，说明后处理校准对置信度可解释性具有直接价值。

        \begin{{table}}[t]
        \caption{{冻结测试集主结果。}}
        \label{{tab:main}}
        \centering
        \scriptsize
        \resizebox{{\columnwidth}}{{!}}{{%
        \begin{{tabular}}{{lrrrrrrr}}
        \toprule
        任务头 & $n$ & Acc. & Macro-F1 & ECE & Brier & Tie-R & Review \\
        \midrule
        {latex_table_main()}
        \bottomrule
        \end{{tabular}}}}
        \end{{table}}

        \begin{{figure}}[t]
          \centering
          \includegraphics[width=\columnwidth]{{fig3_main_results.pdf}}
          \caption{{冻结测试集上成对偏好头与事实性头的主指标。Accuracy 与 Macro-F1 越高越好，ECE 与 Brier 越低越好。}}
          \label{{fig:main_results}}
        \end{{figure}}

        \subsection{{消融实验}}
        表~\ref{{tab:ablation}} 和图~\ref{{fig:ablation}} 展示关键模块消融。事实性头对证据模块高度敏感：去除证据后 accuracy 从 0.7649 降至 0.6928，macro-F1 从 0.7405 降至 0.6542。成对头对校准策略高度敏感：去除校准后 macro-F1 从 0.6730 降至 0.6402，Tie recall 从 0.5231 降至 0.3923。去除偏差模块后成对头 accuracy 和 macro-F1 略高于完整模型，这与项目预设解释一致：偏差模块的主要价值是识别高风险样本和支持复核，而非保证总体指标单调提升。

        \begin{{table*}}[t]
        \caption{{模块消融实验。所有指标均来自测试集。}}
        \label{{tab:ablation}}
        \centering
        \scriptsize
        \resizebox{{\textwidth}}{{!}}{{%
        \begin{{tabular}}{{llrrrrr}}
        \toprule
        变体 & 任务头 & Acc. & Macro-F1 & ECE & Brier & Tie-R \\
        \midrule
        {latex_table_ablation()}
        \bottomrule
        \end{{tabular}}}}
        \end{{table*}}

        \begin{{figure}}[t]
          \centering
          \includegraphics[width=\columnwidth]{{fig4_ablation.pdf}}
          \caption{{关键模块消融的 Macro-F1 对比。证据模块对事实性头贡献最大，校准策略对成对头的 Tie 处理和 macro-F1 贡献明显。}}
          \label{{fig:ablation}}
        \end{{figure}}

        表~\ref{{tab:significance}} 给出配对显著性结果。去除证据模块对事实性 accuracy 的影响为 $+0.0721$，95\% CI 为 $[0.0371,0.1093]$，McNemar $p=0.000224$。去除校准策略对成对 accuracy 的影响为 $+0.0105$，95\% CI 为 $[0.0009,0.0199]$，McNemar $p=0.034690$。去除偏差模块时完整模型低于该消融变体，说明该模块不应作为提升总准确率的充分证据。

        \begin{{table}}[t]
        \caption{{关键消融的配对显著性检验。$\Delta$ 表示完整模型减去对应消融变体。}}
        \label{{tab:significance}}
        \centering
        \scriptsize
        \resizebox{{\columnwidth}}{{!}}{{%
        \begin{{tabular}}{{llrrlr}}
        \toprule
        变体 & 任务头 & $n$ & $\Delta$Acc. & 95\% CI & $p$ \\
        \midrule
        {latex_table_significance()}
        \bottomrule
        \end{{tabular}}}}
        \end{{table}}

        \subsection{{校准方法比较}}
        表~\ref{{tab:calibration}} 报告成对偏好头的扩展校准对比。Platt scaling 在测试集上获得最低 ECE 0.0194，isotonic regression 获得最高 accuracy 0.7692 和较低 ECE 0.0263，但 MCE 较高。本文正式结果采用温度缩放和任务策略组合，因为其参数更少、可解释性更高，并与开发集选择目标一致。该结果说明，若部署场景优先优化置信度校准，可以进一步考虑 Platt 或 isotonic 方法，但需在更大开发集上评估过拟合风险。

        \begin{{table}}[t]
        \caption{{成对偏好头校准方法对比，测试集结果。Coverage 仅适用于 conformal prediction。}}
        \label{{tab:calibration}}
        \centering
        \scriptsize
        \resizebox{{\columnwidth}}{{!}}{{%
        \begin{{tabular}}{{lrrrrrr}}
        \toprule
        方法 & Acc. & ECE & MCE & Brier & NLL & Coverage \\
        \midrule
        {latex_table_calibration()}
        \bottomrule
        \end{{tabular}}}}
        \end{{table}}

        \subsection{{风险覆盖与复核价值}}
        图~\ref{{fig:risk_coverage}} 给出测试集风险覆盖曲线。成对头在复核约 49.95\% 样本时捕获 85.11\% 错误，自动接受部分 accuracy 达到 0.9260；事实性头在复核约 49.90\% 样本时捕获 75.44\% 错误，自动接受部分 accuracy 达到 0.8848。该结果表明，\model 的风险分数可用于实际评审流程中的样本优先级排序。

        \begin{{figure}}[t]
          \centering
          \includegraphics[width=\columnwidth]{{fig5_risk_coverage.pdf}}
          \caption{{风险阈值复核曲线。横轴为进入人工复核队列的样本比例，纵轴为被复核队列捕获的错误比例。}}
          \label{{fig:risk_coverage}}
        \end{{figure}}

        \subsection{{证据错误子类型分析}}
        RAGTruth 测试子集包含 372 条事实性样本，accuracy 为 0.6962，macro-F1 为 0.6363，ECE 为 0.0360，review rate 为 0.6909。图~\ref{{fig:evidence_subtypes}} 显示，numeric evidence gap、low support sentence ratio、low support anchor sentence ratio 等子类型具有较高错误率，同时复核捕获率通常超过 0.87。该结果支持本文核心判断：事实性可靠性的主要提升来自证据增强，而不是单纯依赖基座评判器置信度。

        \begin{{figure*}}[t]
          \centering
          \includegraphics[width=0.94\textwidth]{{fig6_evidence_subtypes.pdf}}
          \caption{{事实性证据错误子类型诊断。错误率反映该子类型下预测失败的比例，复核捕获率反映风险策略捕获错误的能力。}}
          \label{{fig:evidence_subtypes}}
        \end{{figure*}}

        \subsection{{分数据集结果与偏差分析}}
        分数据集结果显示，事实性头在 ARES-NQ 测试切片上达到 0.9912 accuracy 与 0.9911 macro-F1，在 RAGTruth 上为 0.6962 accuracy 与 0.6363 macro-F1，说明 RAGTruth 是当前事实性任务的主要困难来源。成对头在 OffsetBias 上达到 0.8690 accuracy 与 0.8690 macro-F1，在 HelpSteer2 上为 0.6678 accuracy 与 0.5373 macro-F1，在 MT-Bench 与 PandaLM 切片上表现较弱。顺序置换探针共 60 条样本，整体 swap consistency rate 为 0.4667，说明困难样本仍具有明显顺序敏感性，后续需扩大顺序置换训练和评估覆盖。

        偏差子组结果进一步显示，position 组 accuracy 为 0.6962、format 组为 0.7336、rubric sensitivity 组为 0.7500、length 组为 0.8287。position 和 format 组平均偏差风险分别为 0.5474 和 0.5232，review rate 分别为 0.3038 和 0.2664。这一结果说明偏差风险可识别较难子组，但将偏差特征直接作为决策特征并不一定提高总体准确率。

        \section{{讨论}}
        \subsection{{为何证据增强有效}}
        事实性错误通常不是全局语义相似度不足，而是局部证据与回答之间存在实体、数值、日期、否定或比较关系错配。式~\eqref{{eq:support}} 和式~\eqref{{eq:evidencerisk}} 将这些局部错配显式纳入特征空间，使 softmax 头能够学习“回答整体相似但关键事实不被支持”的模式。证据特征组消融显示，从 overlap-only 到加入 numeric/date/entity 特征，accuracy 由 0.7175 提升至 0.7361；加入 sentence/local-risk 后进一步提升至 0.7629；再加入加权校准后达到 0.7649 accuracy 和 0.7405 macro-F1。

        \subsection{{校准与 Tie 策略的作用}}
        成对偏好任务中的 Tie 标签常对应边界样本。未校准模型在开发集的 Tie recall 为 0.3636，完整模型提升到 0.6558；测试集上未校准 Tie recall 为 0.3923，完整模型为 0.5231。虽然 Tie 策略可能牺牲一部分总体 accuracy，但能提升宏平均指标和边界样本可解释性。对评审系统而言，将“不确定但需复核”的样本标记为 Tie 或 review，比强制给出 A>B 或 B>A 更符合质量控制目标。

        \subsection{{方法边界}}
        本文有三个明确边界。第一，\model 不是新训练的大语言模型，无法替代基座评判器的语言理解能力。第二，证据模块是确定性特征抽取与统计校准，不等价于完整的原子声明级事实核验。第三，部分数据集切片样本量较小，例如 JudgeBench、WikiEval 和 PandaLM 的测试样本较少，分数据集结论应作为诊断而非泛化保证。

        \section{{结论}}
        本文提出 \model，一个基于真实 M-Prometheus-3B 输出的偏差感知证据增强评判校准框架。该框架通过轻量 softmax 头、证据风险特征、偏差风险特征和开发集校准策略，在 \dataset 上实现了可复核的生成式内容评估。冻结测试结果显示，\model 在成对偏好与事实性任务上均取得较稳定的 accuracy、macro-F1 和 ECE；消融实验确认，证据增强是事实性可靠性的主要来源，校准和 Tie 策略是成对偏好稳定性的关键。未来工作将扩大顺序置换评估、引入原子声明级事实标注，并在更多中文专业领域数据上验证风险复核策略。

        \section*{{数据与代码可用性}}
        处理后的 \dataset、源数据 manifest、预处理脚本、校验报告、模型输出表和图表应在投稿前归档至可签发 DOI 的仓库。HelpSteer2、OASST1、OffsetBias 和 RAGTruth 的再分发许可已通过项目审计；RewardBench 因混合子集许可限制未纳入正式训练。正式结果使用的关键路径包括 \path{{datasets/processed/bea_judge_cleaned_10000.json}}、\path{{datasets/model_outputs/bea_judge_20260521_110114/validation_report.json}} 和 \path{{datasets/model_outputs/sci_tables_v2_20260521_110114/}}。

        \section*{{致谢}}
        作者感谢公开数据集和开放评判模型社区。本文草稿使用人工智能工具辅助组织语言、生成 LaTeX 和绘制矢量图；所有实验事实、结论边界和投稿版本仍需由作者逐项核验并承担责任。

        \bibliographystyle{{IEEEtran}}
        \bibliography{{references}}

        \end{{document}}
        """
    ).strip() + "\n"


def make_manifest() -> str:
    return dedent(
        """
        # BEA-Judge IEEE Paper Artifact Manifest

        This directory was generated from project experiment outputs, not from any existing manuscript draft.

        ## Primary inputs

        - configs/experiment.json
        - datasets/model_outputs/bea_judge_20260521_110114/validation_report.json
        - datasets/model_outputs/sci_tables_v2_20260521_110114/*.csv
        - datasets/model_outputs/sci_tables_extended_20260522/*.csv
        - datasets/data_availability_v2.md
        - datasets/evidence_fact_report.md

        ## Outputs

        - main.tex
        - references.bib
        - figures/*.pdf
        - figures/*.svg

        ## Notes

        - Figures are generated as vector PDF and SVG.
        - The final PDF should be built with XeLaTeX/BibTeX or latexmk using XeLaTeX.
        - Author metadata, affiliation, funding, and DOI repository identifiers remain placeholders.
        """
    ).strip() + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_figures()
    write_text(OUT / "main.tex", make_tex())
    write_text(OUT / "references.bib", make_bib())
    write_text(OUT / "ARTIFACT_MANIFEST.md", make_manifest())
    print(OUT)


if __name__ == "__main__":
    main()
