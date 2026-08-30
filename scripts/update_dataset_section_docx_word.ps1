param(
    [Parameter(Mandatory = $true)]
    [string]$DocxPath
)

$StatusParagraph = "当前实现状态：截至 2026-05-13，中文专业标注集已扩充至 1000 条，其中 open_qa 400 条、pairwise_bias 400 条、factuality_rag 200 条；输出文件包括 datasets/processed/chinese_professional_annotated_1000.json、datasets/processed/chinese_professional_annotated_latest.json、datasets/splits_zh/train|dev|test.json、datasets/chinese_dataset_statistics.json 和 datasets/chinese_annotation_report.json。"

$ImplementationParagraphs = @(
    $StatusParagraph,
    "数据源整合规则：公开数据继续保留 MT-Bench、PandaLM、JudgeBench、WikiEval 与 ARES 的来源字段、原始记录标识和标签映射；自建中文数据统一写入 self_built_chinese_annotation 来源，并通过 task_type、language、split、human_score、metadata.field_contract 与 metadata.missing_reason 对齐到同一结构标准。",
    "缺失值处理规则：必填文本字段 prompt、answer_a、answer_b 在相应任务契约下必须为非空；非必需的 context、reference 不再使用空字符串占位，统一使用 null，并在 metadata.missing_reason 中记录 not_required_for_task。factuality_rag 样本必须保留 context，reference 建议保留。",
    "质量门槛：生成后必须通过样本数、任务分布、语言分布、重复样本、split 泄漏、score_format/scoring_system 完整性、空字符串扫描和中文标注一致性检查；中文标注报告需记录 Cohen's kappa、仲裁数量、标签分布和领域分布。"
)

function Get-PlainCellText($Cell) {
    return (($Cell.Range.Text -replace "[`r`a]+$", "").Trim())
}

function Set-ParagraphByPrefix($Doc, [string]$Prefix, [string]$Replacement) {
    foreach ($Paragraph in $Doc.Paragraphs) {
        $Text = $Paragraph.Range.Text.Trim()
        if ($Text.StartsWith($Prefix)) {
            $Paragraph.Range.Text = $Replacement + "`r"
            return $true
        }
    }
    return $false
}

function Set-DataSourceTable($Doc) {
    foreach ($Table in $Doc.Tables) {
        if ($Table.Rows.Count -lt 2 -or $Table.Columns.Count -lt 5) {
            continue
        }
        $Header0 = Get-PlainCellText $Table.Cell(1, 1)
        $Header1 = Get-PlainCellText $Table.Cell(1, 2)
        $Header2 = Get-PlainCellText $Table.Cell(1, 3)
        if ($Header0 -ne "数据类型" -or $Header1 -ne "推荐数据源" -or $Header2 -ne "建议样本规模") {
            continue
        }

        for ($RowIndex = 2; $RowIndex -le $Table.Rows.Count; $RowIndex++) {
            $Label = Get-PlainCellText $Table.Cell($RowIndex, 1)
            if ($Label -eq "开放式回答质量") {
                $Table.Cell($RowIndex, 3).Range.Text = "300-800；核心公开集按实验目标抽样，当前可复现构建支持每任务 400/800 条。"
                $Table.Cell($RowIndex, 5).Range.Text = "明确是否使用官方划分；统一标签为 A>B、B>A、Tie；保留 source_url 与原始记录标识。"
            }
            elseif ($Label -eq "Judge 偏差样本") {
                $Table.Cell($RowIndex, 3).Range.Text = "300-800；需覆盖 position、length、format、rubric_sensitivity 扰动。"
                $Table.Cell($RowIndex, 5).Range.Text = "swap 后必须映射回原始实际答案；同一 parent_id 的扰动样本不得跨 split 泄漏。"
            }
            elseif ($Label -eq "RAG/事实性评价") {
                $Table.Cell($RowIndex, 3).Range.Text = "300-800；优先保留带 context 的样本。"
                $Table.Cell($RowIndex, 5).Range.Text = "context 为 factuality_rag 必填字段；claim/evidence 标签和截断策略需可追溯。"
            }
            elseif ($Label -eq "中文专业场景") {
                $Table.Cell($RowIndex, 2).Range.Text = "管理、财经、政策、科研方法问答自建样本；含开放问答、偏差扰动和事实性/RAG。"
                $Table.Cell($RowIndex, 3).Range.Text = "1000（已生成：open_qa 400、pairwise_bias 400、factuality_rag 200）。"
                $Table.Cell($RowIndex, 4).Range.Text = "增强中文与专业场景适用性，并用于验证偏差、证据和校准模块在中文任务上的稳定性。"
                $Table.Cell($RowIndex, 5).Range.Text = "需提供标注规范、匿名化样例、字段契约；可选字段缺失使用 null，避免空字符串。"
            }
        }
        return $true
    }
    return $false
}

function Insert-StatusBlock($Doc) {
    foreach ($Paragraph in $Doc.Paragraphs) {
        if ($Paragraph.Range.Text.Trim() -eq $StatusParagraph) {
            return $false
        }
    }

    foreach ($Paragraph in $Doc.Paragraphs) {
        if ($Paragraph.Range.Text.Trim() -eq "4.2 预处理流程") {
            $Range = $Paragraph.Range.Duplicate
            $Range.Collapse(1)
            $Range.InsertBefore(($ImplementationParagraphs -join "`r") + "`r")
            return $true
        }
    }
    return $false
}

$ResolvedPath = (Resolve-Path -LiteralPath $DocxPath).Path

try {
    $Word = [Runtime.InteropServices.Marshal]::GetActiveObject("Word.Application")
}
catch {
    $Word = New-Object -ComObject Word.Application
}

$Word.DisplayAlerts = 0
$Doc = $null
foreach ($OpenDoc in $Word.Documents) {
    if ($OpenDoc.FullName -ieq $ResolvedPath) {
        $Doc = $OpenDoc
        break
    }
}
if ($null -eq $Doc) {
    $Doc = $Word.Documents.Open($ResolvedPath)
}

[void](Set-ParagraphByPrefix $Doc "数据设计需覆盖" "数据设计需覆盖开放式回答质量、评价偏差、事实性/RAG 和中文专业场景四类任务，以保证模型结论不是单一数据集上的偶然表现。公开数据优先保证可复现性，自建中文数据用于体现应用价值和场景适配；当前中文专业数据已按统一结构标准扩充至 1000 条。")
[void](Set-ParagraphByPrefix $Doc "完整性检查：" "完整性检查：删除必填字段缺失、乱码、重复样本、明显无效回答和过短回答，并输出剔除日志；非必填字段缺失使用 null 表示，不使用空字符串或空格占位。")
[void](Set-ParagraphByPrefix $Doc "字段统一：" "字段统一：将 instruction/query/question 统一为 prompt，将 response/answer/completion 统一为 answer_a/answer_b，并用 field_contract 明确各任务的 context、reference、answer_b 要求。")
[void](Set-ParagraphByPrefix $Doc "数据统计：" "数据统计：报告样本数、任务类型、语言、平均 prompt/answer/context 长度、标签分布、Tie 比例、claim 数量、空字符串扫描结果、字段契约覆盖率和标注一致性。")
[void](Set-ParagraphByPrefix $Doc "每条中文自建样本建议" "中文自建样本按 1000 条规模组织，每条建议由 2 名标注者独立评分；分歧超过 2 分或偏好标签冲突时由第 3 人仲裁，并保留 annotator_votes、arbiter_label 与 agreement 字段。")
[void](Set-ParagraphByPrefix $Doc "中文专业数据样本规模有限" "中文专业数据已由初始小规模样本扩充至 1000 条，但其外部有效性仍需在更多行业文本和真实应用场景中验证。")
[void](Set-DataSourceTable $Doc)
[void](Insert-StatusBlock $Doc)

try {
    $Doc.Save()
    Write-Output $ResolvedPath
}
catch {
    $Directory = Split-Path -Parent $ResolvedPath
    $BaseName = [System.IO.Path]::GetFileNameWithoutExtension($ResolvedPath)
    $FallbackPath = Join-Path $Directory ($BaseName + "_updated.docx")
    $Doc.SaveAs2($FallbackPath)
    Write-Output $FallbackPath
}
