$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$paperDir = Get-ChildItem -LiteralPath $repoRoot -Directory |
    Where-Object { $_.Name -like "QLoRA-BEA-Judge_SCI*20260531" } |
    Select-Object -First 1

if (-not $paperDir) {
    throw "Cannot locate the QLoRA-BEA-Judge SCI paper directory."
}

$vsdxDir = Join-Path $paperDir.FullName "vsdx"
New-Item -ItemType Directory -Force -Path $vsdxDir | Out-Null
$outVsdx = Join-Path $vsdxDir "fig1_bea_judge_framework_ieee_final.vsdx"

$visio = $null
$doc = $null

function Set-CellFormula {
    param($Shape, [string] $Cell, [string] $Formula)
    try {
        $Shape.CellsU($Cell).FormulaU = $Formula
    } catch {
        # Some Visio stencils omit style cells; keep generation compatible.
    }
}

function Set-TextStyle {
    param($Shape, [double] $SizePt = 7.5, [bool] $Bold = $false)
    Set-CellFormula $Shape "Char.Size" ("{0} pt" -f $SizePt)
    Set-CellFormula $Shape "Para.HorzAlign" "1"
    Set-CellFormula $Shape "VerticalAlign" "1"
    Set-CellFormula $Shape "TxtMarginLeft" "0.04 in"
    Set-CellFormula $Shape "TxtMarginRight" "0.04 in"
    Set-CellFormula $Shape "TxtMarginTop" "0.03 in"
    Set-CellFormula $Shape "TxtMarginBottom" "0.03 in"
    if ($Bold) {
        Set-CellFormula $Shape "Char.Style" "1"
    } else {
        Set-CellFormula $Shape "Char.Style" "0"
    }
}

function Set-BoxStyle {
    param($Shape, [string] $Fill = "RGB(255,255,255)", [double] $WeightPt = 0.75, [bool] $Dashed = $false)
    Set-CellFormula $Shape "FillForegnd" $Fill
    Set-CellFormula $Shape "FillPattern" "1"
    Set-CellFormula $Shape "LineColor" "RGB(0,0,0)"
    Set-CellFormula $Shape "LineWeight" ("{0} pt" -f $WeightPt)
    Set-CellFormula $Shape "Rounding" "0.06 in"
    if ($Dashed) {
        Set-CellFormula $Shape "LineColor" "RGB(90,90,90)"
        Set-CellFormula $Shape "LinePattern" "2"
    } else {
        Set-CellFormula $Shape "LinePattern" "1"
    }
}

function Add-Text {
    param($Page, [double] $X, [double] $Top, [double] $W, [double] $H, [string] $Text, [double] $SizePt = 8, [bool] $Bold = $false)
    $shape = $Page.DrawRectangle($X, $script:pageH - $Top - $H, $X + $W, $script:pageH - $Top)
    $shape.Text = $Text
    Set-CellFormula $shape "FillPattern" "0"
    Set-CellFormula $shape "LinePattern" "0"
    Set-TextStyle $shape $SizePt $Bold
    return $shape
}

function Add-GroupBox {
    param($Page, [double] $X, [double] $Top, [double] $W, [double] $H, [string] $Text)
    $shape = $Page.DrawRectangle($X, $script:pageH - $Top - $H, $X + $W, $script:pageH - $Top)
    $shape.Text = $Text
    Set-BoxStyle $shape "RGB(255,255,255)" 0.5 $true
    Set-CellFormula $shape "FillPattern" "0"
    Set-TextStyle $shape 7 $true
    Set-CellFormula $shape "VerticalAlign" "0"
    return $shape
}

function Add-Box {
    param($Page, [hashtable] $Node)
    $shape = $Page.DrawRectangle($Node.X, $script:pageH - $Node.Top - $Node.H, $Node.X + $Node.W, $script:pageH - $Node.Top)
    $shape.Text = $Node.Text
    Set-BoxStyle $shape $Node.Fill $Node.Weight $false
    Set-TextStyle $shape $Node.FontSize $Node.Bold
    $script:nodes[$Node.Id] = @{
        X = [double] $Node.X
        Top = [double] $Node.Top
        W = [double] $Node.W
        H = [double] $Node.H
    }
    return $shape
}

function Get-Point {
    param([string] $NodeId, [ValidateSet("L", "R", "T", "B")] [string] $Side)
    $n = $script:nodes[$NodeId]
    $left = [double] $n.X
    $right = $left + [double] $n.W
    $top = [double] $n.Top
    $bottom = $top + [double] $n.H
    $cx = ($left + $right) / 2.0
    $cyTop = ($top + $bottom) / 2.0
    switch ($Side) {
        "L" { return [pscustomobject]@{ X = $left; Y = ($script:pageH - $cyTop) } }
        "R" { return [pscustomobject]@{ X = $right; Y = ($script:pageH - $cyTop) } }
        "T" { return [pscustomobject]@{ X = $cx; Y = ($script:pageH - $top) } }
        "B" { return [pscustomobject]@{ X = $cx; Y = ($script:pageH - $bottom) } }
    }
}

function Add-LineRaw {
    param($Page, [double] $X1, [double] $Y1, [double] $X2, [double] $Y2, [bool] $Arrow = $true, [bool] $Dashed = $false)
    $line = $Page.DrawLine($X1, $Y1, $X2, $Y2)
    Set-CellFormula $line "LineColor" "RGB(0,0,0)"
    Set-CellFormula $line "LineWeight" "0.75 pt"
    if ($Arrow) {
        Set-CellFormula $line "EndArrow" "4"
        Set-CellFormula $line "EndArrowSize" "2"
    }
    if ($Dashed) {
        Set-CellFormula $line "LinePattern" "2"
    }
    return $line
}

function Add-Arrow {
    param($Page, [string] $From, [string] $FromSide, [string] $To, [string] $ToSide, [bool] $Dashed = $false)
    $p1 = Get-Point $From $FromSide
    $p2 = Get-Point $To $ToSide
    return Add-LineRaw $Page $p1.X $p1.Y $p2.X $p2.Y $true $Dashed
}

function Add-RoutedArrow {
    param($Page, [string] $From, [string] $FromSide, [string] $To, [string] $ToSide, [array] $Waypoints, [bool] $Dashed = $false)
    $p1 = Get-Point $From $FromSide
    $p2 = Get-Point $To $ToSide
    $points = @($p1) + $Waypoints + @($p2)
    for ($i = 0; $i -lt ($points.Count - 1); $i++) {
        $isLast = ($i -eq $points.Count - 2)
        Add-LineRaw $Page $points[$i].X $points[$i].Y $points[$i + 1].X $points[$i + 1].Y $isLast $Dashed | Out-Null
    }
}

try {
    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $true
    $visio.AlertResponse = 7

    $doc = $visio.Documents.Add("")
    $page = $visio.ActivePage
    $page.Name = "IEEE final experimental workflow"

    $script:pageW = 16.8
    $script:pageH = 7.6
    $page.PageSheet.CellsU("PageWidth").FormulaU = ("{0} in" -f $script:pageW)
    $page.PageSheet.CellsU("PageHeight").FormulaU = ("{0} in" -f $script:pageH)
    $script:nodes = @{}

    Add-Text $page 0.3 0.18 16.2 0.28 "Fig. 1. Experimental workflow of the QLoRA-BEA-Judge framework" 10 $true | Out-Null

    Add-GroupBox $page 0.45 0.72 5.60 1.25 "A. Data construction" | Out-Null
    Add-GroupBox $page 6.25 0.72 5.60 1.25 "B. QLoRA training" | Out-Null
    Add-GroupBox $page 12.05 0.72 4.30 1.25 "C. Development selection" | Out-Null
    Add-GroupBox $page 0.45 2.68 15.90 1.42 "D. Inference and calibrated judgment" | Out-Null
    Add-GroupBox $page 0.45 4.88 15.90 1.40 "E. Locked evaluation and reporting" | Out-Null

    $fillWhite = "RGB(255,255,255)"
    $fillLight = "RGB(242,242,242)"
    $fillMid = "RGB(226,226,226)"
    $defs = @(
        @{ Id="raw"; X=0.70; Top=1.12; W=1.50; H=0.58; Text="(1) Pairwise samples`nprompt, response A/B"; Fill=$fillWhite; Weight=0.75; FontSize=7.2; Bold=$false },
        @{ Id="gates"; X=2.50; Top=1.12; W=1.50; H=0.58; Text="(2) Quality gates`nlicense, bias, factuality"; Fill=$fillLight; Weight=0.75; FontSize=7.2; Bold=$false },
        @{ Id="split"; X=4.30; Top=1.12; W=1.50; H=0.58; Text="(3) Stratified split`ntrain / dev / test"; Fill=$fillWhite; Weight=0.75; FontSize=7.2; Bold=$false },
        @{ Id="backbone"; X=6.50; Top=1.12; W=1.50; H=0.58; Text="(4) 3B judge`nbackbone"; Fill=$fillWhite; Weight=0.75; FontSize=7.2; Bold=$false },
        @{ Id="qlora"; X=8.30; Top=1.12; W=1.50; H=0.58; Text="(5) QLoRA SFT`npairwise objective"; Fill=$fillMid; Weight=0.9; FontSize=7.2; Bold=$true },
        @{ Id="ckpt"; X=10.10; Top=1.12; W=1.50; H=0.58; Text="(6) Seeds and`ncheckpoints"; Fill=$fillWhite; Weight=0.75; FontSize=7.2; Bold=$false },
        @{ Id="dev"; X=12.30; Top=1.12; W=1.50; H=0.58; Text="(7) Dev metrics`naccuracy, tie recall"; Fill=$fillWhite; Weight=0.75; FontSize=7.2; Bold=$false },
        @{ Id="calib"; X=14.10; Top=1.12; W=1.70; H=0.58; Text="(8) Calibration`ntemperature, thresholds"; Fill=$fillLight; Weight=0.75; FontSize=7.2; Bold=$false },
        @{ Id="score"; X=1.10; Top=3.08; W=1.70; H=0.62; Text="Base judge scoring`nscore_A, score_B, margin"; Fill=$fillWhite; Weight=0.75; FontSize=7.0; Bold=$false },
        @{ Id="bias"; X=3.45; Top=3.08; W=1.70; H=0.62; Text="Bias-aware features`nposition, length, format"; Fill=$fillLight; Weight=0.75; FontSize=7.0; Bold=$false },
        @{ Id="fact"; X=5.80; Top=3.08; W=1.70; H=0.62; Text="Evidence factuality`nentity, number, date gaps"; Fill=$fillWhite; Weight=0.75; FontSize=7.0; Bold=$false },
        @{ Id="fusion"; X=8.15; Top=3.08; W=1.70; H=0.62; Text="Fusion head`nprobability and confidence"; Fill=$fillMid; Weight=0.9; FontSize=7.0; Bold=$true },
        @{ Id="tie"; X=10.50; Top=3.08; W=1.70; H=0.62; Text="Tie rescue`ndev-only policy search"; Fill=$fillLight; Weight=0.75; FontSize=7.0; Bold=$false },
        @{ Id="output"; X=12.85; Top=3.08; W=2.20; H=0.62; Text="Structured output`nlabel, risk score, review flag"; Fill=$fillWhite; Weight=0.75; FontSize=7.0; Bold=$false },
        @{ Id="lock"; X=1.30; Top=5.28; W=1.85; H=0.62; Text="Locked test split`nused once"; Fill=$fillWhite; Weight=0.9; FontSize=7.0; Bold=$true },
        @{ Id="internal"; X=4.05; Top=5.28; W=1.85; H=0.62; Text="Internal runs`n3-seed mean +/- std"; Fill=$fillLight; Weight=0.75; FontSize=7.0; Bold=$false },
        @{ Id="external"; X=6.80; Top=5.28; W=1.85; H=0.62; Text="External baselines`nsingle full-test run"; Fill=$fillLight; Weight=0.75; FontSize=7.0; Bold=$false },
        @{ Id="metrics"; X=9.55; Top=5.28; W=1.85; H=0.62; Text="Report metrics`naccuracy, ECE, risk"; Fill=$fillWhite; Weight=0.75; FontSize=7.0; Bold=$false },
        @{ Id="claim"; X=12.30; Top=5.28; W=1.70; H=0.62; Text="Final tables`nand figures"; Fill=$fillWhite; Weight=0.75; FontSize=7.0; Bold=$false }
    )
    foreach ($d in $defs) { Add-Box $page $d | Out-Null }

    foreach ($e in @(
        @("raw", "R", "gates", "L"), @("gates", "R", "split", "L"), @("split", "R", "backbone", "L"),
        @("backbone", "R", "qlora", "L"), @("qlora", "R", "ckpt", "L"), @("ckpt", "R", "dev", "L"),
        @("dev", "R", "calib", "L"), @("score", "R", "bias", "L"), @("bias", "R", "fact", "L"),
        @("fact", "R", "fusion", "L"), @("fusion", "R", "tie", "L"), @("tie", "R", "output", "L"),
        @("lock", "R", "internal", "L"), @("internal", "R", "external", "L"), @("external", "R", "metrics", "L"),
        @("metrics", "R", "claim", "L")
    )) {
        Add-Arrow $page $e[0] $e[1] $e[2] $e[3] $false | Out-Null
    }

    Add-RoutedArrow $page "calib" "B" "fusion" "T" @(
        [pscustomobject]@{ X=15.20; Y=5.24 },
        [pscustomobject]@{ X=15.20; Y=4.52 }
    ) $true
    Add-RoutedArrow $page "ckpt" "B" "score" "L" @(
        [pscustomobject]@{ X=10.85; Y=5.22 },
        [pscustomobject]@{ X=0.95; Y=5.22 },
        [pscustomobject]@{ X=0.95; Y=4.21 }
    ) $false
    Add-RoutedArrow $page "split" "B" "lock" "L" @(
        [pscustomobject]@{ X=5.05; Y=5.24 },
        [pscustomobject]@{ X=0.32; Y=5.24 },
        [pscustomobject]@{ X=0.32; Y=2.01 }
    ) $true
    Add-RoutedArrow $page "output" "B" "metrics" "T" @(
        [pscustomobject]@{ X=13.95; Y=2.82 },
        [pscustomobject]@{ X=13.95; Y=1.99 }
    ) $false

    Add-Text $page 0.70 6.78 2.70 0.30 "solid arrow: model/data flow" 6.8 $false | Out-Null
    Add-Text $page 4.40 6.78 3.80 0.30 "dashed arrow: protocol constraints selected on dev" 6.8 $false | Out-Null
    Add-Text $page 5.60 7.18 5.60 0.28 "Test discipline: thresholds and Tie rescue are fixed before the locked test report." 6.8 $false | Out-Null

    if (Test-Path -LiteralPath $outVsdx) {
        Copy-Item -LiteralPath $outVsdx -Destination ($outVsdx + ".bak") -Force
    }
    $doc.SaveAs($outVsdx)
    $doc.Saved = $true

    [pscustomobject]@{ VSDX = $outVsdx } | ConvertTo-Json -Compress
} finally {
    if ($doc) {
        $doc.Saved = $true
    }
}
