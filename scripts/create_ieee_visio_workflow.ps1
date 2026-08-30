$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$paperDir = Get-ChildItem -LiteralPath $repoRoot -Directory |
    Where-Object { $_.Name -like "QLoRA-BEA-Judge_SCI*20260531" } |
    Select-Object -First 1

if (-not $paperDir) {
    throw "Cannot locate the QLoRA-BEA-Judge SCI paper directory."
}

$figDir = Join-Path $paperDir.FullName "figures"
$vsdxDir = Join-Path $paperDir.FullName "vsdx"
New-Item -ItemType Directory -Force -Path $figDir, $vsdxDir | Out-Null

$outVsdx = Join-Path $vsdxDir "fig1_bea_judge_framework_ieee.vsdx"
$outSvg = Join-Path $figDir "fig1_bea_judge_framework_ieee.svg"
$outPng = Join-Path $figDir "fig1_bea_judge_framework_ieee.png"

$visio = $null
$doc = $null

function Set-CellFormula {
    param(
        [Parameter(Mandatory = $true)] $Shape,
        [Parameter(Mandatory = $true)] [string] $Cell,
        [Parameter(Mandatory = $true)] [string] $Formula
    )
    try {
        $Shape.CellsU($Cell).FormulaU = $Formula
    } catch {
        # Older Visio templates can omit some cells; keep generation robust.
    }
}

function Set-TextStyle {
    param(
        [Parameter(Mandatory = $true)] $Shape,
        [double] $SizePt = 7.5,
        [bool] $Bold = $false
    )
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
    param(
        [Parameter(Mandatory = $true)] $Shape,
        [string] $Fill = "RGB(255,255,255)",
        [string] $Line = "RGB(0,0,0)",
        [double] $WeightPt = 0.75,
        [bool] $Dashed = $false
    )
    Set-CellFormula $Shape "FillForegnd" $Fill
    Set-CellFormula $Shape "FillPattern" "1"
    Set-CellFormula $Shape "LineColor" $Line
    Set-CellFormula $Shape "LineWeight" ("{0} pt" -f $WeightPt)
    if ($Dashed) {
        Set-CellFormula $Shape "LinePattern" "2"
    } else {
        Set-CellFormula $Shape "LinePattern" "1"
    }
    Set-CellFormula $Shape "Rounding" "0.06 in"
}

function Add-Box {
    param(
        [Parameter(Mandatory = $true)] $Page,
        [Parameter(Mandatory = $true)] [hashtable] $Node
    )
    $x1 = $Node.X
    $y1 = $script:pageH - $Node.Top - $Node.H
    $x2 = $Node.X + $Node.W
    $y2 = $script:pageH - $Node.Top
    $shape = $Page.DrawRectangle($x1, $y1, $x2, $y2)
    $shape.Text = $Node.Text
    Set-BoxStyle $shape $Node.Fill "RGB(0,0,0)" $Node.Weight $false
    Set-TextStyle $shape $Node.FontSize $Node.Bold
    $script:nodes[$Node.Id] = @{
        Shape = $shape
        X = $Node.X
        Top = $Node.Top
        W = $Node.W
        H = $Node.H
    }
    return $shape
}

function Add-GroupBox {
    param(
        [Parameter(Mandatory = $true)] $Page,
        [double] $X,
        [double] $Top,
        [double] $W,
        [double] $H,
        [string] $Text
    )
    $shape = $Page.DrawRectangle($X, $script:pageH - $Top - $H, $X + $W, $script:pageH - $Top)
    $shape.Text = $Text
    Set-BoxStyle $shape "RGB(255,255,255)" "RGB(90,90,90)" 0.5 $true
    Set-CellFormula $shape "FillPattern" "0"
    Set-TextStyle $shape 7 $true
    Set-CellFormula $shape "VerticalAlign" "0"
    return $shape
}

function Add-Text {
    param(
        [Parameter(Mandatory = $true)] $Page,
        [double] $X,
        [double] $Top,
        [double] $W,
        [double] $H,
        [string] $Text,
        [double] $SizePt = 8,
        [bool] $Bold = $false
    )
    $shape = $Page.DrawRectangle($X, $script:pageH - $Top - $H, $X + $W, $script:pageH - $Top)
    $shape.Text = $Text
    Set-CellFormula $shape "FillPattern" "0"
    Set-CellFormula $shape "LinePattern" "0"
    Set-TextStyle $shape $SizePt $Bold
    return $shape
}

function Get-Point {
    param(
        [string] $NodeId,
        [ValidateSet("L", "R", "T", "B", "C")] [string] $Side
    )
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
        "C" { return [pscustomobject]@{ X = $cx; Y = ($script:pageH - $cyTop) } }
    }
}

function Add-LineRaw {
    param(
        [Parameter(Mandatory = $true)] $Page,
        [double] $X1,
        [double] $Y1,
        [double] $X2,
        [double] $Y2,
        [bool] $Arrow = $true,
        [bool] $Dashed = $false,
        [double] $WeightPt = 0.75
    )
    $line = $Page.DrawLine($X1, $Y1, $X2, $Y2)
    Set-CellFormula $line "LineColor" "RGB(0,0,0)"
    Set-CellFormula $line "LineWeight" ("{0} pt" -f $WeightPt)
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
    param(
        [Parameter(Mandatory = $true)] $Page,
        [string] $From,
        [string] $FromSide,
        [string] $To,
        [string] $ToSide,
        [bool] $Dashed = $false
    )
    $p1 = Get-Point $From $FromSide
    $p2 = Get-Point $To $ToSide
    return Add-LineRaw $Page $p1.X $p1.Y $p2.X $p2.Y $true $Dashed 0.75
}

function Add-ElbowArrow {
    param(
        [Parameter(Mandatory = $true)] $Page,
        [string] $From,
        [string] $FromSide,
        [string] $To,
        [string] $ToSide,
        [double] $MidX,
        [bool] $Dashed = $false
    )
    $p1 = Get-Point $From $FromSide
    $p2 = Get-Point $To $ToSide
    Add-LineRaw $Page $p1.X $p1.Y $MidX $p1.Y $false $Dashed 0.75 | Out-Null
    Add-LineRaw $Page $MidX $p1.Y $MidX $p2.Y $false $Dashed 0.75 | Out-Null
    return Add-LineRaw $Page $MidX $p2.Y $p2.X $p2.Y $true $Dashed 0.75
}

function Xml-Escape {
    param([string] $Text)
    return [System.Security.SecurityElement]::Escape($Text)
}

function Add-SvgTextLines {
    param(
        [System.Collections.Generic.List[string]] $Lines,
        [double] $Cx,
        [double] $Y,
        [string[]] $TextLines,
        [double] $FontSize = 13,
        [bool] $BoldFirst = $false
    )
    for ($i = 0; $i -lt $TextLines.Count; $i++) {
        $weight = "400"
        if ($BoldFirst -and $i -eq 0) { $weight = "700" }
        $dy = $Y + ($i * ($FontSize + 3))
        $Lines.Add(('  <text x="{0:F1}" y="{1:F1}" class="txt" font-size="{2:F1}" font-weight="{3}" text-anchor="middle">{4}</text>' -f $Cx, $dy, $FontSize, $weight, (Xml-Escape $TextLines[$i])))
    }
}

function Write-IeeeSvg {
    param([string] $Path)

    $sx = 100.0
    $sy = 100.0
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add('<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="640" viewBox="0 0 1120 640">')
    $lines.Add('  <defs>')
    $lines.Add('    <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 Z" fill="#111"/></marker>')
    $lines.Add('    <style><![CDATA[')
    $lines.Add('      .txt { font-family: "Times New Roman", SimSun, serif; fill: #111; }')
    $lines.Add('      .box { stroke: #111; stroke-width: 1.1; rx: 6; }')
    $lines.Add('      .group { fill: none; stroke: #555; stroke-width: 0.9; stroke-dasharray: 6 4; rx: 8; }')
    $lines.Add('      .edge { fill: none; stroke: #111; stroke-width: 1.1; marker-end: url(#arrow); }')
    $lines.Add('      .edge-soft { fill: none; stroke: #111; stroke-width: 1.0; stroke-dasharray: 5 4; marker-end: url(#arrow); }')
    $lines.Add('    ]]></style>')
    $lines.Add('  </defs>')
    $lines.Add('  <rect width="1120" height="640" fill="#fff"/>')
    $lines.Add('  <text x="560" y="42" class="txt" font-size="18" font-weight="700" text-anchor="middle">Fig. 1. Experimental workflow of the QLoRA-BEA-Judge framework</text>')

    $groups = @(
        @{ X=25; Y=62; W=342; H=125; Text="A. Data construction" },
        @{ X=388; Y=62; W=352; H=125; Text="B. QLoRA training" },
        @{ X=760; Y=62; W=335; H=125; Text="C. Development selection" },
        @{ X=25; Y=218; W=1070; H=142; Text="D. Inference and calibrated judgment" },
        @{ X=25; Y=392; W=1070; H=140; Text="E. Locked evaluation and reporting" }
    )
    foreach ($g in $groups) {
        $lines.Add(('  <rect class="group" x="{0}" y="{1}" width="{2}" height="{3}"/>' -f $g.X, $g.Y, $g.W, $g.H))
        $lines.Add(('  <text x="{0}" y="{1}" class="txt" font-size="13" font-weight="700">{2}</text>' -f ($g.X + 12), ($g.Y + 20), (Xml-Escape $g.Text)))
    }

    $svgNodes = @(
        @{ Id="raw"; X=45; Y=102; W=118; H=58; Text=@("(1) Pairwise samples", "prompt, response A/B"); Fill="#fff" },
        @{ Id="gates"; X=172; Y=102; W=118; H=58; Text=@("(2) Quality gates", "license, bias, factuality"); Fill="#f2f2f2" },
        @{ Id="split"; X=299; Y=102; W=118; H=58; Text=@("(3) Stratified split", "train / dev / test"); Fill="#fff" },
        @{ Id="backbone"; X=408; Y=102; W=118; H=58; Text=@("(4) 3B judge", "backbone"); Fill="#fff" },
        @{ Id="qlora"; X=535; Y=102; W=118; H=58; Text=@("(5) QLoRA SFT", "pairwise objective"); Fill="#e2e2e2" },
        @{ Id="ckpt"; X=662; Y=102; W=118; H=58; Text=@("(6) Seeds and", "checkpoints"); Fill="#fff" },
        @{ Id="dev"; X=782; Y=102; W=118; H=58; Text=@("(7) Dev metrics", "accuracy, tie recall"); Fill="#fff" },
        @{ Id="calib"; X=909; Y=102; W=118; H=58; Text=@("(8) Calibration", "temperature, thresholds"); Fill="#f2f2f2" },
        @{ Id="score"; X=70; Y=258; W=128; H=62; Text=@("Base judge scoring", "score_A, score_B, margin"); Fill="#fff" },
        @{ Id="bias"; X=220; Y=258; W=128; H=62; Text=@("Bias-aware features", "position, length, format"); Fill="#f2f2f2" },
        @{ Id="fact"; X=370; Y=258; W=128; H=62; Text=@("Evidence factuality", "entity, number, date gaps"); Fill="#fff" },
        @{ Id="fusion"; X=520; Y=258; W=128; H=62; Text=@("Fusion head", "probability and confidence"); Fill="#e2e2e2" },
        @{ Id="tie"; X=670; Y=258; W=128; H=62; Text=@("Tie rescue", "dev-only policy search"); Fill="#f2f2f2" },
        @{ Id="output"; X=820; Y=258; W=178; H=62; Text=@("Structured output", "label, risk score, review flag"); Fill="#fff" },
        @{ Id="lock"; X=100; Y=430; W=155; H=62; Text=@("Locked test split", "used once"); Fill="#fff" },
        @{ Id="internal"; X=305; Y=430; W=155; H=62; Text=@("Internal runs", "3-seed mean +/- std"); Fill="#f2f2f2" },
        @{ Id="external"; X=510; Y=430; W=155; H=62; Text=@("External baselines", "single full-test run"); Fill="#f2f2f2" },
        @{ Id="metrics"; X=715; Y=430; W=155; H=62; Text=@("Report metrics", "accuracy, ECE, risk"); Fill="#fff" },
        @{ Id="claim"; X=920; Y=430; W=115; H=62; Text=@("Final tables", "and figures"); Fill="#fff" }
    )

    $index = @{}
    foreach ($n in $svgNodes) {
        $index[$n.Id] = $n
        $lines.Add(('  <rect class="box" x="{0}" y="{1}" width="{2}" height="{3}" fill="{4}"/>' -f $n.X, $n.Y, $n.W, $n.H, $n.Fill))
        Add-SvgTextLines $lines ($n.X + $n.W / 2.0) ($n.Y + 24) $n.Text 12 $true
    }

    function P {
        param([string] $Id, [string] $Side)
        $n = $index[$Id]
        $left = [double] $n.X
        $right = $left + [double] $n.W
        $top = [double] $n.Y
        $bottom = $top + [double] $n.H
        $cx = ($left + $right) / 2.0
        $cy = ($top + $bottom) / 2.0
        switch ($Side) {
            "L" { return [pscustomobject]@{X=$left; Y=$cy} }
            "R" { return [pscustomobject]@{X=$right; Y=$cy} }
            "T" { return [pscustomobject]@{X=$cx; Y=$top} }
            "B" { return [pscustomobject]@{X=$cx; Y=$bottom} }
        }
    }

    function SvgArrow {
        param([string] $From, [string] $FromSide, [string] $To, [string] $ToSide, [bool] $Dashed = $false)
        $a = P $From $FromSide
        $b = P $To $ToSide
        $cls = if ($Dashed) { "edge-soft" } else { "edge" }
        $lines.Add(('  <path class="{0}" d="M {1:F1},{2:F1} L {3:F1},{4:F1}"/>' -f $cls, $a.X, $a.Y, $b.X, $b.Y))
    }
    function SvgElbow {
        param([string] $From, [string] $FromSide, [string] $To, [string] $ToSide, [double] $MidX, [bool] $Dashed = $false)
        $a = P $From $FromSide
        $b = P $To $ToSide
        $cls = if ($Dashed) { "edge-soft" } else { "edge" }
        $lines.Add(('  <path class="{0}" d="M {1:F1},{2:F1} L {3:F1},{2:F1} L {3:F1},{4:F1} L {5:F1},{4:F1}"/>' -f $cls, $a.X, $a.Y, $MidX, $b.Y, $b.X))
    }

    foreach ($e in @(
        @("raw","R","gates","L"), @("gates","R","split","L"), @("split","R","backbone","L"),
        @("backbone","R","qlora","L"), @("qlora","R","ckpt","L"), @("ckpt","R","dev","L"),
        @("dev","R","calib","L"), @("score","R","bias","L"), @("bias","R","fact","L"),
        @("fact","R","fusion","L"), @("fusion","R","tie","L"), @("tie","R","output","L"),
        @("lock","R","internal","L"), @("internal","R","external","L"),
        @("external","R","metrics","L"), @("metrics","R","claim","L")
    )) {
        SvgArrow $e[0] $e[1] $e[2] $e[3] $false
    }
    SvgElbow "calib" "B" "fusion" "T" 967 $true
    SvgElbow "ckpt" "B" "score" "T" 720 $false
    SvgElbow "split" "B" "lock" "T" 205 $true
    SvgElbow "output" "B" "metrics" "T" 908 $false

    $lines.Add('  <line x1="54" y1="574" x2="128" y2="574" class="edge"/>')
    $lines.Add('  <text x="145" y="579" class="txt" font-size="12">model/data flow</text>')
    $lines.Add('  <line x1="322" y1="574" x2="396" y2="574" class="edge-soft"/>')
    $lines.Add('  <text x="413" y="579" class="txt" font-size="12">protocol constraints selected on dev</text>')
    $lines.Add('  <text x="560" y="612" class="txt" font-size="12" text-anchor="middle">Test discipline: thresholds and Tie rescue are fixed before the locked test report.</text>')
    $lines.Add('</svg>')
    [System.IO.File]::WriteAllLines($Path, $lines, [System.Text.UTF8Encoding]::new($false))
}

try {
    $visio = New-Object -ComObject Visio.Application
    $visio.Visible = $true
    $visio.AlertResponse = 7

    $doc = $visio.Documents.Add("")
    $page = $visio.ActivePage
    $page.Name = "IEEE experimental workflow"

    $script:pageW = 11.2
    $script:pageH = 6.4
    $page.PageSheet.CellsU("PageWidth").FormulaU = ("{0} in" -f $script:pageW)
    $page.PageSheet.CellsU("PageHeight").FormulaU = ("{0} in" -f $script:pageH)
    $script:nodes = @{}

    Add-Text $page 0.3 0.16 10.6 0.28 "Fig. 1. Experimental workflow of the QLoRA-BEA-Judge framework" 10 $true | Out-Null

    Add-GroupBox $page 0.25 0.62 3.42 1.25 "A. Data construction" | Out-Null
    Add-GroupBox $page 3.88 0.62 3.52 1.25 "B. QLoRA training" | Out-Null
    Add-GroupBox $page 7.60 0.62 3.35 1.25 "C. Development selection" | Out-Null
    Add-GroupBox $page 0.25 2.18 10.70 1.42 "D. Inference and calibrated judgment" | Out-Null
    Add-GroupBox $page 0.25 3.92 10.70 1.40 "E. Locked evaluation and reporting" | Out-Null

    $boxW = 1.18
    $boxH = 0.58
    $fillWhite = "RGB(255,255,255)"
    $fillLight = "RGB(242,242,242)"
    $fillMid = "RGB(226,226,226)"

    $defs = @(
        @{ Id="raw"; X=0.45; Top=1.02; W=$boxW; H=$boxH; Text="(1) Pairwise samples`nprompt, response A/B"; Fill=$fillWhite; Weight=0.75; FontSize=7.2; Bold=$false },
        @{ Id="gates"; X=1.72; Top=1.02; W=$boxW; H=$boxH; Text="(2) Quality gates`nlicense, bias, factuality"; Fill=$fillLight; Weight=0.75; FontSize=7.2; Bold=$false },
        @{ Id="split"; X=2.99; Top=1.02; W=$boxW; H=$boxH; Text="(3) Stratified split`ntrain / dev / test"; Fill=$fillWhite; Weight=0.75; FontSize=7.2; Bold=$false },
        @{ Id="backbone"; X=4.08; Top=1.02; W=$boxW; H=$boxH; Text="(4) 3B judge`nbackbone"; Fill=$fillWhite; Weight=0.75; FontSize=7.2; Bold=$false },
        @{ Id="qlora"; X=5.35; Top=1.02; W=$boxW; H=$boxH; Text="(5) QLoRA SFT`npairwise objective"; Fill=$fillMid; Weight=0.9; FontSize=7.2; Bold=$true },
        @{ Id="ckpt"; X=6.62; Top=1.02; W=$boxW; H=$boxH; Text="(6) Seeds and`ncheckpoints"; Fill=$fillWhite; Weight=0.75; FontSize=7.2; Bold=$false },
        @{ Id="dev"; X=7.82; Top=1.02; W=$boxW; H=$boxH; Text="(7) Dev metrics`naccuracy, tie recall"; Fill=$fillWhite; Weight=0.75; FontSize=7.2; Bold=$false },
        @{ Id="calib"; X=9.09; Top=1.02; W=$boxW; H=$boxH; Text="(8) Calibration`ntemperature, thresholds"; Fill=$fillLight; Weight=0.75; FontSize=7.2; Bold=$false }
    )
    foreach ($d in $defs) { Add-Box $page $d | Out-Null }

    $defs2 = @(
        @{ Id="score"; X=0.70; Top=2.58; W=1.28; H=0.62; Text="Base judge scoring`nscore_A, score_B, margin"; Fill=$fillWhite; Weight=0.75; FontSize=7.0; Bold=$false },
        @{ Id="bias"; X=2.20; Top=2.58; W=1.28; H=0.62; Text="Bias-aware features`nposition, length, format"; Fill=$fillLight; Weight=0.75; FontSize=7.0; Bold=$false },
        @{ Id="fact"; X=3.70; Top=2.58; W=1.28; H=0.62; Text="Evidence factuality`nentity, number, date gaps"; Fill=$fillWhite; Weight=0.75; FontSize=7.0; Bold=$false },
        @{ Id="fusion"; X=5.20; Top=2.58; W=1.28; H=0.62; Text="Fusion head`nprobability and confidence"; Fill=$fillMid; Weight=0.9; FontSize=7.0; Bold=$true },
        @{ Id="tie"; X=6.70; Top=2.58; W=1.28; H=0.62; Text="Tie rescue`ndev-only policy search"; Fill=$fillLight; Weight=0.75; FontSize=7.0; Bold=$false },
        @{ Id="output"; X=8.20; Top=2.58; W=1.78; H=0.62; Text="Structured output`nlabel, risk score, review flag"; Fill=$fillWhite; Weight=0.75; FontSize=7.0; Bold=$false }
    )
    foreach ($d in $defs2) { Add-Box $page $d | Out-Null }

    $defs3 = @(
        @{ Id="lock"; X=1.00; Top=4.30; W=1.55; H=0.62; Text="Locked test split`nused once"; Fill=$fillWhite; Weight=0.9; FontSize=7.0; Bold=$true },
        @{ Id="internal"; X=3.05; Top=4.30; W=1.55; H=0.62; Text="Internal runs`n3-seed mean +/- std"; Fill=$fillLight; Weight=0.75; FontSize=7.0; Bold=$false },
        @{ Id="external"; X=5.10; Top=4.30; W=1.55; H=0.62; Text="External baselines`nsingle full-test run"; Fill=$fillLight; Weight=0.75; FontSize=7.0; Bold=$false },
        @{ Id="metrics"; X=7.15; Top=4.30; W=1.55; H=0.62; Text="Report metrics`naccuracy, ECE, risk"; Fill=$fillWhite; Weight=0.75; FontSize=7.0; Bold=$false },
        @{ Id="claim"; X=9.20; Top=4.30; W=1.15; H=0.62; Text="Final tables`nand figures"; Fill=$fillWhite; Weight=0.75; FontSize=7.0; Bold=$false }
    )
    foreach ($d in $defs3) { Add-Box $page $d | Out-Null }

    $solidEdges = @(
        @("raw", "R", "gates", "L"),
        @("gates", "R", "split", "L"),
        @("split", "R", "backbone", "L"),
        @("backbone", "R", "qlora", "L"),
        @("qlora", "R", "ckpt", "L"),
        @("ckpt", "R", "dev", "L"),
        @("dev", "R", "calib", "L"),
        @("score", "R", "bias", "L"),
        @("bias", "R", "fact", "L"),
        @("fact", "R", "fusion", "L"),
        @("fusion", "R", "tie", "L"),
        @("tie", "R", "output", "L"),
        @("lock", "R", "internal", "L"),
        @("internal", "R", "external", "L"),
        @("external", "R", "metrics", "L"),
        @("metrics", "R", "claim", "L")
    )
    foreach ($e in $solidEdges) { Add-Arrow $page $e[0] $e[1] $e[2] $e[3] $false | Out-Null }

    Add-ElbowArrow $page "calib" "B" "fusion" "T" 9.67 $true | Out-Null
    Add-ElbowArrow $page "ckpt" "B" "score" "T" 7.20 $false | Out-Null
    Add-ElbowArrow $page "split" "B" "lock" "T" 2.05 $true | Out-Null
    Add-ElbowArrow $page "output" "B" "metrics" "T" 9.08 $false | Out-Null

    Add-Text $page 0.52 5.58 4.10 0.34 "Solid arrows: model/data flow. Dashed arrows: protocol constraints selected on dev." 6.8 $false | Out-Null
    Add-Text $page 5.15 5.58 5.10 0.34 "Test discipline: thresholds and Tie rescue are fixed before the locked test report." 6.8 $false | Out-Null

    if (Test-Path -LiteralPath $outVsdx) {
        Copy-Item -LiteralPath $outVsdx -Destination ($outVsdx + ".bak") -Force
    }

    $doc.SaveAs($outVsdx)
    Write-IeeeSvg $outSvg

    [pscustomobject]@{
        VSDX = $outVsdx
        SVG = $outSvg
        PNG = $outPng
    } | ConvertTo-Json -Compress
} finally {
    if ($doc) {
        $doc.Saved = $true
    }
}
