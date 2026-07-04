$ErrorActionPreference = 'Stop'
Set-Location 'c:\Working folders\BITS Pillani M Tech AI ML\Study Material\Semester 3\MLOps\Assignment\new\heart-disease-mlops'
$word = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $docPath = (Resolve-Path 'report.docx').Path
    $pdfPath = [string](Join-Path (Get-Location) 'report.pdf')
    $doc = $word.Documents.Open($docPath)
    $doc.SaveAs2($pdfPath, 17)
    $doc.Close()
    Write-Output "Saved PDF to $pdfPath"
} catch {
    Write-Error "Conversion failed: $($_.Exception.Message)"
    exit 2
} finally {
    if ($word -ne $null) { $word.Quit() }
}
