# ORBIT auto commit + push helper
# Usage:  .\git-push.ps1 [message]
#   without message: opens editor prompt for a commit message (default: "Update")
# Stops safely if there is nothing to commit.

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

git status --porcelain
if (-not $?) { exit 1 }

$staged = git status --porcelain
if (-not $staged) {
    Write-Host "Nothing to commit - working tree clean." -ForegroundColor Yellow
    exit 0
}

if ($args.Count -ge 1) {
    $message = $args[0]
} else {
    $message = Read-Host "Commit message (default: Update)"
    if (-not $message) { $message = "Update" }
}

Write-Host "Staging all changes..." -ForegroundColor Cyan
git add -A
if (-not $?) { exit 1 }

git commit -m $message
if (-not $?) { exit 1 }

Write-Host "Pushing to origin..." -ForegroundColor Cyan
git push
if ($?) {
    Write-Host "Done. Committed and pushed." -ForegroundColor Green
} else {
    Write-Host "Push failed - commit exists locally, retry push later." -ForegroundColor Red
}
