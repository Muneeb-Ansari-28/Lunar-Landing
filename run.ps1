# Windows PowerShell Workflow & Interactive Launcher for Lunar Lander

param (
    [Parameter(Position=0)]
    [string]$Target = "menu",
    [string]$Config = "baseline",
    [int]$Seed = 0
)

$ErrorActionPreference = "Stop"

function Show-Menu {
    python main.py
}

function Launch-Viewer {
    Write-Host ">>> Launching Real-Time Landing Simulation ($Config seed $Seed)..." -ForegroundColor Green
    python -m ll.live_viewer --config $Config --seed $Seed
}

function Check-Install {
    Write-Host ">>> Running environment check..." -ForegroundColor Cyan
    python scripts/00_check_install.py
}

function Run-Explore {
    Write-Host ">>> Running random agent baseline..." -ForegroundColor Cyan
    python scripts/01_random_agent.py
    Write-Host ">>> Explaining reward step calculation..." -ForegroundColor Cyan
    python scripts/02_explain_reward.py
}

function Run-Tests {
    Write-Host ">>> Running pytest test suite..." -ForegroundColor Cyan
    python -m pytest tests/ -v
}

function Train-Quick {
    Write-Host ">>> Training quick smoke test ($Config seed $Seed)..." -ForegroundColor Cyan
    python -m ll.train --config $Config --seed $Seed --quick
}

function Train-Full {
    Write-Host ">>> Training full matrix (4 configs x 3 seeds)..." -ForegroundColor Cyan
    $configs = @("baseline", "time_penalty", "hover_hack", "fixed")
    $seeds = @(0, 1, 2)
    foreach ($c in $configs) {
        foreach ($s in $seeds) {
            Write-Host ">>> Training $c (seed $s)..." -ForegroundColor Yellow
            python -m ll.train --config $c --seed $s
        }
    }
}

function Run-Eval {
    Write-Host ">>> Evaluating all trained models on raw environment..." -ForegroundColor Cyan
    python -m ll.evaluate
}

function Run-Rollouts {
    Write-Host ">>> Collecting trajectories..." -ForegroundColor Cyan
    python -m ll.rollouts
}

function Run-Plots {
    Write-Host ">>> Generating scientific matplotlib figures..." -ForegroundColor Cyan
    python -m ll.plots
}

function Run-Report {
    Write-Host ">>> Generating self-contained HTML report..." -ForegroundColor Cyan
    python -m ll.report
}

switch ($Target.ToLower()) {
    "menu"        { Show-Menu }
    "launch"      { Launch-Viewer }
    "viewer"      { Launch-Viewer }
    "check"       { Check-Install }
    "explore"     { Run-Explore }
    "test"        { Run-Tests }
    "train-quick" { Train-Quick }
    "train"       { Train-Full }
    "evaluate"    { Run-Eval }
    "rollouts"    { Run-Rollouts }
    "plots"       { Run-Plots }
    "report"      { Run-Report }
    "first-run"   {
        Check-Install
        Run-Explore
        Run-Tests
        Train-Quick
        Run-Plots
        Run-Report
        Write-Host ">>> First-run verification complete!" -ForegroundColor Green
    }
    "all"         {
        Check-Install
        Run-Explore
        Run-Tests
        Train-Full
        Run-Eval
        Run-Rollouts
        Run-Plots
        Run-Report
    }
    default {
        Write-Host "Unknown target: $Target." -ForegroundColor Red
        Write-Host "Available: menu, launch, viewer, check, explore, test, train-quick, train, evaluate, rollouts, plots, report, first-run, all" -ForegroundColor Yellow
    }
}
