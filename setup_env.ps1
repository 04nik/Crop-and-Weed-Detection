# setup_env.ps1
# Script to install necessary Python dependencies for Crop and Weed Detection

Write-Host "Checking for required Python libraries..." -ForegroundColor Cyan

$libraries = @("opencv-python", "numpy", "matplotlib", "ultralytics", "torch")
$to_install = @()

foreach ($lib in $libraries) {
    python -c "import $lib" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "$lib is missing, queueing for installation..." -ForegroundColor Yellow
        $to_install += $lib
    } else {
        Write-Host "$lib is already installed!" -ForegroundColor Green
    }
}

if ($to_install.Count -gt 0) {
    Write-Host "Installing missing libraries: $to_install..." -ForegroundColor Yellow
    python -m pip install --quiet $to_install
    if ($LASTEXITCODE -eq 0) {
        Write-Host "All libraries installed successfully!" -ForegroundColor Green
    } else {
        Write-Warning "Failed to install some libraries. Please run: pip install opencv-python numpy matplotlib manually."
    }
} else {
    Write-Host "Environment is ready!" -ForegroundColor Green
}
