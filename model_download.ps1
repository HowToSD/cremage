# This script downloads models to appropriate directories under cremage installation.
# If you already have models on your system that you like, you do not
# need to run this script.

Write-Host "Downloading main models"
$TARGET_DIR = "models/ldm"
if (-not (Test-Path $TARGET_DIR)) {
    New-Item -ItemType Directory -Force -Path $TARGET_DIR
}
Invoke-WebRequest -Uri "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors?download=true" -OutFile "${TARGET_DIR}/DreamShaper_8_pruned.safetensors"
Invoke-WebRequest -Uri "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_INPAINTING.inpainting.safetensors?download=true" -OutFile "${TARGET_DIR}/DreamShaper_8_INPAINTING.inpainting.safetensors"
Invoke-WebRequest -Uri "https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_NV_B1_fp16.safetensors?download=true" -OutFile "${TARGET_DIR}/Realistic_Vision_V6.0_NV_B1_fp16.safetensors"
Invoke-WebRequest -Uri "https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_NV_B1_inpainting_fp16.safetensors?download=true" -OutFile "${TARGET_DIR}/Realistic_Vision_V6.0_NV_B1_inpainting_fp16.safetensors"

Write-Host "Downloading VAE"
$TARGET_DIR = "models/vae"
if (-not (Test-Path $TARGET_DIR)) {
    New-Item -ItemType Directory -Force -Path $TARGET_DIR
}
Invoke-WebRequest -Uri "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors?download=true" -OutFile "${TARGET_DIR}/vae-ft-mse-840000-ema-pruned.safetensors"

Write-Host "Downloading IP-Adapter-FaceID model"
$TARGET_DIR = "models/control_net"
if (-not (Test-Path $TARGET_DIR)) {
    New-Item -ItemType Directory -Force -Path $TARGET_DIR
}
Invoke-WebRequest -Uri "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin?download=true" -OutFile "${TARGET_DIR}/ip-adapter-faceid-plusv2_sd15.bin"
