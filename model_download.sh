#!/bin/bash
# This script downloads models to appropriate directories under cremage installation.
# If you already have models on your system that you like, you do not
# need to run this script.

echo "Downloading main models"
export TARGET_DIR=models/ldm
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
fi
wget -O ${TARGET_DIR}/DreamShaper_8_pruned.safetensors https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors?download=true
wget -O ${TARGET_DIR}/DreamShaper_8_INPAINTING.inpainting.safetensors https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_INPAINTING.inpainting.safetensors?download=true
wget -O ${TARGET_DIR}/Realistic_Vision_V6.0_NV_B1_fp16.safetensors https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_NV_B1_fp16.safetensors?download=true
wget -O ${TARGET_DIR}/Realistic_Vision_V6.0_NV_B1_inpainting_fp16.safetensors https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_NV_B1_inpainting_fp16.safetensors?download=true

echo "Downloading VAE"
export TARGET_DIR=models/vae
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
fi
wget -O ${TARGET_DIR}/vae-ft-mse-840000-ema-pruned.safetensors https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors?download=true

echo "Downloading IP-Adapter-FaceID model"
export TARGET_DIR=models/control_net
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
fi
wget -O ${TARGET_DIR}/ip-adapter-faceid-plusv2_sd15.bin https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin?download=true