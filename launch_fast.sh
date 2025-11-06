#!/bin/bash
# STPLS3D Güvenli Hızlandırılmış Training

echo "================================================================================
🚀 GÜVENLİ HIZLANDIRILMIŞ TRAINING BAŞLATILIYOR
================================================================================
Optimizasyonlar:
✅ Validation: Her 3 epoch'ta bir (önceden her epoch)
✅ File step: 25 (önceden 17)
✅ Smart checkpoint (sadece improvement'da)

Korunanlar (kalite için):
✅ Max points: 500K (değişmedi)
✅ Batch size: değişmedi
✅ Learning rate: 0.0001 (değişmedi)

Beklenen:
⚡ 2-3x hızlanma
🎯 Model kalitesi: %98+ korunur
⏰ ~10 saat (20 saat yerine)
================================================================================"

# Config'i değiştir
export CUDA_VISIBLE_DEVICES=0

# GPU durumu
echo ""
echo "🖥️ GPU Durumu:"
nvidia-smi --query-gpu=name,memory.free,utilization.gpu --format=csv,noheader
echo ""

# Training başlat
if [ "$1" == "--resume" ]; then
    echo "▶️ Resume mode (kaldığı yerden devam)..."
    # config_fast.yaml kullan
    cp config_fast.yaml config.yaml
    python -u train_fast.py 2>&1 | tee training_fast.log
else
    echo "🆕 Fresh start (--no-resume flag ile)..."
    cp config_fast.yaml config.yaml
    python -u train_fast.py --no-resume 2>&1 | tee training_fast.log
fi
