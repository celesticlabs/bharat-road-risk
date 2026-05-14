# Bharat Road Risk Intelligence

A predictive road-safety intelligence system for Indian roads. Detects near-misses, wrong-way driving, stalled vehicles, and pedestrian conflicts in real-time from CCTV feeds.

> **Urgent Access**: For deployment credentials, API keys, or commercial licensing, contact: **celesticlabs@gmail.com**

## Features

- **Real-time Object Detection**: YOLOv8s fine-tuned on Indian Driving Dataset (IDD)
- **Multi-object Tracking**: ByteTrack with trajectory analysis
- **Risk Assessment**: TTC/PET-based collision risk scoring + vulnerable road user weighting
- **Evidence Generation**: Fine-tuned Gemma 4 31B for incident summarization and dispatch recommendations

## Architecture

```
CCTV Feed → YOLOv8s Detector → ByteTrack Tracker → Risk Engine → Gemma 4 VLM
                                    ↓                               ↓
                              Trajectories              Incident Summaries + Actions
```

## Performance

| Component | Metric | Value |
|-----------|--------|-------|
| Detector (YOLOv8s on IDD) | mAP50 | **65.8%** |
| Detector | Precision | **83.3%** |
| Detector | Recall | **58.0%** |
| VLM (Gemma 4 31B + QLoRA) | Severity Accuracy | **100%** |
| VLM | Risk Precision | **100%** |
| VLM | Risk Recall | **100%** |
| Full Pipeline | Risk Events (10s video) | **22,029** |

## Quick Start

```bash
# Clone
git clone https://github.com/celestic-labs/bharat-road-risk.git
cd bharat-road-risk

# Install dependencies
pip install -r requirements.txt

# Run inference
python -m src.pipeline --input demo/kolkata_traffic.mp4
```

## Files

- `src/pipeline.py` — Full inference pipeline
- `src/detector.py` — YOLOv8s detector
- `src/tracker.py` — ByteTrack wrapper
- `src/risk_engine.py` — Risk assessment
- `scripts/train_gemma4.py` — Gemma 4 fine-tuning
- `demo/app.py` — Gradio web demo
- `models/` — Model configs (download weights separately)

## Deployment

Requires:
- GPU with 8GB+ VRAM (RTX PRO 6000 Blackwell recommended for training)
- Python 3.10+
- CUDA 12+

Contact `celesticlabs@gmail.com` for deployment assistance, API access, or commercial licensing.

## License

Proprietary. Contact for access.

## Pitch Deck

Contact `celesticlabs@gmail.com` for the pitch deck and latest demo.

## Contact

- **Email**: celesticlabs@gmail.com
- **Website**: celesticlabs.ai (coming soon)
- **GitHub**: https://github.com/celestic-labs/bharat-road-risk