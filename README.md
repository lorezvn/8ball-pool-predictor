## 8 Ball Pool Shot Predictor

Project developed for the Computer Vision Course 2025/2026

- Lorenzo Zanda (2006432)
- Davide Vittucci (1903954)
- Paolo Marchetti (1986485)

## How to run
1. Install dependencies 
    ```python
    pip install -r requirements.txt
    ```

## Repository structure

```text
repo/
├── src/                    
│   ├── detection/        # Detection modules
│   │   ├── detect_balls.py   
│   │   ├── detect_cue.py     
│   │   ├── detect_pockets.py 
│   │   └── detect_table.py   
│   ├── tools/            # Utility scripts, QA, and visualization tools
│   │   ├── compare_images.py 
│   │   ├── inspect_video.py  
│   │   ├── qa_labels.py     
│   │   └── render_video.py   
│   ├── config.py             # System configurations
│   ├── dataset.py            # Dataset preparation
│   ├── train.py              # Model training
│   └── utils.py              # Helper functions
├── main.py                 
└── README.md
```

## TO-DO 
- [x] Ottimizzare processi di merging datasets
- [x] Detect tavolo
- [x] Detect pockets
- [x] Detect palle 
- [x] Detect stecca
- [x] Detect direzione
- [x] Sistemare la visualizzazione dell'overlay generale
- [x] Sistemare struttura del progetto
- [ ] Sistemare codice stecca e palle
- [ ] Traiettoria
- [ ] Mappa 2D
- [ ] Dimostrazione visiva
- [ ] Angolazione 