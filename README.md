# Carb Inspector

A website in which the user can upload pictures of a food item. 

It will classify the food item and return the amount of carbohydrates(g) in the food.

## Model code

The source code for the model is a .ipynb file: carb_inspector_model.ipynb


## Carb Inspector - API/UI

Single-file app that serves a browser UI and a JSON API for food-image classification + carb estimation.

### GitHub Repository Download (unconvential):
The model is stored on GitHub via LFS, so the user needs to use LFS to download it to their PC as setup. Instructions (in CMD):
1) install git LFS:
   - git lfs install
2) clone the git repo (donnot directly download it from github)
   - cd path\to\where\you\want\to\download\the\prohect\to
   - git clone https://github.com/HilaHorizon/Carb-Inspector.git
   - cd Carb-Inspector
3) Fetch LFS contact
   - git lfs pull
   - git lfs fetch --all
   - git lfs checkout
4) you are now set to begin :)

### Windows run:
- One command to run:  python carb_inspector.py
- Auto-opens your browser to the app
- Model files: model.pth and class_names.txt in the same folder as carb_inspector.py

 
### Linux run:
- One command to run:  ./carb_inspector_linux
- prints the link to the website on your terminal - need to open it in browser
- doesn't require additional files

------------------------------------------------------------
## Requirements
------------------------------------------------------------
- Python 3.9 or newer
- Packages (the carb_inspector.py file self-installs them):
  * fastapi
  * uvicorn
  * pillow
  * torch
  * torchvision
- Files:
  * model.pth (trained model weights)
  * class_names.txt (list of class labels, one per line, same order as training)
  * carb_inspector.py


------------------------------------------------------------
## API Features
------------------------------------------------------------
1) POST /predict   →  Send a food photo, get predictions
   - Upload an image (PNG/JPG/WEBP).
   - Optional setting: tta=off | on | auto
       * auto (default): only use extra checks if the model is unsure.
       * on: always use extra checks (slower, maybe better).
       * off: no extra checks (fastest).
   - Returns: best label, carbs (g), confidence, and a list of top guesses.

   Example result:
   {
     "label": "sandwich",
     "carbs_g": 11.4,
     "confidence": 0.14,
     "uncertain": true,
     "top_preds": [
       {"label": "sandwich", "prob": 0.14, "carbs_g": 11.4},
       {"label": "pasta",    "prob": 0.12, "carbs_g": 48.1}
     ]
   }

   You can try it from the command line:
   Linux:
     curl -X POST "http://127.0.0.1:8000/predict" -F "image=@food.jpg"
   Windows PowerShell:
     Invoke-WebRequest -Uri "http://127.0.0.1:8000/predict" -Method Post -Form @{ image = Get-Item "C:\food.jpg" }

2) GET /info   →  Quick check if it’s running
   Example: { "ok": true, "device": "cpu", "num_classes": 193 }

3) GET /       →  Opens the web app (drag & drop UI in your browser).


------------------------------------------------------------
## Configuration (env vars)
------------------------------------------------------------
You can tweak behavior without editing code:

MODEL_PATH       (default model.pth)
CLASS_NAMES_PATH (default class_names.txt)
CONF_THRESHOLD   (default 0.10)
TOPK_MIN         (default 3)
TOPK_MAX         (default 6)
CONF_LOW_FLAG    (default 0.50)

Linux:
CONF_THRESHOLD=0.15 TOPK_MAX=5 python3 carb_inspector.py

Windows PowerShell:
$env:CONF_THRESHOLD="0.15"; $env:TOPK_MAX="5"; python carb_inspector.py

------------------------------------------------------------
## Troubleshooting
------------------------------------------------------------
- "model file not found": ensure model.pth is in the same folder
- Wrong classes/mismatch: ensure class_names.txt matches the provided class, following the training order
- Browser didn’t open: open the URL shown in the console
- Windows firewall prompt: allow local access; the server runs on 127.0.0.1

------------------------------------------------------------
## Credits
------------------------------------------------------------
© Gur Abraham & Hila Ofek, 2025  @ TAU - Carb Inspector Model Demo.
© Tam & Le, 2019 @ Google Research - EfficientNet-B4.
© Thames at al, 2021 @ Google Research - Nutrition5K.




