# LungDetect AI

LungDetect AI is an educational prototype for two independent experiments: symptom-based risk scoring and lung histopathology image classification. It includes a React/Vite dashboard, FastAPI inference service, local AI chat through Ollama, and optional Supabase authentication and persistence.

> **Medical and data notice:** This application is not a diagnostic device and must not guide treatment decisions. Model scores are experimental, not clinically calibrated probabilities. Use synthetic or de-identified data only; this project is not designed for identifiable medical records or HIPAA-regulated use.

## Features

- Symptom-based Gaussian Naive Bayes prediction with score and confidence metrics.
- ResNet18 histopathology classification for `lung_aca`, `lung_n`, and `lung_scc`.
- Configurable image confidence threshold; low-confidence images are classified as `unknown` for review.
- Risk- and pathology-specific chat using local Ollama by default; OpenAI is optional.
- Supabase email/password authentication, user-scoped analysis and chat records, and private pathology-image storage protected by row-level security (RLS).
- Training and preprocessing scripts with evaluation plots and saved artifacts.

## Technology

| Area | Technology |
| --- | --- |
| Frontend | React 18, Vite 5, JavaScript/JSX |
| Frontend auth/data/storage | Supabase JavaScript client, Supabase Auth, PostgreSQL, Supabase Storage, RLS |
| Backend API | Python, FastAPI, Pydantic, Uvicorn, `python-multipart` |
| Tabular ML | pandas, NumPy, scikit-learn, joblib; Gaussian Naive Bayes and preprocessing encoders/scaler |
| Image ML | PyTorch, torchvision, ResNet18, Pillow |
| Chat | Ollama `llama3.2:3b` by default; optional OpenAI-compatible chat API |
| Data/model analysis | matplotlib, seaborn, Plotly, XGBoost, LightGBM, CatBoost, SHAP, Optuna |

## Project Structure

The tree below shows source plus expected local data/model paths. Dataset files, histopathology images, trained weights, generated plots, local environments, build output, `.env.local`, and `node_modules` are excluded from the source-only GitHub release. Empty/generated paths may not appear in a fresh clone.

```text
Early Cancer Detection/
├── .gitignore
├── README.md
├── start-app.ps1
├── lung_cancer.csv
├── Backend/
│   ├── app.py
│   ├── image_predict.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── requirements.txt
│   ├── test_image.py
│   ├── train.py
│   ├── train_image_model.py
│   ├── models/                 # Reserved; currently empty
│   └── plots/                  # Reserved; currently empty
├── Data/
│   ├── lung_cancer.csv
│   ├── processed.csv
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   ├── y_test.pkl
│   └── histopathology/
│       ├── lung_aca/           # Histopathology images
│       ├── lung_n/
│       ├── lung_scc/
│       └── unknown/            # Low-confidence/manual-review images
├── frontend/
│   ├── .env.example
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       ├── components/
│       │   ├── Assistant.jsx
│       │   ├── Auth.jsx
│       │   ├── Histo.jsx
│       │   ├── Pages.jsx
│       │   ├── Risk.jsx
│       │   └── ui.jsx
│       ├── services/
│       │   ├── api.js
│       │   ├── persistence.js
│       │   └── supabase.js
│       └── styles/
│           └── app.css
├── models/
│   └── # Locally generated model artifacts; not distributed
├── plots/
│   └── # Locally generated plots; not distributed
└── supabase/
    └── schema.sql
```

### Key modules

- `Backend/app.py`: health, tabular risk, image-analysis, and AI-chat API routes.
- `Backend/predict.py`: loads the saved tabular model, scaler, and encoders.
- `Backend/image_predict.py`: loads ResNet18, applies the image threshold, and saves images locally under `Data/histopathology/<class>/` or `unknown/`.
- `Backend/preprocess.py`, `train.py`, `train_image_model.py`: preprocessing and model training/evaluation pipelines.
- `frontend/src/App.jsx`: navigation, Supabase session handling, saved history, and analysis gating.
- `frontend/src/services/api.js`: browser-to-FastAPI requests.
- `frontend/src/services/supabase.js`, `persistence.js`: Supabase client and user-scoped persistence.
- `frontend/src/components/Risk.jsx`, `Histo.jsx`, `Assistant.jsx`: prediction forms/results and both chat contexts.
- `supabase/schema.sql`: tables, grants, RLS rules, private bucket and storage policies; also requests a PostgREST schema-cache refresh.

## Requirements

- Windows PowerShell (the launcher is `start-app.ps1`).
- Python with dependencies from `Backend/requirements.txt` and a supported PyTorch/torchvision installation.
- Node.js and npm.
- Ollama plus the configured chat model for local AI chat.
- A Supabase project for accounts and cloud persistence.

The public source release does **not** include training datasets or trained model files. The backend currently loads tabular artifacts from `models/` and the ResNet18 checkpoint from `models/lung_resnet18.pth`; inference will not start until these are obtained under suitable terms or generated locally from licensed training data. The image-training script expects the class folders under `Data/histopathology/`.

## Setup

### 1. Install Python and JavaScript dependencies

From the project root, create/activate a Python virtual environment if one is not already present, then install backend requirements:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install -r Backend\requirements.txt
```

Install frontend packages:

```powershell
Push-Location frontend
npm install
Pop-Location
```

The supplied launcher also recognizes `.venv-2` and `.venv` at the project root.

### 1a. Provide licensed datasets and model artifacts

Obtain each dataset and pre-trained artifact from its authorized source, review its license, and place it in the paths shown above. The app is educational and the repository does not redistribute these files. For the tabular pipeline, provide the expected `.csv` input and train using the scripts below. For histopathology, place images into the `lung_aca`, `lung_n`, and `lung_scc` folders before training; inference requires the resulting ResNet18 checkpoint.

### 2. Configure Supabase

1. Create a Supabase project and enable email/password sign-in in **Authentication**.
2. Set the local site/redirect URL to `http://localhost:3000` in the Supabase URL configuration.
3. In **SQL Editor**, run the full contents of [`supabase/schema.sql`](supabase/schema.sql). It creates `analyses` and `chat_messages`, enables RLS, and provisions the private `pathology-images` bucket.
4. Copy the frontend environment template:

   ```powershell
   Copy-Item frontend\.env.example frontend\.env.local
   ```

5. In `frontend/.env.local`, set `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` from the Supabase project API settings. The Vite prefix must be `VITE_`. Never put a database password or `service_role` key in frontend configuration.
6. Restart Vite after changing the env file. If PostgREST still reports that `public.analyses` is missing from its schema cache, rerun the SQL script and refresh the app.

Supabase stores full risk answers/results, histopathology result metadata, chat messages, and the original pathology image in a private bucket. RLS scopes rows and image paths to the signed-in user. The backend also keeps a separate local image archive in `Data/histopathology`; deleting a Supabase account or object does not remove that local archive.

### 3. Install and configure local AI chat

Install Ollama from its official distribution, then pull the default model:

```powershell
ollama pull llama3.2:3b
```

The backend defaults to `AI_PROVIDER=ollama`, model `llama3.2:3b`, and `http://127.0.0.1:11434`. Optional settings are `OLLAMA_MODEL` and `OLLAMA_BASE_URL`. For OpenAI instead, set `AI_PROVIDER=openai`, `OPENAI_API_KEY`, and optionally `OPENAI_MODEL` in the backend process environment. OpenAI mode sends prompt context to OpenAI; local Ollama mode runs locally. Histopathology chat sends class/result metadata, not the image itself.

### 4. Start the app

From the project root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\start-app.ps1
```

Open `http://localhost:3000`. The API is at `http://127.0.0.1:5050`; interactive API docs are at `http://127.0.0.1:5050/docs`.

**Launcher note:** `start-app.ps1` stops processes listening on ports 5050 and 3000 before starting both servers. Use manual startup if those ports contain processes you need to keep.

Manual startup, in separate PowerShell terminals from the project root:

```powershell
# Backend (activate the project venv first)
$env:AI_PROVIDER = "ollama"
python -m uvicorn --app-dir Backend app:app --host 127.0.0.1 --port 5050
```

```powershell
# Frontend
Push-Location frontend
npm run dev -- --host 0.0.0.0 --port 3000
```

## Using the app

1. Configure Supabase and sign in or create an account.
2. In **Risk Assessment**, submit age, gender, and symptom indicators. Indicator values are encoded as `1 = No`, `2 = Yes`; gender is `M` or `F`.
3. The backend returns the Gaussian Naive Bayes result. The result screen offers a context-aware chat; medication questions are directed to a clinician/pharmacist, not prescribed.
4. In **Histopathology**, upload a JPG/JPEG, PNG, or WEBP image and analyze it. Results below the configurable threshold (default 70%) are stored as `unknown` and marked for manual verification.
5. Histopathology chat receives class scores and result metadata, not the uploaded image. The model scores are uncalibrated outputs and are not clinical disease probabilities.
6. **History** and dashboard totals show saved analyses for the signed-in Supabase user.

## API

Base URL: `http://127.0.0.1:5050`.

| Method and route | Purpose |
| --- | --- |
| `GET /` | API welcome/status response |
| `GET /health` | Health check |
| `POST /predict` | Accepts the patient JSON fields and returns `{ "success": true, "prediction": { ... } }` with model, prediction, cancer score, survival estimate, and confidence |
| `POST /predict-image` | Accepts multipart form field `file`; supports JPEG, PNG, WEBP; returns class scores, threshold, storage class, and manual-review flag |
| `POST /assistant/chat` | Accepts a message, recent chat history, result context, `context_type` (`risk` or `histopathology`), and optional patient indicators; returns an AI reply |

The image confidence threshold is configurable with `HISTOPATHOLOGY_CONFIDENCE_THRESHOLD` in the backend environment. The default is 70 percent. This is an experimental review threshold, not proof that an image is valid or in-distribution.

## Training and data pipeline

Run commands from the project root with the Python environment activated:

```powershell
python Backend\preprocess.py
python Backend\train.py
python Backend\train_image_model.py
```

`preprocess.py` prepares the tabular data and plots. `train.py` evaluates the tabular classifiers and writes fitted model/scaler/encoder artifacts under `models/` and split data under `Data/`. `train_image_model.py` trains ResNet18 from `Data/histopathology/` and writes `models/lung_resnet18.pth` plus training plots/history. Trained artifacts are required to run inference; they are not regenerated when the app starts.

## Publishing and large files

This source-only release excludes datasets, patient/image data, trained model artifacts, and generated plots because redistribution rights have not been confirmed and the image corpus is large. Never commit `.env.local`, API keys, database passwords, or Supabase `service_role` credentials. If you later confirm redistribution rights, review file sizes and licenses before changing `.gitignore`; large files may require Git LFS.

## Safety and privacy

- Educational/research prototype only; not a diagnosis, screening recommendation, prognosis, or treatment plan.
- Do not use the prototype to decide whether to start, stop, or change medication. Chat is not medical care; seek a qualified clinician/pathologist for interpretation.
- High tabular risk scores trigger clinician follow-up guidance; severe breathing difficulty, coughing blood, severe chest pain, or sudden deterioration warrants urgent medical care.
- Histopathology `unknown` means the model score fell below its experimental threshold. It cannot determine whether the input was invalid, out of distribution, or a model failure.
- Supabase Auth and RLS protect per-user cloud data, but the project has not been security-audited or certified for clinical records. Use synthetic or de-identified data only.
- By default the chat model runs locally via Ollama. If OpenAI is configured, risk inputs/chat context are sent to that provider. Review provider data terms before use.
