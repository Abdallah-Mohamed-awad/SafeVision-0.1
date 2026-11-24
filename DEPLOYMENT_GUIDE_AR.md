# SafeVision AI — GitHub + Free-tier Cloud Run Deployment

This kit is for the FastAPI + HTML frontend (`server.py` + `web/index.html`). It does not replace the existing Streamlit app or `requirements.txt`.

## 1) Required project layout

```text
SafeVision_AI_Torch/
├── server.py
├── Dockerfile
├── requirements-api.txt
├── app/
│   ├── __init__.py
│   └── inference.py
├── web/
│   └── index.html
└── models/
    └── best.pt
```

Keep `app/inference.py` and `app/__init__.py` from the project. Copy the uploaded HTML file to `web/index.html` (create the `web` folder if it does not exist).

**Model check:** `server.py` searches `models/best.pt` first. Put the selected, fine-tuned YOLO26s checkpoint at exactly `models/best.pt`. Do not use the generic pretrained `yolo26s.pt` as the trained project checkpoint. Do not upload every intermediate checkpoint; keep one final deployment weight.

## 2) Install the files from this kit

Copy `Dockerfile`, `requirements-api.txt`, `.dockerignore`, `.gitignore`, `server.py`, and the `web` folder into the project root. If a `.gitignore` already exists, merge the supplied rules instead of blindly overwriting your own rules.

The patched server and web UI enforce a 10 MB image / 25 MB video limit. The 25 MB video ceiling is intentionally below Cloud Run's documented 32 MiB HTTP/1 request limit. The camera polling loop is throttled to reduce inference pressure. The result message no longer equates “no missing-PPE label returned” with proof of full compliance.

## 3) Test the FastAPI site locally (PowerShell)

From `C:\NTI\SafeVision_AI_Torch`:

```powershell
py -3.11 -m venv .venv-api
.\.venv-api\Scripts\python.exe -m pip install --upgrade pip
.\.venv-api\Scripts\python.exe -m pip install -r requirements-api.txt
.\.venv-api\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

This creates a separate API environment so the existing Streamlit `.venv` and its package versions remain untouched.

Open `http://127.0.0.1:8000` and check `http://127.0.0.1:8000/api/health`.

## 4) Prepare GitHub safely

Before `git add`, check that `.venv`, `datasets`, `runs`, `.env`, API keys, credentials, and private images are not staged. The `.gitignore` excludes these. The final `models/best.pt` is intentionally included. GitHub warns for normal Git files above 50 MiB and blocks files above 100 MiB; the previously reported ~19 MB selected checkpoint is below those limits.

PowerShell:

```powershell
cd C:\NTI\SafeVision_AI_Torch
git status
git init
git branch -M main
git add .
git status --short
```

Review the status carefully. If it shows `.venv`, `datasets`, `runs`, `.env`, or files you do not intend to publish, stop and fix `.gitignore` before committing.

```powershell
git commit -m "Initial SafeVision AI project"
```

Create a new repository on GitHub. For the first push, create it empty (do not pre-add README, license, or .gitignore). Then run, replacing the URL with your repository URL:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/SafeVision_AI_Torch.git
git push -u origin main
```

If `origin` already exists, inspect it with `git remote -v` and use `git remote set-url origin YOUR_REPOSITORY_URL` instead of adding a duplicate.

## 5) Free-tier deployment: Google Cloud Run

Cloud Run is a managed HTTPS service and can build this repository from its Dockerfile. Its request-based free tier currently includes monthly CPU, RAM, and request quotas. It still requires a Google Cloud billing account, and usage beyond free quotas or other billable Google Cloud resources can incur charges. Configure a small budget alert and keep the maximum instance count at one for this demo.

1. Create a Google Cloud project and attach a billing account.
2. Enable Cloud Run and Cloud Build APIs when prompted.
3. Open Google Cloud Shell (or install and authenticate the Google Cloud CLI locally).
4. Clone the GitHub repository and enter the project folder:

```bash
git clone https://github.com/YOUR_USERNAME/SafeVision_AI_Torch.git
cd SafeVision_AI_Torch
```

5. Deploy:

```bash
gcloud run deploy safevision-ai \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --cpu 1 \
  --memory 2Gi \
  --concurrency 1 \
  --min-instances 0 \
  --max-instances 1 \
  --timeout 900
```

The Dockerfile binds to the `PORT` environment variable (Cloud Run normally supplies 8080) and runs a single Uvicorn worker to avoid loading multiple copies of the model.

6. When deployment finishes, open the service URL and check:
   - `/` — frontend
   - `/api/health` — model file presence

The HTTPS URL also provides the secure context required by browser camera access.

## 6) Important free-tier / demo limits

- Cloud Run's HTTP/1 request limit is 32 MiB. This build caps video uploads at 25 MB and image uploads at 10 MB.
- Free-tier quotas are shared with other Cloud Run usage on the same billing account and reset monthly. A budget alert is not a hard spending cap.
- `--max-instances=1` and `--concurrency=1` limit resource use but do not mathematically guarantee a $0 bill.
- Cloud Run instances have ephemeral local storage. Uploaded video jobs and generated MP4s are temporary and can disappear after a restart or scale-to-zero. This app is designed as a demo, not permanent media storage.
- The app is publicly reachable with `--allow-unauthenticated`. Do not upload confidential worker footage or personal data. A real production launch needs authentication, rate limiting, access control, monitoring, retention policy, and external storage for files.
- CPU inference is slower than the Colab T4. Real-time camera analysis may run at a lower frame rate; the frontend polling has been throttled.

## 7) GitHub size guidance

GitHub warns for regular Git files larger than 50 MiB and blocks files larger than 100 MiB. A ~19 MB checkpoint can be stored directly in the repository. If the final weight exceeds 100 MB, use Git LFS or keep the weight in a separate model artifact store and download it during deployment. Do not commit the complete dataset, `.venv`, or generated `runs` directory.
