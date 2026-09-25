# MonsoonIQ Cloud Deployment Guide

This guide details the simplest and most effective ways to deploy **MonsoonIQ** live on the internet for demonstrations, hackathons, and judging.

---

## Option 1: Render.com (Recommended — Free, Single Service Full-Stack)

Because MonsoonIQ includes a production multi-stage `Dockerfile` that builds the React frontend and mounts it directly inside FastAPI, **you only need to deploy a single Web Service** on Render! This eliminates CORS issues and gives you one unified URL (e.g. `https://monsooniq.onrender.com`).

### Step-by-Step Instructions:

1. Go to **[Render.com](https://render.com)** and sign in with your GitHub account (`Harsh007engineering`).
2. Click **New +** in the top navigation and select **Web Service**.
3. Choose **Build and deploy from a Git repository**, and select:
   `Harsh007engineering/MonsoonIQ-Regime-Aware-AI-Post-Processing-System`
4. Configure the service:
   - **Name**: `monsooniq` (or any name you prefer)
   - **Region**: Choose closest to you (e.g., `Singapore` or `Frankfurt`)
   - **Branch**: `main`
   - **Runtime / Environment**: Select **Docker**
   - **Instance Type**: Select **Free**
5. Click **Create Web Service**.
6. Render will automatically:
   - Run the Node.js builder to compile the React 19 + Tailwind dashboard
   - Install the Python ML dependencies (FastAPI, LightGBM, scikit-learn, ReportLab)
   - Launch `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
7. Once the build completes (usually ~3-4 minutes), your live application will be accessible at:
   `https://monsooniq.onrender.com`

---

## Option 2: Railway.app (Fastest Alternative — 2 Minutes)

1. Go to **[Railway.app](https://railway.app)** and log in with GitHub.
2. Click **New Project** → **Deploy from GitHub repo**.
3. Select `Harsh007engineering/MonsoonIQ-Regime-Aware-AI-Post-Processing-System`.
4. Railway automatically detects the `Dockerfile` and begins deployment.
5. In the service **Settings** tab:
   - Under **Networking**, click **Generate Domain**.
6. Your live app is instantly accessible on the generated Railway domain.

---

## Option 3: Split Deployment (Vercel Frontend + Render Backend)

If you prefer hosting the React frontend on Vercel:

### 1. Deploy the Backend on Render:
- Create a new **Web Service** on Render pointing to your repo.
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
- Copy the resulting backend URL (e.g., `https://monsooniq-api.onrender.com`).

### 2. Deploy the Frontend on Vercel:
- Go to **[Vercel.com](https://vercel.com)** → **Add New Project**.
- Select `MonsoonIQ-Regime-Aware-AI-Post-Processing-System`.
- Set **Root Directory** to: `frontend`.
- Under **Environment Variables**, add:
  - `VITE_API_BASE`: `https://monsooniq-api.onrender.com`
- Click **Deploy**.

---

## Option 4: Linux Virtual Machine (AWS EC2 / DigitalOcean / Azure)

To run the entire system on your own cloud VM using Docker Compose:

1. SSH into your VM:
   ```bash
   ssh ubuntu@your-vm-ip
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/Harsh007engineering/MonsoonIQ-Regime-Aware-AI-Post-Processing-System.git
   cd MonsoonIQ-Regime-Aware-AI-Post-Processing-System
   ```
3. Launch with Docker Compose:
   ```bash
   docker compose up -d --build
   ```
4. Access the application on `http://your-vm-ip:8000`.

---

## Live Endpoints Available After Deployment

- **Frontend Dashboard**: `https://your-domain/`
- **Interactive OpenAPI / Swagger Docs**: `https://your-domain/docs`
- **System Health Check**: `https://your-domain/health`
- **Download Official PDF Report**: `https://your-domain/verification/report.pdf`
- **District Forecasts**: `https://your-domain/forecast/corrected?date=2023-07-15&lead_time_days=1`
