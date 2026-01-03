# Quick Start Guide - SocialVision (Dec 2025)

**Get up and running in 5 minutes!**

---

## 🚀 Option 1: Docker (Easiest)

### Step 1: Build and Start

```bash
# Navigate to project directory
cd SocialVision-Facial-Recognition-Search

# Build the Docker image (first time: 10-15 min)
docker compose build

# Start the application
docker compose up -d

# Or use the automated script
./docker-demo.sh
```

### Step 2: Access the Application

Open your browser and go to:

```text
http://localhost:8501
```

### Step 3: Test It Out

1. **Add a Face:**
   - Go to "📤 Add Faces" tab
   - Upload an image with a face
   - Enter username: "test_user"
   - Click "➕ Add to Database"

   Notes:
   - If the username already exists, SocialVision will **only upload missing embedding keys** (new “dimensions”, e.g. add `dlib` later without re-uploading `deepface`).
   - Repeated uploads with no new embedding keys will still create new face samples (useful for improving robustness).

2. **Search for Faces:**
   - Go to "🔎 Search" tab
   - Upload the same (or similar) image
   - Click "🔍 Search"
   - See the results!

   Tip:
   - You can also switch the search input to **Video upload** (samples frames) or **Live camera** (WebRTC when available; otherwise capture-based mode).

3. **View Analytics:**
   - Go to "📈 Analytics" tab
   - See database statistics

### Useful Commands

```bash
# View logs
docker compose logs -f

# Stop application
docker compose down

# Restart
docker compose restart

# Check status
docker compose ps
```

### IBM MAX Upscaling (Optional; Disabled by Default in Compose)

The repo includes optional wiring for the [IBM MAX Image Resolution Enhancer](https://github.com/IBM/MAX-Image-Resolution-Enhancer), but the default `docker-compose.yml` ships with IBM MAX **disabled** (and the `ibm-max` service commented out) because the upstream image is **x86-only** and often fails on Apple Silicon.

To enable IBM MAX on an x86 host:

1. Uncomment the `ibm-max` service in `docker-compose.yml`.
2. Set `IBM_MAX_ENABLED=true` (in `.env` or as an environment override).
3. Keep `IBM_MAX_URL=http://ibm-max:5000` (the compose network hostname).

For manual/local (non-Docker) runs, export the same variables to prefer MAX before the Real-ESRGAN fallbacks:

```bash
export IBM_MAX_ENABLED=true
export IBM_MAX_URL="http://localhost:5000"
export IBM_MAX_TIMEOUT=180
# Optional: override when running the container yourself
export IBM_MAX_PLATFORM=linux/amd64
```

> **Apple Silicon tip:** If IBM MAX crash-loops with `Illegal instruction`, keep `IBM_MAX_ENABLED=false` and rely on the built-in Real-ESRGAN/OpenCV/Lanczos pipeline.

If the microservice is temporarily unreachable, set `IBM_MAX_FAILURE_THRESHOLD=1` so the Streamlit process disables IBM MAX after the first failure and immediately jumps to the Real-ESRGAN stack. Leave `IBM_MAX_PROBE_ON_START=true` (default) so the app pings `/model/metadata` once at boot; if the local container is already crash-looping, IBM MAX will be disabled before the first upload. Set it to `false` only when you intentionally rely on a remote endpoint that might take extra time to come online.

### Real-ESRGAN Defaults & Firestore Mode

- `IMAGE_UPSCALING_TARGET_TILES=25` forces roughly a 5×5 grid so every upload benefits from tiled inference even on CPU-constrained Docker.
- `IMAGE_UPSCALING_MIN_REALESRGAN_SCALE=1.0` keeps Real-ESRGAN active for even minor touch-ups; raise it if you want to skip AI passes for tiny images.
- To run against Google Firestore instead of the local JSON DB, set `DB_TYPE=firestore`, `FIREBASE_ENABLED=true`, and provide a service account at `config/firebase_config.json` (mount it into the container).
- To run against Firebase Realtime Database, set `DB_TYPE=realtime`, plus `FIREBASE_DATABASE_URL` and credentials.
- To prefer Firebase Realtime Database but fall back to Firestore automatically (and then to local JSON as a last resort), set `DB_TYPE=firebase`.

---

## 🐍 Option 2: Local Installation

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Run Application

```bash
streamlit run src/app.py
```

### Live Camera on macOS (Recommended)

For the best live camera experience (WebRTC), use the included local runner:

```bash
./run_local.sh
```

This exists because webcam/WebRTC support is often limited inside Docker on macOS due to VM/browser isolation.

The application will open automatically in your browser.

---

## 🧪 Quick Test

### Test 1: Add Face

1. Upload image with face
2. Enter username
3. Click "Add to Database"
4. ✅ Should see success message

### Test 2: Search

1. Upload same/similar image
2. Click "Search"
3. ✅ Should see matching results

### Test 3: Analytics

1. Go to Analytics tab
2. ✅ Should see statistics

---

## 📚 Next Steps

- **Full Testing Guide:** [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md)
- **Docker Guide:** [docs/DOCKER_TESTING_GUIDE.md](docs/DOCKER_TESTING_GUIDE.md)
- **What It Can Do:** [docs/CURRENT_CAPABILITIES.md](docs/CURRENT_CAPABILITIES.md)
- **Demo Guide:** [docs/DEMONSTRATION_GUIDE.md](docs/DEMONSTRATION_GUIDE.md)

---

## 🆘 Troubleshooting

### Docker Issues

```bash
# Check if Docker is running
docker ps

# View container logs
docker compose logs

# Rebuild from scratch
docker compose build --no-cache
```

### Local Installation Issues

```bash
# Check Python version (need 3.9+)
python --version

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Application Not Loading

- Wait 30-60 seconds after starting
- Check logs for errors
- Verify port 8501 is not in use
- Try different port in docker-compose.yml

---

## 📞 Need Help?

- Check [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md)
- Review [docs/DOCKER_TESTING_GUIDE.md](docs/DOCKER_TESTING_GUIDE.md)
- Contact: [mtabdevt@gmail.com](mailto:mtabdevt@gmail.com)

---

Last Updated: January 2026 (video/live camera + multi-backend DB + staged Docker installs)

