# Docker Testing Guide for SocialVision

**Version:** 1.0.0  
**Last Updated:** December 2024

---

## 🐳 Quick Start with Docker

### Prerequisites

- Docker installed (version 20.10+)
- Docker Compose installed (version 2.0+)
- At least 4GB free disk space
- Internet connection (for downloading dependencies)

### Verify Docker Installation

```bash
docker --version
docker-compose --version
```

---

## 🚀 Step-by-Step Docker Testing

### Step 1: Build the Docker Image

```bash
# Navigate to project directory
cd /path/to/SocialVision-Facial-Recognition-Search

# Build the Docker image
docker-compose build

# Or build with no cache (if you want a fresh build)
docker-compose build --no-cache
```

**Expected Output:**
- Docker will download Python base image
- Install system dependencies
- Install Python packages (this may take 10-15 minutes)
- Copy application code
- Create necessary directories

**Build Time:** ~10-15 minutes (first time), ~2-5 minutes (subsequent builds)

### Step 2: Start the Container

```bash
# Start the application
docker-compose up

# Or run in detached mode (background)
docker-compose up -d
```

**Expected Output:**
```
Creating network "socialvision-facial-recognition-search_socialvision-network" ... done
Creating socialvision-app ... done
Attaching to socialvision-app
socialvision-app  | 
socialvision-app  |   You can now view your Streamlit app in your browser.
socialvision-app  | 
socialvision-app  |   Local URL: http://localhost:8501
socialvision-app  |   Network URL: http://0.0.0.0:8501
```

### Step 3: Access the Application

Open your web browser and navigate to:
```
http://localhost:8501
```

You should see the SocialVision interface with:
- 🔍 Search tab
- 📤 Add Faces tab
- 📈 Analytics tab

### Step 4: Check Container Status

```bash
# Check if container is running
docker-compose ps

# View logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f
```

### Step 5: Stop the Container

```bash
# Stop the container
docker-compose down

# Stop and remove volumes (cleans up data)
docker-compose down -v
```

---

## 🧪 Testing the Application

### Test 1: Application Startup

**Goal:** Verify the application starts correctly

**Steps:**
1. Build and start: `docker-compose up -d`
2. Wait 30-60 seconds for startup
3. Check logs: `docker-compose logs socialvision`
4. Verify health: `curl http://localhost:8501/_stcore/health`

**Expected Result:**
- Container starts successfully
- No error messages in logs
- Health check returns 200 OK
- Application accessible at http://localhost:8501

### Test 2: Face Detection

**Goal:** Test face detection functionality

**Steps:**
1. Open http://localhost:8501
2. Go to "📤 Add Faces" tab
3. Upload an image with a clear face (JPG or PNG)
4. Enter a test username (e.g., "test_user_1")
5. Select source type (e.g., "profile_pic")
6. Click "➕ Add to Database"

**Expected Result:**
- Image uploads successfully
- Face is detected
- Success message: "Added X face(s) to database"
- Database statistics update

### Test 3: Face Search

**Goal:** Test similarity search functionality

**Steps:**
1. Go to "🔎 Search" tab
2. Upload the same image (or similar face)
3. Click "🔍 Search"
4. Adjust similarity threshold if needed

**Expected Result:**
- Image processes successfully
- Faces are detected
- Search results appear
- Shows matching usernames with similarity scores

### Test 4: Analytics Dashboard

**Goal:** Verify analytics display

**Steps:**
1. Go to "📈 Analytics" tab
2. View database statistics

**Expected Result:**
- Total faces count displayed
- Unique users count displayed
- Source distribution chart visible
- All data matches what was added

### Test 5: Multiple Faces

**Goal:** Test with multiple faces in one image

**Steps:**
1. Upload an image with multiple faces
2. Add to database
3. Search with similar image

**Expected Result:**
- All faces detected
- All faces added to database
- Search finds all matches

---

## 🔍 Troubleshooting Docker Issues

### Issue 1: Build Fails

**Symptoms:**
```
ERROR: failed to solve: process "/bin/sh -c pip install..." did not complete successfully
```

**Solutions:**
```bash
# Clean build (no cache)
docker-compose build --no-cache

# Check Docker resources
docker system df

# Free up space if needed
docker system prune -a
```

### Issue 2: Container Won't Start

**Symptoms:**
```
Error: container exited with code 1
```

**Solutions:**
```bash
# Check logs
docker-compose logs socialvision

# Check if port is already in use
lsof -i :8501

# Try different port (edit docker-compose.yml)
ports:
  - "8502:8501"
```

### Issue 3: Application Not Accessible

**Symptoms:**
- Browser shows "Connection refused"
- Container is running but can't access

**Solutions:**
```bash
# Check container status
docker-compose ps

# Check if port is exposed
docker port socialvision-app

# Verify network
docker network ls
docker network inspect socialvision-network
```

### Issue 4: DeepFace Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'deepface'
```

**Solutions:**
- This is expected if DeepFace dependencies fail
- Application will run in degraded mode
- Check logs for warnings
- Face detection may not work, but other features should

### Issue 5: Out of Memory

**Symptoms:**
```
Killed process
Container exits unexpectedly
```

**Solutions:**
```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory (set to 4GB+)

# Check memory usage
docker stats socialvision-app

# Restart Docker daemon
```

---

## 📊 Monitoring Container Health

### View Container Stats

```bash
# Real-time resource usage
docker stats socialvision-app

# Container information
docker inspect socialvision-app

# Health check status
docker inspect --format='{{.State.Health.Status}}' socialvision-app
```

### View Logs

```bash
# All logs
docker-compose logs

# Last 100 lines
docker-compose logs --tail=100

# Follow logs
docker-compose logs -f

# Specific service logs
docker-compose logs socialvision
```

### Access Container Shell

```bash
# Execute command in container
docker-compose exec socialvision bash

# Check Python version
docker-compose exec socialvision python --version

# Check installed packages
docker-compose exec socialvision pip list

# Check database file
docker-compose exec socialvision ls -la /app/data/
```

---

## 🔄 Common Docker Commands

### Build and Run

```bash
# Build only
docker-compose build

# Build and start
docker-compose up --build

# Start in background
docker-compose up -d

# Stop
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Restart
docker-compose restart
```

### Debugging

```bash
# View logs
docker-compose logs -f

# Execute shell in container
docker-compose exec socialvision bash

# Check container status
docker-compose ps

# Inspect container
docker inspect socialvision-app
```

### Cleanup

```bash
# Remove containers
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Remove images
docker-compose down --rmi all

# Full cleanup
docker system prune -a
```

---

## 📁 Data Persistence

### Volumes

The following directories are mounted as volumes:
- `./data` → `/app/data` (Database files)
- `./uploads` → `/app/uploads` (Uploaded images)
- `./logs` → `/app/logs` (Log files)
- `./models` → `/app/models` (ML models)
- `./config` → `/app/config` (Configuration)

### Backup Data

```bash
# Backup database
docker-compose exec socialvision cp /app/data/faces_database.json /app/data/backup_$(date +%Y%m%d).json

# Copy from container to host
docker cp socialvision-app:/app/data/faces_database.json ./backup/
```

---

## 🎯 Performance Testing

### Load Test

```bash
# Monitor resource usage
docker stats socialvision-app

# Test with multiple requests
for i in {1..10}; do
  curl http://localhost:8501/_stcore/health
done
```

### Memory Usage

```bash
# Check memory
docker stats --no-stream socialvision-app

# Expected: ~500MB-2GB depending on models loaded
```

---

## ✅ Testing Checklist

### Pre-Deployment

- [ ] Docker image builds successfully
- [ ] Container starts without errors
- [ ] Application is accessible at http://localhost:8501
- [ ] Health check passes
- [ ] Logs show no critical errors

### Functionality Tests

- [ ] Can upload images
- [ ] Face detection works
- [ ] Can add faces to database
- [ ] Search functionality works
- [ ] Analytics display correctly
- [ ] Multiple faces handled correctly
- [ ] Error handling works

### Performance Tests

- [ ] Application starts within 60 seconds
- [ ] Face detection completes within 10 seconds
- [ ] Search completes within 5 seconds
- [ ] Memory usage is reasonable (<4GB)
- [ ] No memory leaks during extended use

---

## 🚀 Production Deployment Tips

### Environment Variables

Create a `.env` file:
```env
DEBUG=False
LOG_LEVEL=INFO
STREAMLIT_SERVER_PORT=8501
FIREBASE_ENABLED=False
```

### Resource Limits

Add to `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

### Security

- Don't expose ports publicly without authentication
- Use Docker secrets for sensitive data
- Regularly update base images
- Scan images for vulnerabilities

---

## 📞 Support

If you encounter issues:

1. Check logs: `docker-compose logs`
2. Review this guide
3. Check [TESTING_GUIDE.md](TESTING_GUIDE.md)
4. Contact: mtabdevt@gmail.com

---

*Last Updated: December 2024*

