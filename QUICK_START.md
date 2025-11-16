# Quick Start Guide - SocialVision

**Get up and running in 5 minutes!**

---

## 🚀 Option 1: Docker (Easiest)

### Step 1: Build and Start

```bash
# Navigate to project directory
cd SocialVision-Facial-Recognition-Search

# Build the Docker image (first time: 10-15 min)
docker-compose build

# Start the application
docker-compose up -d

# Or use the automated script
./docker-demo.sh
```

### Step 2: Access the Application

Open your browser and go to:
```
http://localhost:8501
```

### Step 3: Test It Out

1. **Add a Face:**
   - Go to "📤 Add Faces" tab
   - Upload an image with a face
   - Enter username: "test_user"
   - Click "➕ Add to Database"

2. **Search for Faces:**
   - Go to "🔎 Search" tab
   - Upload the same (or similar) image
   - Click "🔍 Search"
   - See the results!

3. **View Analytics:**
   - Go to "📈 Analytics" tab
   - See database statistics

### Useful Commands

```bash
# View logs
docker-compose logs -f

# Stop application
docker-compose down

# Restart
docker-compose restart

# Check status
docker-compose ps
```

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
docker-compose logs

# Rebuild from scratch
docker-compose build --no-cache
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
- Contact: mtabdevt@gmail.com

---

*Last Updated: December 2024*

