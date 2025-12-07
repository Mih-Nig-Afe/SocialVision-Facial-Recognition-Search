# SocialVision Testing Guide

**Version:** 1.0.0  
**Last Updated:** December 2024

---

## 📋 Table of Contents

1. [Quick Start Testing](#quick-start-testing)
2. [Unit Testing](#unit-testing)
3. [Integration Testing](#integration-testing)
4. [Manual Testing](#manual-testing)
5. [Testing Current Capabilities](#testing-current-capabilities)
6. [Test Data Preparation](#test-data-preparation)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start Testing

### Prerequisites

```bash
# Ensure you have Python 3.9+ installed
python --version

# Install dependencies
pip install -r requirements.txt

# Install test dependencies (if not in requirements.txt)
pip install pytest pytest-cov
```

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_database.py -v
```

---

## 🧪 Unit Testing

### Test Structure

```
tests/
├── test_face_recognition.py    # Face recognition engine tests
├── test_database.py            # Database tests
└── test_search_engine.py       # Search engine tests
```

### Running Unit Tests

#### 1. Face Recognition Engine Tests

```bash
pytest tests/test_face_recognition.py -v
```

**What it tests:**
- Face engine initialization
- Face detection (with empty/sample images)
- Embedding extraction
- Face comparison
- Distance calculations
- Batch processing

**Expected Output:**
```
test_face_recognition.py::test_face_engine_initialization PASSED
test_face_recognition.py::test_detect_faces_empty_image PASSED
test_face_recognition.py::test_extract_embeddings_empty PASSED
test_face_recognition.py::test_compare_faces PASSED
test_face_recognition.py::test_face_distance PASSED
test_face_recognition.py::test_process_image_nonexistent PASSED
test_face_recognition.py::test_batch_process_images PASSED
```

#### 2. Database Tests

```bash
pytest tests/test_database.py -v
```

**What it tests:**
- Database initialization
- Adding faces
- Retrieving embeddings
- Searching similar faces
- Statistics generation
- Database clearing

**Expected Output:**
```
test_database.py::test_database_initialization PASSED
test_database.py::test_add_face PASSED
test_database.py::test_get_all_embeddings PASSED
test_database.py::test_get_face_by_id PASSED
test_database.py::test_get_faces_by_username PASSED
test_database.py::test_search_similar_faces PASSED
test_database.py::test_get_statistics PASSED
test_database.py::test_clear_database PASSED
```

#### 3. Search Engine Tests

```bash
pytest tests/test_search_engine.py -v
```

**What it tests:**
- Search engine initialization
- Search by embedding
- Search by image
- Username extraction
- Result grouping
- Top username ranking

**Expected Output:**
```
test_search_engine.py::test_search_engine_initialization PASSED
test_search_engine.py::test_search_by_embedding_empty PASSED
test_search_engine.py::test_search_by_embedding_with_data PASSED
test_search_engine.py::test_get_unique_usernames PASSED
test_search_engine.py::test_get_results_by_username PASSED
test_search_engine.py::test_get_top_usernames PASSED
```

### Test Coverage

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

**Current Coverage:** ~80% (estimated)

---

## 🔗 Integration Testing

### Manual Integration Tests

Since Firebase integration is not yet implemented, integration tests focus on the local system.

#### Test Workflow: Add Face → Search

```python
# Example integration test script
from src.database import FaceDatabase
from src.face_recognition_engine import FaceRecognitionEngine
from src.search_engine import SearchEngine
from src.image_utils import ImageProcessor
import numpy as np

# 1. Initialize components
db = FaceDatabase()
face_engine = FaceRecognitionEngine()
search_engine = SearchEngine(db)

# 2. Load test image
image = ImageProcessor.load_image("test_image.jpg")
if image is None:
    print("Failed to load image")
    exit(1)

# 3. Detect and extract faces
face_locations = face_engine.detect_faces(image)
if not face_locations:
    print("No faces detected")
    exit(1)

embeddings = face_engine.extract_face_embeddings(image, face_locations)

# 4. Add to database
for i, embedding in enumerate(embeddings):
    db.add_face(
        embedding.tolist(),
        f"test_user_{i}",
        "test_source"
    )

# 5. Search for similar faces
results = search_engine.search_by_image(image, threshold=0.6, top_k=10)
print(f"Found {results['total_matches']} matches")

# 6. Verify results
assert results['total_matches'] > 0, "Should find at least one match"
print("Integration test passed!")
```

---

## 🖱️ Manual Testing

### Testing the Streamlit Application

#### 1. Start the Application

```bash
streamlit run src/app.py
```

The application will open in your browser at `http://localhost:8501`

#### 2. Test Search Functionality

**Steps:**
1. Navigate to the "🔎 Search" tab
2. Upload an image with faces (JPG, PNG, etc.)
3. Click "🔍 Search"
4. Verify results are displayed
5. Adjust similarity threshold in sidebar
6. Test with different images

**Expected Behavior:**
- Image uploads successfully
- Faces are detected
- Search results are displayed
- Results show usernames, match counts, and similarity scores
- Adjusting threshold changes results

**Test Cases:**
- ✅ Upload image with one face
- ✅ Upload image with multiple faces
- ✅ Upload image with no faces (should show warning)
- ✅ Upload invalid file format (should show error)
- ✅ Upload very large image (should handle gracefully)
- ✅ Search with empty database (should show "no matches")

#### 3. Test Add Faces Functionality

**Steps:**
1. Navigate to "📤 Add Faces" tab
2. Upload an image with faces
3. Enter username
4. Select source type (profile_pic, post, story, reel)
5. Click "➕ Add to Database"
6. Verify success message

**Expected Behavior:**
- Image uploads successfully
- Faces are detected
- Faces are added to database
- Success message is displayed
- Database statistics update

**Test Cases:**
- ✅ Add single face
- ✅ Add multiple faces from one image
- ✅ Add face without username (should show error)
- ✅ Add face without image (should show error)
- ✅ Verify face appears in search results

#### 4. Test Analytics Dashboard

**Steps:**
1. Navigate to "📈 Analytics" tab
2. View database statistics
3. Check total faces count
4. Check unique users count
5. View sources breakdown

**Expected Behavior:**
- Statistics are displayed correctly
- Charts are rendered
- Data matches actual database content

**Test Cases:**
- ✅ View statistics with empty database
- ✅ View statistics with populated database
- ✅ Verify counts are accurate
- ✅ Check source distribution chart

---

## 🎯 Testing Current Capabilities

### What the Current Version Can Do

#### ✅ Working Features

1. **Face Detection**
   - Detects faces in uploaded images
   - Handles multiple faces per image
   - Works with various image formats

2. **Face Embedding Extraction**
   - Extracts 512-dimensional embeddings
   - Uses VGGFace2 model via DeepFace
   - Processes batch images

3. **Local Database Storage**
   - Stores face embeddings in JSON format
   - Persists data between sessions
   - Supports metadata (username, source, timestamps)

4. **Similarity Search**
   - Searches for similar faces
   - Ranks results by similarity
   - Configurable threshold
   - Top-K results

5. **Web Interface**
   - Streamlit-based UI
   - Image upload
   - Search interface
   - Analytics dashboard

#### ⚠️ Limitations

1. **No Firebase Integration**
   - Only local JSON database
   - No cloud storage
   - No real-time sync

2. **No Automated Ingestion**
    - Manual face addition only
    - No automatic data collection

3. **Embedding Dimension Mismatch**
   - Database expects 128-dim embeddings
   - Engine produces 512-dim embeddings
   - **Note:** This needs to be fixed

4. **Limited Test Data**
    - No real production data feed
    - Limited sample images

---

## 📊 Test Data Preparation

### Creating Test Images

#### Option 1: Use Sample Images

```bash
# Create test_images directory
mkdir -p test_images

# Add sample images with faces
# Images should be in JPG, PNG format
# Recommended: 640x480 or larger
```

#### Option 2: Generate Test Data Programmatically

```python
# test_data_generator.py
from src.database import FaceDatabase
from src.face_recognition_engine import FaceRecognitionEngine
from src.image_utils import ImageProcessor
import numpy as np

def create_test_database():
    """Create a test database with sample faces"""
    db = FaceDatabase("data/test_database.json")
    face_engine = FaceRecognitionEngine()
    
    # Add test faces
    test_images = [
        "test_images/person1.jpg",
        "test_images/person2.jpg",
        # ... more images
    ]
    
    for img_path in test_images:
        image = ImageProcessor.load_image(img_path)
        if image:
            face_locations = face_engine.detect_faces(image)
            embeddings = face_engine.extract_face_embeddings(image, face_locations)
            
            for embedding in embeddings:
                db.add_face(
                    embedding.tolist(),
                    f"test_user_{len(db.data['faces'])}",
                    "test_source"
                )
    
    print(f"Created test database with {len(db.data['faces'])} faces")

if __name__ == "__main__":
    create_test_database()
```

### Test Scenarios

#### Scenario 1: Empty Database
- Start with fresh database
- Verify no matches found
- Test error handling

#### Scenario 2: Single User
- Add multiple faces of same person
- Search for that person
- Verify all faces are found

#### Scenario 3: Multiple Users
- Add faces from different people
- Search for specific person
- Verify correct matches

#### Scenario 4: Similar Faces
- Add faces of similar-looking people
- Test threshold sensitivity
- Verify ranking accuracy

---

## 🔧 Troubleshooting

### Common Issues

#### 1. DeepFace Import Error

**Error:**
```
ModuleNotFoundError: No module named 'deepface'
```

**Solution:**
```bash
pip install deepface
```

**Note:** DeepFace has heavy dependencies (TensorFlow). If installation fails, the system will run in degraded mode (no face detection).

#### 2. OpenCV Not Available

**Warning:**
```
cv2 not available, using PIL fallback
```

**Solution:**
```bash
pip install opencv-python-headless
```

**Note:** The system works without OpenCV but with reduced functionality.

#### 3. Database Not Found

**Error:**
```
FileNotFoundError: data/faces_database.json
```

**Solution:**
- The database is created automatically on first use
- Ensure `data/` directory exists and is writable

#### 4. No Faces Detected

**Possible Causes:**
- Image has no faces
- Image quality too low
- Face too small or obscured
- DeepFace not properly installed

**Solution:**
- Use clear images with visible faces
- Ensure faces are front-facing
- Check DeepFace installation

#### 5. Search Returns No Results

**Possible Causes:**
- Database is empty
- Similarity threshold too high
- Embeddings don't match

**Solution:**
- Add faces to database first
- Lower similarity threshold
- Verify embeddings are being stored correctly

### Debug Mode

Enable debug logging:

```python
# In src/config.py or environment variable
DEBUG = True
LOG_LEVEL = "DEBUG"
```

Or set environment variable:
```bash
export DEBUG=True
export LOG_LEVEL=DEBUG
streamlit run src/app.py
```

### View Logs

```bash
# Check log files
tail -f logs/socialvision.log
tail -f logs/src_face_recognition_engine.log
tail -f logs/src_database.log
```

---

## 📈 Performance Testing

### Benchmark Tests

```python
# performance_test.py
import time
from src.database import FaceDatabase
from src.face_recognition_engine import FaceRecognitionEngine
from src.search_engine import SearchEngine

def benchmark_search():
    """Benchmark search performance"""
    db = FaceDatabase()
    search_engine = SearchEngine(db)
    
    # Add test data
    # ... (add faces)
    
    # Benchmark search
    start = time.time()
    results = search_engine.search_by_embedding(
        test_embedding,
        threshold=0.6,
        top_k=50
    )
    elapsed = time.time() - start
    
    print(f"Search completed in {elapsed:.2f}s")
    print(f"Found {len(results)} results")
    print(f"Throughput: {len(results)/elapsed:.2f} results/sec")
```

### Load Testing

```python
# load_test.py
import concurrent.futures
from src.database import FaceDatabase
from src.search_engine import SearchEngine

def load_test_search(num_searches=100):
    """Test search under load"""
    db = FaceDatabase()
    search_engine = SearchEngine(db)
    
    def perform_search():
        # Perform search
        results = search_engine.search_by_embedding(
            test_embedding,
            threshold=0.6
        )
        return len(results)
    
    # Run concurrent searches
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(perform_search) for _ in range(num_searches)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    print(f"Completed {num_searches} searches")
    print(f"Average results: {sum(results)/len(results):.2f}")
```

---

## ✅ Test Checklist

### Pre-Release Testing

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Manual UI testing completed
- [ ] Error handling tested
- [ ] Edge cases tested
- [ ] Performance acceptable
- [ ] Documentation updated
- [ ] Logs reviewed for errors

### Feature Testing

- [ ] Face detection works
- [ ] Embedding extraction works
- [ ] Database operations work
- [ ] Search functionality works
- [ ] UI is responsive
- [ ] Analytics display correctly
- [ ] Error messages are clear

---

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Streamlit Testing](https://docs.streamlit.io/)
- [DeepFace Documentation](https://github.com/serengil/deepface)

---

*Last Updated: December 2024*

