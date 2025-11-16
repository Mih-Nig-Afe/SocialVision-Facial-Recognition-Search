# SocialVision: Advanced Facial Recognition Search Engine

**Advanced Facial Recognition Search Engine for Instagram Content Analysis**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39+-red.svg)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18+-orange.svg)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-green.svg)](https://opencv.org)
[![Firebase](https://img.shields.io/badge/Firebase-10.0+-orange.svg)](https://firebase.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

SocialVision is an advanced facial recognition search engine designed for Instagram content analysis. This academic research project demonstrates the practical application of computer vision, machine learning, and cloud computing technologies using exclusively free resources.

**Developer:** Mihretab N. Afework  
**GitHub:** [@Mih-Nig-Afe](https://github.com/Mih-Nig-Afe)  
**Email:** <mtabdevt@gmail.com>  

## 🚀 Key Features

- **Advanced Facial Recognition** using Python's face_recognition library
- **Real-time Search Engine** with Streamlit web interface
- **Cloud-based Storage** using Firebase Firestore
- **Zero-Budget Implementation** using only free resources
- **Ethical AI Development** with privacy-first approach
- **Academic Research Focus** with comprehensive documentation

## 🛠️ Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Frontend** | Streamlit | 1.39+ | Web application framework |
| **Alternative UI** | Gradio | 4.44+ | ML-focused web interface |
| **Backend** | FastAPI/Flask | 0.115+/3.1+ | API development |
| **Database** | Firebase Firestore | 2.19+ | NoSQL document database |
| **ML/AI** | OpenCV + face_recognition | 4.10+/1.3+ | Computer vision and facial recognition |
| **Deep Learning** | TensorFlow/PyTorch | 2.18+/2.5+ | Neural networks and ML models |
| **Data Processing** | NumPy + Pandas | 2.1+/2.2+ | Numerical computing and data analysis |
| **Cloud Storage** | Firebase Storage | 2.18+ | Image and media storage |
| **Image Processing** | Pillow + scikit-image | 11.0+/0.24+ | Advanced image manipulation |

## 📋 Project Structure

```text
SocialVision-Facial-Recognition-Search/
├── docs/                          # Documentation directory
│   ├── README.md                  # Documentation index
│   ├── PROJECT_STATUS.md          # Current project status and capabilities
│   ├── TESTING_GUIDE.md           # Comprehensive testing guide
│   ├── DEVELOPMENT_ROADMAP.md     # Development plan and roadmap
│   └── SocialVision_Technical_Proposal.md  # Technical proposal
├── src/                           # Source code
│   ├── app.py                     # Main Streamlit application
│   ├── face_recognition_engine.py # Face recognition engine
│   ├── database.py                # Local JSON database
│   ├── search_engine.py           # Search engine
│   ├── image_utils.py             # Image processing utilities
│   ├── config.py                  # Configuration management
│   └── logger.py                  # Logging setup
├── tests/                         # Test suite
│   ├── test_face_recognition.py   # Face recognition tests
│   ├── test_database.py           # Database tests
│   └── test_search_engine.py      # Search engine tests
├── data/                          # Data directory (database, etc.)
├── config/                        # Configuration files
├── logs/                          # Log files
├── uploads/                       # Uploaded images
├── models/                        # ML models directory
├── requirements.txt               # Python dependencies
├── pytest.ini                    # Pytest configuration
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose config
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## 🎓 Academic Objectives

- **Technical Mastery:** Advanced Python development with ML/AI libraries
- **Research Contribution:** Novel approach to social media content analysis
- **Ethical AI:** Responsible development practices and privacy considerations
- **Open Source:** Community contribution and knowledge sharing

## 📖 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **[📊 Project Status](docs/PROJECT_STATUS.md)** - Current project status, completed features, and capabilities
- **[🧪 Testing Guide](docs/TESTING_GUIDE.md)** - How to test the current version and capabilities
- **[🗺️ Development Roadmap](docs/DEVELOPMENT_ROADMAP.md)** - Development plan, phases, and future goals
- **[📋 Technical Proposal](docs/SocialVision_Technical_Proposal.md)** - Comprehensive technical proposal and architecture
- **[📚 Documentation Index](docs/README.md)** - Complete documentation index

### Quick Links

- **Current Status:** 60% Complete - See [PROJECT_STATUS.md](docs/PROJECT_STATUS.md)
- **How to Test:** See [TESTING_GUIDE.md](docs/TESTING_GUIDE.md)
- **What's Next:** See [DEVELOPMENT_ROADMAP.md](docs/DEVELOPMENT_ROADMAP.md)

## 🔧 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Installation Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/Mih-Nig-Afe/SocialVision-Facial-Recognition-Search.git
   cd SocialVision-Facial-Recognition-Search
   ```

2. **Create virtual environment (recommended)**

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure Firebase**

   ```bash
   # Add your Firebase configuration
   cp config/firebase_config_template.json config/firebase_config.json
   ```

5. **Run the application**

   **Option A: Using Docker (Recommended)**
   ```bash
   # Build and start with Docker
   docker-compose build
   docker-compose up -d
   
   # Or use the quick demo script
   ./docker-demo.sh
   
   # Access at http://localhost:8501
   ```

   **Option B: Local Installation**
   ```bash
   streamlit run src/app.py
   ```

   **For detailed Docker instructions, see [DOCKER_TESTING_GUIDE.md](docs/DOCKER_TESTING_GUIDE.md)**

### System Requirements

- **RAM**: Minimum 4GB (8GB recommended for ML operations)
- **Storage**: 2GB free space
- **Internet**: Required for Firebase and model downloads

## 📊 Project Status

**Current Development Level:** 60% Complete  
**Current Phase:** Phase 4 (User Interface Development) - In Progress

### Completed Phases ✅
- **Phase 1:** Foundation and Setup (100%)
- **Phase 3:** Search Engine Development (100%)

### In Progress ⚠️
- **Phase 2:** Data Collection and Processing (50%)
- **Phase 4:** User Interface Development (90%)
- **Phase 5:** Testing and Optimization (40%)

### Key Features Completed
- ✅ Face detection and embedding extraction
- ✅ Local JSON database system
- ✅ Similarity search engine
- ✅ Streamlit web interface
- ✅ Image processing utilities
- ✅ Unit test suite

For detailed status, see [PROJECT_STATUS.md](docs/PROJECT_STATUS.md)

## 🤝 Contributing

This is an academic research project. Contributions, suggestions, and feedback are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

### Mihretab N. Afework

- **GitHub**: [@Mih-Nig-Afe](https://github.com/Mih-Nig-Afe)
- **Email**: <mtabdevt@gmail.com>
- **Project**: [SocialVision-Facial-Recognition-Search](https://github.com/Mih-Nig-Afe/SocialVision-Facial-Recognition-Search)
- **LinkedIn**: [Connect for collaboration](https://linkedin.com/in/mihretab-afework)

---

*This project is developed for academic and research purposes, demonstrating ethical AI development practices and responsible use of facial recognition technology.*
