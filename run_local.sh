#!/bin/bash
# Run Streamlit directly on macOS for live camera WebRTC support
# Docker containers on macOS don't support WebRTC properly due to VM isolation

echo "🔧 Setting up local environment for live camera support..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Set environment variables from .env
export $(grep -v '^#' .env | xargs)

# Run Streamlit
echo "🚀 Starting Streamlit with live camera support..."
echo "Open http://localhost:8501 in your browser"
streamlit run src/app.py --server.port 8501 --server.address localhost

