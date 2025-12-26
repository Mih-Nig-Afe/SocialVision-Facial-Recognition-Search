"""
Live video component for continuous face recognition.
Uses JavaScript to capture frames from webcam and stream to Python for processing.
"""

import streamlit as st
import streamlit.components.v1 as components
import base64
import numpy as np
import cv2
from typing import Optional, List, Tuple, Dict, Any


def get_continuous_video_html(component_id: str = "live_video") -> str:
    """
    Generate HTML/JS for continuous video capture that works with Streamlit.
    Uses a hidden form to submit frames for processing.
    """
    return f"""
<div id="{component_id}_container" style="position: relative; max-width: 640px; margin: 0 auto;">
    <video id="{component_id}_video" autoplay playsinline muted
           style="width: 100%; border-radius: 8px; background: #1a1a2e;"></video>
    <canvas id="{component_id}_canvas" style="display: none;"></canvas>
    <div id="{component_id}_overlay" style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; pointer-events: none;"></div>
    <div id="{component_id}_status" style="position: absolute; top: 10px; left: 10px;
         background: rgba(0,0,0,0.7); color: #fff; padding: 5px 12px; border-radius: 20px; font-size: 13px;">
        📷 Initializing...
    </div>
</div>
<script>
(function() {{
    const id = "{component_id}";
    const video = document.getElementById(id + "_video");
    const canvas = document.getElementById(id + "_canvas");
    const status = document.getElementById(id + "_status");
    const ctx = canvas.getContext("2d");
    let stream = null;
    let captureInterval = null;

    async function startCamera() {{
        try {{
            stream = await navigator.mediaDevices.getUserMedia({{
                video: {{ width: 640, height: 480, facingMode: "user" }},
                audio: false
            }});
            video.srcObject = stream;
            await video.play();
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            status.innerHTML = "🟢 Live - Capturing frames";
            status.style.background = "rgba(0,128,0,0.8)";

            // Start capturing frames
            startCapture();
        }} catch (err) {{
            status.innerHTML = "❌ Camera error: " + err.message;
            status.style.background = "rgba(255,0,0,0.8)";
        }}
    }}

    function captureFrame() {{
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL("image/jpeg", 0.7);
    }}

    function startCapture() {{
        // Store frame in session storage for Streamlit to read
        captureInterval = setInterval(() => {{
            const frame = captureFrame();
            try {{
                sessionStorage.setItem("{component_id}_frame", frame);
                sessionStorage.setItem("{component_id}_timestamp", Date.now().toString());
            }} catch(e) {{
                console.warn("Storage full, clearing...");
                sessionStorage.clear();
            }}
        }}, 200);  // Capture every 200ms
    }}

    // Cleanup on page unload
    window.addEventListener("beforeunload", () => {{
        if (stream) stream.getTracks().forEach(t => t.stop());
        if (captureInterval) clearInterval(captureInterval);
    }});

    startCamera();
}})();
</script>
"""


def decode_frame(base64_data: str) -> Optional[np.ndarray]:
    """Decode a base64 JPEG frame to numpy array (BGR format)."""
    try:
        if not base64_data:
            return None
        # Remove data URL prefix if present
        if "," in base64_data:
            base64_data = base64_data.split(",")[1]

        img_bytes = base64.b64decode(base64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def encode_frame(image: np.ndarray, quality: int = 80) -> str:
    """Encode numpy array (BGR) to base64 JPEG."""
    try:
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode("utf-8")
    except Exception:
        return ""
