import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import io
import base64

# st.set_option('server.maxUploadSize', 400)

from utils.detector import detect
from utils.preprocess import preprocess
from utils.ocr import read_plate
from utils.draw import draw

# Page configuration
st.set_page_config(
    page_title="License Plate Recognition", 
    page_icon="🚗", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Card styling */
    .card {
        background: linear-gradient(145deg, #ffffff, #f0f2f6);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
    }
    
    /* Meter styling */
    .meter-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 20px;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    /* Stat card styling */
    .stat-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
    }
    
    .stat-card-green {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
    }
    
    .stat-card-purple {
        background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
        box-shadow: 0 10px 30px rgba(161, 140, 209, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Image container */
    .image-container {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    
    /* Title gradient */
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
    }
    
    /* Success badge */
    .success-badge {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Download button container */
    .download-btn-container {
        margin-top: 10px;
    }
    
    /* Custom download link styling */
    .custom-download-link {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 12px 24px;
        border-radius: 50px;
        text-decoration: none;
        font-weight: 600;
        text-align: center;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .custom-download-link:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        color: white !important;
        text-decoration: none;
    }
    </style>
""", unsafe_allow_html=True)

# Title with gradient
st.markdown('<p class="gradient-text">🚗 License Plate Recognition System</p>', unsafe_allow_html=True)
st.markdown("*Powered by YOLOv8 + EasyOCR*")

# Sidebar for settings and information
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for detection"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    
    # Placeholder for stats
    if 'detections' in st.session_state and len(st.session_state.detections) > 0:
        total_detections = len(st.session_state.detections)
        avg_confidence = np.mean([d['confidence'] for d in st.session_state.detections]) if total_detections > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Plates", total_detections)
        with col2:
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    else:
        st.info("No detections yet. Upload an image to get started!")
    
    st.markdown("---")
    st.markdown("### 📁 Export Data")
    
    # Export functionality - Fixed button
    if 'detections' in st.session_state and len(st.session_state.detections) > 0:
        # Create DataFrame
        df = pd.DataFrame(st.session_state.detections)
        
        # Add timestamp columns
        df['detection_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df['confidence_percent'] = (df['confidence'] * 100).round(1)
        
        # Create CSV
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        filename = f"license_plates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Create download button
        st.markdown(f"""
        <div class="download-btn-container">
            <a href="data:file/csv;base64,{b64}" 
               download="{filename}"
               class="custom-download-link">
                📥 Download CSV ({len(st.session_state.detections)} plates)
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        # Show preview
        with st.expander("📋 Preview Data"):
            st.dataframe(df[['plate_number', 'confidence_percent', 'detection_time']], use_container_width=True)
    else:
        st.warning("No data to export. Upload an image first!")

# Main upload section
uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"],
    max_upload_size=800,
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file:
    # Initialize detections in session state
    if 'detections' not in st.session_state:
        st.session_state.detections = []
    
    # Process image
    with st.spinner("🔄 Processing image..."):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Clear previous detections
        st.session_state.detections = []
        
        # Detection
        results = detect(image)
        
        # Process each detection
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            if conf >= confidence_threshold:
                plate = image[y1:y2, x1:x2]
                
                if plate.size > 0:
                    processed = preprocess(plate)
                    text = read_plate(processed)
                    
                    # Draw on image
                    draw(image, (x1, y1, x2, y2), text, conf)
                    
                    # Store detection data
                    st.session_state.detections.append({
                        'plate_number': text if text else "Unknown",
                        'confidence': conf,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    })
        
        # Convert image for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display results in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📸 Processed Image")
        st.image(
            image_rgb,
            caption=f"Detected {len(st.session_state.detections)} License Plate(s)",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Detection Summary")
        
        total_plates = len(st.session_state.detections)
        st.markdown(f"**Total Plates Detected:** {total_plates}")
        
        if total_plates > 0:
            # Confidence meter
            confidences = [d['confidence'] for d in st.session_state.detections]
            avg_conf = np.mean(confidences) * 100
            
            st.markdown("#### Confidence Meter")
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = avg_conf,
                title = {'text': "Average Confidence"},
                delta = {'reference': 80, 'increasing': {'color': "RebeccaPurple"}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#764ba2"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(255, 0, 0, 0.1)'},
                        {'range': [50, 80], 'color': 'rgba(255, 165, 0, 0.1)'},
                        {'range': [80, 100], 'color': 'rgba(0, 255, 0, 0.1)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#2c3e50"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detected plates list
            st.markdown("#### 📋 Detected Plates")
            for i, det in enumerate(st.session_state.detections, 1):
                plate_text = det['plate_number'] if det['plate_number'] != "Unknown" else "❓ Unknown"
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                                border-radius: 10px;
                                padding: 10px;
                                margin: 5px 0;
                                display: flex;
                                justify-content: space-between;
                                align-items: center;">
                        <span><b>#{i}</b> {plate_text}</span>
                        <span class="success-badge">{det['confidence']:.1%}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Confidence distribution chart
            st.markdown("#### 📊 Confidence Distribution")
            fig_dist = go.Figure(data=[
                go.Bar(
                    x=[f"Plate {i+1}" for i in range(len(confidences))],
                    y=[c * 100 for c in confidences],
                    marker_color='#667eea',
                    marker_line_color='#764ba2',
                    marker_line_width=2,
                    text=[f"{c:.1%}" for c in confidences],
                    textposition='auto',
                )
            ])
            
            fig_dist.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#2c3e50"},
                xaxis_title="",
                yaxis_title="Confidence (%)",
                yaxis_range=[0, 100]
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("No plates detected in this image. Try adjusting the confidence threshold.")
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Welcome screen with instructions
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 50px 20px;">
            <h1 style="font-size: 4rem; margin-bottom: 0;">🚗</h1>
            <h2 style="color: #667eea;">Ready to Detect License Plates</h2>
            <p style="color: #666; font-size: 1.2rem;">
                Upload an image to start the recognition process.
                <br>Our AI will detect and read license plates automatically.
            </p>
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 20px;
                        padding: 20px;
                        margin-top: 20px;
                        color: white;">
                <h4>✨ Features</h4>
                <p>✅ Real-time detection using YOLOv8<br>
                   ✅ OCR powered by EasyOCR<br>
                   ✅ Confidence scoring for each detection<br>
                   ✅ Export results to CSV<br>
                   ✅ Interactive visualizations</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 20px;">
        Made with ❤️ using Streamlit | YOLOv8 | EasyOCR
    </div>
    """,
    unsafe_allow_html=True
)