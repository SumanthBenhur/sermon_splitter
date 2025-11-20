import streamlit as st
from sermon_splitter import SermonSplitterApp, VideoUtils
from pathlib import Path

# --- Page Config ---
st.set_page_config(page_title="Sermon Splitter AI", layout="wide", page_icon="⚡")

# --- Custom CSS for Futuristic Theme ---
CUSTOM_CSS = """
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        background-image: radial-gradient(circle at 50% 0%, #1c2333 0%, #0e1117 70%);
        color: #e0e0e0;
        font-family: 'Roboto', sans-serif;
    }

    /* Title Styling */
    .title-text {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 0 20px rgba(0, 210, 255, 0.3);
    }
    
    .subtitle-text {
        font-size: 1.2rem;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }

    /* Input Fields */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.05);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px 15px;
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00d2ff;
        box-shadow: 0 0 10px rgba(0, 210, 255, 0.2);
        background-color: rgba(255, 255, 255, 0.08);
    }

    /* Number Input */
    .stNumberInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.05);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.5);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* Cards/Containers */
    .css-1r6slb0, .stExpander {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Expander Header */
    .streamlit-expanderHeader {
        background-color: transparent;
        color: #e0e0e0;
        font-weight: 600;
    }

    /* Success/Info Messages */
    .stAlert {
        background-color: rgba(0, 210, 255, 0.1);
        border: 1px solid rgba(0, 210, 255, 0.2);
        color: #e0e0e0;
        border-radius: 10px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="title-text">SERMON SPLITTER AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Advanced Video Processing & Transcription Engine</div>', unsafe_allow_html=True)

# --- Main Content ---
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📂 Source File")
    input_source = st.text_input("Absolute path to local video file:", placeholder="/path/to/video.mp4")

source_video_path = ""
if input_source:
    path_obj = Path(input_source)
    if not path_obj.is_file():
        st.error("❌ Invalid file path.")
    else:
        source_video_path = input_source

if source_video_path:
    with col2:
        st.markdown("### 🎥 Preview")
        st.video(str(source_video_path))

    st.markdown("---")
    st.markdown("### ✂️ Clip Configuration")

    c1, c2 = st.columns([1, 3])
    with c1:
        num_clips = st.number_input("Number of clips to extract", min_value=1, value=1, step=1)
        output_filename = st.text_input("Output Filename", value="clip.mp4")

    with c2:
        clips_data = []
        
        # Get video duration
        video_duration = VideoUtils.get_video_duration(str(source_video_path))
        
        with st.expander("Clip Timestamps", expanded=True):
            for i in range(num_clips):
                st.markdown(f"**Clip {i+1}**")
                
                # Slider for selection
                start_sec, end_sec = st.slider(
                    f"Select Range for Clip {i+1}",
                    min_value=0.0,
                    max_value=video_duration,
                    value=(0.0, min(10.0, video_duration)),
                    step=1.0,
                    key=f"slider_{i}"
                )
                
                # Convert to HH:MM:SS
                def seconds_to_hms(seconds):
                    m, s = divmod(seconds, 60)
                    h, m = divmod(m, 60)
                    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

                start_time = seconds_to_hms(start_sec)
                end_time = seconds_to_hms(end_sec)
                
                st.caption(f"Selected: {start_time} - {end_time}")
                clips_data.append({"start_time": start_time, "end_time": end_time})

    st.markdown("---")
    
    # Center the button
    b1, b2, b3 = st.columns([1, 2, 1])
    with b2:
        process_btn = st.button("🚀 INITIALIZE PROCESSING SEQUENCE")

    if process_btn:
        if any(not clip["start_time"] or not clip["end_time"] for clip in clips_data):
            st.warning("⚠️ Please define start and end times for all clips.")
        else:
            try:
                with st.spinner("⚡ Processing video... AI models engaging..."):
                    app = SermonSplitterApp(source_path=str(source_video_path))
                    final_output_path = app.run(num_clips, clips_data, output_filename)
                    
                    st.success("✅ Processing Complete!")
                    st.balloons()
                    
                    # Result Area
                    r1, r2 = st.columns([1, 1])
                    with r1:
                        st.info(f"Output saved to: `{final_output_path}`")
                    with r2:
                        with open(final_output_path, "rb") as f:
                            st.download_button(
                                label="⬇️ Download Processed Video",
                                data=f,
                                file_name=Path(final_output_path).name,
                                mime="video/mp4"
                            )
            except Exception as e:
                st.error(f"🛑 System Error: {e}")