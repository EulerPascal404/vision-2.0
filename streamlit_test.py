import streamlit as st
from PIL import Image
from io import BytesIO
import time
import random
from datetime import datetime
import uuid
import requests
import base64
import numpy as np
from cartesia import Cartesia
import wave

# ----------------------------- #
#     PAGE CONFIGURATION        #
# ----------------------------- #

# Set page configuration at the very top, before any other Streamlit commands
st.set_page_config(
    page_title="Summary Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------- #
#         CONFIGURATION         #
# ----------------------------- #

# PHP Server API Endpoint (Commented out for simulation)
# API_ENDPOINT = "https://yourphpserver.com/api/get_latest_summary"

# Refresh interval in seconds
REFRESH_INTERVAL = 5  # Adjust as needed

# ----------------------------- #
#          STYLING CSS          #
# ----------------------------- #

# Custom CSS for sleek and modern design
def add_custom_css():
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Lato:wght@400;700&display=swap" rel="stylesheet">
        <style>
        /* Apply Google Fonts */
        body, .title, .section-header, .text-summary, .footer {
            font-family: 'Roboto', sans-serif;
        }
        /* Background */
        .stApp {
            background-color: #F5F7FA;
            color: #333333;
        }
        /* Title */
        .title {
            font-size: 3em;
            text-align: center;
            color: #2C3E50;
            margin-bottom: 20px;
            font-weight: 700;
        }
        /* Section Headers */
        .section-header {
            font-size: 2em;
            color: #34495E;
            border-bottom: 3px solid #1ABC9C;
            padding-bottom: 10px;
            margin-top: 50px;
            margin-bottom: 20px;
            font-weight: 500;
        }
        /* Summary Card */
        .summary-card {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .summary-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.15);
        }
        /* Text Summary */
        .text-summary {
            font-size: 1.1em;
            color: #555555;
            line-height: 1.6;
        }
        /* Image Styling */
        .summary-image {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            margin-top: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        /* Button Styling */
        .stButton > button {
            background-color: #1ABC9C;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 8px;
            font-size: 1em;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.3s, transform 0.2s;
        }
        .stButton > button:hover {
            background-color: #16A085;
            transform: scale(1.05);
        }
        /* Footer */
        .footer {
            text-align: center;
            font-size: 0.9em;
            color: #7F8C8D;
            margin-top: 60px;
            padding-top: 20px;
            border-top: 1px solid #DDDDDD;
        }
        /* Responsive Columns */
        @media (max-width: 768px) {
            .summary-container {
                flex-direction: column;
            }
        }
        /* Remove default padding/margin for summary cards */
        .summary-card-container {
            margin-top: 0px;
        }
        /* Optional: Remove horizontal rules styling */
        hr {
            border: none;
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ----------------------------- #
#          DATA FETCHING        #
# ----------------------------- #

@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_latest_summary_simulated():
    """
    Simulate fetching the latest summary from the PHP server.
    Replace this function with actual API calls when the PHP endpoint is available.
    """
    # Simulate a delay as if fetching from an API
    time.sleep(1)  # Simulate network latency

    # Generate simulated data
    generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    unique_id = str(uuid.uuid4())  # Generate unique id
    simulated_data = {
        "id": unique_id,
        "image_url": f"https://picsum.photos/seed/{random.randint(1, 1000)}/600/400",
        "text_summary": f"This is a simulated summary generated at {generated_time}."
    }
    return simulated_data

# Uncomment the following function and comment out `fetch_latest_summary_simulated` when the PHP API is ready

# @st.cache_data(ttl=REFRESH_INTERVAL)
# def fetch_latest_summary():
#     try:
#         response = requests.get(API_ENDPOINT)
#         if response.status_code == 200:
#             data = response.json()
#             return data
#         else:
#             st.error(f"Error fetching data: {response.status_code}")
#             return None
#     except Exception as e:
#         st.error(f"Exception occurred: {e}")
#         return None


# ----------------------------- #
#       TEXT TO SPEECH FUNCTION  #
# ----------------------------- #
def cartesia_text_to_speech(text, api_key):
    """
    Convert text to speech using Cartesia AI Voice API.
    
    Args:
        text (str): The text to convert to speech.
        api_key (str): Your Cartesia AI Voice API key.
    
    Returns:
        str: Base64-encoded WAV audio content or None if failed.
    """
    client = Cartesia(api_key=api_key)
    
    voice_id = "a0e99841-438c-4a64-b679-ae501e7d6091"  # Barbershop Man
    model_id = "sonic-english"
    transcript = text
    
    output_format = {
        "container": "raw",
        "encoding": "pcm_f32le",
        "sample_rate": 44100,
    }
    
    # Set up a WebSocket connection.
    ws = client.tts.websocket()
    
    # Create a BytesIO buffer to hold PCM data
    pcm_buffer = BytesIO()
    
    # Generate and stream audio.
    try:
        for output in ws.send(
            model_id=model_id,
            transcript=transcript,
            voice_id=voice_id,
            stream=True,
            output_format=output_format,
        ):
            buffer = output.get("audio")
            if buffer:
                pcm_buffer.write(buffer)
    except Exception as e:
        st.error(f"Error during Text-to-Speech streaming: {e}")
        ws.close()
        return None
    
    # Close the connection to release resources
    ws.close()
    
    # Retrieve PCM data
    pcm_data = pcm_buffer.getvalue()
    
    if not pcm_data:
        st.error("No audio data received from Cartesia.")
        return None
    
    # Convert PCM float32 to int16 for WAV
    try:
        float_data = np.frombuffer(pcm_data, dtype=np.float32)
        int_data = np.int16(float_data * 32767)
    except Exception as e:
        st.error(f"Error converting PCM data: {e}")
        return None
    
    # Create a BytesIO buffer for WAV
    wav_buffer = BytesIO()
    
    # Write WAV header and frames
    try:
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16 bits
            wf.setframerate(44100)
            wf.writeframes(int_data.tobytes())
    except Exception as e:
        st.error(f"Error writing WAV data: {e}")
        return None
    
    # Get WAV bytes
    wav_bytes = wav_buffer.getvalue()
    
    # Encode to base64
    wav_base64 = base64.b64encode(wav_bytes).decode('utf-8')
    
    return wav_base64

# ----------------------------- #
#           MAIN APP            #
# ----------------------------- #

def main():
    # Apply custom CSS
    add_custom_css()

    # Title
    st.markdown('<div class="title">📈 Summary Dashboard</div>', unsafe_allow_html=True)

    # Initialize Session State for Past Summaries
    if 'past_summaries' not in st.session_state:
        st.session_state.past_summaries = []

    # Button to manually fetch new data (optional)
    fetch_button = st.button("Fetch Latest Summary Now")

    if fetch_button or 'auto_fetch' not in st.session_state:
        # Fetch the latest summary
        latest_data = fetch_latest_summary_simulated()
        # For actual API call, replace the above line with:
        # latest_data = fetch_latest_summary()

        if latest_data:
            # Check if the latest data is already in past summaries
            if not st.session_state.past_summaries or \
               (st.session_state.past_summaries and latest_data != st.session_state.past_summaries[-1]):
                # Append the new summary to past summaries
                st.session_state.past_summaries.append(latest_data)

    # Insert your Cartesia AI Voice API key here
    CARTESIA_API_KEY = "19fbcc89-191b-487a-aca3-11b32496413e"  # Replace with your actual API key

    # Display Latest Summary
    if st.session_state.past_summaries:
        latest = st.session_state.past_summaries[-1]
        st.markdown('<div class="section-header">Latest Summary</div>', unsafe_allow_html=True)

        # Display the summary card with the button
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                # Embed the image using HTML to apply CSS classes
                st.markdown(
                    f"""
                    <img src="{latest['image_url']}" alt="Latest Image" class="summary-image" />
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(f"<p class='text-summary'>{latest['text_summary']}</p>", unsafe_allow_html=True)
                
                # Read Aloud button with unique key
                read_button_key = f"tts_button_{latest['id']}"
                if f"audio_{latest['id']}" not in st.session_state:
                    st.session_state[f"audio_{latest['id']}"] = None

                if st.button("🔊 Read Aloud", key=read_button_key):
                    audio_base64 = cartesia_text_to_speech(latest["text_summary"], CARTESIA_API_KEY)
                    if audio_base64:
                        st.session_state[f"audio_{latest['id']}"] = audio_base64

        # Display the audio player if audio is available
        if st.session_state[f"audio_{latest['id']}"]:
            audio_bytes = base64.b64decode(st.session_state[f"audio_{latest['id']}"])
            st.audio(audio_bytes, format='audio/mp3', start_time=0)

    # Display Past Summaries
    if len(st.session_state.past_summaries) > 1:
        st.markdown('<div class="section-header">Past Summaries</div>', unsafe_allow_html=True)
        for past in reversed(st.session_state.past_summaries[:-1]):
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    # Embed the image using HTML to apply CSS classes
                    st.markdown(
                        f"""
                        <img src="{past['image_url']}" alt="Past Image" class="summary-image" />
                        """,
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(f"<p class='text-summary'>{past['text_summary']}</p>", unsafe_allow_html=True)
                    
                    # Read Aloud button with unique key
                    read_button_key = f"tts_button_{past['id']}"
                    if f"audio_{past['id']}" not in st.session_state:
                        st.session_state[f"audio_{past['id']}"] = None

                    if st.button("🔊 Read Aloud", key=read_button_key):
                        audio_base64 = cartesia_text_to_speech(past["text_summary"], CARTESIA_API_KEY)
                        if audio_base64:
                            st.session_state[f"audio_{past['id']}"] = audio_base64

            # Display the audio player if audio is available
            if st.session_state[f"audio_{past['id']}"]:
                audio_bytes = base64.b64decode(st.session_state[f"audio_{past['id']}"])
                st.audio(audio_bytes, format='audio/mp3', start_time=0)

    # Footer
    st.markdown('<div class="footer">© 2024 Your Company. All rights reserved.</div>', unsafe_allow_html=True)

    # Auto-Refresh Mechanism
    # Using JavaScript to refresh the page
    refresh_script = f"""
    <script>
    setTimeout(function() {{
        window.location.reload();
    }}, {REFRESH_INTERVAL * 1000});
    </script>
    """
    st.markdown(refresh_script, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# import streamlit as st
# from PIL import Image
# from io import BytesIO
# import time
# import random
# from datetime import datetime
# import uuid
# import requests
# import base64
# import numpy as np
# import wave  # Ensure 'wave' is imported
# import asyncio
# import websockets
# import json
# from cartesia import Cartesia  # Ensure this is the correct import based on Cartesia's SDK

# # ----------------------------- #
# #     PAGE CONFIGURATION        #
# # ----------------------------- #

# st.set_page_config(
#     page_title="Summary Dashboard",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

# # ----------------------------- #
# #         CONFIGURATION         #
# # ----------------------------- #

# REFRESH_INTERVAL = 10  # seconds

# # ----------------------------- #
# #          STYLING CSS          #
# # ----------------------------- #

# def add_custom_css():
#     st.markdown(
#         """
#         <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Lato:wght@400;700&display=swap" rel="stylesheet">
#         <style>
#         /* Apply Google Fonts */
#         body, .title, .section-header, .text-summary, .footer {
#             font-family: 'Roboto', sans-serif;
#         }
#         /* Background */
#         .stApp {
#             background-color: #F5F7FA;
#             color: #333333;
#         }
#         /* Title */
#         .title {
#             font-size: 3em;
#             text-align: center;
#             color: #2C3E50;
#             margin-bottom: 20px;
#             font-weight: 700;
#         }
#         /* Section Headers */
#         .section-header {
#             font-size: 2em;
#             color: #34495E;
#             border-bottom: 3px solid #1ABC9C;
#             padding-bottom: 10px;
#             margin-top: 50px;
#             margin-bottom: 20px;
#             font-weight: 500;
#         }
#         /* Summary Card */
#         .summary-card {
#             background-color: #FFFFFF;
#             padding: 20px;
#             border-radius: 15px;
#             box-shadow: 0 8px 16px rgba(0,0,0,0.1);
#             margin-bottom: 30px;
#             transition: transform 0.2s, box-shadow 0.2s;
#         }
#         .summary-card:hover {
#             transform: translateY(-5px);
#             box-shadow: 0 12px 24px rgba(0,0,0,0.15);
#         }
#         /* Text Summary */
#         .text-summary {
#             font-size: 1.1em;
#             color: #555555;
#             line-height: 1.6;
#         }
#         /* Image Styling */
#         .summary-image {
#             max-width: 100%;
#             height: auto;
#             border-radius: 10px;
#             margin-top: 15px;
#             box-shadow: 0 4px 8px rgba(0,0,0,0.1);
#         }
#         /* Button Styling */
#         .stButton > button {
#             background-color: #1ABC9C;
#             color: white;
#             border: none;
#             padding: 10px 24px;
#             border-radius: 8px;
#             font-size: 1em;
#             font-weight: 500;
#             cursor: pointer;
#             transition: background-color 0.3s, transform 0.2s;
#         }
#         .stButton > button:hover {
#             background-color: #16A085;
#             transform: scale(1.05);
#         }
#         /* Footer */
#         .footer {
#             text-align: center;
#             font-size: 0.9em;
#             color: #7F8C8D;
#             margin-top: 60px;
#             padding-top: 20px;
#             border-top: 1px solid #DDDDDD;
#         }
#         /* Responsive Columns */
#         @media (max-width: 768px) {
#             .summary-container {
#                 flex-direction: column;
#             }
#         }
#         /* Remove default padding/margin for summary cards */
#         .summary-card-container {
#             margin-top: 0px;
#         }
#         /* Optional: Remove horizontal rules styling */
#         hr {
#             border: none;
#             margin: 0;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # ----------------------------- #
# #          DATA FETCHING        #
# # ----------------------------- #

# @st.cache_data(ttl=REFRESH_INTERVAL)
# def fetch_latest_summary_simulated():
#     """
#     Simulate fetching the latest summary from the PHP server.
#     Replace this function with actual API calls when the PHP endpoint is available.
#     """
#     # Simulate a delay as if fetching from an API
#     time.sleep(1)  # Simulate network latency

#     # Generate simulated data
#     generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     unique_id = str(uuid.uuid4())  # Generate unique id
#     simulated_data = {
#         "id": unique_id,
#         "image_url": f"https://picsum.photos/seed/{random.randint(1, 1000)}/600/400",
#         "text_summary": f"This is a simulated summary generated at {generated_time}."
#     }
#     return simulated_data

# # Uncomment the following function and comment out `fetch_latest_summary_simulated` when the PHP API is ready
# # @st.cache_data(ttl=REFRESH_INTERVAL)
# # def fetch_latest_summary():
# #     try:
# #         response = requests.get(API_ENDPOINT)
# #         if response.status_code == 200:
# #             data = response.json()
# #             return data
# #         else:
# #             st.error(f"Error fetching data: {response.status_code}")
# #             return None
# #     except Exception as e:
# #         st.error(f"Exception occurred: {e}")
# #         return None

# # ----------------------------- #
# #       TEXT TO SPEECH FUNCTION  #
# # ----------------------------- #

# def cartesia_text_to_speech(text, api_key):
#     """
#     Convert text to speech using Cartesia AI Voice API via WebSockets.
    
#     Args:
#         text (str): The text to convert to speech.
#         api_key (str): Your Cartesia AI Voice API key.
    
#     Returns:
#         str: Base64-encoded WAV audio content or None if failed.
#     """
#     async def tts():
#         uri = "wss://api.cartesia.ai/voice/v1/text-to-speech"  # Replace with actual WebSocket endpoint
        
#         headers = {
#             "Authorization": f"Bearer {api_key}",
#             "Content-Type": "application/json"
#         }
        
#         voice_id = "a0e99841-438c-4a64-b679-ae501e7d6091"  # Barbershop Man
#         model_id = "sonic-english"
        
#         output_format = {
#             "container": "raw",
#             "encoding": "pcm_f32le",
#             "sample_rate": 44100,
#         }
        
#         pcm_buffer = BytesIO()
        
#         try:
#             async with websockets.connect(uri, extra_headers=headers) as websocket:
#                 # Send the TTS request
#                 request = {
#                     "model_id": model_id,
#                     "transcript": text,
#                     "voice_id": voice_id,
#                     "stream": True,
#                     "output_format": output_format
#                 }
#                 await websocket.send(json.dumps(request))
                
#                 # Receive the streamed audio data
#                 while True:
#                     try:
#                         message = await asyncio.wait_for(websocket.recv(), timeout=30)
#                     except asyncio.TimeoutError:
#                         st.error("Timeout: No audio data received.")
#                         break
#                     data = json.loads(message)
#                     audio_chunk = data.get("audio")
#                     if audio_chunk:
#                         pcm_buffer.write(audio_chunk)
#                     else:
#                         break  # No more audio data
                        
#         except Exception as e:
#             st.error(f"Error during Text-to-Speech streaming: {e}")
#             return None
        
#         # Retrieve PCM data
#         pcm_data = pcm_buffer.getvalue()
        
#         if not pcm_data:
#             st.error("No audio data received from Cartesia.")
#             return None
        
#         # Convert PCM float32 to int16 for WAV
#         try:
#             float_data = np.frombuffer(pcm_data, dtype=np.float32)
#             int_data = np.int16(float_data * 32767)
#         except Exception as e:
#             st.error(f"Error converting PCM data: {e}")
#             return None
        
#         # Create a BytesIO buffer for WAV
#         wav_buffer = BytesIO()
        
#         # Write WAV header and frames
#         try:
#             with wave.open(wav_buffer, 'wb') as wf:
#                 wf.setnchannels(1)  # Mono
#                 wf.setsampwidth(2)  # 16 bits
#                 wf.setframerate(44100)
#                 wf.writeframes(int_data.tobytes())
#         except Exception as e:
#             st.error(f"Error writing WAV data: {e}")
#             return None
        
#         # Get WAV bytes
#         wav_bytes = wav_buffer.getvalue()
        
#         # Encode to base64
#         wav_base64 = base64.b64encode(wav_bytes).decode('utf-8')
        
#         return wav_base64

#     return asyncio.run(tts())

# # ----------------------------- #
# #           MAIN APP            #
# # ----------------------------- #

# def main():
#     # Apply custom CSS
#     add_custom_css()

#     # Title
#     st.markdown('<div class="title">📈 Summary Dashboard</div>', unsafe_allow_html=True)

#     # Initialize Session State for Past Summaries
#     if 'past_summaries' not in st.session_state:
#         st.session_state.past_summaries = []

#     # Button to manually fetch new data (optional)
#     fetch_button = st.button("Fetch Latest Summary Now")

#     if fetch_button or 'auto_fetch' not in st.session_state:
#         # Fetch the latest summary
#         latest_data = fetch_latest_summary_simulated()
#         # For actual API call, replace the above line with:
#         # latest_data = fetch_latest_summary()

#         if latest_data:
#             # Check if the latest data is already in past summaries
#             if not st.session_state.past_summaries or \
#                (st.session_state.past_summaries and latest_data != st.session_state.past_summaries[-1]):
#                 # Append the new summary to past summaries
#                 st.session_state.past_summaries.append(latest_data)

#     # Insert your Cartesia AI Voice API key here
#     CARTESIA_API_KEY = "19fbcc89-191b-487a-aca3-11b32496413e"  # Replace with your actual API key

#     # Display Latest Summary
#     if st.session_state.past_summaries:
#         latest = st.session_state.past_summaries[-1]
#         st.markdown('<div class="section-header">Latest Summary</div>', unsafe_allow_html=True)

#         # Display the summary card with the button
#         with st.container():
#             col1, col2 = st.columns([1, 3])
#             with col1:
#                 # Embed the image using HTML to apply CSS classes
#                 st.markdown(
#                     f"""
#                     <img src="{latest['image_url']}" alt="Latest Image" class="summary-image" />
#                     """,
#                     unsafe_allow_html=True
#                 )
#             with col2:
#                 st.markdown(f"<p class='text-summary'>{latest['text_summary']}</p>", unsafe_allow_html=True)
                
#                 # Read Aloud button with unique key
#                 read_button_key = f"tts_button_{latest['id']}"
#                 if f"audio_{latest['id']}" not in st.session_state:
#                     st.session_state[f"audio_{latest['id']}"] = None

#                 if st.button("🔊 Read Aloud", key=read_button_key):
#                     if CARTESIA_API_KEY != "YOUR_CARTESIA_API_KEY_HERE":
#                         audio_base64 = cartesia_text_to_speech(latest["text_summary"], CARTESIA_API_KEY)
#                         if audio_base64:
#                             st.session_state[f"audio_{latest['id']}"] = audio_base64
#                     else:
#                         st.error("Please replace 'YOUR_CARTESIA_API_KEY_HERE' with your actual Cartesia API key.")

#         # Display the audio player with autoplay (hidden controls)
#         if st.session_state[f"audio_{latest['id']}"]:
#             # Create the audio tag with autoplay and without controls
#             audio_html = f"""
#             <audio autoplay style="display:none;">
#                 <source src="data:audio/wav;base64,{st.session_state[f"audio_{latest['id']}"]}" type="audio/wav">
#                 Your browser does not support the audio element.
#             </audio>
#             """
#             st.markdown(audio_html, unsafe_allow_html=True)

#     # Display Past Summaries
#     if len(st.session_state.past_summaries) > 1:
#         st.markdown('<div class="section-header">Past Summaries</div>', unsafe_allow_html=True)
#         for past in reversed(st.session_state.past_summaries[:-1]):
#             with st.container():
#                 col1, col2 = st.columns([1, 3])
#                 with col1:
#                     # Embed the image using HTML to apply CSS classes
#                     st.markdown(
#                         f"""
#                         <img src="{past['image_url']}" alt="Past Image" class="summary-image" />
#                         """,
#                         unsafe_allow_html=True
#                     )
#                 with col2:
#                     st.markdown(f"<p class='text-summary'>{past['text_summary']}</p>", unsafe_allow_html=True)
                    
#                     # Read Aloud button with unique key
#                     read_button_key = f"tts_button_{past['id']}"
#                     if f"audio_{past['id']}" not in st.session_state:
#                         st.session_state[f"audio_{past['id']}"] = None

#                     if st.button("🔊 Read Aloud", key=read_button_key):
#                         if CARTESIA_API_KEY != "YOUR_CARTESIA_API_KEY_HERE":
#                             audio_base64 = cartesia_text_to_speech(past["text_summary"], CARTESIA_API_KEY)
#                             if audio_base64:
#                                 st.session_state[f"audio_{past['id']}"] = audio_base64
#                         else:
#                             st.error("Please replace 'YOUR_CARTESIA_API_KEY_HERE' with your actual Cartesia API key.")

#             # Display the audio player with autoplay (hidden controls)
#             if st.session_state[f"audio_{past['id']}"]:
#                 # Create the audio tag with autoplay and without controls
#                 audio_html = f"""
#                 <audio autoplay style="display:none;">
#                     <source src="data:audio/wav;base64,{st.session_state[f"audio_{past['id']}"]}" type="audio/wav">
#                     Your browser does not support the audio element.
#                 </audio>
#                 """
#                 st.markdown(audio_html, unsafe_allow_html=True)

#     # Footer
#     st.markdown('<div class="footer">© 2024 Your Company. All rights reserved.</div>', unsafe_allow_html=True)

#     # Auto-Refresh Mechanism
#     # Using JavaScript to refresh the page
#     refresh_script = f"""
#     <script>
#     setTimeout(function() {{
#         window.location.reload();
#     }}, {REFRESH_INTERVAL * 1000});
#     </script>
#     """
#     st.markdown(refresh_script, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()
