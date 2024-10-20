import reflex as rx
# import aiohttp 
from typing import List

from cartesia import Cartesia
import os
import subprocess
import ffmpeg

client = Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))

voice_id = "a0e99841-438c-4a64-b679-ae501e7d6091" 
model_id = "sonic-english"
transcript = "Hello! Welcome to Cartesia"

output_format = {
    "container": "raw",
    "encoding": "pcm_f32le",
    "sample_rate": 44100,
}

ws = client.tts.websocket()

# f = open("sonic.pcm", "wb")

# for output in ws.send(
#     model_id=model_id,
#     transcript=transcript,
#     voice_id=voice_id,
#     stream=True,
#     output_format=output_format,
# ):
#     buffer = output["audio"]  # buffer contains raw PCM audio bytes
#     f.write(buffer)

# # Close the connection to release resources
# ws.close()
# f.close()

# ffmpeg.input("sonic.pcm", format="f32le").output("sonic.wav").run()

# subprocess.run(["ffplay", "-autoexit", "-nodisp", "sonic.wav"])

class AppState(rx.State):
    """Application state management."""

    summary_text: str = ""
    past_results: List[str] = []

    def add_summary(self, summary: str):
        """Add a new summary to the state and past results."""
        self.summary_text = summary
        self.past_results.insert(0, summary)

    def speak_summary(self):
        """Use the browser's text-to-speech to read the summary."""
        # if not self.summary_text:
        #     return rx.alert("No summary available to read.")
        # Use JavaScript's SpeechSynthesis API to read the summary aloud
        # return rx.window_eval(f"""
        #     const utterance = new SpeechSynthesisUtterance(`{self.summary_text}`);
        #     speechSynthesis.speak(utterance);
        # """)
        return "hello"

    async def load_latest_summary(self):
        """Load the latest summary from the backend or database."""
        try:
            # Simulate an API call delay
            await rx.sleep(1)
            # Example of fetching data from an API endpoint
            # api_url = "https://your-backend-api.com/latest-summary"
            # async with aiohttp.ClientSession() as session:
            #     async with session.get(api_url) as response:
            #         if response.status == 200:
            #             data = await response.json()
            #             new_summary = data.get("summary", "")
            #         else:
            #             new_summary = "Failed to load the latest summary."
            # For now, we'll use a simulated summary
            new_summary = "This is a new navigational summary generated just now."
            self.add_summary(new_summary)
        except Exception as e:
            rx.alert(f"An error occurred while loading the summary: {e}")

    async def load_past_results(self):
        """Load past summaries from the database or backend."""
        # Implement logic to load past results
        # For now, we'll simulate with placeholder data
        try:
            await rx.sleep(1)
            # Simulated past results
            self.past_results = [
                "Previous summary 1.",
                "Previous summary 2.",
                "Previous summary 3.",
            ]
        except Exception as e:
            rx.alert(f"An error occurred while loading past summaries: {e}")