# import reflex as rx

# class AppState(rx.State):
#     """Application state management."""

#     summary_text: str = ""
#     past_results: list = []

#     def add_summary(self, summary: str):
#         """Add a new summary to the state and past results."""
#         self.summary_text = summary
#         self.past_results.insert(0, summary)

#     def speak_summary(self):
#         """Use the browser's text-to-speech to read the summary."""
#         return rx.window_eval(f"""
#             const utterance = new SpeechSynthesisUtterance(`{self.summary_text}`);
#             speechSynthesis.speak(utterance);
#         """)

# calhacks/app_state.py

import reflex as rx
# import aiohttp  # For making asynchronous HTTP requests (if needed)
from typing import List

class AppState(rx.State):
    """Application state management."""

    # State variables
    summary_text: str = ""
    past_results: List[str] = []

    def add_summary(self, summary: str):
        """Add a new summary to the state and past results."""
        self.summary_text = summary
        self.past_results.insert(0, summary)

    def speak_summary(self):
        """Use the browser's text-to-speech to read the summary."""
        if not self.summary_text:
            return rx.alert("No summary available to read.")
        # Use JavaScript's SpeechSynthesis API to read the summary aloud
        return rx.window_eval(f"""
            const utterance = new SpeechSynthesisUtterance(`{self.summary_text}`);
            speechSynthesis.speak(utterance);
        """)

    async def load_latest_summary(self):
        """Load the latest summary from the backend or database."""
        # Replace the placeholder code with your actual data fetching logic
        # For example, make an asynchronous HTTP request to your backend API
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
