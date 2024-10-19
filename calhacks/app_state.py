# calhacks/app_state.py

import reflex as rx

class AppState(rx.State):
    """Application state management."""

    summary_text: str = ""
    past_results: list = []

    def add_summary(self, summary: str):
        """Add a new summary to the state and past results."""
        self.summary_text = summary
        self.past_results.insert(0, summary)

    def speak_summary(self):
        """Use the browser's text-to-speech to read the summary."""
        return rx.window_eval(f"""
            const utterance = new SpeechSynthesisUtterance(`{self.summary_text}`);
            speechSynthesis.speak(utterance);
        """)
