"""The main page of the app."""

import reflex as rx
from calhacks import styles
from calhacks.templates.template import template
from calhacks.app_state import AppState
# from calhacks.components.navbar import navbar  # Import the navbar component


@template(route="/", title="MetaEye - Navigational Guidance")
def index() -> rx.Component:
    """The main page displaying navigational guidance."""
    
    async def load_summaries():
        await AppState.load_latest_summary()
        await AppState.load_past_results()

    return rx.fragment(
        rx.box(
            rx.box(
                rx.vstack(
                    rx.box(
                        rx.heading(
                            "Latest Summary",
                            size="2xl",
                            text_align="left",
                            color=styles.primary_color,
                            margin_bottom="1rem",
                        ),
                        rx.box(
                            rx.text(
                                AppState.summary_text | "No summary available.",
                                size="md",
                                color=styles.text_color,
                                style={
                                    "padding": "1.5rem",
                                    "backgroundColor": styles.card_bg_color,
                                    "borderRadius": "8px",
                                    "boxShadow": styles.box_shadow,
                                },
                            ),
                        ),
                        rx.hstack(
                            rx.button(
                                "Listen",
                                on_click=AppState.speak_summary,
                                size="lg",
                                color_scheme="teal",
                                # left_icon=rx.icon("volume-2", color="white"),
                                aria_label="Listen to the summary",
                            ),
                            rx.button(
                                "Refresh",
                                on_click=AppState.load_latest_summary,
                                size="lg",
                                color_scheme="blue",
                                # left_icon=rx.icon("refresh-ccw", color="white"),
                                aria_label="Refresh summary",
                            ),
                            spacing="1rem",
                            margin_top="1rem",
                            justify="center",
                        ),
                        width="100%",
                    ),
                    rx.divider(margin_y="2rem"),
                    rx.heading(
                        "Past Summaries",
                        size="2xl",
                        text_align="left",
                        color=styles.primary_color,
                        margin_bottom="1rem",
                    ),
                    rx.box(
                        rx.foreach(
                            AppState.past_results,
                            lambda summary: rx.box(
                                rx.text(
                                    summary,
                                    size="md",
                                    color=styles.text_color,
                                    style={
                                        "padding": "1rem",
                                        "backgroundColor": styles.card_bg_color,
                                        "borderRadius": "8px",
                                        "margin_bottom": "1rem",
                                        "boxShadow": styles.box_shadow,
                                    },
                                )
                            ),
                        ),
                        width="100%",
                    ),
                ),
            ),
            width="100%",
            max_width="800px",
            margin="0 auto",
            padding="2rem",
        ),

    )




