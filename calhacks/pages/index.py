# """The overview page of the app."""

# import reflex as rx
# from .. import styles
# from ..templates import template
# from ..views.stats_cards import stats_cards
# from ..views.charts import (
#     users_chart,
#     revenue_chart,
#     orders_chart,
#     area_toggle,
#     pie_chart,
#     timeframe_select,
#     StatsState,
# )
# from ..views.adquisition_view import adquisition
# # from ..components.notification import notification
# from ..components.card import card
# from .profile import ProfileState
# import datetime


# def _time_data() -> rx.Component:
#     return rx.hstack(
#         rx.tooltip(
#             rx.icon("info", size=20),
#             content=f"{(datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%b %d, %Y')} - {datetime.datetime.now().strftime('%b %d, %Y')}",
#         ),
#         rx.text("Last 30 days", size="4", weight="medium"),
#         align="center",
#         spacing="2",
#         display=["none", "none", "flex"],
#     )


# def tab_content_header() -> rx.Component:
#     return rx.hstack(
#         _time_data(),
#         area_toggle(),
#         align="center",
#         width="100%",
#         spacing="4",
#     )


# @template(route="/", title="Overview", on_load=StatsState.randomize_data)
# def index() -> rx.Component:
#     """The overview page.

#     Returns:
#         The UI for the overview page.
#     """
#     return rx.vstack(
#         rx.heading(f"Welcome, {ProfileState.profile.name}", size="5"),
#         rx.flex(
#             rx.input(
#                 rx.input.slot(rx.icon("search"), padding_left="0"),
#                 placeholder="Search here...",
#                 size="3",
#                 width="100%",
#                 max_width="450px",
#                 radius="large",
#                 style=styles.ghost_input_style,
#             ),
#             # rx.flex(
#             #     notification("bell", "cyan", 12),
#             #     notification("message-square-text", "plum", 6),
#             #     spacing="4",
#             #     width="100%",
#             #     wrap="nowrap",
#             #     justify="end",
#             # ),
#             justify="between",
#             align="center",
#             width="100%",
#         ),
#         stats_cards(),
#         card(
#             rx.hstack(
#                 tab_content_header(),
#                 rx.segmented_control.root(
#                     rx.segmented_control.item("Users", value="users"),
#                     rx.segmented_control.item("Revenue", value="revenue"),
#                     rx.segmented_control.item("Orders", value="orders"),
#                     margin_bottom="1.5em",
#                     default_value="users",
#                     on_change=StatsState.set_selected_tab,
#                 ),
#                 width="100%",
#                 justify="between",
#             ),
#             rx.match(
#                 StatsState.selected_tab,
#                 ("users", users_chart()),
#                 ("revenue", revenue_chart()),
#                 ("orders", orders_chart()),
#             ),
#         ),
#         rx.grid(
#             card(
#                 rx.hstack(
#                     rx.hstack(
#                         rx.icon("user-round-search", size=20),
#                         rx.text("Visitors Analytics", size="4", weight="medium"),
#                         align="center",
#                         spacing="2",
#                     ),
#                     timeframe_select(),
#                     align="center",
#                     width="100%",
#                     justify="between",
#                 ),
#                 pie_chart(),
#             ),
#             card(
#                 rx.hstack(
#                     rx.icon("globe", size=20),
#                     rx.text("Acquisition Overview", size="4", weight="medium"),
#                     align="center",
#                     spacing="2",
#                     margin_bottom="2.5em",
#                 ),
#                 rx.vstack(
#                     adquisition(),
#                 ),
#             ),
#             gap="1rem",
#             grid_template_columns=[
#                 "1fr",
#                 "repeat(1, 1fr)",
#                 "repeat(2, 1fr)",
#                 "repeat(2, 1fr)",
#                 "repeat(2, 1fr)",
#             ],
#             width="100%",
#         ),
#         spacing="8",
#         width="100%",
#     )


## new code

# calhacks/pages/index.py

"""The main page of the app."""

import reflex as rx
from calhacks import styles
from calhacks.templates.template import template
from calhacks.app_state import AppState
# from calhacks.components.navbar import navbar  # Import the navbar component


@template(route="/", title="MetaEye - Navigational Guidance")
def index() -> rx.Component:
    """The main page displaying navigational guidance."""

    return rx.fragment(
        # navbar(),  # Include the navbar at the top
        rx.box(
            rx.vstack(
                # rx.heading(
                #     "MetaEye",
                #     size="3xl",
                #     text_align="center",
                #     color=styles.primary_color,
                #     margin_bottom="1rem",
                # ),
                # rx.text(
                #     "Your Personal Navigational Assistant",
                #     size="lg",
                #     text_align="center",
                #     color=styles.secondary_color,
                #     margin_bottom="2rem",
                # ),
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
                            left_icon=rx.icon("volume-2", color="white"),
                            aria_label="Listen to the summary",
                        ),
                        rx.button(
                            "Refresh",
                            on_click=AppState.load_latest_summary,
                            size="lg",
                            color_scheme="blue",
                            left_icon=rx.icon("refresh-ccw", color="white"),
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
            width="100%",
            max_width="800px",
            margin="0 auto",
            padding="2rem",
        ),
    )