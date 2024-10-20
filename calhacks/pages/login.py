# # calhacks/pages/login.py

# import reflex as rx
# from calhacks.templates.template import template
# from calhacks.app_state import AppState

# @template(route="/login", title="MetaEye - Login")
# def login() -> rx.Component:
#     """Login page where users enter their login information."""
#     return rx.box(
#         rx.vstack(
#             rx.heading(
#                 "Welcome to MetaEye",
#                 size="2xl",
#                 text_align="center",
#                 margin_bottom="1rem",
#             ),
#             rx.text(
#                 "Please enter your login information:",
#                 size="md",
#                 text_align="center",
#                 margin_bottom="1rem",
#             ),
#             rx.form(
#                 rx.input(
#                     placeholder="Username",
#                     type="text",
#                     name="username",
#                     required=True,
#                     value=AppState.username,
#                     on_change=AppState.set_username,
#                     style={
#                         "padding": "0.5rem",
#                         "width": "100%",
#                         "max_width": "300px",
#                         "margin_bottom": "1rem",
#                     },
#                 ),
#                 rx.input(
#                     placeholder="Password",
#                     type="password",
#                     name="password",
#                     required=True,
#                     value=AppState.password,
#                     on_change=AppState.set_password,
#                     style={
#                         "padding": "0.5rem",
#                         "width": "100%",
#                         "max_width": "300px",
#                         "margin_bottom": "1rem",
#                     },
#                 ),
#                 rx.button(
#                     "Login",
#                     type="submit",
#                     on_click=AppState.login,
#                     size="lg",
#                     color_scheme="teal",
#                 ),
#                 method="POST",
#                 on_submit=AppState.login,
#             ),
#             spacing="1rem",
#             align_items="center",
#         ),
#         width="100%",
#         max_width="400px",
#         margin="0 auto",
#         padding="2rem",
#     )
