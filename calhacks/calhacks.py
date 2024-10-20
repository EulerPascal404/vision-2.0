"""Welcome to Reflex!."""

# Import all the pages.
from .pages import *
from . import styles
from .pages import login

import reflex as rx
from .app_state import AppState


# Create the app.
app = rx.App(
    style=styles.base_style,
    stylesheets=styles.base_stylesheets,
    title="Dashboard Template",
    description="A dashboard template for Reflex.",
)  

# app.add_page(login.login, route="/login", title="MetaEye - Login")
# app.add_page(index.index, route="/", title="MetaEye - Navigational Guidance")