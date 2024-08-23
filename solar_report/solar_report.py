import reflex as rx

from solar_report import styles
from solar_report.pages import about, index

# Create the app.
app = rx.App(
    style=styles.base_style,
    stylesheets=styles.base_stylesheets,
    title="Global AI week Statisics",
    description="Dashboard for global AI participants and thier API usage statistics",
)
