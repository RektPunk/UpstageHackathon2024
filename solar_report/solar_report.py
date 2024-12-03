import reflex as rx

from solar_report import styles
from solar_report.pages import about, index, insight

# Create the app.
app = rx.App(
    style=styles.base_style,
    stylesheets=styles.base_stylesheets,
)
app.add_page(index)
app.add_page(insight)
app.add_page(about)
