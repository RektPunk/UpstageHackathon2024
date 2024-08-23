"""The about page."""

import reflex as rx

from solar_report import styles
from solar_report.templates import template


@template(route="/about", title="About")
def about() -> rx.Component:
    """The about page.

    Returns:
        The UI for the about page.
    """
    with open("README.md", encoding="utf-8") as readme:
        content = readme.read()
    return rx.markdown(content, component_map=styles.markdown_style)
