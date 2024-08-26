"""The code insight page."""

import reflex as rx

from solar_report import styles
from solar_report.templates import template


@template(route="/insight", title="Code insight")
def insight() -> rx.Component:
    with open("CODE_INSIGHT.md", encoding="utf-8") as readme:
        content = readme.read()
    return rx.markdown(content, component_map=styles.markdown_style)
