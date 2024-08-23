"""The overview page of the app."""

import reflex as rx
from ..templates import template
from ..views.stats_cards import api_stats_cards, participants_stats_cards
from ..views.charts import (
    users_chart,
    revenue_chart,
    area_toggle,
    track_pie_chart,
    tech_pie_chart,
    StatsState,
)
from ..views.adquisition_view import adquisition
from ..components.card import card
import datetime


def _time_data() -> rx.Component:
    return rx.hstack(
        rx.tooltip(
            rx.icon("info", size=20),
            content=f"{(datetime.datetime.now() - datetime.timedelta(days=60)).strftime('%b %d, %Y')} - {datetime.datetime.now().strftime('%b %d, %Y')}",
        ),
        rx.text("Last 60 days", size="4", weight="medium"),
        align="center",
        spacing="2",
        display=["none", "none", "flex"],
    )


def tab_content_header() -> rx.Component:
    return rx.hstack(
        _time_data(),
        area_toggle(),
        align="center",
        width="100%",
        spacing="4",
    )


@template(route="/", title="Overview", on_load=StatsState.randomize_data)
def index() -> rx.Component:
    """The overview page.

    Returns:
        The UI for the overview page.
    """
    return rx.vstack(
        rx.heading(f"Participants Analysis", size="5"),
        participants_stats_cards(),
        rx.grid(
            card(
                rx.hstack(
                    rx.hstack(
                        rx.icon("user-round-search", size=20),
                        rx.text("Track", size="4", weight="medium"),
                        align="center",
                        spacing="2",
                    ),
                    align="center",
                    width="100%",
                    justify="between",
                ),
                track_pie_chart(),
            ),
            gap="1rem",
            grid_template_columns=[
                "1fr",
                "repeat(1, 1fr)",
            ],
            width="100%",
        ),
        rx.grid(
            card(
                rx.hstack(
                    rx.hstack(
                        rx.icon("user-round-search", size=20),
                        rx.text("Used Tool", size="4", weight="medium"),
                        align="center",
                        spacing="2",
                    ),
                    align="center",
                    width="100%",
                    justify="between",
                ),
                tech_pie_chart(),
            ),
            card(
                rx.hstack(
                    rx.icon("globe", size=20),
                    rx.text("Participants by geography", size="4", weight="medium"),
                    align="center",
                    spacing="2",
                    margin_bottom="2.5em",
                ),
                rx.vstack(
                    adquisition(),
                ),
            ),
            gap="1rem",
            grid_template_columns=[
                "1fr",
                "repeat(1, 1fr)",
                "repeat(2, 1fr)",
                "repeat(2, 1fr)",
                "repeat(2, 1fr)",
            ],
            width="100%",
        ),
        rx.heading(f"API usage", size="5"),
        api_stats_cards(),
        card(
            rx.hstack(
                tab_content_header(),
                rx.segmented_control.root(
                    rx.segmented_control.item("API Calls", value="api_call"),
                    rx.segmented_control.item("Unique keys", value="unique_keys"),
                    margin_bottom="1.5em",
                    default_value="api_call",
                    on_change=StatsState.set_selected_tab,
                ),
                width="100%",
                justify="between",
            ),
            rx.match(
                StatsState.selected_tab,
                ("api_call", users_chart()),
                ("unique_keys", revenue_chart()),
            ),
        ),
        spacing="8",
        width="100%",
    )
    
