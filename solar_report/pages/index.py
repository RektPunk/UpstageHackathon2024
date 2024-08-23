"""The overview page of the app."""

import datetime

import reflex as rx

from solar_report.components.card import card
from solar_report.templates import template
from solar_report.views.charts import (
    StatsState,
    api_call_chart,
    area_toggle,
    pie_chart,
    pie_type_select,
    tech_pie_chart,
    track_pie_chart,
    unique_key_chart,
)
from solar_report.views.geography_view import geography
from solar_report.views.stats_cards import (
    api_stats_cards,
    participants_stats_cards,
    submission_stats_cards,
)


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
        rx.heading(f"Welcome", size="9"),
        rx.text("greetings & introduction on hackathon blah blah"),
        rx.heading(f"Participants", size="8"),
        rx.text("blah blah"),
        participants_stats_cards(),
        rx.grid(
            card(
                rx.hstack(
                    rx.icon("globe", size=20),
                    rx.text("Participants by geography", size="4", weight="medium"),
                    align="center",
                    spacing="2",
                    margin_bottom="2.5em",
                ),
                rx.vstack(
                    geography(),
                ),
            ),
            gap="1rem",
            grid_template_columns=[
                "1fr",
                "repeat(1, 1fr)",
            ],
            width="100%",
        ),
        rx.heading("Submission", size="8"),
        rx.text("there are blah blah"),
        submission_stats_cards(),
        rx.grid(
            card(
                rx.hstack(
                    rx.hstack(
                        rx.icon("train-track", size=20),
                        rx.text("Track", size="4", weight="medium"),
                        align="center",
                        spacing="2",
                    ),
                    align="center",
                    width="50%",
                    justify="between",
                ),
                track_pie_chart(),
            ),
            card(
                rx.hstack(
                    rx.hstack(
                        rx.icon("wrench", size=20),
                        rx.text("Used Tool", size="4", weight="medium"),
                        align="center",
                        spacing="2",
                    ),
                    align="center",
                    width="50%",
                    justify="between",
                ),
                tech_pie_chart(),
            ),
            gap="1rem",
            grid_template_columns=[
                "1fr",
                "repeat(1, 1fr)",
                "repeat(2, 1fr)",
            ],
            width="100%",
        ),
        rx.heading(f"API usage", size="8"),
        rx.text("There are blah blah"),
        api_stats_cards(),
        rx.grid(
            card(
                rx.hstack(
                    rx.hstack(
                        rx.icon("user-round-search", size=20),
                        rx.text(
                            "API used with respect to Track", size="4", weight="medium"
                        ),
                        align="center",
                        spacing="2",
                    ),
                    pie_type_select(),
                    align="center",
                    width="100%",
                    justify="between",
                ),
                pie_chart(),
                gap="1rem",
                grid_template_columns=[
                    "1fr",
                    "repeat(1, 1fr)",
                    "repeat(2, 1fr)",
                ],
            ),
            width="100%",
        ),
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
                ("api_call", api_call_chart()),
                ("unique_keys", unique_key_chart()),
            ),
        ),
        spacing="8",
        width="100%",
    )
