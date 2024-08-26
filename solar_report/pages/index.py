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
        rx.text("Pre, during, post Hackathon", size="4", weight="medium"),
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
    return rx.vstack(
        rx.heading("Welcome to the Global AI Hackathon at a Glance", size="8"),
        rx.markdown("This document provides a comprehensive overview of the **Global AI Hackathon**, capturing key insights and data from the event. The hackathon brought together AI enthusiasts from across the globe, and this summary aims to highlight the most important aspects of their participation."),
        rx.heading("Participants", size="7"),
        rx.markdown("An analysis of the participant data from our recent global hackathon is presented. We had an incredible turnout, with participants joining us from all around the world. Below, you'll find key statistics that highlight the diversity and reach of this event, including the total number of participants and the number of countries represented. These figures not only showcase the scale of our hackathon but also emphasize the global community that came together to innovate and collaborate in the field of AI."),
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
        rx.heading("Submission Statistics", size="7"),
        rx.markdown("""
            We delve into the key metrics related to the projects submitted during the Global AI Hackathon. 
            Participants, organized into teams, tackled various challenges across different tracks, showcasing their creativity and technical expertise.
            We received submissions from numerous teams, each contributing unique projects. 
        """),
        rx.markdown("""
            Participants had the opportunity to choose from several tracks, each focusing on different aspects of LLM.
            An analysis of the tools and frameworks that were most frequently used by teams in their projects. 
            This includes a summary of the programming languages, libraries, and APIs that powered the innovative solutions submitted.
        """),
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
        rx.heading(f"API usage", size="7"),
        rx.markdown("""
            We focuse on the usage of APIs throughout the **Global AI Hackathon**, providing insights into how participants leveraged external services to power their projects.
            We tracked the volume of API calls made by participants before, during, and after the hackathon. This analysis highlights the surge in activity as the event progressed, showing how teams increasingly relied on APIs to build and refine their solutions.
            Each track presented unique challenges that required different sets of APIs. We break down which APIs were most popular within each track, offering insights into the specific technologies that teams favored to address their chosen challenges.
        """),
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
