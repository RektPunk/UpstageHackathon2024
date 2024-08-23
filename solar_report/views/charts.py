import datetime
import random

import reflex as rx
from reflex.components.radix.themes.base import LiteralAccentColor

from solar_report.views.data import (
    FINANCE_DATA,
    HEALTHCARE_DATA,
    INNOVATION_DATA,
    LEGAL_DATA,
    OVERALL_DATA,
    TECH_DATA,
    TRACK_DATA,
    TRAVEL_DATA,
)


class StatsState(rx.State):
    area_toggle: bool = True
    selected_tab: str = "api_call"

    pie_type: str = "Overall"
    api_calls = []
    unique_keys = []

    tech_data = TECH_DATA
    track_data = TRACK_DATA

    overall_data = OVERALL_DATA
    travel_data = TRAVEL_DATA
    innovation_data = INNOVATION_DATA
    healthcare_data = HEALTHCARE_DATA
    finance_data = FINANCE_DATA
    legal_data = LEGAL_DATA

    def toggle_areachart(self):
        self.area_toggle = not self.area_toggle

    def randomize_data(self):
        # FIXME: fix it
        if self.api_calls != []:
            return

        for i in range(30, -1, -1):  # Include today's data
            self.api_calls.append(
                {
                    "Date": (
                        datetime.datetime.now() - datetime.timedelta(days=i)
                    ).strftime("%m-%d"),
                    "API Calls": random.randint(1000, 5000),
                }
            )
        for i in range(30, -1, -1):
            self.unique_keys.append(
                {
                    "Date": (
                        datetime.datetime.now() - datetime.timedelta(days=i)
                    ).strftime("%m-%d"),
                    "Unique Keys": random.randint(100, 500),
                }
            )


def area_toggle() -> rx.Component:
    return rx.cond(
        StatsState.area_toggle,
        rx.icon_button(
            rx.icon("area-chart"),
            size="2",
            cursor="pointer",
            variant="surface",
            on_click=StatsState.toggle_areachart,
        ),
        rx.icon_button(
            rx.icon("bar-chart-3"),
            size="2",
            cursor="pointer",
            variant="surface",
            on_click=StatsState.toggle_areachart,
        ),
    )


def _create_gradient(color: LiteralAccentColor, id: str) -> rx.Component:
    return (
        rx.el.svg.defs(
            rx.el.svg.linear_gradient(
                rx.el.svg.stop(
                    stop_color=rx.color(color, 7), offset="5%", stop_opacity=0.8
                ),
                rx.el.svg.stop(
                    stop_color=rx.color(color, 7), offset="95%", stop_opacity=0
                ),
                x1=0,
                x2=0,
                y1=0,
                y2=1,
                id=id,
            ),
        ),
    )


def _custom_tooltip(color: LiteralAccentColor) -> rx.Component:
    return (
        rx.recharts.graphing_tooltip(
            separator=" : ",
            content_style={
                "backgroundColor": rx.color("gray", 1),
                "borderRadius": "var(--radius-2)",
                "borderWidth": "1px",
                "borderColor": rx.color(color, 7),
                "padding": "0.5rem",
                "boxShadow": "0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)",
            },
            is_animation_active=True,
        ),
    )


def api_call_chart() -> rx.Component:
    return rx.cond(
        StatsState.area_toggle,
        rx.recharts.area_chart(
            _create_gradient("blue", "colorBlue"),
            _custom_tooltip("blue"),
            rx.recharts.cartesian_grid(
                stroke_dasharray="3 3",
            ),
            rx.recharts.area(
                data_key="API Calls",
                stroke=rx.color("blue", 9),
                fill="url(#colorBlue)",
                type_="monotone",
            ),
            rx.recharts.x_axis(data_key="Date", scale="auto"),
            rx.recharts.y_axis(),
            rx.recharts.legend(),
            data=StatsState.api_calls,
            height=425,
        ),
        rx.recharts.bar_chart(
            rx.recharts.cartesian_grid(
                stroke_dasharray="3 3",
            ),
            _custom_tooltip("blue"),
            rx.recharts.bar(
                data_key="API Calls",
                stroke=rx.color("blue", 9),
                fill=rx.color("blue", 7),
            ),
            rx.recharts.x_axis(data_key="Date", scale="auto"),
            rx.recharts.y_axis(),
            rx.recharts.legend(),
            data=StatsState.api_calls,
            height=425,
        ),
    )


def unique_key_chart() -> rx.Component:
    return rx.cond(
        StatsState.area_toggle,
        rx.recharts.area_chart(
            _create_gradient("green", "colorGreen"),
            _custom_tooltip("green"),
            rx.recharts.cartesian_grid(
                stroke_dasharray="3 3",
            ),
            rx.recharts.area(
                data_key="Unique Keys",
                stroke=rx.color("green", 9),
                fill="url(#colorGreen)",
                type_="monotone",
            ),
            rx.recharts.x_axis(data_key="Date", scale="auto"),
            rx.recharts.y_axis(),
            rx.recharts.legend(),
            data=StatsState.unique_keys,
            height=425,
        ),
        rx.recharts.bar_chart(
            _custom_tooltip("green"),
            rx.recharts.cartesian_grid(
                stroke_dasharray="3 3",
            ),
            rx.recharts.bar(
                data_key="Unique Keys",
                stroke=rx.color("green", 9),
                fill=rx.color("green", 7),
            ),
            rx.recharts.x_axis(data_key="Date", scale="auto"),
            rx.recharts.y_axis(),
            rx.recharts.legend(),
            data=StatsState.unique_keys,
            height=425,
        ),
    )


def tech_pie_chart() -> rx.Component:
    return rx.recharts.pie_chart(
        rx.recharts.pie(
            data=StatsState.tech_data,
            data_key="value",
            name_key="name",
            cx="50%",
            cy="50%",
            padding_angle=1,
            inner_radius="60",
            outer_radius="80",
            label=True,
        ),
        rx.recharts.legend(),
        height=300,
    )


def track_pie_chart() -> rx.Component:
    return rx.recharts.pie_chart(
        rx.recharts.pie(
            data=StatsState.track_data,
            data_key="value",
            name_key="name",
            cx="50%",
            cy="50%",
            padding_angle=1,
            inner_radius="70",
            outer_radius="100",
            label=True,
        ),
        rx.recharts.legend(),
        height=300,
    )


def _sub_pie_chart(data):
    return rx.recharts.pie_chart(
        rx.recharts.pie(
            data=data,
            data_key="value",
            name_key="name",
            cx="50%",
            cy="50%",
            padding_angle=1,
            inner_radius="70",
            outer_radius="100",
            label=True,
        ),
        rx.recharts.legend(),
        height=300,
    )


def pie_chart() -> rx.Component:
    return rx.match(
        StatsState.pie_type,
        ("Travel", _sub_pie_chart(data=StatsState.travel_data)),
        ("Innovation", _sub_pie_chart(data=StatsState.innovation_data)),
        ("Healthcare", _sub_pie_chart(data=StatsState.healthcare_data)),
        ("Finance", _sub_pie_chart(data=StatsState.finance_data)),
        ("Legal", _sub_pie_chart(data=StatsState.legal_data)),
        _sub_pie_chart(data=StatsState.overall_data),
    )


def pie_type_select() -> rx.Component:
    return rx.select(
        ["Overall", "Travel", "Innovation", "Healthcare", "Finance", "Legal"],
        default_value="Overall",
        value=StatsState.pie_type,
        variant="surface",
        on_change=StatsState.set_pie_type,
    )
