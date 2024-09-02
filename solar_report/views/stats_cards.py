import reflex as rx
from reflex.components.radix.themes.base import LiteralAccentColor

from solar_report import styles


def api_stats_card(
    stat_name: str,
    value: int,
    prev_value: int,
    icon: str,
    icon_color: LiteralAccentColor,
    extra_char: str = "",
) -> rx.Component:
    percentage_change = (
        round(((value - prev_value) / prev_value) * 100, 2)
        if prev_value != 0
        else 0 if value == 0 else float("inf")
    )
    change = "increase" if value > prev_value else "decrease"
    arrow_icon = "trending-up" if value > prev_value else "trending-down"
    arrow_color = "grass" if value > prev_value else "tomato"
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.badge(
                    rx.icon(tag=icon, size=34),
                    color_scheme=icon_color,
                    radius="full",
                    padding="0.7rem",
                ),
                rx.vstack(
                    rx.heading(
                        f"{extra_char}{value:,}",
                        size="6",
                        weight="bold",
                    ),
                    rx.text(stat_name, size="4", weight="medium"),
                    spacing="1",
                    height="100%",
                    align_items="start",
                    width="100%",
                ),
                height="100%",
                spacing="4",
                align="center",
                width="100%",
            ),
            rx.hstack(
                rx.hstack(
                    rx.icon(
                        tag=arrow_icon,
                        size=24,
                        color=rx.color(arrow_color, 9),
                    ),
                    rx.text(
                        f"{percentage_change}%",
                        size="3",
                        color=rx.color(arrow_color, 9),
                        weight="medium",
                    ),
                    spacing="2",
                    align="center",
                ),
                rx.text(
                    f"{change} during Hackathon",
                    size="2",
                    color=rx.color("gray", 10),
                ),
                align="center",
                width="100%",
            ),
            spacing="3",
        ),
        size="3",
        width="100%",
        box_shadow=styles.box_shadow_style,
    )


def participants_stats_card(
    stat_name: str,
    value: int,
    icon: str,
    icon_color: LiteralAccentColor,
    extra_char: str = "",
) -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.badge(
                    rx.icon(tag=icon, size=34),
                    color_scheme=icon_color,
                    radius="full",
                    padding="0.7rem",
                ),
                rx.vstack(
                    rx.heading(
                        f"{extra_char}{value:,}",
                        size="6",
                        weight="bold",
                    ),
                    rx.text(stat_name, size="4", weight="medium"),
                    spacing="1",
                    height="100%",
                    align_items="start",
                    width="100%",
                ),
                height="100%",
                spacing="4",
                align="center",
                width="100%",
            ),
            spacing="3",
        ),
        size="3",
        width="100%",
        box_shadow=styles.box_shadow_style,
    )


def api_stats_cards() -> rx.Component:
    return rx.grid(
        api_stats_card(
            stat_name="API Calls",
            value=243186,
            prev_value=120625,
            icon="send",
            icon_color="blue",
        ),
        api_stats_card(
            stat_name="Token Usage",
            value=154932430,
            prev_value=40928588,
            icon="file_key",
            icon_color="green",
        ),
        gap="1rem",
        grid_template_columns=[
            "1fr",
            "repeat(1, 1fr)",
            "repeat(2, 1fr)",
        ],
        width="100%",
    )


def participants_stats_cards() -> rx.Component:
    return rx.grid(
        participants_stats_card(
            stat_name="# of participants",
            value=612,
            icon="users",
            icon_color="blue",
        ),
        participants_stats_card(
            stat_name="# of countries",
            value=43,
            icon="earth",
            icon_color="green",
        ),
        gap="1rem",
        grid_template_columns=[
            "1fr",
            "repeat(1, 1fr)",
            "repeat(2, 1fr)",
        ],
        width="100%",
    )


def submission_stats_cards() -> rx.Component:
    return rx.grid(
        participants_stats_card(
            stat_name="# of teams",
            value=70,
            icon="building",
            icon_color="blue",
        ),
        participants_stats_card(
            stat_name="# of submission",
            value=61,
            icon="star",
            icon_color="red",
        ),
        gap="1rem",
        grid_template_columns=[
            "1fr",
            "repeat(1, 1fr)",
            "repeat(2, 1fr)",
        ],
        width="100%",
    )
