from typing import TypedDict


class CTBRControllerParams(TypedDict):
    dt: float
    rate_pid_gains: list[list[float]]


class CTBRControllerConfig(TypedDict):
    ctbr_controller_params: CTBRControllerParams
