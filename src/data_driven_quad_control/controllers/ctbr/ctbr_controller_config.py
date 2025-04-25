from typing import TypedDict


class CTBRControllerParams(TypedDict):
    rate_pid_gains: list[list[float]]


class CTBRControllerConfig(TypedDict):
    ctbr_controller_params: CTBRControllerParams
