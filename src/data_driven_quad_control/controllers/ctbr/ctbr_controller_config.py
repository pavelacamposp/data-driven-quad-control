from typing import TypedDict


class CTBRControllerParams(TypedDict):
    dt: float
    pid_coefficients: list[list[float]]


class CTBRControllerConfig(TypedDict):
    ctbr_controller_params: CTBRControllerParams
