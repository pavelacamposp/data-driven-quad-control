from typing import TypedDict


class DroneInertia(TypedDict):
    Jxx: float
    Jxy: float
    Jxz: float
    Jyy: float
    Jyz: float
    Jzz: float


class DronePhysicalParams(TypedDict):
    mass: float
    inertia: DroneInertia


class DroneRotorParams(TypedDict):
    kf: float
    km: float
    arm_length: float
    rotor_angles_deg: list[float]
    rotor_spin_directions: list[int]


class DroneConfig(TypedDict):
    drone_params: DronePhysicalParams
    drone_rotor_params: DroneRotorParams
