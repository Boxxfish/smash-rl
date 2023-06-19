from typing import List


class CharState:
    StartupHeavy: int
    RecoveryHeavy: int
    StartupLight: int
    RecoveryLight: int
    Grab: int
    Shield: int
    Hitstun: int
    Other: int

class HBox:
    is_hit: bool
    x: int
    y: int
    w: int
    h: int
    angle: float
    is_player: bool
    damage: int
    char_state: int

class MicroFighter:
    def __init__(self, human: bool) -> None: ...
    def run(self) -> None: ...
    def step(self, action_id: int) -> List[HBox]: ...
    def get_screen_size(self) -> int: ...
