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

class StepOutput:
    hboxes: List[HBox]
    round_over: bool
    player_won: bool

class MicroFighter:
    def __init__(self, human: bool) -> None: ...
    def run(self) -> None: ...
    def step(self, action_id: int) -> StepOutput: ...
    def bot_step(self, action_id: int) -> None: ...
    def reset(self) -> StepOutput: ...
    def get_screen_size(self) -> int: ...
