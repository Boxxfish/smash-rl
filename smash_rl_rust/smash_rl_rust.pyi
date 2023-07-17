from typing import List


class MoveState:
    Idle: int
    Run: int
    Jump: int
    Fall: int
    Shield: int
    Hitstun: int
    LightAttackStartup: int
    LightAttackHit: int
    LightAttackRecovery: int
    HeavyAttackStartup: int
    HeavyAttackHit: int
    HeavyAttackRecovery: int
    SpecialAttackStartup: int
    SpecialAttackHit: int
    SpecialAttackRecovery: int
    Grab: int

class HBox:
    is_hit: bool
    x: int
    y: int
    w: int
    h: int
    angle: float
    is_player: bool
    damage: int

class StepOutput:
    hboxes: List[HBox]
    round_over: bool
    player_won: bool
    game_state: GameState
    player_damage: int
    player_state: int
    player_dir: int
    opponent_damage: int
    opponent_state: int
    opponent_dir: int
    net_damage: int

class GameState:
    pass

class MicroFighter:
    def __init__(self, human: bool) -> None: ...
    def run(self) -> None: ...
    def step(self, action_id: int) -> StepOutput: ...
    def bot_step(self, action_id: int) -> None: ...
    def load_state(self, state: GameState) -> StepOutput: ...
    def get_game_state(self) -> GameState: ...
    def reset(self) -> StepOutput: ...
    def get_screen_size(self) -> int: ...
