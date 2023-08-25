from typing import List

from torch import Tensor

def test_jit() -> None: ...

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
    player_pos: tuple[int, int]
    opponent_damage: int
    opponent_state: int
    opponent_dir: int
    opponent_pos: tuple[int, int]
    net_damage: int

class GameState:
    pass

class BotData:
    elo: int

class RolloutContext:
    def __init__(
        self,
        total_num_envs: int,
        num_workers: int,
        num_steps: int,
        max_skip_frames: int,
        num_frames: int,
        time_limit: int,
        first_bot_path: str,
        top_k: int,
        initial_elo: float,
    ) -> None: ...
    def rollout(
        self,
        latest_policy_path: str,
    ) -> tuple[list[Tensor], Tensor, Tensor, Tensor, Tensor, Tensor, float]: ...
    def push_bot(self, path: str) -> None: ...
    def insert_bot(self, path: str, index: int) -> None: ...
    def set_expl_reward_amount(self, amount: float) -> None: ...
    def perform_eval(self, eval_steps: int, max_eval_steps: int, elo_k: int) -> None: ...
    def current_elo(self) -> float: ...
    def bot_data(self) -> list[BotData]: ...

class MicroFighter:
    def __init__(self, human: bool) -> None: ...
    def run(self) -> None: ...
    def step(self, action_id: int) -> StepOutput: ...
    def bot_step(self, action_id: int) -> None: ...
    def load_state(self, state: GameState) -> StepOutput: ...
    def get_game_state(self) -> GameState: ...
    def reset(self) -> StepOutput: ...
    def get_screen_size(self) -> int: ...