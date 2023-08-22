use rand::Rng;
use smash_rl_rust::melee::{Button, Gamepad, Stick};

/// Tests that connecting to Dolphin works.
fn main() {
    let mut gamepad = Gamepad::new("~/.config/SlippiOnline/Pipes/pipes1");
    let btn_list = [Button::A, Button::B, Button::L, Button::X, Button::Z];
    let mut rng = rand::thread_rng();
    loop {
        for btn in btn_list {
            gamepad.set_btn_state(btn, rng.gen_bool(0.5));
        }
        for stick in [Stick::Main, Stick::Control] {
            gamepad.set_stick_value(stick, rng.gen(), rng.gen());
        }
        gamepad.flush();
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}
