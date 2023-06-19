use smash_rl_rust::micro_fighter::MicroFighter;

fn main() {
    let mut env = MicroFighter::new(true);
    env.run();
}