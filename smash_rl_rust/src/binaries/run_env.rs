use rand::Rng;
use smash_rl_rust::env::MFEnv;

fn main() {
    let mut env = MFEnv::new(1, 4, true, (0, 2, 3), 500);
    let mut rng = rand::thread_rng();
    loop {
        env.bot_step(rng.gen_range(0..9));
        let (_, _, done, trunc, _) = env.step(rng.gen_range(0..9));
        env.render();
        if done || trunc {
            _ = env.reset();
        }
    }
}
