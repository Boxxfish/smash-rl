use std::sync::{RwLock, Arc};

use rand::Rng;
use smash_rl_rust::{env::MFEnv, training::{PCA, RetrievalContext}};

fn main() {
    let encoder = tch::jit::CModule::load("temp/encoder.ptc").unwrap();
    let pca = PCA::load("temp/pca.json");
    let retrieval_ctx = Arc::new(RwLock::new(RetrievalContext::new(
        128,
        "temp/index.bin",
        "temp/generated",
        "temp/episode_data.json",
        encoder,
        pca,
    )));

    let mut env = MFEnv::new(1, 4, true, (0, 2, 3), 500, 4, retrieval_ctx);
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
