use smash_rl_rust::training::{RetrievalContext, PCA};

/// Tests that the retrieval mechanism is working.

fn main() {
    let encoder = tch::jit::CModule::load("temp/encoder.ptc").unwrap();
    let pca = PCA::load("temp/pca.json");
    RetrievalContext::new(
        128,
        "temp/index.bin",
        "temp/generated",
        "temp/episode_data.json",
        encoder,
        pca,
    );
}
