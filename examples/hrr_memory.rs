//! Holographic memory: episodes, patterns, and working memory
//!
//! Demonstrates: bind → hologram → project, plus failure patterns and salience decay.
//! Uses random vectors as stand-in embeddings (swap in any real embedder).
//!
//! Run with:
//! ```bash
//! cargo run --example hrr_memory
//! ```

use an_tensor_compiler::holographic::memory::{
    Action, ActionOutcome, Episode, EpisodeId, ExperienceHologram, FocusItem, Observation,
    PatternLibrary, WorkingMemory,
};
use an_tensor_compiler::holographic::ops::{bind, project};
use an_tensor_compiler::{Device, Result, Tensor};
use std::time::{Duration, Instant};

fn rand_vec(dim: usize, device: &Device) -> Result<Tensor> {
    Ok(Tensor::randn(0f32, 1f32, dim, device)?)
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let dim = 256;

    // 1. Bind an observation-action-outcome episode into one hologram
    let obs = rand_vec(dim, &device)?;
    let act = rand_vec(dim, &device)?;
    let out = rand_vec(dim, &device)?;
    let hologram = bind(&obs, &bind(&act, &out)?)?;

    let mut experience = ExperienceHologram::new(device.clone(), dim)?;
    experience.encode_episode(&Episode {
        id: EpisodeId("deploy-001".into()),
        observation: Observation::new("deploy", "Service crashed on startup"),
        action: Action::new("rollback", "Rolled back to previous version"),
        outcome: ActionOutcome::success("Service recovered"),
        obs_tensor: obs.clone(),
        action_tensor: act.clone(),
        outcome_tensor: out.clone(),
        hologram,
        novelty_bits: 1.0,
        regime: None,
    })?;
    println!("Encoded 1 episode into a fixed-size hologram");

    // 2. Project: recall from the hologram with a query (no index, no search)
    let (recall, confidence) = experience.recall(&obs)?;
    println!(
        "Recall confidence: {confidence:.3} (shape: {:?})",
        recall.shape()
    );

    // 3. Pattern library: register a failure signature + recovery, then match
    let mut patterns = PatternLibrary::new(device.clone(), 0.7);
    patterns.register_failure(
        "oom_crash".to_string(),
        "Out of memory during batch processing".to_string(),
        obs.clone(),
    );
    patterns.register_recovery(
        "oom_crash".to_string(),
        "Halve the batch size and retry".to_string(),
        act.clone(),
        true,
    );
    if let Some((failure, score)) = patterns.match_failure(&obs) {
        println!("Matched failure '{}' (score {score:.3})", failure.name);
        if let Some(recovery) = patterns.get_best_recovery(&failure.name) {
            println!(
                "  recovery: {} ({:.0}% success)",
                recovery.description,
                recovery.success_rate * 100.0
            );
        }
    }

    // 4. Working memory: focus, decay, attend
    let mut working = WorkingMemory::new(device.clone(), dim, 10)?;
    working.focus(FocusItem {
        tensor: obs.clone(),
        description: "Deployment alert".to_string(),
        salience: 1.0,
        added_at: Instant::now(),
        source: Some("demo".to_string()),
    })?;
    working.tick(Duration::from_secs(30))?;
    working.attend(&obs, 0.5)?;
    println!("Working context shape: {:?}", working.context.shape());

    // 5. Hologram health: signal quality degrades gracefully as it fills
    println!(
        "Estimated SNR: {:.2}, episodes: {}",
        experience.estimated_snr(),
        experience.episode_ids().len()
    );

    // 6. Raw ops: bind and project directly
    let pair = bind(&obs, &act)?;
    let unbound = project(&pair, &obs)?;
    println!("Direct bind/project shape: {:?}", unbound.shape());

    Ok(())
}
