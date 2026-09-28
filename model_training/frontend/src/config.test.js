import assert from "node:assert/strict";
import test from "node:test";
import { buildTrainingConfig, parseClassList } from "./config.js";

test("training form conversion preserves the documented deblur defaults", () => {
  const result = buildTrainingConfig({
    architecture: "GhostDeblurGAN", image_size: "256", epochs: "200", batch_size: "4",
    learning_rate: "0.0001", final_learning_rate: "0.000001", weight_decay: "0", workers: "0",
    seed: "42", generator_updates: "1", discriminator_updates: "1", linear_decay_start_epoch: "10",
    perceptual_weight: "0.7", l1_weight: "0.3", adversarial_weight: "0.005",
    vgg19_weights_path: "", resume_checkpoint: "", auto_resume: true, checkpoint_interval_steps: "100"
  });
  assert.equal(result.image_size, 256);
  assert.equal(result.model.architecture, "GhostDeblurGAN");
  assert.equal("architecture" in result, false);
  assert.equal(result.perceptual_weight, 0.7);
  assert.equal(result.adversarial_weight, 0.005);
  assert.equal(result.vgg19_weights_path, null);
  assert.equal(result.auto_resume, true);
  assert.equal(result.checkpoint_interval_steps, 100);
});

test("class risk lists are trimmed and empty values removed", () => {
  assert.deepEqual(parseClassList("Class 1, Class 2, , Class 3"), ["Class 1", "Class 2", "Class 3"]);
});
