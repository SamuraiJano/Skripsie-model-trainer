export function parseClassList(value) {
  return String(value).split(",").map((item) => item.trim()).filter(Boolean);
}

export function buildTrainingConfig(form) {
  const { architecture, ...values } = form;
  return {
    ...values,
    image_size: Number(form.image_size), epochs: Number(form.epochs), batch_size: Number(form.batch_size),
    learning_rate: Number(form.learning_rate), final_learning_rate: Number(form.final_learning_rate),
    weight_decay: Number(form.weight_decay), workers: Number(form.workers), seed: Number(form.seed),
    generator_updates: Number(form.generator_updates), discriminator_updates: Number(form.discriminator_updates),
    checkpoint_interval_steps: Number(form.checkpoint_interval_steps),
    linear_decay_start_epoch: Number(form.linear_decay_start_epoch), perceptual_weight: Number(form.perceptual_weight),
    l1_weight: Number(form.l1_weight), adversarial_weight: Number(form.adversarial_weight),
    vgg19_weights_path: form.vgg19_weights_path || null, resume_checkpoint: form.resume_checkpoint || null,
    model: { architecture }
  };
}
