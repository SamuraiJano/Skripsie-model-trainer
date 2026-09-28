# Deblurring references and attribution

The implementation in `model_training/deblur/` is self-contained project code. It is informed by, but does not vendor verbatim source files from:

- York-SDCNLab, **Ghost-DeblurGAN** (IROS 2022), MIT licence. The baseline follows its GhostNet hierarchy, Ghost convolution, HIN, FPN heads, and residual restoration concepts. See `LICENSES/Ghost-DeblurGAN-MIT.txt`.
- VITA-Group, **DeblurGAN-v2** (ICCV 2019), BSD licence. The trainer follows its paired-data, FPN, double-scale discriminator, and relativistic least-squares GAN concepts. See `LICENSES/DeblurGANv2-BSD.txt`.
- Huang et al., **Lightweight GAN for Restoring Blurred Images to Enhance Citrus Detection**, *Plants* 14(19), 3085 (2025), DOI `10.3390/plants14193085`.

`AGGDeblurGANReproduction` is a paper-guided reproduction. No verified official public implementation was available when it was written. The paper describes the main hierarchy and modules but not every expansion ratio, padding choice, or block-level ordering; those approximations are documented in `deblur/models.py` and exposed through `DeblurModelConfig`. Do not describe this implementation as official AGG-DeblurGAN code.

No reference repository is cloned or downloaded during normal runtime or tests. VGG19 weights must be supplied as a local file; automatic downloads are disabled.
