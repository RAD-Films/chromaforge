## Develop

- **No fringe on bright edges:** process V9's demosaic and its fit-size binning take red and blue as a ratio to green across every edge, not only next to clipped areas, so a dark edge against a bright, unclipped sky no longer gets a violet line at fit size or a neon fringe at full size. The rule runs only where green steps between neighbouring samples and where the file's noise (from its black and white levels and ISO) lets the ratio be trusted, so flat areas and noisy shadows are unchanged.
