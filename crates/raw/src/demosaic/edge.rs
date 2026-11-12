//! Process V9: across a luminance edge, red and blue follow green's shape. AHD (`ahd.rs`) completes red and blue as
//! green plus their difference to green at the samples they are interpolated from, and the fit-size binning
//! (`binned.rs`) reads them between neighbouring blocks. On a steep edge both go wrong the same way: the difference of a
//! colour to green grows with the brightness, so a pixel on the edge, whose green lies between its sites' greens, takes
//! the bright side's large difference (a neon fringe at full size), and the binning takes red from more of one side than
//! green and blue from more of the other (a violet line at fit size). Green times the colour's ratio to green at the
//! same sites stays put across such an edge.
//!
//! The rule runs only on an edge: where the pixel's green and the green at one of its sites lie a step apart
//! ([`EDGE_STOPS`]), more than noise could have made ([`EDGE_SIGMAS`]). Flat and smoothly shaded areas develop bit for
//! bit as without it (there the ratio and the difference give nearly the same colour anyway). A ratio to a noisy green
//! comes out too high on average (the mean of `1 / g` exceeds `1 / mean(g)` by about the green's squared relative
//! noise), which turned dark areas magenta, so the ratio is trusted by the green's relative noise, from a sensor noise
//! model ([`SensorNoise`]: the file's black and white levels and its ISO), not by fixed dark levels ([`NOISE_CV`]), and
//! it is summed over more sites than the nearest ones to bring that noise down ([`SUPPORT_OUTER`]).
//!
//! Next to clipped samples the sums leave out the sites whose green the clip understates (a clipped sample on or beside
//! the site): a ratio to such a green is too high, and on the bright side of dark bars against a clipped sky it drew a
//! magenta line (worst at fit size at -3 EV). The outer sites reach into the clipping first. At full size the nearest
//! sites are left out too (none left: the colour difference stays); at fit size only the outer blocks, since the
//! alignment's own blocks carry the colour being corrected and the result stays within their means.
//!
//! Every tunable of the rule is here, so a round of tuning against Lightroom Classic changes this file only:
//!
//! | constant | what it sets |
//! |---|---|
//! | [`EDGE_STOPS`] | the edge gate: the step (in stops) between a pixel's green and the green at its sites |
//! | [`EDGE_SIGMAS`] | the edge gate's noise allowance: how much of a step noise may have made |
//! | [`NOISE_CV`] | the noise weight: the relative noise of the sites' green at which the ratio is trusted fully / not |
//! | [`SUPPORT_OUTER`] | the support: weight of the next sites out, beside the nearest ones |
//! | [`GUIDE_BRIGHT`] | the near-clip fade (full size): back to the difference as the pixel's green nears the clip |
//! | [`WHITE_ELECTRONS_ISO100`], [`READ_NOISE_ELECTRONS`], [`READ_NOISE_DN`], [`DEFAULT_ISO`] | the noise model |
//!
//! Sources: own design. Interpolating a colour as a ratio to green (constant hue) goes back to D. R. Cok's
//! constant-hue interpolation (US patent 4,642,678, 1987); the demosaic it corrects is Hirakawa & Parks' AHD (IEEE TIP
//! 14(3), 2005). The noise model is the usual shot-plus-read one (variance = signal / electrons at white + read
//! variance; e.g. Foi et al., "Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data",
//! IEEE TIP 17(10), 2008).

/// The edge gate, in stops: the ratio blends in as the largest step between a pixel's green and the green at the
/// nearest sites a colour is interpolated from (1 px away at full size, the alignment's blocks at fit size) grows from
/// `.0` (none: the colour difference or the aligned mean as before, bit for bit) to `.1` (all). Smooth shading steps
/// far less than this from one pixel to the next (a sky that brightens by a stop over 100 px: 0.01 stops), an edge as
/// a lens draws it (a bridge against a bright sky) a fraction of a stop to several stops at each of its pixels. Centred
/// on 0.2 stops, a 15 % step. At fit size the steps are read between blocks `k` sensor pixels apart, so the gate is
/// `k / 2` times as wide there (×2 as here, ×4 twice as wide): a gradient steps `k` times as far between blocks.
pub(crate) const EDGE_STOPS: (f32, f32) = (0.1, 0.3);

/// The edge gate's noise allowance, in standard deviations of the difference of two greens ([`SensorNoise`], over the
/// samples each green is the mean of): each step is first shrunk by this much. Where noise alone makes the steps, the
/// gate would pick out the pixels whose greens noise has pushed apart, and the ratio's bias with them (a ratio to a
/// green that noise pushed down is too high): red came out 1 % too high around glints on a dark noisy surface without
/// it. Pure noise steps 3 σ past a neighbour's about once in 400.
pub(crate) const EDGE_SIGMAS: f32 = 3.0;

/// The noise weight: the ratio is trusted fully where the relative noise (standard deviation over mean) of the green
/// summed over its sites is at most `.0`, and not at all from `.1`. The ratio's mean comes out too high by about that
/// relative noise squared: 0.1 % at 3 %, 0.36 % at 6 % (the ratio everywhere at fit size, with no such weight, turned
/// shadows magenta: a* +2.3 at L* < 20). From [`SensorNoise`] at the sites' mean green.
pub(crate) const NOISE_CV: (f32, f32) = (0.03, 0.06);

/// The support: the ratio's sums take the nearest sites of the colour (weight 1, the ones the colour difference reads)
/// and the next ones out with this weight (0: the nearest only). At full size, a green pixel's horizontal neighbours'
/// colour adds the same columns two rows up and down, its vertical neighbours' colour the same rows two columns left
/// and right, and a red or blue pixel's diagonal neighbours' colour the 8 sites at (±1, ±3) and (±3, ±1); at fit size
/// the blocks one further out on either side of the alignment's two (weighted so a linear ramp is still read at the
/// block centre). 0.5 cuts the green sum's noise by about 1.6× at full size (2 → 5.3 sites' worth for a green pixel, 4
/// → 10.7 for a red or blue one), which lets the ratio work about 2.5× darker for the same bias; the outer sites
/// still lie within 3 px, so the colour they read is the edge's own.
pub(crate) const SUPPORT_OUTER: f32 = 0.5;

/// The near-clip fade (full size only): the ratio fades back to the colour difference as the pixel's own green nears
/// the clip (from `.0` to `.1` of the white). The greens around it are clipped or about to clip there, and a ratio to
/// an understated green overstates the colour; these pixels are also what the highlight rebuild takes as a clipped
/// surface's colour just below its clip (with the ratio there a frame's partly clipped foliage came out 0.4 dE further
/// from Lightroom Classic). The fit-size binning keeps each colour within the neighbouring blocks' means instead.
pub(crate) const GUIDE_BRIGHT: (f32, f32) = (0.7, 0.95);

/// Added to both sides of the ratio, and to both greens of the edge gate (of the white), so that they stay finite at
/// black.
pub(crate) const RATIO_FLOOR: f32 = 1e-3;

/// Noise model: photo-electrons a photosite holds at the file's white level at ISO 100, scaled by 100 / ISO. A
/// deliberately small sensor's worth (about a 4 µm pixel; a full-frame 24-45 MP sensor holds 2-3 times as many), so
/// that the model errs toward too much noise: the ratio then stays out of shadows a little longer than it needs to.
pub(crate) const WHITE_ELECTRONS_ISO100: f32 = 20_000.0;
/// Noise model: read noise referred to the sensor (electrons, standard deviation), which dominates the shadows at
/// high ISO.
pub(crate) const READ_NOISE_ELECTRONS: f32 = 3.0;
/// Noise model: read noise after the amplifier (raw units, standard deviation: the converter, pattern noise and
/// rounding), which dominates the shadows at base ISO.
pub(crate) const READ_NOISE_DN: f32 = 2.0;
/// Noise model: the ISO assumed when the file names none (a high one, so the ratio errs toward staying out).
pub(crate) const DEFAULT_ISO: u32 = 800;
/// Noise model: the raw range (white - black, raw units) assumed without a file (a 14-bit converter) and for float
/// raws (whose rounding is negligible).
pub(crate) const DEFAULT_RANGE_DN: f32 = 16_383.0 - 512.0;

/// Noise of a raw file's samples, normalised to its white level (1.0): variance `shot · level + read` (shot noise and
/// read noise). Built from the file's raw range (white - black level) and ISO ([`SensorNoise::new`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SensorNoise {
    /// Variance per unit of signal: 1 / the electrons at white.
    shot: f32,
    /// Variance at black.
    read: f32,
}

impl SensorNoise {
    /// The model for a file whose white level lies `range_dn` raw units above its black level, shot at `iso` (`None`:
    /// [`DEFAULT_ISO`]). A range or ISO that isn't a positive, finite number takes the default.
    pub fn new(range_dn: f32, iso: Option<u32>) -> SensorNoise {
        let range = if range_dn.is_finite() && range_dn >= 1.0 { range_dn } else { DEFAULT_RANGE_DN };
        let iso = iso.filter(|&i| i > 0).unwrap_or(DEFAULT_ISO) as f32;
        let electrons = (WHITE_ELECTRONS_ISO100 * 100.0 / iso).max(1.0);
        let read = (READ_NOISE_DN / range).powi(2) + (READ_NOISE_ELECTRONS / electrons).powi(2);
        SensorNoise { shot: 1.0 / electrons, read }
    }

    /// Variance of one sample at `level` (of the white).
    #[inline]
    pub(crate) fn variance(&self, level: f32) -> f32 {
        self.shot * level.max(0.0) + self.read
    }
}

impl Default for SensorNoise {
    /// [`SensorNoise::new`] for a 14-bit file of unknown ISO.
    fn default() -> SensorNoise {
        SensorNoise::new(DEFAULT_RANGE_DN, None)
    }
}

#[inline]
pub(crate) fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The rule for one file: [`EDGE_STOPS`] as ratios, and the file's noise.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Follow {
    edge: (f32, f32),
    noise: SensorNoise,
}

impl Follow {
    /// The rule at full size.
    pub(crate) fn new(noise: SensorNoise) -> Follow {
        Follow::wider(noise, 1.0)
    }

    /// The rule with its edge gate `scale` times as wide in stops (fit size: `k / 2` for `k × k` blocks).
    pub(crate) fn wider(noise: SensorNoise, scale: f32) -> Follow {
        let scale = if scale.is_finite() { scale.max(1.0) } else { 1.0 };
        Follow { edge: ((EDGE_STOPS.0 * scale).exp2(), (EDGE_STOPS.1 * scale).exp2()), noise }
    }

    /// Whether the greens step far enough for [`Follow::edge`] to pass anything before its noise allowance: a cheap
    /// first test, false for nearly every pixel of a flat or smooth area.
    #[inline(always)]
    pub(crate) fn may_step(&self, green: f32, lo: f32, hi: f32) -> bool {
        let g = green.max(0.0) + RATIO_FLOOR;
        // (NaN fails both comparisons)
        g > self.edge.0 * (lo.max(0.0) + RATIO_FLOOR) || hi.max(0.0) + RATIO_FLOOR > self.edge.0 * g
    }

    /// [`Follow::edge`]'s first test for greens that are each the mean of `samples` samples ([`FirstTest::passes`]).
    pub(crate) fn first_test(&self, samples: f32) -> FirstTest {
        let k = 0.99 * EDGE_SIGMAS * EDGE_SIGMAS / if samples > 1.0 { samples } else { 1.0 };
        let e = self.edge.0;
        FirstTest { e, k, e2k: e * e * k, noise: self.noise }
    }

    /// The edge gate ([`EDGE_STOPS`]) between a pixel's `green` and the green at its sites, which spans `lo..=hi`
    /// (each green the mean of `samples` samples' worth), each step shrunk by what noise could have made of it
    /// ([`EDGE_SIGMAS`]): 0 (exactly) below the gate's start, 1 past its end.
    #[inline]
    pub(crate) fn edge(&self, green: f32, lo: f32, hi: f32, samples: f32) -> f32 {
        if !self.may_step(green, lo, hi) {
            return 0.0;
        }
        let (g, lo, hi) = (green.max(0.0), lo.max(0.0), hi.max(0.0));
        let step = |a: f32, b: f32| (a + RATIO_FLOOR) / (b + RATIO_FLOOR);
        let samples = if samples > 1.0 { samples } else { 1.0 };
        let allow = |a: f32, b: f32| EDGE_SIGMAS * ((self.noise.variance(a) + self.noise.variance(b)) / samples).sqrt();
        let (lo, hi) = ((lo + allow(g, lo)).min(g), (hi - allow(g, hi)).max(g));
        let step = step(g, lo).max(step(hi, g));
        if step.is_nan() || step <= self.edge.0 {
            return 0.0;
        }
        smoothstep(self.edge.0, self.edge.1, step)
    }

    /// How far a ratio to a green of mean `level`, measured over `samples` samples' worth, can be trusted
    /// ([`NOISE_CV`]): 1 where its relative noise is small, 0 where it would bias the colour.
    #[inline]
    pub(crate) fn trust(&self, level: f32, samples: f32) -> f32 {
        if !(level > 0.0 && samples > 0.0) {
            return 0.0;
        }
        let cv = (self.noise.variance(level) / samples).sqrt() / level;
        if cv.is_nan() || cv >= NOISE_CV.1 {
            return 0.0;
        }
        1.0 - smoothstep(NOISE_CV.0, NOISE_CV.1, cv)
    }
}

/// Whether [`Follow::edge`] may pass anything for some greens, noise allowance included, without a square root, a
/// division or a branch (noise makes branches unpredictable): true wherever the gate is above 0 and, by a hair of
/// slack against rounding, a little beyond. Made by [`Follow::first_test`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct FirstTest {
    e: f32,
    k: f32,
    e2k: f32,
    noise: SensorNoise,
}

impl FirstTest {
    /// For a pixel's `green` and the greens around it, which span `lo..=hi`.
    #[inline]
    pub(crate) fn passes(&self, green: f32, lo: f32, hi: f32) -> bool {
        let (g, lo, hi) = (green.max(0.0), lo.max(0.0), hi.max(0.0));
        // each step passes when it outgrows its noise allowance (see `Follow::edge`), squared
        let (d_lo, d_hi) = ((g + RATIO_FLOOR) - self.e * (lo + RATIO_FLOOR), (hi + RATIO_FLOOR) - self.e * (g + RATIO_FLOOR));
        let var = |a: f32| self.noise.variance(a) + self.noise.variance(g);
        ((d_lo > 0.0) & (d_lo * d_lo > self.e2k * var(lo))) | ((d_hi > 0.0) & (d_hi * d_hi > self.k * var(hi)))
    }
}

/// The near-clip fade ([`GUIDE_BRIGHT`]) at a pixel whose own green is `green`: 1 below it, 0 at the clip.
#[inline]
pub(crate) fn fade(green: f32) -> f32 {
    1.0 - smoothstep(GUIDE_BRIGHT.0, GUIDE_BRIGHT.1, green)
}

/// `green` times a colour's ratio to green at its sites (`own` and `guide`: the colour's and green's weighted means
/// there).
#[inline]
pub(crate) fn ratio(green: f32, own: f32, guide: f32) -> f32 {
    green * (own + RATIO_FLOOR) / (guide.max(0.0) + RATIO_FLOOR)
}

/// Samples' worth of a weighted mean whose weights sum to `sum` and their squares to `sum_sq`, each term worth
/// `per_term` samples.
#[inline]
pub(crate) fn samples(sum: f32, sum_sq: f32, per_term: f32) -> f32 {
    if sum_sq > 0.0 { sum * sum / sum_sq * per_term } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// More ISO or a shorter raw range means more noise, a darker level more relative noise; defaults stand in for
    /// nonsense.
    #[test]
    fn the_noise_model_orders_sensibly() {
        let (base, high) = (SensorNoise::new(15_000.0, Some(100)), SensorNoise::new(15_000.0, Some(6400)));
        assert!(high.variance(0.1) > 10.0 * base.variance(0.1));
        assert!(SensorNoise::new(4000.0, Some(100)).variance(0.0) > base.variance(0.0));
        let cv = |n: SensorNoise, l: f32| n.variance(l).sqrt() / l;
        assert!(cv(base, 0.01) > cv(base, 0.1) && cv(base, 0.1) > cv(base, 0.5));
        for bad in [SensorNoise::new(f32::NAN, Some(0)), SensorNoise::new(-3.0, None), SensorNoise::new(0.0, None)] {
            assert_eq!(bad, SensorNoise::default());
        }
    }

    /// The gate is exactly 0 for a small step (both ways), rises with the step, and never yields NaN.
    #[test]
    fn the_edge_gate_and_the_trust_are_bounded() {
        let f = Follow::new(SensorNoise::new(15_000.0, Some(100)));
        // a step that noise could have made doesn't count
        let noisy = Follow::new(SensorNoise::new(15_000.0, Some(12_800)));
        assert!(f.edge(0.05, 0.05, 0.065, 1.0) > 0.0);
        assert_eq!(noisy.edge(0.05, 0.05, 0.065, 1.0), 0.0);
        // a wider gate (fit size, ×4) passes a step the full-size one takes
        assert!(f.edge(0.5, 0.5, 0.58, 1.0) > 0.0);
        assert_eq!(Follow::wider(SensorNoise::new(15_000.0, Some(100)), 2.0).edge(0.5, 0.5, 0.58, 1.0), 0.0);
        assert_eq!(f.edge(0.5, 0.5, 0.5, 1.0), 0.0);
        assert_eq!(f.edge(0.5, 0.5 / 1.05, 0.5 * 1.05, 1.0), 0.0);
        assert!(f.edge(0.5, 0.5, 0.6, 1.0) > 0.0 && f.edge(0.5, 0.5, 0.6, 1.0) < f.edge(0.5, 0.5, 0.65, 1.0));
        // the mean of more samples is less noisy: less of its step is allowed for
        assert!(noisy.edge(0.05, 0.05, 0.08, 64.0) > 0.0 && noisy.edge(0.05, 0.05, 0.08, 1.0) == 0.0);
        assert!(f.edge(0.5, 0.42, 0.5, 1.0) > 0.0);
        assert_eq!(f.edge(0.2, 0.2, 0.5, 1.0), 1.0);
        assert_eq!(f.edge(0.5, 0.2, 0.5, 1.0), 1.0);
        // the first test passes wherever the gate does
        for (g, lo, hi, n) in
            [(0.5, 0.42, 0.5, 1.0), (0.05, 0.05, 0.08, 64.0), (0.05, 0.05, 0.08, 1.0), (0.3, 0.2, 0.31, 2.0), (0.01, 0.002, 0.01, 8.0)]
        {
            for r in [f, noisy] {
                assert!(r.edge(g, lo, hi, n) == 0.0 || r.first_test(n).passes(g, lo, hi), "{g} {lo} {hi} {n}");
            }
        }
        assert!(!noisy.first_test(1.0).passes(0.05, 0.05, 0.065));
        for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0, 0.0] {
            for w in [f32::NAN, 0.3, f32::INFINITY] {
                assert!((0.0..=1.0).contains(&f.edge(v, w, w, 1.0)), "{v} {w}");
                assert!((0.0..=1.0).contains(&f.edge(w, v, w, 1.0)), "{v} {w}");
                assert!((0.0..=1.0).contains(&f.trust(v, w)), "{v} {w}");
            }
        }
        // bright: trusted; deep shadow at high ISO: not
        assert_eq!(f.trust(0.5, 4.0), 1.0);
        assert_eq!(Follow::new(SensorNoise::new(15_000.0, Some(6400))).trust(0.02, 4.0), 0.0);
    }
}
