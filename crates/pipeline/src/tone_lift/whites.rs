//! Generated from the raw Whites +100 fit to Lightroom Classic exports of our own Canon EOS R5 C raws (rounds 7
//! and 8, 7 Nov 2026, OneSlips; see `super`) by the calibration scripts kept with them. Do not edit by hand.

use super::Whites;

pub(super) const WHITES_UP_RAW: Whites = Whites {
    pct: 99.9,
    white: 2.47393,
    gap_max: 2.0492,
    knot_lo: -10.0,
    c0: [
        -0.313712, -0.309218, -0.276398, -0.154565, -0.013512, 0.072034, 0.147753, 0.194738, 0.218119, 0.241071, 0.24949, 0.275253, 0.346997,
        0.44119, 0.527572, 0.596743, 0.621247, 0.593044, 0.504812, 0.361045, 0.180645, 0.000246, -0.180154,
    ],
    per_gap: [
        1.18618, 1.12703, 1.03629, 0.976725, 0.949436, 0.957078, 1.07035, 1.26092, 1.50199, 1.78068, 2.0113, 2.16545, 2.21567, 2.15756, 2.02544,
        1.88633, 1.78373, 1.71588, 1.62979, 1.52523, 1.41785, 1.31047, 1.20309,
    ],
};
