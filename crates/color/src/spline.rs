//! Curves for tone mapping: monotone cubic Hermite splines (Fritsch–Carlson) and lookup tables.
//!
//! Monotone interpolation never overshoots between control points, so a point curve can't invert
//! tones by accident (a common complaint with natural cubic splines).

use serde::{Deserialize, Serialize};

/// A curve through control points in `0..1 × 0..1` (x strictly increasing after normalization).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MonotoneCurve {
    xs: Vec<f64>,
    ys: Vec<f64>,
    ms: Vec<f64>,
}

impl MonotoneCurve {
    /// Build from points; sorts by x, merges duplicates. Fewer than two points → identity.
    pub fn new(points: &[(f64, f64)]) -> MonotoneCurve {
        let mut p: Vec<(f64, f64)> = points.iter().copied().filter(|(x, y)| x.is_finite() && y.is_finite()).collect();
        p.sort_by(|a, b| a.0.total_cmp(&b.0));
        p.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9);
        if p.len() < 2 {
            p = vec![(0.0, 0.0), (1.0, 1.0)];
        }
        let n = p.len();
        let xs: Vec<f64> = p.iter().map(|q| q.0).collect();
        let ys: Vec<f64> = p.iter().map(|q| q.1).collect();
        let d: Vec<f64> = (0..n - 1).map(|i| (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])).collect();
        let mut ms = vec![0.0; n];
        ms[0] = d[0];
        ms[n - 1] = d[n - 2];
        for i in 1..n - 1 {
            ms[i] = if d[i - 1] * d[i] <= 0.0 { 0.0 } else { (d[i - 1] + d[i]) / 2.0 };
        }
        // Fritsch–Carlson limiter where the data is monotone within a segment.
        for i in 0..n - 1 {
            if d[i] == 0.0 {
                ms[i] = 0.0;
                ms[i + 1] = 0.0;
                continue;
            }
            let a = ms[i] / d[i];
            let b = ms[i + 1] / d[i];
            let s = a * a + b * b;
            if s > 9.0 {
                let t = 3.0 / s.sqrt();
                ms[i] = t * a * d[i];
                ms[i + 1] = t * b * d[i];
            }
        }
        MonotoneCurve { xs, ys, ms }
    }

    pub fn identity() -> MonotoneCurve {
        MonotoneCurve::new(&[(0.0, 0.0), (1.0, 1.0)])
    }

    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if x.is_nan() {
            return self.ys[0];
        }
        if x <= self.xs[0] {
            return self.ys[0];
        }
        if x >= self.xs[n - 1] {
            return self.ys[n - 1];
        }
        let i = match self.xs.binary_search_by(|v| v.total_cmp(&x)) {
            Ok(i) => return self.ys[i],
            Err(i) => i.saturating_sub(1).min(n - 2),
        };
        let h = self.xs[i + 1] - self.xs[i];
        let t = (x - self.xs[i]) / h;
        let (t2, t3) = (t * t, t * t * t);
        (2.0 * t3 - 3.0 * t2 + 1.0) * self.ys[i]
            + (t3 - 2.0 * t2 + t) * h * self.ms[i]
            + (-2.0 * t3 + 3.0 * t2) * self.ys[i + 1]
            + (t3 - t2) * h * self.ms[i + 1]
    }

    /// Sample into a LUT of `n` entries over `0..1` (at least 2).
    pub fn to_lut(&self, n: usize) -> Lut1 {
        let n = n.max(2);
        Lut1 { v: (0..n).map(|i| self.eval(i as f64 / (n - 1) as f64) as f32).collect() }
    }

    pub fn points(&self) -> Vec<(f64, f64)> {
        self.xs.iter().copied().zip(self.ys.iter().copied()).collect()
    }
}

/// A natural cubic spline through control points in `0..1 × 0..1` (zero second derivative at both ends), flat
/// outside the first and last point. Unlike [`MonotoneCurve`] it can overshoot between points; [`Self::to_lut`]
/// clamps to `0..1`. Lightroom Classic's point curves measure as this spline (round 18 charts: ΔE76 ≤ 0.07 on
/// one-point curves, where the monotone spline is off by about 1).
#[derive(Clone, Debug, PartialEq)]
pub struct NaturalCurve {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Second derivatives at the points.
    m2: Vec<f64>,
}

impl NaturalCurve {
    /// Build from points; sorts by x, merges duplicates, drops non-finite ones. Fewer than two points → identity.
    pub fn new(points: &[(f64, f64)]) -> NaturalCurve {
        let mut p: Vec<(f64, f64)> = points.iter().copied().filter(|(x, y)| x.is_finite() && y.is_finite()).collect();
        p.sort_by(|a, b| a.0.total_cmp(&b.0));
        p.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9);
        if p.len() < 2 {
            p = vec![(0.0, 0.0), (1.0, 1.0)];
        }
        let n = p.len();
        let xs: Vec<f64> = p.iter().map(|q| q.0).collect();
        let ys: Vec<f64> = p.iter().map(|q| q.1).collect();
        let mut m2 = vec![0.0; n];
        if n > 2 {
            // tridiagonal system for the inner second derivatives (Thomas algorithm), natural ends
            let h: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).collect();
            let mut diag = vec![0.0; n];
            let mut rhs = vec![0.0; n];
            let mut upper = vec![0.0; n];
            for i in 1..n - 1 {
                diag[i] = 2.0 * (h[i - 1] + h[i]);
                upper[i] = h[i];
                rhs[i] = 6.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);
            }
            for i in 2..n - 1 {
                let w = h[i - 1] / diag[i - 1];
                diag[i] -= w * upper[i - 1];
                rhs[i] -= w * rhs[i - 1];
            }
            for i in (1..n - 1).rev() {
                m2[i] = (rhs[i] - upper[i] * m2[i + 1]) / diag[i];
            }
        }
        NaturalCurve { xs, ys, m2 }
    }

    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if x.is_nan() || x <= self.xs[0] {
            return self.ys[0];
        }
        if x >= self.xs[n - 1] {
            return self.ys[n - 1];
        }
        let i = match self.xs.binary_search_by(|v| v.total_cmp(&x)) {
            Ok(i) => return self.ys[i],
            Err(i) => i.saturating_sub(1).min(n - 2),
        };
        let h = self.xs[i + 1] - self.xs[i];
        let a = (self.xs[i + 1] - x) / h;
        let b = (x - self.xs[i]) / h;
        a * self.ys[i] + b * self.ys[i + 1] + ((a * a * a - a) * self.m2[i] + (b * b * b - b) * self.m2[i + 1]) * h * h / 6.0
    }

    /// Sample into a LUT of `n` entries over `0..1` (at least 2), clamped to `0..1`.
    pub fn to_lut(&self, n: usize) -> Lut1 {
        let n = n.max(2);
        Lut1 { v: (0..n).map(|i| self.eval(i as f64 / (n - 1) as f64).clamp(0.0, 1.0) as f32).collect() }
    }
}

/// A point curve's spline: [`MonotoneCurve`] (masks' Curves, and the Tone Curve before process V9) or
/// [`NaturalCurve`] (the Tone Curve from V9), so the curve editor draws what the render applies.
#[derive(Clone, Debug, PartialEq)]
pub enum PointSpline {
    Monotone(MonotoneCurve),
    Natural(NaturalCurve),
}

impl PointSpline {
    pub fn new(points: &[(f64, f64)], natural: bool) -> PointSpline {
        if natural { PointSpline::Natural(NaturalCurve::new(points)) } else { PointSpline::Monotone(MonotoneCurve::new(points)) }
    }

    pub fn eval(&self, x: f64) -> f64 {
        match self {
            PointSpline::Monotone(c) => c.eval(x),
            PointSpline::Natural(c) => c.eval(x),
        }
    }
}

/// A 1-D lookup table over `0..1` with linear interpolation; inputs outside clamp.
#[derive(Clone, Debug, PartialEq)]
pub struct Lut1 {
    pub v: Vec<f32>,
}

impl Lut1 {
    pub fn identity(n: usize) -> Lut1 {
        let n = n.max(2);
        Lut1 { v: (0..n).map(|i| i as f32 / (n - 1) as f32).collect() }
    }
    pub fn from_fn(n: usize, f: impl Fn(f32) -> f32) -> Lut1 {
        let n = n.max(2);
        Lut1 { v: (0..n).map(|i| f(i as f32 / (n - 1) as f32)).collect() }
    }
    #[inline]
    pub fn eval(&self, x: f32) -> f32 {
        let n = self.v.len();
        if n < 2 {
            // Only reachable by building `v` by hand; a constant table.
            return self.v.first().copied().unwrap_or(x);
        }
        let f = x.clamp(0.0, 1.0) * (n - 1) as f32;
        let i = (f as usize).min(n - 2);
        let t = f - i as f32;
        self.v[i] + (self.v[i + 1] - self.v[i]) * t
    }
    /// `self ∘ inner` (apply `inner` first).
    pub fn compose(&self, inner: &Lut1) -> Lut1 {
        Lut1 { v: inner.v.iter().map(|&x| self.eval(x)).collect() }
    }
    pub fn is_identity(&self) -> bool {
        let n = self.v.len();
        n >= 2 && self.v.iter().enumerate().all(|(i, &y)| (y - i as f32 / (n - 1) as f32).abs() < 1e-6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_curve() {
        let c = MonotoneCurve::identity();
        for i in 0..=10 {
            let x = i as f64 / 10.0;
            assert!((c.eval(x) - x).abs() < 1e-12);
        }
        assert!(c.to_lut(256).is_identity());
    }

    #[test]
    fn passes_through_points_and_is_monotone() {
        let pts = [(0.0, 0.0), (0.25, 0.15), (0.5, 0.5), (0.75, 0.88), (1.0, 1.0)];
        let c = MonotoneCurve::new(&pts);
        for (x, y) in pts {
            assert!((c.eval(x) - y).abs() < 1e-12);
        }
        let mut prev = -1.0;
        for i in 0..=1000 {
            let y = c.eval(i as f64 / 1000.0);
            assert!(y >= prev - 1e-12);
            prev = y;
        }
    }

    #[test]
    fn no_overshoot_on_flat_segment() {
        let c = MonotoneCurve::new(&[(0.0, 0.0), (0.4, 0.5), (0.6, 0.5), (1.0, 1.0)]);
        for i in 400..=600 {
            let y = c.eval(i as f64 / 1000.0);
            assert!((y - 0.5).abs() < 1e-9, "{y}");
        }
    }

    #[test]
    fn hostile_inputs_do_not_panic() {
        let c = MonotoneCurve::new(&[(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)]);
        for x in [f64::NAN, -f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e300, 1e300] {
            assert!(c.eval(x).is_finite(), "{x}");
        }
        // Non-finite control points are dropped; too few left means identity.
        let bad = MonotoneCurve::new(&[(f64::NAN, 0.0), (0.5, f64::INFINITY)]);
        assert!((bad.eval(0.3) - 0.3).abs() < 1e-12);
        for n in [0, 1] {
            let l = MonotoneCurve::identity().to_lut(n);
            assert!(l.eval(0.5).is_finite());
            assert!(Lut1::identity(n).eval(0.5).is_finite());
        }
        assert!(Lut1 { v: vec![] }.eval(0.5).is_finite());
        assert!(Lut1 { v: vec![0.25] }.eval(0.5).is_finite());
    }

    #[test]
    fn natural_curve_passes_through_points_and_matches_the_textbook_spline() {
        let pts = [(0.0, 0.0), (128.0 / 255.0, 160.0 / 255.0), (1.0, 1.0)];
        let c = NaturalCurve::new(&pts);
        for (x, y) in pts {
            assert!((c.eval(x) - y).abs() < 1e-12);
        }
        // one inner point: m2 = 6·(d1 − d0)/(2·(h0 + h1)); at x = 0.25 the textbook value
        let (h0, h1) = (128.0 / 255.0, 127.0 / 255.0);
        let (d0, d1) = ((160.0 / 255.0) / h0, (95.0 / 255.0) / h1);
        let m = 6.0 * (d1 - d0) / (2.0 * (h0 + h1));
        let (a, b) = ((h0 - 0.25) / h0, 0.25 / h0);
        let want = b * (160.0 / 255.0) + (b * b * b - b) * m * h0 * h0 / 6.0 + (a * a * a - a) * 0.0;
        assert!((c.eval(0.25) - want).abs() < 1e-12, "{} vs {want}", c.eval(0.25));
        // two points: a straight line; outside the points: flat
        let l = NaturalCurve::new(&[(0.0, 30.0 / 255.0), (1.0, 225.0 / 255.0)]);
        assert!((l.eval(0.5) - 127.5 / 255.0).abs() < 1e-12);
        let e = NaturalCurve::new(&[(0.2, 0.1), (0.8, 0.9)]);
        assert_eq!((e.eval(0.0), e.eval(1.0)), (0.1, 0.9));
        // overshoot is clamped in the table only
        let s = NaturalCurve::new(&[(0.0, 0.0), (32.0 / 255.0, 96.0 / 255.0), (1.0, 1.0)]);
        assert!(s.eval(0.6) > 1.0 && s.to_lut(1024).v.iter().all(|v| (0.0..=1.0).contains(v)));
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e300, 1e300] {
            assert!(c.eval(x).is_finite(), "{x}");
        }
        assert!((NaturalCurve::new(&[(f64::NAN, 0.0)]).eval(0.3) - 0.3).abs() < 1e-12);
    }

    #[test]
    fn lut_compose() {
        let a = Lut1::from_fn(1024, |x| x * x);
        let b = Lut1::from_fn(1024, |x| x.sqrt());
        let c = a.compose(&b);
        assert!((c.eval(0.3) - 0.3).abs() < 2e-3);
    }

    proptest::proptest! {
        #[test]
        fn monotone_for_increasing_data(ys in proptest::collection::vec(0.0f64..1.0, 3..8)) {
            let mut ys = ys; ys.sort_by(|a, b| a.total_cmp(b));
            let n = ys.len();
            let pts: Vec<(f64, f64)> = ys.iter().enumerate().map(|(i, y)| (i as f64 / (n - 1) as f64, *y)).collect();
            let c = MonotoneCurve::new(&pts);
            let mut prev = f64::MIN;
            for i in 0..=200 { let y = c.eval(i as f64 / 200.0); proptest::prop_assert!(y >= prev - 1e-9); prev = y; }
        }
    }
}
