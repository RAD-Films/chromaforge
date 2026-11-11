// Element-wise kernels (one thread per pixel, 1-D on a 2-D grid). Bindings: a, b, c (inputs), dst.
// P[0] = pixel count; further parameters per kernel. Each mirrors a CPU expression in
// `lcc_pipeline::local` (named in the comment).

fn rgb_a(i: u32) -> vec3<f32> {
    return vec3<f32>(a[3u * i], a[3u * i + 1u], a[3u * i + 2u]);
}

fn put_rgb(i: u32, v: vec3<f32>) {
    dst[3u * i] = v.x;
    dst[3u * i + 1u] = v.y;
    dst[3u * i + 2u] = v.z;
}

// `log_lum` of an RGB image.
@compute @workgroup_size(256)
fn log_lum_k(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    dst[i] = log_lum(rgb_a(i));
}

// `dark_of`: the dark channel (dehaze).
@compute @workgroup_size(256)
fn dark_k(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let c = rgb_a(i);
    dst[i] = min(min(c.x, c.y), c.z);
}

// Guided filter input pair (p, p²).
@compute @workgroup_size(256)
fn guided_pre(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let p = a[i];
    dst[2u * i] = p;
    dst[2u * i + 1u] = p * p;
}

// Guided filter coefficients (a, b) from blurred (mean, corr). P[1] = eps.
@compute @workgroup_size(256)
fn guided_ab(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let m = a[2u * i];
    let c = a[2u * i + 1u];
    let v = max(c - m * m, 0.0);
    let k = v / (v + pf(1u));
    dst[2u * i] = k;
    dst[2u * i + 1u] = m - k * m;
}

// `q = a·p + b` (a: p, b: interleaved blurred coefficients).
@compute @workgroup_size(256)
fn guided_apply(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    dst[i] = b[2u * i] * a[i] + b[2u * i + 1u];
}

// Cross guided filter (`masks::guided_cross`), a: guide, b: input. P[1] = 0: (guide, input);
// 1: (guide·input, guide²).
@compute @workgroup_size(256)
fn xguided_pre(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let gi = a[i];
    let p = b[i];
    if (pu(1u) == 0u) {
        dst[2u * i] = gi;
        dst[2u * i + 1u] = p;
    } else {
        dst[2u * i] = gi * p;
        dst[2u * i + 1u] = gi * gi;
    }
}

// Coefficients (a, b) from blurred means (a: guide, input) and correlations (b: guide·input,
// guide²). P[1] = eps.
@compute @workgroup_size(256)
fn xguided_ab(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let mi = a[2u * i];
    let mp = a[2u * i + 1u];
    let v = max(b[2u * i + 1u] - mi * mi, 0.0);
    let k = (b[2u * i] - mi * mp) / (v + pf(1u));
    dst[2u * i] = k;
    dst[2u * i + 1u] = mp - k * mi;
}

// q = clamp(a·guide + b, 0, 1) (a: guide, b: blurred coefficients, c: the input p), then
// P[1] = 0: max(p, q) (Auto Mask strokes); 1: p + (q − p)·P[2] (Refine Edges).
@compute @workgroup_size(256)
fn xguided_apply(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let q = clamp(b[2u * i] * a[i] + b[2u * i + 1u], 0.0, 1.0);
    if (pu(1u) == 0u) {
        dst[i] = max(c[i], q);
    } else {
        dst[i] = c[i] + (q - c[i]) * pf(2u);
    }
}

// White balance: 3×3 matrix (P[1] = has matrix, P[2..11] row-major), clamped at 0.
@compute @workgroup_size(256)
fn wb_k(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    var c = rgb_a(i);
    if (pu(1u) != 0u) {
        c = vec3<f32>(
            pf(2u) * c.x + pf(3u) * c.y + pf(4u) * c.z,
            pf(5u) * c.x + pf(6u) * c.y + pf(7u) * c.z,
            pf(8u) * c.x + pf(9u) * c.y + pf(10u) * c.z,
        );
    }
    put_rgb(i, max(c * 1.0, vec3<f32>(0.0)));
}

// White balance, then (process V9) a camera profile's tables: `wb_k`, then `lcc_raw::profile::ProfileTables::apply`
// (into linear ProPhoto, hue/saturation map, baseline gain, look table, back), as `lcc_pipeline::local::wb_gain` does;
// non-finite channels become 0, then clamped at 0. P[1] = has matrix, P[2..11] the matrix, P[11..20] into ProPhoto,
// P[20..29] out of it, P[29] the baseline gain, P[30..36] the hue/saturation map's and P[36..42] the look table's
// header (present, hue / saturation / value divisions, sRGB-coded value, first entry in b). b: the tables' entries
// (hue shift, saturation scale, value scale).
fn tab_wrap(x: f32, p: f32) -> f32 {
    return x - p * floor(x / p);
}

fn tab_l2s(v: f32) -> f32 {
    if (v <= 0.0031308) {
        return v * 12.92;
    }
    return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

fn tab_s2l(v: f32) -> f32 {
    if (v <= 0.04045) {
        return v / 12.92;
    }
    return pow((v + 0.055) / 1.055, 2.4);
}

fn tab_limited(x: f32, k: f32) -> f32 {
    if (x <= 1.0) {
        return min(x * k, 1.0);
    }
    return x * min(k, 1.0);
}

fn tab_entry(t: u32, v: u32, h: u32, s: u32) -> vec3<f32> {
    let k = pu(t + 5u) + (v * pu(t + 1u) + h) * pu(t + 2u) + s;
    return vec3<f32>(b[3u * k], b[3u * k + 1u], b[3u * k + 2u]);
}

fn tab_lerp(x: vec3<f32>, y: vec3<f32>, t: f32) -> vec3<f32> {
    return x + (y - x) * t;
}

// (i0, i1, t) of a table axis with n ≥ 1 points
fn tab_axis(x: f32, n: u32) -> vec3<f32> {
    let f = clamp(x, 0.0, 1.0) * f32(n - 1u);
    let i0 = min(u32(f), select(0u, n - 2u, n >= 2u));
    let i1 = min(i0 + 1u, n - 1u);
    return vec3<f32>(f32(i0), f32(i1), clamp(f - f32(i0), 0.0, 1.0));
}

fn tab_plane(t: u32, vi: u32, h0: u32, h1: u32, s0: u32, s1: u32, ts: f32, th: f32) -> vec3<f32> {
    let lo = tab_lerp(tab_entry(t, vi, h0, s0), tab_entry(t, vi, h0, s1), ts);
    let hi = tab_lerp(tab_entry(t, vi, h1, s0), tab_entry(t, vi, h1, s1), ts);
    return tab_lerp(lo, hi, th);
}

fn tab_lookup(t: u32, h: f32, s: f32, v: f32) -> vec3<f32> {
    let hd = pu(t + 1u);
    let hf = clamp(tab_wrap(h, 360.0) / 360.0 * f32(hd), 0.0, f32(hd));
    let h0 = min(u32(hf), hd - 1u);
    let h1 = (h0 + 1u) % hd;
    let th = clamp(hf - f32(h0), 0.0, 1.0);
    let sa = tab_axis(s, pu(t + 2u));
    let s0 = u32(sa.x);
    let s1 = u32(sa.y);
    let vd = pu(t + 3u);
    if (vd <= 1u) {
        return tab_plane(t, 0u, h0, h1, s0, s1, sa.z, th);
    }
    let va = tab_axis(v, vd);
    let p0 = tab_plane(t, u32(va.x), h0, h1, s0, s1, sa.z, th);
    if (u32(va.x) == u32(va.y)) {
        return p0;
    }
    return tab_lerp(p0, tab_plane(t, u32(va.y), h0, h1, s0, s1, sa.z, th), va.z);
}

fn tab_hue(c: vec3<f32>, mx: f32, d: f32) -> f32 {
    if (d <= 0.0) {
        return 0.0;
    }
    var h = 0.0;
    if (mx == c.x) {
        h = tab_wrap((c.y - c.z) / d, 6.0);
    } else if (mx == c.y) {
        h = (c.z - c.x) / d + 2.0;
    } else {
        h = (c.x - c.y) / d + 4.0;
    }
    return h * 60.0;
}

fn tab_hsv_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let hp = tab_wrap(h, 360.0) / 60.0;
    let sector = min(u32(hp), 5u);
    let f = hp - f32(sector);
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    switch sector {
        case 0u: { return vec3<f32>(v, t, p); }
        case 1u: { return vec3<f32>(q, v, p); }
        case 2u: { return vec3<f32>(p, v, t); }
        case 3u: { return vec3<f32>(p, q, v); }
        case 4u: { return vec3<f32>(t, p, v); }
        default: { return vec3<f32>(v, p, q); }
    }
}

// `HsvTable::apply` of the table whose header starts at P[t] (the colour unchanged without one)
fn tab_apply(t: u32, c: vec3<f32>) -> vec3<f32> {
    if (pu(t) == 0u) {
        return c;
    }
    let mx = max(max(c.x, c.y), c.z);
    if (!(mx > 0.0) || !(mx <= 3.4028235e38)) {
        return c;
    }
    let mn = min(min(c.x, c.y), c.z);
    let d = mx - mn;
    let hue = tab_hue(c, mx, d);
    let sat = d / mx;
    let srgb = pu(t + 4u) != 0u;
    let vi = select(mx, tab_l2s(min(mx, 1.0)), srgb);
    let e = tab_lookup(t, hue, sat, vi);
    let s2 = tab_limited(sat, e.y);
    var v2 = 0.0;
    if (srgb && mx <= 1.0) {
        v2 = select(tab_s2l(min(vi * e.z, 1.0)), mx, e.z == 1.0);
    } else if (srgb) {
        v2 = mx * min(tab_s2l(min(e.z, 1.0)), 1.0);
    } else {
        v2 = tab_limited(mx, e.z);
    }
    return tab_hsv_rgb(hue + e.x, s2, v2);
}

fn tab_mat(o: u32, c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        pf(o) * c.x + pf(o + 1u) * c.y + pf(o + 2u) * c.z,
        pf(o + 3u) * c.x + pf(o + 4u) * c.y + pf(o + 5u) * c.z,
        pf(o + 6u) * c.x + pf(o + 7u) * c.y + pf(o + 8u) * c.z,
    );
}

fn tab_finite(v: f32) -> f32 {
    return select(0.0, v, abs(v) <= 3.4028235e38);
}

@compute @workgroup_size(256)
fn wb_tables_k(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    var c = rgb_a(i);
    if (pu(1u) != 0u) {
        c = tab_mat(2u, c);
    }
    let gain = pf(29u);
    var p = tab_mat(11u, c / gain);
    p = tab_apply(30u, p);
    p = p * gain;
    p = tab_apply(36u, p);
    c = tab_mat(20u, p);
    c = vec3<f32>(tab_finite(c.x), tab_finite(c.y), tab_finite(c.z));
    put_rgb(i, max(c, vec3<f32>(0.0)));
}

// Luminance NR: scale by 2^((f − l)·k) (a: image, b: log luminance l, c: filtered f). P[1] = k.
@compute @workgroup_size(256)
fn nr_lum(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let d = (c[i] - b[i]) * pf(1u);
    put_rgb(i, rgb_a(i) * exp2(d));
}

// Chromaticity rgb / Y (local Moire / Noise plane).
@compute @workgroup_size(256)
fn chroma_k(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let c0 = rgb_a(i);
    let y = max(lum2020(c0), 1e-6);
    put_rgb(i, vec3<f32>(c0.x / y, c0.y / y, c0.z / y));
}

// The image clamped at 0 (blurred for colour NR's luminance-weighted chromaticity).
@compute @workgroup_size(256)
fn clamp0_k(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    put_rgb(i, max(rgb_a(i), vec3<f32>(0.0)));
}

// The chromaticity (rgb / Y of max(rgb, 0)) of pixel (x, y), edges clamped (`local::chroma_plane`).
fn chroma_at(x: i32, y: i32, w: u32, h: u32) -> vec3<f32> {
    let xx = u32(min(max(x, 0), i32(w) - 1));
    let yy = u32(min(max(y, 0), i32(h) - 1));
    let c = max(rgb_a(yy * w + xx), vec3<f32>(0.0));
    return c / max(lum2020(c), 1e-6);
}

// One comparator of the median network: `a` takes the smaller, `b` the larger, per channel.
fn cmp_swap(a: ptr<function, vec3<f32>>, b: ptr<function, vec3<f32>>) {
    let lo = min(*a, *b);
    *b = max(*a, *b);
    *a = lo;
}

// Per-pixel chromaticity's 3 × 3 median, per channel: the plane colour NR's speckle median reads
// (`local::median_chroma3x3`). P[1] = width. The median is a selection, so the 19-comparator
// median-of-9 network on plain variables gives the CPU sort's fifth value. (An insertion sort over a
// private array, as this was, came out wrong on Vulkan: issue #934 moved Windows' compute onto it.)
@compute @workgroup_size(256)
fn chroma_med(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let w = pu(1u);
    let h = pu(0u) / w;
    let x = i32(i % w);
    let y = i32(i / w);
    var p0 = chroma_at(x - 1, y - 1, w, h);
    var p1 = chroma_at(x, y - 1, w, h);
    var p2 = chroma_at(x + 1, y - 1, w, h);
    var p3 = chroma_at(x - 1, y, w, h);
    var p4 = chroma_at(x, y, w, h);
    var p5 = chroma_at(x + 1, y, w, h);
    var p6 = chroma_at(x - 1, y + 1, w, h);
    var p7 = chroma_at(x, y + 1, w, h);
    var p8 = chroma_at(x + 1, y + 1, w, h);
    cmp_swap(&p1, &p2);
    cmp_swap(&p4, &p5);
    cmp_swap(&p7, &p8);
    cmp_swap(&p0, &p1);
    cmp_swap(&p3, &p4);
    cmp_swap(&p6, &p7);
    cmp_swap(&p1, &p2);
    cmp_swap(&p4, &p5);
    cmp_swap(&p7, &p8);
    cmp_swap(&p0, &p3);
    cmp_swap(&p5, &p8);
    cmp_swap(&p4, &p7);
    cmp_swap(&p3, &p6);
    cmp_swap(&p1, &p4);
    cmp_swap(&p2, &p5);
    cmp_swap(&p4, &p7);
    cmp_swap(&p4, &p2);
    cmp_swap(&p6, &p4);
    cmp_swap(&p4, &p2);
    put_rgb(i, p4);
}

// Colour NR: blend chromaticity towards the luminance-weighted blurred one and re-apply luminance
// (a: image, b: 3 × 3 median chromaticity, c: blurred max(rgb, 0)). P[1] = t, P[2] / P[3] = the
// speckle deviation window (`local::SPECKLE_LO` / `SPECKLE_HI`): a channel that disagrees with all
// nine of its 3 × 3 median samples at once is speckle and is pulled all the way in.
@compute @workgroup_size(256)
fn nr_col(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let p = rgb_a(i);
    let yl = lum2020(p);
    let t = pf(1u);
    let lo = pf(2u);
    let hi = pf(3u);
    let c0 = p / max(yl, 1e-6);
    let bl = vec3<f32>(c[3u * i], c[3u * i + 1u], c[3u * i + 2u]);
    let cb = bl / max(lum2020(bl), 1e-6);
    let md = vec3<f32>(b[3u * i], b[3u * i + 1u], b[3u * i + 2u]);
    var out = vec3<f32>(0.0);
    for (var k = 0u; k < 3u; k++) {
        var s = clamp((abs(c0[k] - md[k]) - lo) / max(hi - lo, 1e-6), 0.0, 1.0);
        s = s * s * (3.0 - 2.0 * s);
        let tk = t + (1.0 - t) * s;
        out[k] = max((c0[k] + (cb[k] - c0[k]) * tk) * yl, 0.0);
    }
    put_rgb(i, out);
}

// Every P[1]-th value of `a` (airlight sampling). P[0] = output count.
@compute @workgroup_size(256)
fn subsample(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    dst[i] = a[i * pu(1u)];
}

// `redeye::eye_pixel` (a: image). P[1] = width, P[2] = eye count, then EYE_WORDS per eye:
// cx, cy, r, darken, pet, has catchlight, catchlight x, y, r (output pixels).
@compute @workgroup_size(256)
fn redeye_k(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let w = pu(1u);
    let x = f32(i % w) + 0.5;
    let y = f32(i / w) + 0.5;
    var c = rgb_a(i);
    for (var e = 0u; e < pu(2u); e++) {
        let o = 3u + e * EYE_WORDS;
        let ex = x - pf(o);
        let ey = y - pf(o + 1u);
        let d = sqrt(ex * ex + ey * ey) / pf(o + 2u);
        var m = 1.0 - sstep(1.0, 1.35, d);
        if (pf(o + 4u) != 0.0) {
            m = 1.0 - sstep(0.95, 1.2, d);
        }
        let darken = pf(o + 3u);
        if (m > 0.0) {
            if (pf(o + 4u) != 0.0) {
                let k = min(lum2020(c), 0.04) * (1.0 - 0.85 * darken);
                c = c + (vec3<f32>(k) - c) * m;
            } else {
                let red = (c.x - max(c.y, c.z)) / max(c.x, 1e-6);
                let a = m * sstep(0.25, 0.5, red);
                let n = 0.5 * (c.y + c.z) * (1.0 - 0.9 * darken);
                c = c + (vec3<f32>(n) - c) * a;
            }
        }
        if (pf(o + 5u) != 0.0) {
            let lx = x - pf(o + 6u);
            let ly = y - pf(o + 7u);
            let dc = sqrt(lx * lx + ly * ly) / pf(o + 8u);
            let mc = 1.0 - sstep(0.5, 1.0, dc);
            if (mc > 0.0) {
                c = c + (vec3<f32>(0.9) - c) * mc;
            }
        }
    }
    put_rgb(i, c);
}

// Defringe, first step (`optics::defringe`): OkLab of the clamped colour (a: image) → dst
// (lightness, a, b interleaved).
@compute @workgroup_size(256)
fn fringe_lab(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    put_rgb(i, oklab(max(rgb_a(i), vec3<f32>(0.0))));
}

// `optics::hue_weight`: membership of hue h (deg) in [lo, hi] with 8° shoulders.
fn fringe_hue_weight(h: f32, lo: f32, hi: f32) -> f32 {
    let mid = (lo + hi) / 2.0;
    let half = (hi - lo) / 2.0;
    var d = abs(h - mid) % 360.0;
    if (d > 180.0) {
        d = 360.0 - d;
    }
    return 1.0 - sstep(half, half + 8.0, d);
}

// Defringe, last step: desaturate fringe colours at edges (a: OkLab, b / c: lightness min / max
// around each pixel; dst: the image, written only where it changes, so it starts as a copy).
// P[1..7] = purple lo, hi, green lo, hi, purple strength, green strength.
@compute @workgroup_size(256)
fn fringe_apply(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = lin_index(g, nw);
    if (i >= pu(0u)) {
        return;
    }
    let edge = sstep(0.08, 0.25, c[i] - b[i]);
    if (edge <= 0.0) {
        return;
    }
    let lab = rgb_a(i);
    let ch = sqrt(lab.y * lab.y + lab.z * lab.z);
    let chroma_w = sstep(0.015, 0.05, ch);
    if (chroma_w <= 0.0) {
        return;
    }
    let hue = rem_euclid(degrees(atan2(lab.z, lab.y)), 360.0);
    let k = max(pf(5u) * fringe_hue_weight(hue, pf(1u), pf(2u)), pf(6u) * fringe_hue_weight(hue, pf(3u), pf(4u))) * edge * chroma_w;
    if (k <= 0.0) {
        return;
    }
    let f = 1.0 - min(k, 1.0);
    put_rgb(i, max(oklab_inv(vec3<f32>(lab.x, lab.y * f, lab.z * f)), vec3<f32>(0.0)));
}
