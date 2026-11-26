use std::fmt::Write as _;

const BUCKETS: usize = 1 << 15;
const MAX_COLORS: usize = 256;

#[inline]
fn bucket(r: u8, g: u8, b: u8) -> usize {
    ((r as usize >> 3) << 10) | ((g as usize >> 3) << 5) | (b as usize >> 3)
}

// Encode an RGBA image (alpha composited over black) as a sixel string.
pub(crate) fn encode(rgba: &[u8], width: usize, height: usize) -> String {
    assert_eq!(rgba.len(), width * height * 4, "buffer size does not match dimensions");
    if width == 0 || height == 0 {
        return String::new();
    }

    // Composite, bucket and histogram every pixel.
    let mut idx = vec![0u16; width * height];
    let mut count = vec![0u32; BUCKETS];
    let mut sum = vec![[0u64; 3]; BUCKETS];
    for (i, px) in rgba.as_chunks::<4>().0.iter().enumerate() {
        let a = px[3] as u32;
        let (r, g, b) =
            ((px[0] as u32 * a / 255) as u8, (px[1] as u32 * a / 255) as u8, (px[2] as u32 * a / 255) as u8);
        let k = bucket(r, g, b);
        idx[i] = k as u16;
        count[k] += 1;
        let s = &mut sum[k];
        s[0] += r as u64;
        s[1] += g as u64;
        s[2] += b as u64;
    }

    // Palette: the most frequent buckets, at their mean colour.
    let mut used: Vec<usize> = (0..BUCKETS).filter(|&k| count[k] > 0).collect();
    used.sort_unstable_by(|&a, &b| count[b].cmp(&count[a]));
    let palette: Vec<[u8; 3]> = used
        .iter()
        .take(MAX_COLORS)
        .map(|&k| {
            let n = count[k] as u64;
            [(sum[k][0] / n) as u8, (sum[k][1] / n) as u8, (sum[k][2] / n) as u8]
        })
        .collect();

    // Bucket -> nearest palette entry (only for buckets that occur).
    let mut map = vec![0u8; BUCKETS];
    for (rank, &k) in used.iter().enumerate() {
        map[k] = if rank < palette.len() {
            rank as u8
        } else {
            let n = count[k] as u64;
            let c = [(sum[k][0] / n) as i32, (sum[k][1] / n) as i32, (sum[k][2] / n) as i32];
            let mut best = (i32::MAX, 0u8);
            for (j, p) in palette.iter().enumerate() {
                let d: i32 = (0..3).map(|ch| (c[ch] - p[ch] as i32).pow(2)).sum();
                if d < best.0 {
                    best = (d, j as u8);
                }
            }
            best.1
        };
    }
    for v in idx.iter_mut() {
        *v = map[*v as usize] as u16;
    }

    // Header: P2 = 1 (zero bits leave pixels untouched), size, palette.
    let mut out = String::with_capacity(width * height / 2);
    let _ = write!(out, "\x1bP0;1;0q\"1;1;{width};{height}");
    for (i, p) in palette.iter().enumerate() {
        let pct = |v: u8| (v as u32 * 100 + 127) / 255;
        let _ = write!(out, "#{i};2;{};{};{}", pct(p[0]), pct(p[1]), pct(p[2]));
    }

    // Six-row bands: one run-length-encoded row per colour present.
    let n_colors = palette.len();
    let mut masks = vec![0u8; n_colors * width];
    let mut present = vec![false; n_colors];
    let mut order = Vec::with_capacity(n_colors);
    for y0 in (0..height).step_by(6) {
        let rows = (height - y0).min(6);
        for dy in 0..rows {
            let row = &idx[(y0 + dy) * width..(y0 + dy + 1) * width];
            for (x, &c) in row.iter().enumerate() {
                let c = c as usize;
                if !present[c] {
                    present[c] = true;
                    order.push(c);
                }
                masks[c * width + x] |= 1 << dy;
            }
        }
        for (n, &c) in order.iter().enumerate() {
            if n > 0 {
                out.push('$');
            }
            let _ = write!(out, "#{c}");
            let row = &mut masks[c * width..(c + 1) * width];
            // drop trailing empty columns
            let end = row.iter().rposition(|&m| m != 0).map_or(0, |p| p + 1);
            let mut x = 0;
            while x < end {
                let m = row[x];
                let mut run = 1;
                while x + run < end && row[x + run] == m {
                    run += 1;
                }
                let ch = (63 + m) as char;
                if run > 3 {
                    let _ = write!(out, "!{run}{ch}");
                } else {
                    for _ in 0..run {
                        out.push(ch);
                    }
                }
                x += run;
            }
            row.fill(0);
            present[c] = false;
        }
        order.clear();
        if y0 + 6 < height {
            out.push('-');
        }
    }
    out.push_str("\x1b\\");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tiny decoder for the subset encode emits, returning palette indices.
    fn decode(s: &str, width: usize, height: usize) -> (Vec<[u8; 3]>, Vec<Option<usize>>) {
        let body = s.strip_prefix("\x1bP0;1;0q").unwrap().strip_suffix("\x1b\\").unwrap();
        let mut img = vec![None; width * height];
        let mut pal = vec![[0u8; 3]; 256];
        let b = body.as_bytes();
        let (mut i, mut x, mut y0, mut color) = (0usize, 0usize, 0usize, 0usize);
        let num = |i: &mut usize| -> usize {
            let st = *i;
            while *i < b.len() && b[*i].is_ascii_digit() {
                *i += 1;
            }
            body[st..*i].parse().unwrap()
        };
        while i < b.len() {
            match b[i] {
                b'"' => {
                    i += 1;
                    for _ in 0..4 {
                        num(&mut i);
                        if i < b.len() && b[i] == b';' {
                            i += 1;
                        }
                    }
                }
                b'#' => {
                    i += 1;
                    let c = num(&mut i);
                    if i < b.len() && b[i] == b';' {
                        i += 3;
                        let r = num(&mut i);
                        i += 1;
                        let g = num(&mut i);
                        i += 1;
                        let bl = num(&mut i);
                        pal[c] = [r as u8, g as u8, bl as u8];
                    } else {
                        color = c;
                    }
                }
                b'$' => {
                    x = 0;
                    i += 1;
                }
                b'-' => {
                    x = 0;
                    y0 += 6;
                    i += 1;
                }
                b'!' => {
                    i += 1;
                    let n = num(&mut i);
                    let m = b[i] - 63;
                    for _ in 0..n {
                        put(&mut img, width, height, x, y0, m, color);
                        x += 1;
                    }
                    i += 1;
                }
                ch @ 63..=126 => {
                    put(&mut img, width, height, x, y0, ch - 63, color);
                    x += 1;
                    i += 1;
                }
                other => panic!("unexpected byte {other}"),
            }
        }
        (pal, img)
    }

    fn put(img: &mut [Option<usize>], w: usize, h: usize, x: usize, y0: usize, m: u8, c: usize) {
        for dy in 0..6 {
            if m & (1 << dy) != 0 && y0 + dy < h {
                assert!(img[(y0 + dy) * w + x].is_none(), "pixel painted twice");
                img[(y0 + dy) * w + x] = Some(c);
            }
        }
    }

    #[test]
    fn round_trips_a_few_colours() {
        let (w, h) = (23, 13); // not multiples of 6 or of run lengths
        let colors = [[68u8, 68, 68], [236, 158, 158], [55, 187, 191], [255, 255, 198]];
        let mut rgba = Vec::new();
        let mut want = Vec::new();
        for y in 0..h {
            for x in 0..w {
                #[rustfmt::skip]
                let c = if y == 5 { 1 } else if x == 7 || x == 8 { 2 } else if (x + y) % 11 == 0 { 3 } else { 0 };
                want.push(c);
                rgba.extend_from_slice(&colors[c]);
                rgba.push(255);
            }
        }
        let s = encode(&rgba, w, h);
        let (pal, img) = decode(&s, w, h);
        for (i, &c) in want.iter().enumerate() {
            let got = img[i].expect("every pixel painted");
            let p = pal[got];
            let expect = colors[c].map(|v| ((v as u32 * 100 + 127) / 255) as u8);
            // 15-bit bucketing: within 8/255 ~= 3 percentage points
            for ch in 0..3 {
                assert!((p[ch] as i32 - expect[ch] as i32).abs() <= 3, "pixel {i}");
            }
        }
    }

    #[test]
    fn more_than_256_colours_map_to_nearest() {
        let (w, h) = (64, 8);
        let mut rgba = Vec::new();
        for y in 0..h {
            for x in 0..w {
                rgba.extend_from_slice(&[(x * 4) as u8, (y * 32) as u8, ((x * y * 7) % 256) as u8, 255]);
            }
        }
        let distinct: std::collections::HashSet<usize> =
            rgba.as_chunks::<4>().0.iter().map(|p| bucket(p[0], p[1], p[2])).collect();
        assert!(distinct.len() > MAX_COLORS);
        let s = encode(&rgba, w, h);
        let (_, img) = decode(&s, w, h);
        assert!(img.iter().all(|p| p.is_some()));
    }
}
