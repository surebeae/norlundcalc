use num_complex::Complex;

pub(crate) fn solve_tridiagonal(xs: &[f64], yvals: &[f64]) -> Vec<f64> {
    let n = xs.len();
    if n <= 2 {
        return vec![0.0; n];
    }

    let n_int = n - 2;
    let mut a = vec![0.0f64; n_int]; // subdiagonal
    let mut b = vec![0.0f64; n_int]; // diagonal
    let mut c = vec![0.0f64; n_int]; // superdiagonal
    let mut d = vec![0.0f64; n_int]; // right-hand side

    for i in 1..=n_int {
        let idx = i; // interior knot index
        let h_i = xs[idx] - xs[idx - 1];
        let h_ip1 = xs[idx + 1] - xs[idx];
        let j = i - 1;

        a[j] = if j == 0 { 0.0 } else { h_i };
        b[j] = 2.0 * (h_i + h_ip1);
        c[j] = if j + 1 < n_int { h_ip1 } else { 0.0 };

        d[j] = 6.0 * ((yvals[idx + 1] - yvals[idx]) / h_ip1 - (yvals[idx] - yvals[idx - 1]) / h_i);
    }

    // Thomas algorithm for the tridiagonal system
    for i in 1..n_int {
        let w = a[i] / b[i - 1];
        b[i] -= w * c[i - 1];
        d[i] -= w * d[i - 1];
    }

    let mut m = vec![0.0f64; n];

    // Last interior unknown
    m[n_int] = d[n_int - 1] / b[n_int - 1];

    // Back substitution
    for i in (0..n_int - 1).rev() {
        m[i + 1] = (d[i] - c[i] * m[i + 2]) / b[i];
    }

    m
}

pub(crate) struct RealSpline {
    xs: Vec<f64>,
    ys: Vec<f64>,
    m: Vec<f64>,
}

impl RealSpline {
    pub(crate) fn new(xs: Vec<f64>, ys: Vec<f64>) -> Self {
        let m = solve_tridiagonal(&xs, &ys);
        RealSpline { xs, ys, m }
    }

    pub(crate) fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();

        if x <= self.xs[0] {
            let h = self.xs[1] - self.xs[0];
            let t = (x - self.xs[0]) / h;
            return self.ys[0] + t * (self.ys[1] - self.ys[0]);
        }

        if x >= self.xs[n - 1] {
            let h = self.xs[n - 1] - self.xs[n - 2];
            let t = (x - self.xs[n - 2]) / h;
            return self.ys[n - 2] + t * (self.ys[n - 1] - self.ys[n - 2]);
        }

        let i = self.xs.partition_point(|&v| v < x) - 1;
        let h = self.xs[i + 1] - self.xs[i];
        let t = (x - self.xs[i]) / h;
        let a = 1.0 - t;
        let b = t;

        a * self.ys[i]
            + b * self.ys[i + 1]
            + (h * h / 6.0) * ((a * a * a - a) * self.m[i] + (b * b * b - b) * self.m[i + 1])
    }
}

pub(crate) struct ComplexSpline {
    real: RealSpline,
    imag: RealSpline,
}

impl ComplexSpline {
    pub(crate) fn new(xs: Vec<f64>, ys: Vec<Complex<f64>>) -> Self {
        let real: Vec<f64> = ys.iter().map(|c| c.re).collect();
        let imag: Vec<f64> = ys.iter().map(|c| c.im).collect();

        ComplexSpline {
            real: RealSpline::new(xs.clone(), real),
            imag: RealSpline::new(xs, imag),
        }
    }

    pub(crate) fn eval(&self, x: f64) -> Complex<f64> {
        Complex::new(self.real.eval(x), self.imag.eval(x))
    }
}
