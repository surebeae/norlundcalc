use gauss_quad::legendre::GaussLegendre;
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;

type NodesWeights = (Vec<f64>, Vec<f64>);

thread_local! {
    static GL_CACHE: RefCell<HashMap<usize, NodesWeights>> = RefCell::new(HashMap::new());
}

fn get_gl(order: usize) -> NodesWeights {
    GL_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();

        cache
            .entry(order)
            .or_insert_with(|| {
                let gl = GaussLegendre::new(order.try_into().unwrap());
                let nodes: Vec<f64> = gl.nodes().map(|x| 0.5 * x + 0.5).collect();
                let weights: Vec<f64> = gl.weights().map(|w| 0.5 * w).collect();

                (nodes, weights)
            })
            .clone()
    })
}

pub(crate) fn composite_gl<F: Fn(f64) -> f64 + Sync>(
    a: f64,
    b: f64,
    n_sub: usize,
    order: usize,
    f: &F,
) -> f64 {
    let (nodes, weights) = get_gl(order);
    let h = (b - a) / n_sub as f64;

    (0..n_sub)
        .into_par_iter()
        .map(|i| {
            let left = a + i as f64 * h;
            let right = left + h;
            let mut sum = 0.0;

            for (node, weight) in nodes.iter().zip(weights.iter()) {
                let x = left + (right - left) * node;
                sum += weight * f(x);
            }

            sum * (right - left)
        })
        .sum()
}
