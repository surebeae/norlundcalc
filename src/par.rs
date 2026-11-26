// rayon when the `parallel` feature is on; sequential otherwise (wasm32-wasip1 has no threads).
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
pub fn map<I, T, F>(items: I, f: F) -> Vec<T>
where
    I: IntoParallelIterator,
    F: Fn(I::Item) -> T + Sync + Send,
    T: Send,
{
    items.into_par_iter().map(f).collect()
}

#[cfg(feature = "parallel")]
pub fn any<I, F>(items: I, f: F) -> bool
where
    I: IntoParallelIterator,
    F: Fn(I::Item) -> bool + Sync + Send,
{
    items.into_par_iter().any(f)
}

#[cfg(not(feature = "parallel"))]
pub fn map<I: IntoIterator, T, F: Fn(I::Item) -> T>(items: I, f: F) -> Vec<T> {
    items.into_iter().map(f).collect()
}

#[cfg(not(feature = "parallel"))]
pub fn any<I: IntoIterator, F: Fn(I::Item) -> bool>(items: I, f: F) -> bool {
    items.into_iter().any(f)
}
