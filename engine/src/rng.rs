//! Small, fast, seedable RNG (xoshiro256++ seeded through splitmix64).
//! Kept dependency-free so a seed reproduces the exact same deal on every machine.

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        Rng { s: [splitmix(&mut x), splitmix(&mut x), splitmix(&mut x), splitmix(&mut x)] }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = (s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform integer in 0..n (n > 0).
    pub fn below(&mut self, n: u64) -> u64 {
        // Lemire's multiply-shift; bias is negligible for the small n used here.
        ((self.next_u64() as u128 * n as u128) >> 64) as u64
    }

    /// Uniform float in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            v.swap(i, j);
        }
    }

    /// Derive an independent stream, e.g. one per hand in a batch.
    pub fn derive(seed: u64, stream: u64) -> Self {
        let mut x = seed ^ stream.wrapping_mul(0xD1B54A32D192ED03);
        let _ = splitmix(&mut x);
        Rng::new(splitmix(&mut x))
    }
}
