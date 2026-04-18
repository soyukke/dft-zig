/// Return the next FFT-friendly size whose factors are only 2, 3, and 5.
pub fn nextFftSize(value: usize) usize {
    if (value <= 1) return 1;
    var n = value;
    while (!isFftSize(n)) : (n += 1) {}
    return n;
}

fn isFftSize(value: usize) bool {
    var n = value;
    while (n % 2 == 0) n /= 2;
    while (n % 3 == 0) n /= 3;
    while (n % 5 == 0) n /= 5;
    return n == 1;
}
