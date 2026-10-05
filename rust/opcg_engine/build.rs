// `src/theory/**` の原文のハッシュを `RD_KERNEL_SRC_HASH` として埋める（FNV-1a 64・相対パス順）。
// 古い wheel の取り違え検出（Python `rd_kernel.py` が同じ式で作る）と、計画の覚え書きの鍵に使う。
use std::fs;
use std::path::{Path, PathBuf};

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                collect(&p, out);
            } else if p.extension().map(|x| x == "rs").unwrap_or(false) {
                out.push(p);
            }
        }
    }
}

fn main() {
    let root = Path::new("src/theory");
    let mut files = Vec::new();
    collect(root, &mut files);
    let mut keyed: Vec<(String, PathBuf)> = files
        .into_iter()
        .map(|p| (p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/"), p))
        .collect();
    keyed.sort();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for (rel, p) in &keyed {
        feed(rel.as_bytes());
        feed(&[0]);
        feed(&fs::read(p).unwrap_or_default());
        feed(&[0]);
    }
    println!("cargo:rustc-env=RD_KERNEL_SRC_HASH={:016x}", h);
    println!("cargo:rerun-if-changed=src/theory");
    println!("cargo:rerun-if-changed=build.rs");
}
