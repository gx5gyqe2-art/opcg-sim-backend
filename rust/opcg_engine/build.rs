// 守る側の計算（`rd_solve`）の原文のハッシュを `RD_KERNEL_SRC_HASH` として埋める（FNV-1a 64・相対パス順）。
// 古い wheel の取り違え検出（Python `rd_kernel.py` が同じ式で作る）と、計画の覚え書きの鍵に使う。
// **全移植・段 3（2026-10-07）から `KERNEL_FILES` の 8 ファイルだけ**を読む——理論の移植のファイル（葉・値付けの核
// `core/`）を足し直すたびに `SOLVER_VERSION` を上げずに済む（守る側の解き方の値が変わり得るのはこの 8 つだけ）。
// 並びは Python の `rd_kernel.KERNEL_FILES` と同じ。
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

const KERNEL_FILES: &[&str] =
    &["defender.rs", "layers.rs", "numeric.rs", "plans.rs", "pyapi.rs", "sched.rs", "succ.rs", "table.rs"];

fn main() {
    let root = Path::new("src/theory");
    let mut files = Vec::new();
    collect(root, &mut files);
    files.retain(|p| p.parent() == Some(root) && KERNEL_FILES.iter().any(|k| p.file_name().map(|f| f == *k).unwrap_or(false)));
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
