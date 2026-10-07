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
    // **全移植・段 4（2026-10-07）**: Rust の計画のディスクの覚え書き（`core/store.rs`）の鍵に入る**解き方の原文の全部**のハッシュ
    // （`src/theory/**` の `.rs` のうち試験〔`tests*.rs`〕でないもの全部・相対パス順・FNV-1a 64 を 2 本）——`rule_don_solve` の外側が
    // Rust に移ったので、どのファイルを直しても古い計画は返らない（版の上げ忘れは `test_rd_speed` の指紋が見張る）。
    let mut all = Vec::new();
    collect(root, &mut all);
    let mut keyed: Vec<(String, PathBuf)> = all
        .into_iter()
        .map(|p| (p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/"), p))
        .filter(|(rel, _)| !rel.rsplit('/').next().unwrap_or("").starts_with("tests"))
        .collect();
    keyed.sort();
    let (mut h1, mut h2): (u64, u64) = (0xcbf2_9ce4_8422_2325, 0x8422_2325_cbf2_9ce4);
    for (rel, p) in &keyed {
        let body = fs::read(p).unwrap_or_default();
        for &b in rel.as_bytes().iter().chain([0u8].iter()).chain(body.iter()).chain([0u8].iter()) {
            h1 ^= b as u64;
            h1 = h1.wrapping_mul(0x0000_0100_0000_01b3);
            h2 = (h2 ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(5);
        }
    }
    println!("cargo:rustc-env=RD_SOLVER_SRC_HASH={:016x}{:016x}", h1, h2);
    println!("cargo:rerun-if-changed=src/theory");
    println!("cargo:rerun-if-changed=build.rs");
}
