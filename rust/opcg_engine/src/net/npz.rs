//! npz（zip の stored/deflate ＋ npy v1/v2 ヘッダ）の最小読み取り（P4・WP `rs-p4-net`）。
//!
//! Python 側の正本は `numpy.savez_compressed`（`n_rel.NRelNet.save`）。読めればよいのは
//! **numpy が書いた npz** だけなので、汎用 zip 実装は作らない。対応範囲:
//!
//! - zip: 中央ディレクトリ（EOCD → central directory）から辿る。**local header は zip64 で
//!   サイズが 0xFFFFFFFF になっている**（numpy は `force_zip64=True` で書く）ので、
//!   サイズ・オフセットは中央ディレクトリ側の値（必要なら zip64 拡張フィールド）を使い、
//!   local header は「名前長＋拡張長を読んでデータ位置へ進む」ためだけに読む。
//! - 圧縮: stored(0) と deflate(8)。deflate の展開だけ `miniz_oxide`（§12.4 が許した唯一の依存）。
//! - npy: v1.0/v2.0（magic `\x93NUMPY`・ヘッダ長 u16/u32）・`fortran_order: False` のみ。
//!   dtype は `<f4`／`<f8`／`<i8`／`<i4`／`<U<n>`（UCS-4 LE・NUL 詰め）。
//!   object 配列（`allow_pickle`）は読まない＝`meta` は Python 側が JSON 文字列（`<U…`）で書く。

use crate::state::EngineError;
use std::collections::BTreeMap;

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadPayload(msg.into())
}

fn u16le(b: &[u8], at: usize) -> Result<u16, EngineError> {
    let s = b.get(at..at + 2).ok_or_else(|| bad("npz: 途中で終端（u16）"))?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn u32le(b: &[u8], at: usize) -> Result<u32, EngineError> {
    let s = b.get(at..at + 4).ok_or_else(|| bad("npz: 途中で終端（u32）"))?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn u64le(b: &[u8], at: usize) -> Result<u64, EngineError> {
    let s = b.get(at..at + 8).ok_or_else(|| bad("npz: 途中で終端（u64）"))?;
    let mut a = [0u8; 8];
    a.copy_from_slice(s);
    Ok(u64::from_le_bytes(a))
}

/// npy 1 配列の中身（dtype ごと）。
#[derive(Debug, Clone)]
pub enum NpyData {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    Str(Vec<String>),
}

/// npy 1 配列（形＋中身・C 順）。
#[derive(Debug, Clone)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: NpyData,
}

impl NpyArray {
    /// 要素数（形の積・0 次元は 1）。
    pub fn len(&self) -> usize {
        self.shape.iter().product::<usize>().max(if self.shape.is_empty() { 1 } else { 0 })
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// float32 として取り出す（f8/i8/i4 は変換する＝Python 側の dtype 揺れに強くする）。
    pub fn to_f32(&self, name: &str) -> Result<Vec<f32>, EngineError> {
        match &self.data {
            NpyData::F32(v) => Ok(v.clone()),
            NpyData::F64(v) => Ok(v.iter().map(|x| *x as f32).collect()),
            NpyData::I64(v) => Ok(v.iter().map(|x| *x as f32).collect()),
            NpyData::I32(v) => Ok(v.iter().map(|x| *x as f32).collect()),
            NpyData::Str(_) => Err(bad(format!("npz: {name} は数値配列ではない"))),
        }
    }

    /// 文字列として取り出す（`<U…`）。
    pub fn to_strings(&self, name: &str) -> Result<Vec<String>, EngineError> {
        match &self.data {
            NpyData::Str(v) => Ok(v.clone()),
            _ => Err(bad(format!("npz: {name} は文字列配列ではない"))),
        }
    }

    /// 2 次元（rows, cols）として検査する。
    pub fn dims2(&self, name: &str) -> Result<(usize, usize), EngineError> {
        match self.shape.as_slice() {
            [r, c] => Ok((*r, *c)),
            other => Err(bad(format!("npz: {name} は 2 次元でない（shape={other:?}）"))),
        }
    }
}

/// npy バイト列 → 配列。
pub fn parse_npy(name: &str, b: &[u8]) -> Result<NpyArray, EngineError> {
    if b.len() < 10 || &b[0..6] != b"\x93NUMPY" {
        return Err(bad(format!("npy: {name} のマジックが違う")));
    }
    let major = b[6];
    let (hlen, body) = match major {
        1 => (u16le(b, 8)? as usize, 10usize),
        2 | 3 => (u32le(b, 8)? as usize, 12usize),
        v => return Err(bad(format!("npy: {name} の版 {v} は読まない"))),
    };
    let head = b
        .get(body..body + hlen)
        .ok_or_else(|| bad(format!("npy: {name} のヘッダが切れている")))?;
    let head = std::str::from_utf8(head).map_err(|_| bad(format!("npy: {name} のヘッダが UTF-8 でない")))?;
    let data = &b[body + hlen..];

    if !head.contains("'fortran_order': False") {
        return Err(bad(format!("npy: {name} は fortran_order=True（読まない）")));
    }
    let descr = head
        .split("'descr':")
        .nth(1)
        .and_then(|s| s.split('\'').nth(1))
        .ok_or_else(|| bad(format!("npy: {name} の descr が読めない")))?
        .to_string();
    let shape_s = head
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| bad(format!("npy: {name} の shape が読めない")))?;
    let mut shape = Vec::new();
    for part in shape_s.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        shape.push(
            t.parse::<usize>()
                .map_err(|_| bad(format!("npy: {name} の shape に数でない要素 {t:?}")))?,
        );
    }
    let n: usize = shape.iter().product();

    let need = |unit: usize| -> Result<(), EngineError> {
        if data.len() < n * unit {
            return Err(bad(format!("npy: {name} のデータが短い（{} < {}）", data.len(), n * unit)));
        }
        Ok(())
    };
    let out = match descr.as_str() {
        "<f4" | "=f4" | "f4" => {
            need(4)?;
            NpyData::F32(
                data.chunks_exact(4)
                    .take(n)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
        }
        "<f8" | "=f8" | "f8" => {
            need(8)?;
            NpyData::F64(
                data.chunks_exact(8)
                    .take(n)
                    .map(|c| {
                        let mut a = [0u8; 8];
                        a.copy_from_slice(c);
                        f64::from_le_bytes(a)
                    })
                    .collect(),
            )
        }
        "<i8" | "=i8" | "i8" => {
            need(8)?;
            NpyData::I64(
                data.chunks_exact(8)
                    .take(n)
                    .map(|c| {
                        let mut a = [0u8; 8];
                        a.copy_from_slice(c);
                        i64::from_le_bytes(a)
                    })
                    .collect(),
            )
        }
        "<i4" | "=i4" | "i4" => {
            need(4)?;
            NpyData::I32(
                data.chunks_exact(4)
                    .take(n)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
        }
        d if d.starts_with("<U") || d.starts_with("=U") || d.starts_with('U') => {
            let width: usize = d
                .trim_start_matches(['<', '=', 'U'])
                .parse()
                .map_err(|_| bad(format!("npy: {name} の dtype {d} の幅が読めない")))?;
            need(width * 4)?;
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let s = &data[i * width * 4..(i + 1) * width * 4];
                let mut t = String::new();
                for c in s.chunks_exact(4) {
                    let cp = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    if cp == 0 {
                        break; // NUL 詰め（numpy の固定長文字列）
                    }
                    t.push(char::from_u32(cp).ok_or_else(|| bad(format!("npy: {name} に不正な符号位置")))?);
                }
                v.push(t);
            }
            NpyData::Str(v)
        }
        d => return Err(bad(format!("npy: {name} の dtype {d} は読まない"))),
    };
    Ok(NpyArray { shape, data: out })
}

/// zip の 1 エントリ（中央ディレクトリから読んだ位置と大きさ）。
struct Entry {
    name: String,
    method: u16,
    comp_size: u64,
    local_offset: u64,
}

/// zip の中央ディレクトリを辿ってエントリ一覧を作る。
fn central_directory(b: &[u8]) -> Result<Vec<Entry>, EngineError> {
    // EOCD（0x06054b50）を末尾から探す（コメント長は最大 65535）。
    let eocd = (0..b.len().saturating_sub(21))
        .rev()
        .take(65_557)
        .find(|&i| b[i..].starts_with(&[0x50, 0x4b, 0x05, 0x06]))
        .ok_or_else(|| bad("npz: EOCD が見つからない（zip でない）"))?;
    let mut count = u16le(b, eocd + 10)? as u64;
    let mut cd_off = u32le(b, eocd + 16)? as u64;
    if count == 0xFFFF || cd_off == 0xFFFF_FFFF {
        // zip64 EOCD locator（0x07064b50）→ zip64 EOCD（0x06064b50）
        let loc = eocd
            .checked_sub(20)
            .filter(|&i| b[i..].starts_with(&[0x50, 0x4b, 0x06, 0x07]))
            .ok_or_else(|| bad("npz: zip64 EOCD locator が無い"))?;
        let z64 = u64le(b, loc + 8)? as usize;
        if !b[z64..].starts_with(&[0x50, 0x4b, 0x06, 0x06]) {
            return Err(bad("npz: zip64 EOCD の署名が違う"));
        }
        count = u64le(b, z64 + 32)?;
        cd_off = u64le(b, z64 + 48)?;
    }
    let mut out = Vec::with_capacity(count as usize);
    let mut p = cd_off as usize;
    for _ in 0..count {
        if !b[p..].starts_with(&[0x50, 0x4b, 0x01, 0x02]) {
            return Err(bad("npz: 中央ディレクトリの署名が違う"));
        }
        let method = u16le(b, p + 10)?;
        let mut comp_size = u32le(b, p + 20)? as u64;
        let uncomp_size = u32le(b, p + 24)? as u64;
        let nlen = u16le(b, p + 28)? as usize;
        let elen = u16le(b, p + 30)? as usize;
        let clen = u16le(b, p + 32)? as usize;
        let mut local_offset = u32le(b, p + 42)? as u64;
        let name = String::from_utf8_lossy(
            b.get(p + 46..p + 46 + nlen).ok_or_else(|| bad("npz: 名前が切れている"))?,
        )
        .into_owned();
        // zip64 拡張フィールド（0x0001）: 0xFFFFFFFF の欄だけがこの順で並ぶ。
        if comp_size == 0xFFFF_FFFF || local_offset == 0xFFFF_FFFF || uncomp_size == 0xFFFF_FFFF {
            let ex = b
                .get(p + 46 + nlen..p + 46 + nlen + elen)
                .ok_or_else(|| bad("npz: 拡張フィールドが切れている"))?;
            let mut q = 0usize;
            while q + 4 <= ex.len() {
                let tag = u16le(ex, q)?;
                let sz = u16le(ex, q + 2)? as usize;
                if tag == 0x0001 {
                    let mut r = q + 4;
                    if uncomp_size == 0xFFFF_FFFF {
                        r += 8;
                    }
                    if comp_size == 0xFFFF_FFFF {
                        comp_size = u64le(ex, r)?;
                        r += 8;
                    }
                    if local_offset == 0xFFFF_FFFF {
                        local_offset = u64le(ex, r)?;
                    }
                    break;
                }
                q += 4 + sz;
            }
        }
        out.push(Entry { name, method, comp_size, local_offset });
        p += 46 + nlen + elen + clen;
    }
    Ok(out)
}

/// エントリの生バイト列を取り出す（stored はそのまま・deflate は展開）。
fn entry_bytes(b: &[u8], e: &Entry) -> Result<Vec<u8>, EngineError> {
    let lo = e.local_offset as usize;
    if !b.get(lo..).is_some_and(|s| s.starts_with(&[0x50, 0x4b, 0x03, 0x04])) {
        return Err(bad(format!("npz: {} の local header が無い", e.name)));
    }
    let nlen = u16le(b, lo + 26)? as usize;
    let elen = u16le(b, lo + 28)? as usize;
    let start = lo + 30 + nlen + elen;
    let raw = b
        .get(start..start + e.comp_size as usize)
        .ok_or_else(|| bad(format!("npz: {} のデータが切れている", e.name)))?;
    match e.method {
        0 => Ok(raw.to_vec()),
        8 => miniz_oxide::inflate::decompress_to_vec(raw)
            .map_err(|err| bad(format!("npz: {} の展開に失敗（{err:?}）", e.name))),
        m => Err(bad(format!("npz: {} の圧縮方式 {m} は読まない", e.name))),
    }
}

/// npz ファイル → 配列名（`.npy` を落とす）→ 配列。
pub fn read_npz(path: &str) -> Result<BTreeMap<String, NpyArray>, EngineError> {
    let b = std::fs::read(path).map_err(|e| bad(format!("npz: {path} が読めない（{e}）")))?;
    read_npz_bytes(&b)
}

/// npz のバイト列 → 配列表（テストが手組みの zip を通すための入口）。
pub fn read_npz_bytes(b: &[u8]) -> Result<BTreeMap<String, NpyArray>, EngineError> {
    let mut out = BTreeMap::new();
    for e in central_directory(b)? {
        let raw = entry_bytes(b, &e)?;
        let key = e.name.strip_suffix(".npy").unwrap_or(&e.name).to_string();
        let arr = parse_npy(&key, &raw)?;
        out.insert(key, arr);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// npy を手組みして往復する（v1 ヘッダ・float32）。
    fn npy_f32(shape: &[usize], v: &[f32]) -> Vec<u8> {
        let shape_s = if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            format!("({})", shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
        };
        let mut head = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_s}, }}");
        while (10 + head.len() + 1) % 64 != 0 {
            head.push(' ');
        }
        head.push('\n');
        let mut b = b"\x93NUMPY\x01\x00".to_vec();
        b.extend_from_slice(&(head.len() as u16).to_le_bytes());
        b.extend_from_slice(head.as_bytes());
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }

    /// stored のみの zip を手組みする（中央ディレクトリ付き）。
    fn zip_stored(files: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        for (name, data) in files {
            let off = out.len() as u32;
            out.extend_from_slice(&[0x50, 0x4b, 0x03, 0x04]);
            out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // ver/flag/method/time/date
            out.extend_from_slice(&0u32.to_le_bytes()); // crc（読まない）
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(data);

            central.extend_from_slice(&[0x50, 0x4b, 0x01, 0x02]);
            central.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            central.extend_from_slice(&0u32.to_le_bytes());
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(name.len() as u16).to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes()); // extra
            central.extend_from_slice(&0u16.to_le_bytes()); // comment
            central.extend_from_slice(&0u16.to_le_bytes()); // disk
            central.extend_from_slice(&0u16.to_le_bytes()); // internal attr
            central.extend_from_slice(&0u32.to_le_bytes()); // external attr
            central.extend_from_slice(&off.to_le_bytes());
            central.extend_from_slice(name.as_bytes());
        }
        let cd_off = out.len() as u32;
        let cd_len = central.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(&[0x50, 0x4b, 0x05, 0x06]);
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(files.len() as u16).to_le_bytes());
        out.extend_from_slice(&(files.len() as u16).to_le_bytes());
        out.extend_from_slice(&cd_len.to_le_bytes());
        out.extend_from_slice(&cd_off.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    #[test]
    fn npy_roundtrip_f32() {
        let v = vec![1.0f32, -2.5, 3.25, 4.0, 0.0, -1.0];
        let b = npy_f32(&[2, 3], &v);
        let a = parse_npy("x", &b).unwrap();
        assert_eq!(a.shape, vec![2, 3]);
        assert_eq!(a.dims2("x").unwrap(), (2, 3));
        assert_eq!(a.to_f32("x").unwrap(), v);
    }

    #[test]
    fn zip_stored_roundtrip() {
        let z = zip_stored(&[
            ("a.npy", npy_f32(&[2], &[1.5, 2.5])),
            ("b.npy", npy_f32(&[1], &[7.0])),
        ]);
        let m = read_npz_bytes(&z).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m["a"].to_f32("a").unwrap(), vec![1.5, 2.5]);
        assert_eq!(m["b"].shape, vec![1]);
    }

    #[test]
    fn npy_rejects_unknown_dtype() {
        let mut head = "{'descr': '<c8', 'fortran_order': False, 'shape': (1,), }".to_string();
        while (10 + head.len() + 1) % 64 != 0 {
            head.push(' ');
        }
        head.push('\n');
        let mut b = b"\x93NUMPY\x01\x00".to_vec();
        b.extend_from_slice(&(head.len() as u16).to_le_bytes());
        b.extend_from_slice(head.as_bytes());
        b.extend_from_slice(&[0u8; 8]);
        assert!(parse_npy("x", &b).is_err());
    }

    /// 同梱の nrel_a1.npz（deflate＋zip64 local header＋`<U` 文字列）を実際に読む。
    #[test]
    fn reads_bundled_nrel_a1() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../opcg_sim/data/learned/nrel_a1.npz");
        let m = read_npz(path).unwrap();
        for k in ["Wa", "Wt", "Wr", "Wc", "W1", "W2", "Wv", "Wp1", "Wp2", "meta", "nrel", "vocab_ids"] {
            assert!(m.contains_key(k), "{k} が無い");
        }
        assert_eq!(m["Wa"].dims2("Wa").unwrap(), (167, 24));
        assert_eq!(m["W1"].dims2("W1").unwrap(), (411, 192));
        assert_eq!(m["Wp1"].dims2("Wp1").unwrap(), (499, 64));
        let ids = m["vocab_ids"].to_strings("vocab_ids").unwrap();
        assert_eq!(ids.len(), 2652);
        assert_eq!(ids[0], "EB01-001");
        let meta = m["meta"].to_strings("meta").unwrap();
        assert_eq!(meta.len(), 1);
        assert!(meta[0].contains("\"ablate\""));
    }
}
