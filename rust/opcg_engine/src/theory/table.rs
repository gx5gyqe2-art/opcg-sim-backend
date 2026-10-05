//! 鍵（`u64` の列）→ 値の表。手書きの開番地法（依存を足さない・クレートの方針）。
//!
//! 鍵は**全体を比べる**（ハッシュは探す位置を決めるだけで、同じ鍵かの判定には使わない）。
//! 鍵の本体は 1 本の連続領域（`arena`）に詰める＝挿入で割り当てが増えない。

/// 鍵 1 本のハッシュ（Fx 風の掛け算＋最後に混ぜる）。
#[inline]
pub fn hash_key(key: &[u64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &w in key {
        h = (h.rotate_left(5) ^ w).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    h ^= h >> 29;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^ (h >> 32)
}

struct Ent<V: Copy> {
    hash: u64,
    koff: u32,
    klen: u32,
    val: V,
}

/// 鍵つきの表。`V` は小さな値（コピーできるもの）。
pub struct KeyTable<V: Copy> {
    slots: Vec<u32>, // 0 = 空き・それ以外は `ents` の番号 + 1
    mask: usize,
    ents: Vec<Ent<V>>,
    arena: Vec<u64>,
}

impl<V: Copy> Default for KeyTable<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Copy> KeyTable<V> {
    pub fn new() -> Self {
        KeyTable { slots: vec![0; 64], mask: 63, ents: Vec::new(), arena: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.ents.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.ents.is_empty()
    }

    fn find(&self, h: u64, key: &[u64]) -> Result<usize, usize> {
        let mut i = (h as usize) & self.mask;
        loop {
            let s = self.slots[i];
            if s == 0 {
                return Err(i);
            }
            let e = &self.ents[(s - 1) as usize];
            if e.hash == h && e.klen as usize == key.len() {
                let o = e.koff as usize;
                if &self.arena[o..o + key.len()] == key {
                    return Ok((s - 1) as usize);
                }
            }
            i = (i + 1) & self.mask;
        }
    }

    pub fn get(&self, key: &[u64]) -> Option<V> {
        match self.find(hash_key(key), key) {
            Ok(ix) => Some(self.ents[ix].val),
            Err(_) => None,
        }
    }

    fn grow(&mut self) {
        let n = self.slots.len() * 2;
        self.slots = vec![0; n];
        self.mask = n - 1;
        for (ix, e) in self.ents.iter().enumerate() {
            let mut i = (e.hash as usize) & self.mask;
            while self.slots[i] != 0 {
                i = (i + 1) & self.mask;
            }
            self.slots[i] = (ix + 1) as u32;
        }
    }

    /// 無い鍵を入れる。入れたら `true`、既に有れば何もせず `false`（集合として使うときの判定）。
    pub fn insert_if_absent(&mut self, key: &[u64], val: V) -> bool {
        if (self.ents.len() + 1) * 2 > self.slots.len() {
            self.grow();
        }
        let h = hash_key(key);
        match self.find(h, key) {
            Ok(_) => false,
            Err(slot) => {
                let koff = self.arena.len() as u32;
                self.arena.extend_from_slice(key);
                self.ents.push(Ent { hash: h, koff, klen: key.len() as u32, val });
                self.slots[slot] = self.ents.len() as u32;
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_grow_and_full_key_compare() {
        let mut t: KeyTable<u32> = KeyTable::new();
        for i in 0..5000u64 {
            assert!(t.insert_if_absent(&[i, i * 7, 3], i as u32));
        }
        assert!(!t.insert_if_absent(&[10, 70, 3], 0));
        assert_eq!(t.len(), 5000);
        for i in 0..5000u64 {
            assert_eq!(t.get(&[i, i * 7, 3]), Some(i as u32));
        }
        assert_eq!(t.get(&[10, 70]), None); // 長さが違えば別の鍵
        assert_eq!(t.get(&[10, 71, 3]), None);
    }
}
