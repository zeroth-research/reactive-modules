use std::cmp::{Ordering, max, min};
use std::fmt;

/// A wiring represents a view over a sequence of indices, each of which is associated
/// with a type.

#[derive(Clone)]
pub(crate) struct Range<D> {
    pub(crate) start: usize,
    pub(crate) end: usize, //inclusive
    pub(crate) dtype: D,
}

#[derive(Clone, Debug)]
pub struct Wire<D> {
    pub(crate) ranges: Vec<Range<D>>,
}

impl<D> Wire<D> {
    pub fn empty() -> Wire<D> {
        Self { ranges: vec![] }
    }

    pub fn scalar(offset: usize, dtype: D) -> Wire<D> {
        Self {
            ranges: vec![Range {
                start: offset,
                end: offset,
                dtype,
            }],
        }
    }

    pub fn vector(offset: usize, dtype: D, length: usize) -> Wire<D> {
        Self {
            ranges: vec![Range {
                start: offset,
                end: offset + length - 1,
                dtype,
            }],
        }
    }

    pub fn size(&self) -> usize {
        self.ranges.iter().map(|r| r.end - r.start + 1).sum()
    }

    // TODO: add a check for contiguity
    fn disjoint_and_sorted(ranges: &[Range<D>]) -> bool {
        let mut i = ranges.iter();
        let Some(mut prev) = i.next() else {
            return true;
        };
        for r in i {
            if r.start <= prev.end {
                return false;
            }
            prev = r;
        }
        true
    }

    fn new_unchecked(ranges: Vec<Range<D>>) -> Wire<D> {
        debug_assert!(Self::disjoint_and_sorted(&ranges));
        Self { ranges }
    }
}

impl<D> Range<D> {
    fn cmp(&self, point: usize) -> Ordering {
        if self.end < point {
            Ordering::Less
        } else if point < self.start {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }

    fn new_unchecked(start: usize, end: usize, dtype: D) -> Self {
        debug_assert!(start <= end);
        Self { start, end, dtype }
    }
}

impl<D: fmt::Debug> fmt::Debug for Range<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{:?}, {:?}] : {:?}", self.start, self.end, self.dtype)
    }
}

impl<D: Eq + Clone> Wire<D> {
    pub fn union(&self, other: &Self) -> Result<Self, &str> {
        let mut ranges: Vec<Range<D>> = Vec::new();
        let mut i = self.ranges.iter().peekable();
        let mut j = other.ranges.iter().peekable();

        loop {
            match (i.peek(), j.peek()) {
                (Some(&a), Some(&b)) => {
                    if a.end + 1 < b.start {
                        ranges.push(a.clone());
                        i.next();
                    } else if b.end + 1 < a.start {
                        ranges.push(b.clone());
                        j.next();
                    } else if (b.end + 1 >= a.start || a.end + 1 >= b.start) && a.dtype == b.dtype {
                        // TODO: what if the a range goes beyond the b range (and viceversa)?
                        ranges.push(Range {
                            start: min(a.start, b.start),
                            end: max(a.end, b.end),
                            dtype: a.dtype.clone(),
                        });
                        i.next();
                        j.next();
                    } else if b.end + 1 == a.start && a.dtype != b.dtype {
                        ranges.push(b.clone());
                        ranges.push(a.clone());
                        i.next();
                        j.next();
                    } else if a.end + 1 == b.start && a.dtype != b.dtype {
                        ranges.push(a.clone());
                        ranges.push(b.clone());
                        i.next();
                        j.next();
                    } else {
                        return Err("dtype mismatch");
                    }
                }
                (Some(&a), None) => {
                    ranges.push(a.clone());
                    i.next();
                }
                (None, Some(&b)) => {
                    ranges.push(b.clone());
                    j.next();
                }
                (None, None) => break,
            }
        }

        Ok(Self::new_unchecked(ranges))
    }

    pub fn intersection(&self, other: &Self) -> Result<Self, &str> {
        let mut ranges: Vec<Range<D>> = Vec::new();
        let mut left: usize = 0;
        for a in self.ranges.iter() {
            // find matching position and make that the new left, working under the assumption
            // that "other" ranges are ordered, and no future start in "self" can be further left
            match other.ranges[left..].binary_search_by(|r| r.cmp(a.start)) {
                Ok(i) => left = i,
                Err(i) => left = i,
            }

            // intersect from the left rightwards until find an empty intersection
            for b in other.ranges[left..].iter() {
                let start = max(a.start, b.start);
                let end = min(a.end, b.end);
                if start <= end {
                    // nonempty intersection
                    if a.dtype != b.dtype {
                        return Err("dtype mismatch");
                    }
                    ranges.push(Range::new_unchecked(start, end, a.dtype.clone()));
                } else {
                    break; // empty intersection
                }
            }
        }
        Ok(Self::new_unchecked(ranges))
    }

    pub fn difference(&self, other: &Self) -> Result<Self, &str> {
        let mut ranges: Vec<Range<D>> = Vec::new();
        let mut left: usize = 0;
        for a in self.ranges.iter() {
            match other.ranges[left..].binary_search_by(|r| r.cmp(a.start)) {
                Ok(i) => left = i,
                Err(i) => left = i,
            }

            let mut start = a.start;
            for b in other.ranges[left..].iter() {
                if start < b.start {
                    if b.dtype != a.dtype {
                        return Err("dtype mismatch");
                    }
                    let end = min(a.end, b.start - 1);
                    ranges.push(Range::new_unchecked(start, end, a.dtype.clone()));
                }
                start = max(start, b.end + 1);
                if start > a.end {
                    break;
                }
            }

            if start <= a.end {
                ranges.push(Range::new_unchecked(start, a.end, a.dtype.clone()));
            }
        }
        Ok(Self::new_unchecked(ranges))
    }
}

impl<D: Eq> Wire<D> {
    pub fn is_subset(&self, other: &Wire<D>) -> bool {
        let mut left: usize = 0;
        for a in self.ranges.iter() {
            match other.ranges[left..].binary_search_by(|r| r.cmp(a.start)) {
                Ok(i) => left = i,
                Err(_) => return false,
            }
            if a.end > other.ranges[left].end || other.ranges[left].dtype != a.dtype {
                return false;
            }
        }
        true
    }

    pub fn is_disjoint(&self, other: &Wire<D>) -> bool {
        for a in self.ranges.iter() {
            match other.ranges.binary_search_by(|r| r.cmp(a.start)) {
                Ok(_) => return false,
                Err(i) => {
                    if i < other.ranges.len() && other.ranges[i].start <= a.end {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn is_twin(&self, other: &Wire<D>) -> bool {
        if self.ranges.len() != other.ranges.len() {
            return false;
        }
        // generalising this later - for now, this works under the following assumption:
        debug_assert!(self.ranges[0].start == 0);
        let offset = match other.ranges.first() {
            Some(range) => range.start,
            None => return true,
        };
        for (a, b) in self.ranges.iter().zip(other.ranges.iter()) {
            if a.start + offset != b.start {
                return false;
            }
            if a.end + offset != b.end {
                return false;
            }
            if a.dtype != b.dtype {
                return false;
            }
        }
        true
    }
}

impl<D: Clone> Wire<D> {
    pub fn twin(&self, offset: isize) -> Result<Wire<D>, &str> {
        let mut twin_ranges: Vec<Range<D>> = Vec::with_capacity(self.ranges.len());
        for range in &self.ranges {
            let start: usize = range.start.checked_add_signed(offset).ok_or("bad offset")?;
            let end: usize = range.end.checked_add_signed(offset).unwrap(); // error cannot happen unless bug

            twin_ranges.push(Range {
                start,
                end,
                dtype: range.dtype.clone(),
            });
        }
        Ok(Self::new_unchecked(twin_ranges))
    }
}
