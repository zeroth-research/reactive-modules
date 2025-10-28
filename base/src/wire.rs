use std::cmp::{Ordering, max, min};
use std::fmt;

/// A typed implementation of a range.
/// Both `start` and `end` are inclusive.
#[derive(Clone)]
struct Range<D> {
    start: usize,
    end: usize, // inclusive
    dtype: D,
}

/// A wire represents a view over a sequence of indices, each of which is associated
/// with a type.
///
/// For example,  we can have the following structure where the computation node `N` is wired with 3 wires
/// `x`, `y`, and `z`:
///
/// ```text
///     -- x -- |    |
///             | N  |  -- z --
///     -- y -- |    |
/// ```
///
/// Wires `x` and `y` connect the node with inputs and `z` connects the output of the node with other nodes
/// (or represents the output of the graph if the wire is not connected to any node).
///
/// If `N` is a multiplication of 3x3 matrices and two matrices `A` and `B` are "sent" into `x` and `y`, then `z`
/// "forwards" the value `A*B`. The wires can be represented in this case as simple variables taking and
/// forwarding the values.
///
/// An important feature of wires is also to create sub-views, e.g., access only parts of a matrix.
/// If, in the example above, `N` should multiply 2x2 sub-matrices of two 3x3 matrices, the wire can specify that it "forwards" to `N` only the 2x2 sub-matrix of a given 3x3 matrix. The wire itself can be now represented as a 2x2 matrix of indices that should be accessed.
#[derive(Debug)]
pub struct Wire<D> {
    ranges: Vec<Range<D>>,
}

impl<D> Wire<D> {
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

    pub fn empty() -> Wire<D> {
        Self { ranges: vec![] }
    }

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

    ///
    /// Populate a wire from a sequence of variables identifiers.
    /// The complexity is not the best right now, but it may not be a bottle-neck.
    pub fn from_iter<I>(iter: I, dtype: D) -> Result<Wire<D>, &'static str>
    where
        I: Iterator<Item = usize>,
        D: Clone,
    {
        let mut identifiers: Vec<usize> = Vec::from_iter(iter);
        identifiers.sort();

        let mut ranges: Vec<Range<D>> = Vec::new();
        for i in identifiers {
            if let Some(range) = ranges.last_mut() {
                assert!(i > range.end, "Repeated value"); // or unsorted array, but we sorted it
                // above
                if i == range.end + 1 {
                    range.end += 1;
                } else {
                    ranges.push(Range {
                        start: i,
                        end: i,
                        dtype: dtype.clone(),
                    });
                }
            } else {
                ranges.push(Range {
                    start: i,
                    end: i,
                    dtype: dtype.clone(),
                });
            }
        }

        Ok(Wire { ranges })
    }

    fn new_unchecked(ranges: Vec<Range<D>>) -> Wire<D> {
        debug_assert!(Self::disjoint_and_sorted(&ranges));
        Self { ranges }
    }
}

impl<T> Range<T> {
    fn cmp(&self, point: usize) -> Ordering {
        if self.end < point {
            Ordering::Less
        } else if point < self.start {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }

    fn overlap(&self, oth: &Range<T>) -> bool {
        self.end >= oth.start || oth.end <= self.start
    }
}

impl<D: fmt::Debug> fmt::Debug for Range<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{:?}..={:?}] : {:?}", self.start, self.end, self.dtype)
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
                    if a.end < b.start {
                        ranges.push(a.clone());
                        i.next();
                    } else if b.end < a.start {
                        ranges.push(b.clone());
                        j.next();
                    } else if a.dtype == b.dtype {
                        ranges.push(Range {
                            start: min(a.start, b.start),
                            end: max(a.end, b.end),
                            dtype: a.dtype.clone(),
                        });
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
                    ranges.push(Range {
                        start,
                        end,
                        dtype: a.dtype.clone(),
                    });
                } else {
                    break; // empty intersection
                }
            }
        }

        Ok(Self::new_unchecked(ranges))
    }
}

// #[macro_export]
// macro_rules! wires {
//     () => (
//         $crate::wire:wireg::empty()
//     );
//     ($($x:expr),+ $(,)?) => (
//         $crate::wire:wireg::from_slice(&[$($x),+])
//     );
// }
