use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

pub trait Variable: Eq + Hash + Clone + Debug {
    fn next(&self) -> Option<Self>;
    fn is_latched(&self) -> bool;
}

pub(crate) trait VarIterExt {
    type Item;
    fn has_duplicates(&self) -> bool;
    fn is_latched(&self) -> bool;
    fn as_set(&self) -> HashSet<&Self::Item>;
    fn into_set(&self) -> HashSet<Self::Item>;
    fn next_set(&self) -> HashSet<Self::Item>;
    fn pairwise_disjoint<V>(sets: &[V]) -> bool
    where
        V: VarIterExt,
        V::Item: Eq + Hash,
    {
        let mut known: HashSet<&V::Item> = HashSet::new();
        for s in sets {
            for i in s.as_set() {
                if !known.contains(&i) {
                    return false;
                }
                known.insert(i);
            }
        }
        true
    }
}

impl<I, V> VarIterExt for I
where
    for<'a> &'a I: IntoIterator<Item = &'a V>,
    for<'a> <&'a I as IntoIterator>::IntoIter: ExactSizeIterator,
    V: Variable,
{
    type Item = V;

    fn has_duplicates(&self) -> bool {
        self.as_set().len() != self.into_iter().len()
    }

    fn is_latched(&self) -> bool {
        self.into_iter().all(|v| v.is_latched())
    }

    fn as_set(&self) -> HashSet<&V> {
        HashSet::<&V>::from_iter(self.into_iter())
    }

    fn into_set(&self) -> HashSet<V> {
        HashSet::<V>::from_iter(self.into_iter().cloned())
    }

    fn next_set(&self) -> HashSet<V> {
        let mut set = HashSet::<V>::new();
        for v in self.into_iter() {
            set.insert(v.next().expect("Variable is latched"));
        }
        set
    }
}
