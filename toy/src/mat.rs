use std::marker::PhantomData;

pub struct MatVecIter<'a> {
    vecs: &'a str,
}

impl<'a> MatVecIter<'a> {
    pub fn new(s: &'a str) -> Self {
        Self {
            vecs: s
                .trim()
                .trim_start_matches(|c| c != '[')
                .trim_end_matches(|c| c != ']'),
        }
    }
}

impl<'a> Iterator for MatVecIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.vecs.is_empty() {
            return None;
        }

        debug_assert!(self.vecs.starts_with("["));

        if let Some(end) = self.vecs.find(']') {
            let res = &self.vecs[..=end].trim();

            // find next vector
            if let Some(start) = self.vecs[1..].find('[') {
                if start < end {
                    // nested vectors
                    return None;
                }
                self.vecs = &self.vecs[start + 1..];
            } else {
                // we have reached the end
                self.vecs = "";
            }

            return Some(res);
        }

        None
    }
}

pub struct VecIter<'a, N: std::str::FromStr> {
    vec: &'a str,
    _phantom: std::marker::PhantomData<N>,
}

impl<'a, N: std::str::FromStr> VecIter<'a, N> {
    pub fn new(s: &'a str) -> Self {
        Self {
            vec: s
                .trim()
                .trim_start_matches("[")
                .trim_end_matches("]")
                .trim(),
            _phantom: PhantomData,
        }
    }
}

impl<'a, N: std::str::FromStr + Copy> Iterator for VecIter<'a, N> {
    type Item = N;

    fn next(&mut self) -> Option<Self::Item> {
        if self.vec.is_empty() {
            return None;
        }

        debug_assert!(!self.vec.starts_with(" "));
        debug_assert!(!self.vec.starts_with(","));

        // find the end of the current element, which we assume is either space or comma
        if let Some(end) = self.vec.find([' ', ',']) {
            let res: N = match &self.vec[..end].trim().parse() {
                Err(_) => return None,
                Ok(n) => *n,
            };

            // find next element
            self.vec = self.vec[end + 1..].trim_start_matches([' ', ',']);

            Some(res)
        } else {
            // this is the last element
            let res: N = match &self.vec.parse() {
                Err(_) => return None,
                Ok(n) => *n,
            };
            self.vec = "";
            Some(res)
        }
    }
}
