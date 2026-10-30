//! Boundaries: the boundary of a morphism is a pair of *polynomials* — its
//! domain and codomain. A polynomial is a `⊕`-sum of `⊗`-monomials, each
//! monomial a word of objects: `⊗` concatenates (all at once), `⊕` offers
//! alternatives (one of), and `⊗` distributes over `⊕`.

/// A monomial: a concatenation of objects — all at once. As a type, an
/// anonymous tuple.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mono<G>(pub Vec<G>);

/// A polynomial: alternative monomials — one of them. As a
/// type, an anonymous variant. The order of the alternatives is tracked, and sums of
/// monomials are the normal form, so equality of boundaries is
/// structural.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly<G>(pub Vec<Mono<G>>);

impl<G> Mono<G> {
    /// The unit monomial `I`: the empty word — the unit of `⊗`.
    pub fn unit() -> Self {
        Self(Vec::new())
    }
}

/// A word of objects is a monomial.
impl<G> FromIterator<G> for Mono<G> {
    fn from_iter<I: IntoIterator<Item = G>>(word: I) -> Self {
        Self(word.into_iter().collect())
    }
}

/// A monomial is a polynomial with a single summand.
impl<G> From<Mono<G>> for Poly<G> {
    fn from(m: Mono<G>) -> Self {
        Self(vec![m])
    }
}

/// A word of objects is a polynomial with a single summand.
impl<G> FromIterator<G> for Poly<G> {
    fn from_iter<I: IntoIterator<Item = G>>(word: I) -> Self {
        Mono::from_iter(word).into()
    }
}

/// A sum of monomials is a polynomial.
impl<G> FromIterator<Mono<G>> for Poly<G> {
    fn from_iter<I: IntoIterator<Item = Mono<G>>>(summands: I) -> Self {
        Self(summands.into_iter().collect())
    }
}

impl<G: Clone> std::ops::Mul for Mono<G> {
    type Output = Mono<G>;

    /// `⊗` of monomials: the concatenation of the words.
    fn mul(mut self, other: Self) -> Mono<G> {
        self.0.extend(other.0);
        self
    }
}

impl<G: Clone> std::ops::Add for Mono<G> {
    type Output = Poly<G>;

    /// `⊕` of monomials: the polynomial with the two summands.
    fn add(self, other: Self) -> Poly<G> {
        Poly(vec![self, other])
    }
}

impl<G: Clone> Poly<G> {
    /// The zero polynomial: the empty sum — the impossible boundary (a
    /// failed guard).
    pub fn zero() -> Self {
        Self(Vec::new())
    }

    /// The unit polynomial `I`: the unit monomial, injected.
    pub fn unit() -> Self {
        Self(vec![Mono::unit()])
    }
}

impl<G: Clone> std::ops::Add for Poly<G> {
    type Output = Poly<G>;

    /// `⊕`: the concatenation of the sums.
    fn add(mut self, other: Self) -> Poly<G> {
        self.0.extend(other.0);
        self
    }
}

impl<G: Clone> std::ops::Mul for Poly<G> {
    type Output = Poly<G>;

    /// `⊗`: pairwise `⊗` of the summands — distributes over `⊕`.
    fn mul(self, other: Self) -> Poly<G> {
        Poly(
            self.0
                .iter()
                .flat_map(|m| other.0.iter().map(|n| m.clone() * n.clone()))
                .collect(),
        )
    }
}

/// The boundary of a morphism: its domain and codomain polynomials over
/// the objects `G`. Boundaries are total: an ill-formed morphism does not exist, by
/// construction. Checking a boundary is simply checking equality of `dom`
/// and `cod`: an application is well-formed iff its boundaries equal the
/// morphism's.
pub trait Boundary<G> {
    fn dom(&self) -> Poly<G>;
    fn cod(&self) -> Poly<G>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundaries() {
        // Bool = I ⊕ I, and ⊗ distributes: Bool ⊗ x has two summands
        let boolean = Poly::<char>::unit() + Poly::unit();
        let paired = boolean * Poly::from_iter(['x']);
        assert_eq!(paired, Poly::from_iter(['x']) + Poly::from_iter(['x']));

        // ⊗ of monomials is concatenation, with the unit I
        let xy = Poly::from_iter(['x']) * Poly::from_iter(['y']);
        assert_eq!(xy, Poly::from_iter(['x', 'y']));
        assert_eq!(xy.clone() * Poly::unit(), xy);

        // 0 annihilates
        assert_eq!(xy.clone() * Poly::zero(), Poly::zero());

        // operator syntax: (x + y) * z = xz ⊕ yz
        let (x, y, z) = (Mono(vec!['x']), Mono(vec!['y']), Mono(vec!['z']));
        assert_eq!(
            (x + y) * Poly::from(z),
            Poly(vec![Mono(vec!['x', 'z']), Mono(vec!['y', 'z'])])
        );
        assert_eq!(Mono(vec!['x']) * Mono(vec!['y']), Mono(vec!['x', 'y']));
    }
}
