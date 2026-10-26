use std::fmt;
use theory::signature::{Dim, Mat, Name, Signature, SignatureElem, Tan, TypeVar, typecheck};
use theory::sort::{MultiSort, TangentMultiSort};

#[derive(Copy, Clone, Debug, PartialEq)]
enum TestSort {
    A(),
    B(usize),
    TB(usize, usize),
    TanZero(),
}

struct TestSignature {
    dom: Vec<SignatureElem<TestSort>>,
    cod: Vec<SignatureElem<TestSort>>,
}

impl Signature<TestSort> for TestSignature {
    fn dom(&self) -> impl Iterator<Item = SignatureElem<TestSort>> {
        self.dom.iter().cloned()
    }

    fn cod(&self) -> impl Iterator<Item = SignatureElem<TestSort>> {
        self.cod.iter().cloned()
    }

    fn dom_arity(&self) -> usize {
        self.dom.len()
    }

    fn cod_arity(&self) -> usize {
        self.cod.len()
    }
}

// Prints, e.g., `(A, B(8)) -> B(8)`; a domain or codomain that is not
// a single element is printed as a tuple, e.g., `() -> A`.
impl fmt::Display for TestSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fmt_elems = |f: &mut fmt::Formatter<'_>, elems: &[SignatureElem<TestSort>]| {
            if let [elem] = elems {
                return write!(f, "{elem}");
            }
            f.write_str("(")?;
            for (i, elem) in elems.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{elem}")?;
            }
            f.write_str(")")
        };
        fmt_elems(f, &self.dom)?;
        f.write_str(" -> ")?;
        fmt_elems(f, &self.cod)
    }
}

impl MultiSort for TestSort {
    const NAME: &'static str = "TestSort";
}

impl fmt::Display for TestSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TestSort::A() => write!(f, "A"),
            TestSort::B(n) => write!(f, "B({n})"),
            TestSort::TB(n, d) => write!(f, "TB({n}, {d})"),
            TestSort::TanZero() => write!(f, "TanZero"),
        }
    }
}

impl TangentMultiSort for TestSort {
    //const NAME = "TestSort";
    fn T(&self) -> TestSort {
        match self {
            TestSort::A() => TestSort::TanZero(),
            TestSort::B(n) => TestSort::TB(*n, 1),
            TestSort::TB(n, d) => TestSort::TB(*n, d + 1),
            TestSort::TanZero() => TestSort::TanZero(),
        }
    }
}

#[test]
fn test_1() {
    let t = TypeVar('a');
    let a = TestSort::A().into();
    let b = TestSort::B(8).into();
    let _tan = Tan(TestSort::B(4).T());
    let m1 = Mat(TestSort::B(3), 'a', 'b');
    let m2 = Mat(TestSort::B(3), 'b', 'c');
    let m3 = Mat(TestSort::B(3), 'a', 'c');

    let sig: TestSignature = TestSignature {
        dom: vec![a, b, m1, m2, t],
        cod: vec![m3],
    };
    assert_eq!(
        sig.to_string(),
        "(A, B(8), Mat(B(3), a, b), Mat(B(3), b, c), 'a) -> Mat(B(3), a, c)"
    );
}

#[test]
fn display_signature_elems() {
    assert_eq!(Elem::from(TestSort::B(8)).to_string(), "B(8)");
    assert_eq!(Tan(TestSort::B(4)).to_string(), "Tan(B(4))");
    assert_eq!(TypeVar::<TestSort>(1).to_string(), "'b");
    assert_eq!(
        Mat(TestSort::A(), Dim::Sym(Name(13)), Dim::Const(5)).to_string(),
        "Mat(A, n, 5)"
    );
}

#[test]
fn name_from() {
    assert_eq!(Name::from(7), Name(7));
    assert_eq!(Name::from('a'), Name(0));
    assert_eq!(Name::from('z'), Name(25));
    assert_eq!(TypeVar::<TestSort>('b'), SignatureElem::TypeVar(Name(1)));
    assert_eq!(TypeVar::<TestSort>(1), SignatureElem::TypeVar(Name(1)));
}

#[test]
fn dim_from() {
    assert_eq!(Dim::from(5), Dim::Const(5));
    assert_eq!(Dim::from('a'), Dim::Sym(Name(0)));
    assert_eq!(
        Mat(TestSort::A(), 'n', 5),
        SignatureElem::Mat(TestSort::A(), Dim::Sym(Name(13)), Dim::Const(5))
    );
}

#[test]
#[should_panic(expected = "variable name must be a..z")]
fn name_from_invalid_char() {
    let _ = Name::from('A');
}

#[test]
fn display_var_names_past_z() {
    assert_eq!(Name(25).to_string(), "z");
    assert_eq!(Name(26).to_string(), "a1");
    assert_eq!(Name(27).to_string(), "b1");
    assert_eq!(Name(255).to_string(), "v9");
    assert_eq!(Dim::Sym(Name(26)).to_string(), "a1");
    assert_eq!(TypeVar::<TestSort>(52).to_string(), "'a2");
}

#[test]
fn display_tuples() {
    let a: Elem = TestSort::A().into();
    let b: Elem = TestSort::B(8).into();
    let sig = |dom: Vec<_>, cod: Vec<_>| TestSignature { dom, cod }.to_string();
    assert_eq!(sig(vec![a], vec![b]), "A -> B(8)");
    assert_eq!(sig(vec![a, a], vec![a]), "(A, A) -> A");
    assert_eq!(sig(vec![], vec![b]), "() -> B(8)");
    assert_eq!(sig(vec![a], vec![]), "A -> ()");
    assert_eq!(sig(vec![], vec![]), "() -> ()");
    assert_eq!(sig(vec![a], vec![a, b]), "A -> (A, B(8))");
}

// Shorthands for the typechecking tests

type Elem = SignatureElem<TestSort>;

fn sig(dom: &[Elem], cod: &[Elem]) -> TestSignature {
    TestSignature {
        dom: dom.to_vec(),
        cod: cod.to_vec(),
    }
}

fn a() -> Elem {
    TestSort::A().into()
}

fn b(n: usize) -> Elem {
    TestSort::B(n).into()
}

fn err(msg: &str) -> Result<(), String> {
    Err(msg.to_string())
}

#[test]
fn typecheck_concrete() {
    assert_eq!(
        typecheck(sig(&[a(), b(8)], &[a()]), sig(&[a(), b(8)], &[a()])),
        Ok(())
    );
    assert_eq!(
        typecheck(sig(&[a(), b(8)], &[a()]), sig(&[a(), b(4)], &[a()])),
        err("Failed unifying `(A, B(8)) -> A` with `(A, B(4)) -> A` on `B(8)` and `B(4)`")
    );
    assert_eq!(
        typecheck(sig(&[a()], &[a()]), sig(&[a()], &[b(8)])),
        err("Failed unifying `A -> A` with `A -> B(8)` on `A` and `B(8)`")
    );
}

#[test]
fn typecheck_nullary() {
    assert_eq!(typecheck(sig(&[], &[]), sig(&[], &[])), Ok(()));
    assert_eq!(typecheck(sig(&[], &[a()]), sig(&[], &[a()])), Ok(()));
    assert_eq!(typecheck(sig(&[a()], &[]), sig(&[a()], &[])), Ok(()));
    assert_eq!(
        typecheck(sig(&[], &[a()]), sig(&[], &[b(8)])),
        err("Failed unifying `() -> A` with `() -> B(8)` on `A` and `B(8)`")
    );
}

#[test]
fn typecheck_arity() {
    assert_eq!(
        typecheck(sig(&[a(), a()], &[a()]), sig(&[a()], &[a()])),
        err("Signatures domains have different arity (2 != 1)")
    );
    assert_eq!(
        typecheck(sig(&[], &[a()]), sig(&[a()], &[a()])),
        err("Signatures domains have different arity (0 != 1)")
    );
    assert_eq!(
        typecheck(sig(&[a()], &[a()]), sig(&[a()], &[a(), a()])),
        err("Signatures codomains have different arity (1 != 2)")
    );
    // arities are checked before the elements
    assert_eq!(
        typecheck(sig(&[a()], &[a()]), sig(&[b(8)], &[])),
        err("Signatures codomains have different arity (1 != 0)")
    );
}

#[test]
fn typecheck_tangent() {
    let tan = Tan(TestSort::B(4));
    assert_eq!(
        typecheck(sig(&[tan], &[b(4)]), sig(&[tan], &[b(4)])),
        Ok(())
    );
    assert_eq!(
        typecheck(sig(&[tan], &[b(4)]), sig(&[b(4)], &[b(4)])),
        err("Failed unifying `Tan(B(4)) -> B(4)` with `B(4) -> B(4)` on `Tan(B(4))` and `B(4)`")
    );
    assert_eq!(
        typecheck(sig(&[TypeVar('a')], &[TypeVar('a')]), sig(&[tan], &[tan])),
        Ok(())
    );
}

#[test]
fn typecheck_type_var() {
    // `'a -> 'a` is instantiated by any sort
    assert_eq!(
        typecheck(sig(&[TypeVar('a')], &[TypeVar('a')]), sig(&[a()], &[a()])),
        Ok(())
    );
    assert_eq!(
        typecheck(sig(&[TypeVar('a')], &[TypeVar('a')]), sig(&[b(8)], &[b(8)])),
        Ok(())
    );
    // ... but consistently
    assert_eq!(
        typecheck(sig(&[TypeVar('a')], &[TypeVar('a')]), sig(&[a()], &[b(8)])),
        err("Failed unifying `'a -> 'a` with `A -> B(8)` on `A` and `B(8)`")
    );
    assert_eq!(
        typecheck(
            sig(&[TypeVar('a'), TypeVar('a')], &[TypeVar('a')]),
            sig(&[a(), b(8)], &[a()])
        ),
        err("Failed unifying `('a, 'a) -> 'a` with `(A, B(8)) -> A` on `A` and `B(8)`")
    );
    // different type variables can be instantiated by different sorts
    assert_eq!(
        typecheck(
            sig(&[TypeVar('a'), TypeVar('b')], &[TypeVar('b')]),
            sig(&[a(), b(8)], &[b(8)])
        ),
        Ok(())
    );
}

#[test]
fn typecheck_type_var_on_the_right() {
    assert_eq!(
        typecheck(
            sig(&[a(), b(8)], &[a()]),
            sig(&[TypeVar('a'), TypeVar('b')], &[TypeVar('a')])
        ),
        Ok(())
    );
    assert_eq!(
        typecheck(
            sig(&[a(), b(8)], &[a()]),
            sig(&[TypeVar('a'), TypeVar('b')], &[TypeVar('b')])
        ),
        err("Failed unifying `(A, B(8)) -> A` with `('a, 'b) -> 'b` on `A` and `B(8)`")
    );
}

#[test]
fn typecheck_type_var_with_type_var() {
    // `'c` unifies with both `'a` and `'b`
    assert_eq!(
        typecheck(
            sig(&[TypeVar('a'), TypeVar('b')], &[TypeVar('a')]),
            sig(&[TypeVar('c'), TypeVar('c')], &[TypeVar('c')])
        ),
        Ok(())
    );
    // `'a` and `'b` unified through `'c` must be instantiated by the same sort
    assert_eq!(
        typecheck(
            sig(&[TypeVar('a'), TypeVar('b'), TypeVar('a')], &[TypeVar('b')]),
            sig(&[TypeVar('c'), TypeVar('c'), a()], &[b(8)])
        ),
        err("Failed unifying `('a, 'b, 'a) -> 'b` with `('c, 'c, A) -> B(8)` on `A` and `B(8)`")
    );
}

#[test]
fn typecheck_type_var_with_matrix() {
    let m23 = Mat(TestSort::A(), 2, 3);
    let m32 = Mat(TestSort::A(), 3, 2);
    assert_eq!(
        typecheck(sig(&[TypeVar('a')], &[TypeVar('a')]), sig(&[m23], &[m23])),
        Ok(())
    );
    assert_eq!(
        typecheck(sig(&[TypeVar('a')], &[TypeVar('a')]), sig(&[m23], &[m32])),
        err(
            "Failed unifying `'a -> 'a` with `Mat(A, 2, 3) -> Mat(A, 3, 2)` on `Mat(A, 2, 3)` and `Mat(A, 3, 2)`"
        )
    );
}

#[test]
fn typecheck_signatures_do_not_share_vars() {
    // `'a` is `A` in the first signature and `B(8)` in the second one
    assert_eq!(
        typecheck(
            sig(&[TypeVar('a'), b(8)], &[TypeVar('a')]),
            sig(&[a(), TypeVar('a')], &[a()])
        ),
        Ok(())
    );
    assert_eq!(
        typecheck(
            sig(&[TypeVar('a')], &[TypeVar('a')]),
            sig(&[TypeVar('a')], &[TypeVar('a')])
        ),
        Ok(())
    );
    // ... but each `'a` is still used consistently within its signature
    assert_eq!(
        typecheck(
            sig(&[TypeVar('a'), b(8)], &[TypeVar('a')]),
            sig(&[TypeVar('a'), TypeVar('a')], &[a()])
        ),
        err("Failed unifying `('a, B(8)) -> 'a` with `('a, 'a) -> A` on `B(8)` and `A`")
    );
    // the same for dimensions: `a` is 3 in the first signature and 2 in the second one
    let a2 = Mat(TestSort::A(), 'a', 2);
    let a3 = Mat(TestSort::A(), 3, 'a');
    assert_eq!(typecheck(sig(&[a2], &[a2]), sig(&[a3], &[a3])), Ok(()));
}

#[test]
fn typecheck_too_many_vars() {
    // no name is left for renaming `'a` of the second signature apart from `'v9` (255)
    assert_eq!(
        typecheck(sig(&[TypeVar(255)], &[]), sig(&[TypeVar('a')], &[])),
        err("Too many variables in the signatures")
    );
}

#[test]
fn typecheck_type_and_dimension_vars_are_separate() {
    // the type variable `'a` and the dimension `a` have the same index
    // but are different variables
    assert_eq!(
        typecheck(
            sig(
                &[TypeVar('a'), Mat(TestSort::A(), 'a', 'b')],
                &[TypeVar('a')]
            ),
            sig(&[b(8), Mat(TestSort::A(), 2, 3)], &[b(8)])
        ),
        Ok(())
    );
}

#[test]
fn typecheck_matrix_const_dims() {
    let m = |rows: usize, cols: usize| Mat(TestSort::A(), rows, cols);
    assert_eq!(
        typecheck(sig(&[m(2, 3)], &[a()]), sig(&[m(2, 3)], &[a()])),
        Ok(())
    );
    assert_eq!(
        typecheck(sig(&[m(2, 3)], &[a()]), sig(&[m(4, 3)], &[a()])),
        err(
            "Failed unifying `Mat(A, 2, 3) -> A` with `Mat(A, 4, 3) -> A` on `Mat(A, 2, 3)` and `Mat(A, 4, 3)`"
        )
    );
    assert_eq!(
        typecheck(sig(&[m(2, 3)], &[a()]), sig(&[m(2, 4)], &[a()])),
        err(
            "Failed unifying `Mat(A, 2, 3) -> A` with `Mat(A, 2, 4) -> A` on `Mat(A, 2, 3)` and `Mat(A, 2, 4)`"
        )
    );
}

#[test]
fn typecheck_matrix_elem_sort() {
    assert_eq!(
        typecheck(
            sig(&[Mat(TestSort::A(), 'a', 'b')], &[a()]),
            sig(&[Mat(TestSort::B(3), 2, 3)], &[a()])
        ),
        err(
            "Failed unifying `Mat(A, a, b) -> A` with `Mat(B(3), 2, 3) -> A` on `Mat(A, 2, 3)` and `Mat(B(3), 2, 3)`"
        )
    );
    assert_eq!(
        typecheck(
            sig(&[Mat(TestSort::A(), 1, 1)], &[a()]),
            sig(&[a()], &[a()])
        ),
        err("Failed unifying `Mat(A, 1, 1) -> A` with `A -> A` on `Mat(A, 1, 1)` and `A`")
    );
}

#[test]
fn typecheck_matmul() {
    // (Mat(a, b), Mat(b, c)) -> Mat(a, c)
    let matmul = || {
        let m = |rows: char, cols: char| Mat(TestSort::B(3), rows, cols);
        sig(&[m('a', 'b'), m('b', 'c')], &[m('a', 'c')])
    };
    let m = |rows: usize, cols: usize| Mat(TestSort::B(3), rows, cols);
    assert_eq!(
        typecheck(matmul(), sig(&[m(2, 3), m(3, 4)], &[m(2, 4)])),
        Ok(())
    );
    assert_eq!(
        typecheck(matmul(), sig(&[m(2, 3), m(4, 5)], &[m(2, 5)])),
        err(
            "Failed unifying `(Mat(B(3), a, b), Mat(B(3), b, c)) -> Mat(B(3), a, c)` \
             with `(Mat(B(3), 2, 3), Mat(B(3), 4, 5)) -> Mat(B(3), 2, 5)` \
             on `Mat(B(3), 3, 5)` and `Mat(B(3), 4, 5)`"
        )
    );
    assert_eq!(
        typecheck(matmul(), sig(&[m(2, 3), m(3, 4)], &[m(2, 5)])),
        err(
            "Failed unifying `(Mat(B(3), a, b), Mat(B(3), b, c)) -> Mat(B(3), a, c)` \
             with `(Mat(B(3), 2, 3), Mat(B(3), 3, 4)) -> Mat(B(3), 2, 5)` \
             on `Mat(B(3), 2, 4)` and `Mat(B(3), 2, 5)`"
        )
    );
}

#[test]
fn typecheck_matrix_symbolic_dims() {
    // Mat(a, b) -> Mat(b, a)
    let transpose = || {
        let m = |rows: char, cols: char| Mat(TestSort::A(), rows, cols);
        sig(&[m('a', 'b')], &[m('b', 'a')])
    };
    // Mat(c, d) -> Mat(d, c)
    let m = |rows: char, cols: char| Mat(TestSort::A(), rows, cols);
    assert_eq!(
        typecheck(transpose(), sig(&[m('c', 'd')], &[m('d', 'c')])),
        Ok(())
    );
    let m = |rows: usize, cols: usize| Mat(TestSort::A(), rows, cols);
    assert_eq!(typecheck(transpose(), sig(&[m(2, 3)], &[m(3, 2)])), Ok(()));
    assert_eq!(
        typecheck(transpose(), sig(&[m(2, 3)], &[m(2, 3)])),
        err(
            "Failed unifying `Mat(A, a, b) -> Mat(A, b, a)` with `Mat(A, 2, 3) -> Mat(A, 2, 3)` \
             on `Mat(A, 3, 2)` and `Mat(A, 2, 3)`"
        )
    );
    // a square matrix
    assert_eq!(
        typecheck(
            sig(&[Mat(TestSort::A(), 'a', 'a')], &[a()]),
            sig(&[m(3, 3)], &[a()])
        ),
        Ok(())
    );
    assert_eq!(
        typecheck(
            sig(&[Mat(TestSort::A(), 'a', 'a')], &[a()]),
            sig(&[m(2, 3)], &[a()])
        ),
        err(
            "Failed unifying `Mat(A, a, a) -> A` with `Mat(A, 2, 3) -> A` on `Mat(A, 2, 2)` and `Mat(A, 2, 3)`"
        )
    );
}
