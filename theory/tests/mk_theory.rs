use theory::*;

// --- mk_theory with default names ---

mod default_simple {
    use theory::mk_theory;

    mk_theory!(
        Types(Nat),
        Add(Nat, Nat) => Nat,
        Id(Nat) => Nat
    );
}

// --- mk_theory with custom names ---

mod custom_simple {
    use theory::mk_theory;

    mk_theory!({NatTypes, NatOps, NatTh}
        Types(Nat),
        Add(Nat, Nat) => Nat,
        Id(Nat) => Nat
    );
}

// --- mk_theory with generics + default names ---

mod generic_default {
    use theory::{Type, mk_theory};

    #[derive(Clone)]
    pub struct BV<const N: usize>();
    impl<const N: usize> Type for BV<N> {}

    mk_theory!([const N: usize]
        Types(BV => BV<N>),
        Add(BV<N>, BV<N>) => BV<N>,
        Id(BV<N>) => BV<N>
    );
}

// --- mk_theory with generics + custom names ---

mod generic_custom {
    use theory::{Type, mk_theory};

    #[derive(Clone)]
    pub struct BV<const N: usize>();
    impl<const N: usize> Type for BV<N> {}

    mk_theory!([const N: usize] {BvTypes, BvOps, BvTh}
        Types(BV => BV<N>),
        Add(BV<N>, BV<N>) => BV<N>,
        Id(BV<N>) => BV<N>
    );
}

// Tests that verify the generated items exist and have the right traits/types.

#[test]
fn test_default_names() {
    use default_simple::*;

    // Theory struct implements Theory with the default DType/IType names
    let _: <Theory as theory::Theory>::DT = Types::Nat;
    let _: <Theory as theory::Theory>::IT = Operations::Add;
    let _: <Theory as theory::Theory>::IT = Operations::Id;

    // Type structs exist and convert into DType
    let _: Types = Nat().into();

    // Operation structs exist and convert into IType
    let _: Operations = Add().into();
    let _: Operations = Id().into();
}

#[test]
fn test_custom_names() {
    use custom_simple::*;

    // Generated items use the custom names
    let _: <NatTh as theory::Theory>::DT = NatTypes::Nat;
    let _: <NatTh as theory::Theory>::IT = NatOps::Add;
    let _: <NatTh as theory::Theory>::IT = NatOps::Id;

    let _: NatTypes = Nat().into();
    let _: NatOps = Add().into();
}

#[test]
fn test_generic_default() {
    use generic_default::*;

    let _: Types = BV::<8>().into();
    let _: Operations = Add().into();
    let _: Operations = Id().into();
}

#[test]
fn test_generic_custom() {
    use generic_custom::*;

    let _: <BvTh as theory::Theory>::DT = BvTypes::BV;
    let _: <BvTh as theory::Theory>::IT = BvOps::Add;

    let _: BvTypes = BV::<32>().into();
    let _: BvOps = Add().into();
}

// Verify TheoryType and TheoryOperation impls work with custom names

#[test]
fn test_theory_type_trait_custom() {
    fn assert_theory_type<T: Theory, X: TheoryType<T>>() {}
    fn assert_theory_op<T: Theory, X: TheoryOperation<T>>() {}

    assert_theory_type::<custom_simple::NatTh, custom_simple::Nat>();
    assert_theory_op::<custom_simple::NatTh, custom_simple::Add>();
    assert_theory_op::<custom_simple::NatTh, custom_simple::Id>();
}

#[test]
fn test_theory_type_trait_default() {
    fn assert_theory_type<T: Theory, X: TheoryType<T>>() {}
    fn assert_theory_op<T: Theory, X: TheoryOperation<T>>() {}

    assert_theory_type::<default_simple::Theory, default_simple::Nat>();
    assert_theory_op::<default_simple::Theory, default_simple::Add>();
    assert_theory_op::<default_simple::Theory, default_simple::Id>();
}
