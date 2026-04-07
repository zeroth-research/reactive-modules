pub mod bool;
pub mod bv;
pub mod int;
pub mod mat;
pub mod nat;
pub mod real;

/// TODO: add/finish doc-commet describing the overall picture
///
/// Typically, [Type] and [Operation] are empty structs or
/// enums and [IType] and [DType] are **non-empty** enums
/// into which [Type]s and [Operation]s map (using `From` traits)

/// [Type]s are used to construct [DType].
///
/// Objects implementing this trait will be those
/// that user will construct, pass into functions, etc.
/// [DType] can be seen as a collection of [Type]s.
pub trait Type: Clone {}

/// [DType] is a an enum of [Type]s used in a [Theory]
pub trait DType: Clone {}

/// [Operation] represents a function on values
/// of a given data type. It is used to construct [IType]
/// (in the same way as [Type] is used to construct [DType]).
pub trait Operation: Copy {}

/// [IType] is an enum of operations present in a theory
pub trait IType: Copy {}

/// Traits representing unary, binary and ternary operations.
/// These traits can be used to do static type-checking.
pub trait Operation1To1<T1: Type, R: Type>: Operation {}
pub trait Operation2To1<T1: Type, T2: Type, R: Type>: Operation {}
pub trait Operation3To1<T1: Type, T2: Type, T3: Type, R: Type>: Operation {}

/// [Theory] binds [IType] and [DType] (and thus [Type]s and [Operation]s
/// together). It is typically and empty struct, possibly with some associated
/// functions.
pub trait Theory {
    type DT: DType;
    type IT: IType;

    // // (optional) run-time type-checking of domain and codomain
    // fn dom_ok(_itype: &Self::IT, _intf: &Interface<Self::DT>) -> bool {
    //     true
    // }
    //
    // fn codom_ok(_itype: &Self::IT, _intf: &Interface<Self::DT>) -> bool {
    //     true
    // }

    //fn check_op_2_to_1<O: Operation, T1: Type, T2: Type, R: Type>(
    //    _op: O,
    //    _read: (T1, T2),
    //    _write: R,
    //) -> bool
    //where
    //    O: Into<Self::IT>,  // the operation is an operation of IType
    //    T1: Into<Self::DT>, // the types of arguments can be turned into DType
    //    T2: Into<Self::DT>,
    //    R: Into<Self::DT>,
    //    O: Operation2To1<T1, T2, R>, // it is a binary operation
    //{
    //    true
    //}
}

// Traits stating that a type and operation belong into a theory.
// It makes writing trait bounds much shorter and easier
pub trait TheoryType<T: Theory>: Type + Into<T::DT> {}
pub trait TheoryOperation<T: Theory>: Operation + Into<T::IT> {}

#[macro_export]
macro_rules! impl_theory_type {
    ([$($gen:tt)*] $theory:ident, $ty:ty) => {
        impl<$($gen)*> $crate::TheoryType<$theory> for $ty {}
    };
    ($theory:ident, $ty:ty) => {
        impl $crate::TheoryType<$theory> for $ty {}
    };
}

#[macro_export]
macro_rules! impl_theory_types {
    ($gen:tt $theory:ident, $($ty:ty),+) => {
        $($crate::impl_theory_type!($gen $theory, $ty);)+
    };
    ($theory:ident, $($ty:ty),+) => {
        $($crate::impl_theory_type!($theory, $ty);)+
    };
}

#[macro_export]
macro_rules! impl_theory_ops {
    ($theory:ident, $($op:ident),*) => {
        $(impl $crate::TheoryOperation<$theory> for $op {})*
    };
}

/// Dispatch helper: emit the correct `OperationNTo1` impl based on
/// argument count. Supports an optional generics clause in `[...]`
/// that defines generic parameters.
///
/// ### Examples
///
/// ```ignore
/// impl_op_trait!(Add(Nat, Nat) => Nat);
/// impl_op_trait!([const N: usize] Add(BV<N>, BV<N>) => BV<N>);
/// ```
#[macro_export]
macro_rules! impl_op_trait {
    ([$($gen:tt)*] $op:ident($a1:ty) => $ret:ty) => {
        impl<$($gen)*> Operation1To1<$a1, $ret> for $op {}
    };
    ([$($gen:tt)*] $op:ident($a1:ty, $a2:ty) => $ret:ty) => {
        impl<$($gen)*> Operation2To1<$a1, $a2, $ret> for $op {}
    };
    ([$($gen:tt)*] $op:ident($a1:ty, $a2:ty, $a3:ty) => $ret:ty) => {
        impl<$($gen)*> Operation3To1<$a1, $a2, $a3, $ret> for $op {}
    };
    ($op:ident($($args:ty),+) => $ret:ty) => {
        impl_op_trait!([] $op($($args),+) => $ret);
    };
}

/// Helper: emit a single `From<$ty> for Types` impl with optional generics.
#[macro_export]
macro_rules! impl_from_type {
    ([$($gen:tt)*] $variant:ident => $ty:ty) => {
        impl<$($gen)*> From<$ty> for Types {
            fn from(_: $ty) -> Types {
                Types::$variant
            }
        }
    };
}

/// Generate a `Types` enum with [`DType`] impl and `From` conversions
/// for each type variant. Supports optional generics in `[...]`.
///
/// Each entry is `VariantName => ConcreteType`. The variant name goes
/// into the enum; the concrete type is used in the `From` impl.
///
/// ### Examples
///
/// ```ignore
/// mk_types_enum!(Nat => Nat);                          // simple
/// mk_types_enum!([const N: usize] BV => BV<N>);        // generic
/// mk_types_enum!(Nat => Nat, Int => Int);              // multiple
/// ```
#[macro_export]
macro_rules! mk_types_enum {
    ($gen:tt $($variant:ident => $ty:ty),+ $(,)?) => {
        #[derive(Copy, Clone)]
        pub enum Types {
            $($variant,)+
        }

        impl DType for Types {}

        $(
            impl_from_type!($gen $variant => $ty);
        )+
    };
    ($($variant:ident => $ty:ty),+ $(,)?) => {
        mk_types_enum!([] $($variant => $ty),+);
    };
}

/// Helper: emit a single operation struct + trait impls with optional generics.
#[macro_export]
macro_rules! mk_one_op {
    ([$($gen:tt)*] $op_name:ident($($arg:ty),+) => $ret:ty) => {
        #[derive(Copy, Clone)]
        pub struct $op_name();
        impl Operation for $op_name {}
        impl_op_trait!([$($gen)*] $op_name($($arg),+) => $ret);
    };
}

/// Generate operation structs, an `Operations` enum with [`IType`] impl,
/// trait impls, and `From` conversions. Supports optional generics in `[...]`.
///
/// ### Examples
///
/// ```ignore
/// mk_ops!(
///     Add(Nat, Nat) => Nat,
///     Id(Nat) => Nat
/// );
///
/// mk_ops!([const N: usize]
///     Add(BV<N>, BV<N>) => BV<N>,
///     Id(BV<N>) => BV<N>
/// );
/// ```
#[macro_export]
macro_rules! mk_ops {
    ($gen:tt $($op_name:ident($($arg:ty),+) => $ret:ty),* $(,)?) => {
        $(
            mk_one_op!($gen $op_name($($arg),+) => $ret);
        )*

        #[derive(Copy, Clone)]
        pub enum Operations {
            $($op_name,)*
        }

        impl $crate::IType for Operations {}

        $(
            impl From<$op_name> for Operations {
                fn from(_: $op_name) -> Operations {
                    Operations::$op_name
                }
            }
        )*
    };
    ($($op_name:ident($($arg:ty),+) => $ret:ty),* $(,)?) => {
        mk_ops!([] $($op_name($($arg),+) => $ret),*);
    };
}

/// Generate a theory **module** with types, operations, enums, and `From`
/// conversions.
///
/// The macro has two forms depending on whether the types are freshly
/// created or pre-existing.
///
/// # Form 1 — Fresh types
///
/// Creates new unit structs for each type listed in `Types(...)`, with
/// `Clone + PartialEq` derives and a [`Type`] trait impl, together with
/// operation structs, a `Types` enum ([`DType`]), an `Operations` enum
/// ([`IType`]), and all `From` conversions.
///
/// ```ignore
/// mk_theory!(
///     <module-name>, Types(<Type1>, <Type2>, ...),
///     <Op>(<Arg>, <Arg>, <Arg>) => <Ret>,   // ternary
///     <Op>(<Arg>, <Arg>) => <Ret>,          // binary
///     <Op>(<Arg>)        => <Ret>,          // unary
/// );
/// ```
///
/// Example — natural numbers:
///
/// ```ignore
/// mk_theory!(
///     nat, Types(Nat),
///     Add(Nat, Nat) => Nat,
///     Mul(Nat, Nat) => Nat,
///     Id(Nat)       => Nat
/// );
/// // Defines:
/// //   nat::Nat         (Type)
/// //   nat::Add,        (Operation)
/// //   nat::Mul,        (Operation)
/// //   nat::Id,         (Operation)
/// //   nat::Types       (DType),
/// //   nat::Operations  (IType)
/// ```
///
/// # Form 2 — Pre-existing / generic types
///
/// Types must already be defined and in scope. A `[generics]` prefix
/// provides the generic parameters, and each type entry uses
/// `Variant => Type` syntax to map an enum variant name to the
/// (possibly generic) type.
///
/// ```ignore
/// mk_theory!([<generics>]
///     <module>, Types(<Variant> => <Type>, ...),
///     <Op>(<ArgType>, <ArgType>) => <RetType>,
///     <Op>(<ArgType>)            => <RetType>,
/// );
/// ```
///
/// The `[generics]` clause is forwarded to every `impl` block, so
/// operation trait impls and `From` conversions are generic. The
/// `Types` and `Operations` enums themselves are not generic — only
/// the impls are.
///
/// Example — bitvectors with a const-generic width:
///
/// ```ignore
/// #[derive(Clone)]
/// struct BV<const N: usize>();
/// impl<const N: usize> Type for BV<N> {}
///
/// mk_theory!([const N: usize]
///     bv, Types(BV => BV<N>),
///     Add(BV<N>, BV<N>) => BV<N>,
///     Mul(BV<N>, BV<N>) => BV<N>,
///     Id(BV<N>)         => BV<N>
/// );
/// // Available: bv::Add, bv::Types, bv::Operations
/// // BV<N> comes from the outer scope via `use super::*`
/// ```
///
/// # Building blocks
///
/// Internally, both forms delegate to [`mk_types_enum!`] and
/// [`mk_ops!`], which can also be used directly for finer control
/// (e.g. mixing pre-existing types with custom enum layouts).
#[macro_export]
macro_rules! mk_theory {
    // Generic arm: pre-existing types, optional generics
    (
        [$($gen:tt)*]
        $mod_name:ident, Types($($variant:ident => $ty:ty),+),
        $($op_name:ident($($arg:ty),+) => $ret:ty),* $(,)?
    ) => {
        pub mod $mod_name {
            use super::*;
            use $crate::*;

            pub struct Theory {}

            $crate::mk_types_enum!([$($gen)*] $($variant => $ty),+);

            $crate::mk_ops!([$($gen)*] $($op_name($($arg),+) => $ret),*);

            $crate::impl_theory_types!([$($gen)*] Theory, $($ty),+);
            $crate::impl_theory_ops!(Theory, $($op_name),*);

            impl $crate::Theory for Theory {
                type DT = Types;
                type IT = Operations;
            }
        }
    };
    // Simple arm: fresh type structs, no generics
    (
        $mod_name:ident, Types($($type_name:ident),+),
        $($op_name:ident($($arg:ident),+) => $ret:ident),* $(,)?
    ) => {
        pub mod $mod_name {
            pub struct Theory {}

            $(
                #[derive(Clone, PartialEq)]
                pub struct $type_name();
                impl $crate::Type for $type_name {}
            )+
            use $crate::*;

            $crate::mk_types_enum!($($type_name => $type_name),+);

            $crate::mk_ops!($($op_name($($arg),+) => $ret),*);

            $crate::impl_theory_types!(Theory, $($type_name),+);
            $crate::impl_theory_ops!(Theory, $($op_name),*);

            impl $crate::Theory for Theory {
                type DT = Types;
                type IT = Operations;
            }
        }
    };
}
