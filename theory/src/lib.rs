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

/// Helper: emit a single `From<$ty> for <enum>` impl with optional generics.
/// The first argument is the target enum name.
#[macro_export]
macro_rules! impl_from_type {
    ($enum_name:ident, [$($gen:tt)*] $variant:ident => $ty:ty) => {
        impl<$($gen)*> From<$ty> for $enum_name {
            fn from(_: $ty) -> $enum_name {
                $enum_name::$variant
            }
        }
    };
}

/// Generate an enum with [`DType`] impl and `From` conversions
/// for each type variant. An optional first argument names the enum
/// (defaults to `Types`). Supports optional generics in `[...]`.
///
/// Each entry is `VariantName => ConcreteType`. The variant name goes
/// into the enum; the concrete type is used in the `From` impl.
///
/// ### Examples
///
/// ```ignore
/// mk_types_enum!(Nat => Nat);                              // enum Types { Nat }
/// mk_types_enum!(MyTypes, Nat => Nat);                     // enum MyTypes { Nat }
/// mk_types_enum!([const N: usize] BV => BV<N>);            // generic
/// mk_types_enum!(MyTypes, [const N: usize] BV => BV<N>);   // generic + custom name
/// ```
#[macro_export]
macro_rules! mk_types_enum {
    // Core: custom name + generics
    ($name:ident, $gen:tt $($variant:ident => $ty:ty),+ $(,)?) => {
        #[derive(Copy, Clone)]
        pub enum $name {
            $($variant,)+
        }

        impl DType for $name {}

        $(
            $crate::impl_from_type!($name, $gen $variant => $ty);
        )+
    };
    // Custom name, no generics
    ($name:ident, $($variant:ident => $ty:ty),+ $(,)?) => {
        $crate::mk_types_enum!($name, [] $($variant => $ty),+);
    };
    // Default name (Types) + generics
    ($gen:tt $($variant:ident => $ty:ty),+ $(,)?) => {
        $crate::mk_types_enum!(Types, $gen $($variant => $ty),+);
    };
    // Default name (Types), no generics
    ($($variant:ident => $ty:ty),+ $(,)?) => {
        $crate::mk_types_enum!(Types, [] $($variant => $ty),+);
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

/// Helper: emit operation structs + trait impls for a list of fresh ops.
/// Generics are passed as a single token tree `[...]`.
#[macro_export]
macro_rules! mk_op_structs {
    ($gen:tt $($op_name:ident($($arg:ty),+) => $ret:ty),*) => {
        $(
            $crate::mk_one_op!($gen $op_name($($arg),+) => $ret);
        )*
    };
}

/// Generate operation structs, a named enum with [`IType`] impl,
/// trait impls, and `From` conversions. An optional first argument
/// names the enum (defaults to `Operations`). Supports optional
/// generics in `[...]`.
///
/// ### Examples
///
/// ```ignore
/// mk_ops!(
///     Add(Nat, Nat) => Nat,
///     Id(Nat) => Nat
/// );
///
/// mk_ops!(NatOps,
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
    // Core: custom name + generics
    ($name:ident, $gen:tt $($op_name:ident($($arg:ty),+) => $ret:ty),* $(,)?) => {
        $(
            $crate::mk_one_op!($gen $op_name($($arg),+) => $ret);
        )*

        #[derive(Copy, Clone)]
        pub enum $name {
            $($op_name,)*
        }

        impl $crate::IType for $name {}

        $(
            impl From<$op_name> for $name {
                fn from(_: $op_name) -> $name {
                    $name::$op_name
                }
            }
        )*
    };
    // Custom name, no generics
    ($name:ident, $($op_name:ident($($arg:ty),+) => $ret:ty),* $(,)?) => {
        $crate::mk_ops!($name, [] $($op_name($($arg),+) => $ret),*);
    };
    // Default name (Operations) + generics
    ($gen:tt $($op_name:ident($($arg:ty),+) => $ret:ty),* $(,)?) => {
        $crate::mk_ops!(Operations, $gen $($op_name($($arg),+) => $ret),*);
    };
    // Default name (Operations), no generics
    ($($op_name:ident($($arg:ty),+) => $ret:ty),* $(,)?) => {
        $crate::mk_ops!(Operations, [] $($op_name($($arg),+) => $ret),*);
    };
}

/// Generate types, operations, enums, and a theory struct with all
/// necessary trait impls and `From` conversions.
///
/// An optional `{DType, IType, Theory}` prefix names the generated
/// enums and struct (defaults to `Types`, `Operations`, `Theory`).
///
/// # Form 1 — Fresh types
///
/// Creates new unit structs for each type listed in `Types(...)`.
///
/// ```ignore
/// mk_theory!(
///     Types(Nat),
///     Add(Nat, Nat) => Nat,
///     Id(Nat)       => Nat
/// );
///
/// // With custom names for the generated DType, IType, and Theory:
/// mk_theory!({NatTypes, NatOps, NatTh}
///     Types(Nat),
///     Add(Nat, Nat) => Nat,
///     Id(Nat)       => Nat
/// );
/// ```
///
/// # Form 2 — Pre-existing / generic types
///
/// Types must already be defined and in scope. A `[generics]` prefix
/// provides generic parameters, and each type entry uses
/// `Variant => Type` syntax.
///
/// ```ignore
/// mk_theory!([const N: usize]
///     Types(BV => BV<N>),
///     Add(BV<N>, BV<N>) => BV<N>
/// );
///
/// // With custom names:
/// mk_theory!([const N: usize] {BvTypes, BvOps, BvTh}
///     Types(BV => BV<N>),
///     Add(BV<N>, BV<N>) => BV<N>
/// );
/// ```
#[macro_export]
macro_rules! mk_theory {
    // Generic + custom names (core)
    (
        [$($gen:tt)*]
        {$dt:ident, $it:ident, $th:ident}
        Types($($variant:ident => $ty:ty),+),
        $($op_name:ident($($arg:ty),+) => $ret:ty),* $(,)?
        $(; $([$($ext_gen:tt)*] $ext_variant:ident => $ext_ty:ty),+ $(,)?)?
    ) => {
        #[allow(unused)]
        use $crate::*;

        pub struct $th {}

        $crate::mk_types_enum!($dt, [$($gen)*] $($variant => $ty),+);

        $crate::mk_op_structs!([$($gen)*] $($op_name($($arg),+) => $ret),*);

        #[derive(Copy, Clone)]
        pub enum $it {
            $($op_name,)*
            $($($ext_variant,)+)?
        }

        impl $crate::IType for $it {}

        $(
            impl From<$op_name> for $it {
                fn from(_: $op_name) -> $it { $it::$op_name }
            }
        )*

        $($(
            impl<$($ext_gen)*> From<$ext_ty> for $it {
                fn from(_: $ext_ty) -> $it { $it::$ext_variant }
            }
        )+)?

        $crate::impl_theory_types!([$($gen)*] $th, $($ty),+);
        $crate::impl_theory_ops!($th, $($op_name),*);

        $($(
            impl<$($ext_gen)*> $crate::TheoryOperation<$th> for $ext_ty {}
        )+)?

        impl $crate::Theory for $th {
            type DT = $dt;
            type IT = $it;
        }
    };
    // Simple + custom names (core)
    (
        {$dt:ident, $it:ident, $th:ident}
        Types($($type_name:ident),+),
        $($op_name:ident($($arg:ident),+) => $ret:ident),* $(,)?
        $(; $([$($ext_gen:tt)*] $ext_variant:ident => $ext_ty:ty),+ $(,)?)?
    ) => {
        pub struct $th {}

        $(
            #[derive(Clone, PartialEq)]
            pub struct $type_name();
            impl $crate::Type for $type_name {}
        )+
        use $crate::*;

        $crate::mk_types_enum!($dt, $($type_name => $type_name),+);

        $crate::mk_op_structs!([] $($op_name($($arg),+) => $ret),*);

        #[derive(Copy, Clone)]
        pub enum $it {
            $($op_name,)*
            $($($ext_variant,)+)?
        }

        impl $crate::IType for $it {}

        $(
            impl From<$op_name> for $it {
                fn from(_: $op_name) -> $it { $it::$op_name }
            }
        )*

        $($(
            impl<$($ext_gen)*> From<$ext_ty> for $it {
                fn from(_: $ext_ty) -> $it { $it::$ext_variant }
            }
        )+)?

        $crate::impl_theory_types!($th, $($type_name),+);
        $crate::impl_theory_ops!($th, $($op_name),*);

        $($(
            impl<$($ext_gen)*> $crate::TheoryOperation<$th> for $ext_ty {}
        )+)?

        impl $crate::Theory for $th {
            type DT = $dt;
            type IT = $it;
        }
    };
    // Generic, default names
    (
        [$($gen:tt)*]
        Types($($variant:ident => $ty:ty),+),
        $($op_name:ident($($arg:ty),+) => $ret:ty),* $(,)?
        $(; $([$($ext_gen:tt)*] $ext_variant:ident => $ext_ty:ty),+ $(,)?)?
    ) => {
        $crate::mk_theory!(
            [$($gen)*]
            {Types, Operations, Theory}
            Types($($variant => $ty),+),
            $($op_name($($arg),+) => $ret),*
            $(; $([$($ext_gen)*] $ext_variant => $ext_ty),+)?
        );
    };
    // Simple, default names
    (
        Types($($type_name:ident),+),
        $($op_name:ident($($arg:ident),+) => $ret:ident),* $(,)?
        $(; $([$($ext_gen:tt)*] $ext_variant:ident => $ext_ty:ty),+ $(,)?)?
    ) => {
        $crate::mk_theory!(
            {Types, Operations, Theory}
            Types($($type_name),+),
            $($op_name($($arg),+) => $ret),*
            $(; $([$($ext_gen)*] $ext_variant => $ext_ty),+)?
        );
    };
}

/// Like [`mk_theory!`], but wraps everything in a `pub mod`.
///
/// An optional `{DType, IType, Theory}` prefix names the generated
/// enums and struct (defaults to `Types`, `Operations`, `Theory`).
///
/// ```ignore
/// mk_theory_mod!(
///     nat, Types(Nat),
///     Add(Nat, Nat) => Nat,
///     Id(Nat)       => Nat
/// );
///
/// // With custom names:
/// mk_theory_mod!({NatTypes, NatOps, NatTh}
///     nat, Types(Nat),
///     Add(Nat, Nat) => Nat,
///     Id(Nat)       => Nat
/// );
///
/// // With generics:
/// mk_theory_mod!([const N: usize]
///     bv, Types(BV => BV<N>),
///     Add(BV<N>, BV<N>) => BV<N>
/// );
///
/// // With generics + custom names:
/// mk_theory_mod!([const N: usize] {BvTypes, BvOps, BvTh}
///     bv, Types(BV => BV<N>),
///     Add(BV<N>, BV<N>) => BV<N>
/// );
/// ```
#[macro_export]
macro_rules! mk_theory_mod {
    // Generic + custom names
    (
        [$($gen:tt)*]
        {$dt:ident, $it:ident, $th:ident}
        $mod_name:ident, Types($($variant:ident => $ty:ty),+),
        $($op_name:ident($($arg:ty),+) => $ret:ty),* $(,)?
        $(; $([$($ext_gen:tt)*] $ext_variant:ident => $ext_ty:ty),+ $(,)?)?
    ) => {
        pub mod $mod_name {
            use super::*;

            $crate::mk_theory!(
                [$($gen)*]
                {$dt, $it, $th}
                Types($($variant => $ty),+),
                $($op_name($($arg),+) => $ret),*
                $(; $([$($ext_gen)*] $ext_variant => $ext_ty),+)?
            );
        }
    };
    // Simple + custom names
    (
        {$dt:ident, $it:ident, $th:ident}
        $mod_name:ident, Types($($type_name:ident),+),
        $($op_name:ident($($arg:ident),+) => $ret:ident),* $(,)?
        $(; $([$($ext_gen:tt)*] $ext_variant:ident => $ext_ty:ty),+ $(,)?)?
    ) => {
        pub mod $mod_name {
            $crate::mk_theory!(
                {$dt, $it, $th}
                Types($($type_name),+),
                $($op_name($($arg),+) => $ret),*
                $(; $([$($ext_gen)*] $ext_variant => $ext_ty),+)?
            );
        }
    };
    // Generic, default names
    (
        [$($gen:tt)*]
        $mod_name:ident, Types($($variant:ident => $ty:ty),+),
        $($op_name:ident($($arg:ty),+) => $ret:ty),* $(,)?
        $(; $([$($ext_gen:tt)*] $ext_variant:ident => $ext_ty:ty),+ $(,)?)?
    ) => {
        pub mod $mod_name {
            use super::*;

            $crate::mk_theory!([$($gen)*]
                Types($($variant => $ty),+),
                $($op_name($($arg),+) => $ret),*
                $(; $([$($ext_gen)*] $ext_variant => $ext_ty),+)?
            );
        }
    };
    // Simple, default names
    (
        $mod_name:ident, Types($($type_name:ident),+),
        $($op_name:ident($($arg:ident),+) => $ret:ident),* $(,)?
        $(; $([$($ext_gen:tt)*] $ext_variant:ident => $ext_ty:ty),+ $(,)?)?
    ) => {
        pub mod $mod_name {
            $crate::mk_theory!(
                Types($($type_name),+),
                $($op_name($($arg),+) => $ret),*
                $(; $([$($ext_gen)*] $ext_variant => $ext_ty),+)?
            );
        }
    };
}
