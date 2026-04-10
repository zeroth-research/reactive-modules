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
    // Shared generics for all types
    ($gen:tt $theory:ident, $($ty:ty),+) => {
        $($crate::impl_theory_type!($gen $theory, $ty);)+
    };
    // Per-type generics
    ($theory:ident, $([$($gen:tt)*] $ty:ty),+) => {
        $(
            impl<$($gen)*> $crate::TheoryType<$theory> for $ty {}
        )+
    };
    // No generics
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
/// mk_types_enum!([const N: usize] BV => BV<N>);            // shared generic
/// mk_types_enum!(MyTypes, [const N: usize] BV => BV<N>);   // shared generic + name
/// // Per-type generics (each entry carries its own [gen]):
/// mk_types_enum!(MyTypes,
///     [] Bool => Bool,
///     [const N: usize] BV => BV<N>,
/// );
/// ```
#[macro_export]
macro_rules! mk_types_enum {
    // Core: custom name + shared generics
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
    // Custom name, per-type generics
    ($name:ident, $([$($gen:tt)*] $variant:ident => $ty:ty),+ $(,)?) => {
        #[derive(Copy, Clone)]
        pub enum $name {
            $($variant,)+
        }

        impl DType for $name {}

        $(
            $crate::impl_from_type!($name, [$($gen)*] $variant => $ty);
        )+
    };
    // Default name (Types) + shared generics
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

/// Helper macro (TT muncher) for processing operation groups in [`mk_theory!`].
///
/// Not intended for direct use. Each `{ }` group is either:
/// - **Fresh ops**: `{ [gen] Op(Arg, ...) => Ret, ... }` — creates structs + impls
/// - **Existing ops**: `{ [gen] Variant => Type, ... }` — wires pre-defined types
#[macro_export]
macro_rules! mk_theory_ops {
    // --- Collect: fresh ops with [gen] ---
    (@collect
        [$th:ident, $dt:ident, $it:ident]
        [enum_variants: $($ev:ident,)*]
        [groups: $($saved:tt)*]
        {
            [$($gen:tt)*]
            $($op:ident($($arg:ty),+) => $ret:ty),+ $(,)?
        }
        $($rest:tt)*
    ) => {
        $crate::mk_theory_ops!(@collect
            [$th, $dt, $it]
            [enum_variants: $($ev,)* $($op,)+]
            [groups: $($saved)* {fresh [$($gen)*] $($op($($arg),+) => $ret),+}]
            $($rest)*
        );
    };
    // --- Collect: existing ops with [gen] ---
    (@collect
        [$th:ident, $dt:ident, $it:ident]
        [enum_variants: $($ev:ident,)*]
        [groups: $($saved:tt)*]
        {
            [$($gen:tt)*]
            $($variant:ident => $ty:ty),+ $(,)?
        }
        $($rest:tt)*
    ) => {
        $crate::mk_theory_ops!(@collect
            [$th, $dt, $it]
            [enum_variants: $($ev,)* $($variant,)+]
            [groups: $($saved)* {existing [$($gen)*] $($variant => $ty),+}]
            $($rest)*
        );
    };
    // --- Collect: fresh ops, no [gen] ---
    (@collect
        [$th:ident, $dt:ident, $it:ident]
        [enum_variants: $($ev:ident,)*]
        [groups: $($saved:tt)*]
        {
            $($op:ident($($arg:ty),+) => $ret:ty),+ $(,)?
        }
        $($rest:tt)*
    ) => {
        $crate::mk_theory_ops!(@collect
            [$th, $dt, $it]
            [enum_variants: $($ev,)* $($op,)+]
            [groups: $($saved)* {fresh [] $($op($($arg),+) => $ret),+}]
            $($rest)*
        );
    };
    // --- Collect: existing ops, no [gen] ---
    (@collect
        [$th:ident, $dt:ident, $it:ident]
        [enum_variants: $($ev:ident,)*]
        [groups: $($saved:tt)*]
        {
            $($variant:ident => $ty:ty),+ $(,)?
        }
        $($rest:tt)*
    ) => {
        $crate::mk_theory_ops!(@collect
            [$th, $dt, $it]
            [enum_variants: $($ev,)* $($variant,)+]
            [groups: $($saved)* {existing [] $($variant => $ty),+}]
            $($rest)*
        );
    };
    // --- Base: all groups collected, emit everything ---
    (@collect
        [$th:ident, $dt:ident, $it:ident]
        [enum_variants: $($ev:ident,)*]
        [groups: $($group:tt)*]
    ) => {
        #[derive(Copy, Clone)]
        pub enum $it {
            $($ev,)*
        }
        impl $crate::IType for $it {}

        $($crate::mk_theory_ops!(@emit [$th, $it] $group);)*

        impl $crate::Theory for $th {
            type DT = $dt;
            type IT = $it;
        }
    };
    // --- Emit: fresh ops ---
    // $gen is a single tt like [const N: usize], avoiding repetition conflict with $op
    (@emit [$th:ident, $it:ident]
        {fresh $gen:tt $($op:ident($($arg:ty),+) => $ret:ty),+}
    ) => {
        $(
            $crate::mk_one_op!($gen $op($($arg),+) => $ret);
            impl From<$op> for $it {
                fn from(_: $op) -> $it { $it::$op }
            }
            impl $crate::TheoryOperation<$th> for $op {}
        )+
    };
    // --- Emit: existing ops (dispatch per-op to avoid gen/variant repetition conflict) ---
    (@emit [$th:ident, $it:ident]
        {existing $gen:tt $($variant:ident => $ty:ty),+}
    ) => {
        $(
            $crate::mk_theory_ops!(@emit_ext [$th, $it] $gen $variant => $ty);
        )+
    };
    // --- Emit: single existing op (destructures gen) ---
    (@emit_ext [$th:ident, $it:ident] [$($gen:tt)*] $variant:ident => $ty:ty) => {
        impl<$($gen)*> From<$ty> for $it {
            fn from(_: $ty) -> $it { $it::$variant }
        }
        impl<$($gen)*> $crate::TheoryOperation<$th> for $ty {}
    };
}

/// Generate types, operations, enums, and a theory struct with all
/// necessary trait impls and `From` conversions.
///
/// An optional `{DType, IType, Theory}` prefix names the generated
/// enums and struct (defaults to `Types`, `Operations`, `Theory`).
///
/// Operations are specified in `{ }` groups. Each group can carry
/// an optional `[generics]` prefix shared by all operations in that
/// group. Groups are distinguished by syntax:
///
/// - **Fresh ops**: `Op(Arg1, Arg2) => Ret` — creates a new unit struct
/// - **Existing ops**: `Variant => Type` — adds a pre-defined operation
///
/// # Form 1 — Fresh types
///
/// ```ignore
/// mk_theory!(
///     Types(Nat),
///     { Add(Nat, Nat) => Nat, Id(Nat) => Nat }
/// );
/// ```
///
/// # Form 2 — Pre-existing / generic types
///
/// Each entry uses `[generics] Variant => Type` (use `[]` for no generics).
///
/// ```ignore
/// mk_theory!(
///     Types([const N: usize] BV => BV<N>),
///     {
///         [const N: usize]
///         Add(BV<N>, BV<N>) => BV<N>
///     }
/// );
/// ```
///
/// # Multiple operation groups
///
/// ```ignore
/// mk_theory!(
///     Types([T: Type, const M: usize, const N: usize] Mat => Mat<T, M, N>),
///     {
///         [T: Type, const M: usize, const N: usize]
///         Add(Mat<T, M, N>, Mat<T, M, N>) => Mat<T, M, N>
///     }
///     {
///         [T: Type + Copy, const A: usize, const B: usize, const C: usize]
///         MatMul => MatMul<T, A, B, C>
///     }
/// );
/// ```
#[macro_export]
macro_rules! mk_theory {
    // === Core: Fresh types + custom names ===
    (
        {$dt:ident, $it:ident, $th:ident}
        Types($($type_name:ident),+ $(,)?),
        $($group:tt)*
    ) => {
        pub struct $th {}
        $(
            #[derive(Clone, Copy, PartialEq)]
            pub struct $type_name();
            impl $crate::Type for $type_name {}
        )+
        #[allow(unused)]
        use $crate::*;
        $crate::mk_types_enum!($dt, $($type_name => $type_name),+);
        $crate::impl_theory_types!($th, $($type_name),+);
        $crate::mk_theory_ops!(@collect [$th, $dt, $it] [enum_variants:] [groups:] $($group)*);
    };
    // === Core: Per-type generics + custom names ===
    (
        {$dt:ident, $it:ident, $th:ident}
        Types($([$($type_gen:tt)*] $variant:ident => $ty:ty),+ $(,)?),
        $($group:tt)*
    ) => {
        #[allow(unused)]
        use $crate::*;
        pub struct $th {}
        $crate::mk_types_enum!($dt, $([$($type_gen)*] $variant => $ty),+);
        $crate::impl_theory_types!($th, $([$($type_gen)*] $ty),+);
        $crate::mk_theory_ops!(@collect [$th, $dt, $it] [enum_variants:] [groups:] $($group)*);
    };
    // === Forwarding: Fresh types, default names ===
    (
        Types($($type_name:ident),+ $(,)?),
        $($group:tt)*
    ) => {
        $crate::mk_theory!(
            {Types, Operations, Theory}
            Types($($type_name),+),
            $($group)*
        );
    };
    // === Forwarding: Per-type generics, default names ===
    (
        Types($([$($type_gen:tt)*] $variant:ident => $ty:ty),+ $(,)?),
        $($group:tt)*
    ) => {
        $crate::mk_theory!(
            {Types, Operations, Theory}
            Types($([$($type_gen)*] $variant => $ty),+),
            $($group)*
        );
    };
}
