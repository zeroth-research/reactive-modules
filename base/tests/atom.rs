// #[test]
// fn can_instantiate_atom_from_module_wire() {
//     let x = Wire::one(0, "real");
//     let y = Wire::one(1, "real");
//     let z = Wire::one(2, "real");
//     let y0 = Wire::one(3, "real_nneg");
//     let z0 = Wire::one(4, "real_nneg");
//
//     let wait = Wire::union(&y0, &z0).unwrap().twin(5).unwrap();
//     let read = x.union(&y).unwrap().union(&z).unwrap();
//     let ctrl = read.twin(5).unwrap();
//
//     let init_term = Term::new(
//         "w[0] = 0, w[1] = a[0], w[2] = a[1]",
//         ctrl.clone(),
//         wait.clone(),
//     );
//     let update_term = Term::new("see report", ctrl.clone(), read.union(&wait).unwrap());
//
//     let latched = x
//         .union(&y)
//         .unwrap()
//         .union(&z)
//         .unwrap()
//         .union(&y0)
//         .unwrap()
//         .union(&z0)
//         .unwrap();
//     let next = latched.twin(5).unwrap();
//
//     let _atom =
        //         Atom::with_module_wire(&[latched, next], vec![init_term], vec![update_term]).unwrap();
// }

// #[test]
// fn cross_check_atom_with_module_wire_and_module_with_atoms() {
//     let x = Wire::one(0, "real");
//     let y = Wire::one(1, "real");
//     let z = Wire::one(2, "real");
//     let y0 = Wire::one(3, "real_nneg");
//     let z0 = Wire::one(4, "real_nneg");
//
//     let x_next = Wire::one(5, "real");
//     let y_next = Wire::one(6, "real");
//     let z_next = Wire::one(7, "real");
//     let y0_next = Wire::one(8, "real_nneg");
//     let z0_next = Wire::one(9, "real_nneg");
//
//     // This usage of from_iter is ugly, we need to find a better system
//     let wait = Wire::from_iter([y0_next.clone(), z0_next.clone()].into_iter().flatten()).unwrap();
//
//     let read = Wire::from_iter([x.clone(), y.clone(), z.clone()].into_iter().flatten()).unwrap();
//
//     let read_wait = Wire::from_iter(
//         [
//             x.clone(),
//             y.clone(),
//             z.clone(),
//             y0_next.clone(),
//             z0_next.clone(),
//         ]
//         .into_iter()
//         .flatten(),
//     )
//     .unwrap();
//
//     let ctrl = Wire::from_iter(
//         [x_next.clone(), y_next.clone(), z_next.clone()]
//             .into_iter()
//             .flatten(),
//     )
//     .unwrap();
//
//     let init_term = Term::new(
//         "w[0] = 0, w[1] = a[0], w[2] = a[1]",
//         ctrl.clone(),
//         wait.clone(),
//     );
//     let update_term = Term::new("see report", ctrl.clone(), read_wait.clone());
//
//     let latched = Wire::from_iter(
//         [x.clone(), y.clone(), z.clone(), y0.clone(), z0.clone()]
//             .into_iter()
//             .flatten(),
//     )
//     .unwrap();
//     let next = Wire::from_iter(
//         [
//             x_next.clone(),
//             y_next.clone(),
//             z_next.clone(),
//             y0_next.clone(),
//             z0_next.clone(),
//         ]
//         .into_iter()
//         .flatten(),
//     )
//     .unwrap();
//     let wire = [latched, next];
//
//     let atom = Atom::with_module_wire(&wire, vec![init_term], vec![update_term]).unwrap();
//
//     let _module = Module::with_atoms(wire, vec![atom]).unwrap();
// }
