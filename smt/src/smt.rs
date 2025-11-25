use crate::dtype::DType;
use crate::itype::IType;

use base::module::Module;

pub fn parse_modules(modules: &[Module<DType, IType>]) {

    for module in modules {
        println!("Module: {}", module.name().unwrap_or("<unnamed>"));

        for atom in module.atoms() {
            let atom_index = module
                .atoms()
                .iter()
                .position(|a| a as *const _ == atom as *const _)
                .unwrap();
            println!(" Atom {}:", atom_index);
            
            let _init = atom.init();
            let _update = atom.update();

            // for term in init.iter() {
            //     println!("{}", term.reads());
            //     println!("{}", term.writes());
            // }

            // for term in init.iter() {
            //     println!("Init: {:#?}", term);
            // }

            // for term in update.iter() {
            //     println!("Update: {:#?}", term);
            // }
        }

    }
}
