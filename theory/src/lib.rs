pub mod bool;
pub mod bv;
pub mod int;
pub mod lia;
pub mod real;

pub trait Theory {
    type DType;

    fn type_check<'a, R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<&'a Self::DType>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
        Self::DType: 'a;
}

// Helpers for type-checking procedures

fn read_nxt<'a, R, D, T>(read: &mut R, i: usize) -> Result<&'a T, String>
where
    R: Iterator<Item = D>,
    D: TryInto<&'a T>,
{
    if let Some(d) = read.next() {
        d.try_into()
            .map_err(|_| format!("Read arg {i} not compatible with Theory"))
    } else {
        Err(format!("Read arg {i} expected, but got none"))
    }
}

fn write_nxt<'a, R, D, T>(write: &mut R, i: usize) -> Result<&'a T, String>
where
    R: Iterator<Item = D>,
    D: TryInto<&'a T>,
{
    if let Some(d) = write.next() {
        d.try_into()
            .map_err(|_| format!("Write arg {i} not compatible with Theory"))
    } else {
        Err(format!("Write arg {i} expected, but got none"))
    }
}
