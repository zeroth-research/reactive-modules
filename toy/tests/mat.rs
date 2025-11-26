use toy::dtype::Type;
use toy::mat::{MatVecIter, VecIter};
use toy::val::Val;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_iter() {
        let mut it = MatVecIter::new("MatInt([1,2], [3,1])");
        assert!(it.next() == Some("[1,2]"));
        assert!(it.next() == Some("[3,1]"));
        assert!(it.next().is_none());

        let mut it = MatVecIter::new("MatReal ( [1,2]; [3,1])");
        assert!(it.next() == Some("[1,2]"));
        assert!(it.next() == Some("[3,1]"));
        assert!(it.next().is_none());

        let mut it = MatVecIter::new("MatInt( [1,2] [3, 1 ] ) ");
        assert!(it.next() == Some("[1,2]"));
        assert!(it.next() == Some("[3, 1 ]"));
        assert!(it.next().is_none());

        let mut it = MatVecIter::new("MatInt( [1,2] [3, 1 ], [3,  4,5] ; [3, 12] ) ");
        assert!(it.next() == Some("[1,2]"));
        assert!(it.next() == Some("[3, 1 ]"));
        assert!(it.next() == Some("[3,  4,5]"));
        assert!(it.next() == Some("[3, 12]"));
        assert!(it.next().is_none());

        let mut it = MatVecIter::new("MatInt( [1 2] [3  1 ], [3   4  5] ; [3  12] ) ");
        assert!(it.next() == Some("[1 2]"));
        assert!(it.next() == Some("[3  1 ]"));
        assert!(it.next() == Some("[3   4  5]"));
        assert!(it.next() == Some("[3  12]"));
        assert!(it.next().is_none());

        let mut it = MatVecIter::new("MatInt( [1,2] [3, [1 ] ) ");
        assert!(it.next() == Some("[1,2]"));
        assert!(it.next().is_none());

        let mut it = MatVecIter::new("MatInt( [1,2] [3, 1] [ [3,2]) ");
        assert!(it.next() == Some("[1,2]"));
        assert!(it.next() == Some("[3, 1]"));
        assert!(it.next().is_none());

        let mut it = MatVecIter::new("MatInt([[1,2] [3, 1] [3,2]]) ");
        assert!(it.next().is_none());
    }

    #[test]
    fn vec_iter() {
        let mut it = VecIter::<i32>::new("[1,2]");
        assert!(it.next() == Some(1));
        assert!(it.next() == Some(2));
        assert!(it.next().is_none());

        let mut it = VecIter::<i32>::new(" [ 1,  2] ");
        assert!(it.next() == Some(1));
        assert!(it.next() == Some(2));
        assert!(it.next().is_none());

        let mut it = VecIter::<i32>::new(" [ 1  ,2 ] ");
        assert!(it.next() == Some(1));
        assert!(it.next() == Some(2));
        assert!(it.next().is_none());

        let mut it = VecIter::<i32>::new(" [ 1  2 ] ");
        assert!(it.next() == Some(1));
        assert!(it.next() == Some(2));
        assert!(it.next().is_none());
    }

    #[test]
    fn vec_iter_space() {
        let mut it = VecIter::<i32>::new("[1 2]");
        assert!(it.next() == Some(1));
        assert!(it.next() == Some(2));
        assert!(it.next().is_none());

        let mut it = VecIter::<i32>::new(" [ 1   2] ");
        assert!(it.next() == Some(1));
        assert!(it.next() == Some(2));
        assert!(it.next().is_none());

        let mut it = VecIter::<i32>::new(" [ 1  ,2 ] ");
        assert!(it.next() == Some(1));
        assert!(it.next() == Some(2));
        assert!(it.next().is_none());

        let mut it = VecIter::<i32>::new(" [ 1  2 ] ");
        assert!(it.next() == Some(1));
        assert!(it.next() == Some(2));
        assert!(it.next().is_none());

        let mut it = VecIter::<i32>::new(" [ 1,  2 ] ");
        assert!(it.next() == Some(1));
        assert!(it.next() == Some(2));
        assert!(it.next().is_none());

        let mut it = VecIter::<i32>::new(" [ 1 #  2 ] ");
        assert!(it.next() == Some(1));
        assert!(it.next().is_none());

        let mut it = VecIter::<i32>::new(" [ 1#  2 ] ");
        assert!(it.next().is_none());
    }

    #[test]
    fn vec_iter_f() {
        let mut it = VecIter::<f32>::new("[1.3 2]");
        assert!(it.next() == Some(1.3));
        assert!(it.next() == Some(2.0));
        assert!(it.next().is_none());

        let mut it = VecIter::<f32>::new(" [ 1.23   2.0] ");
        assert!(it.next() == Some(1.23));
        assert!(it.next() == Some(2.0));
        assert!(it.next().is_none());

        let mut it = VecIter::<f32>::new(" [ 1  ,2.22 ] ");
        assert!(it.next() == Some(1.0));
        assert!(it.next() == Some(2.22));
        assert!(it.next().is_none());

        let mut it = VecIter::<f32>::new(" [ 1.  2 ] ");
        assert!(it.next() == Some(1.0));
        assert!(it.next() == Some(2.0));
        assert!(it.next().is_none());

        let mut it = VecIter::<f32>::new(" [ 1.1   2.3.2 ] ");
        assert!(it.next() == Some(1.1));
        assert!(it.next().is_none());

        let mut it = VecIter::<i32>::new(" [ 1.1#  2 ] ");
        assert!(it.next().is_none());
    }

    #[test]
    fn mat_val() {
        let mat = Val::from_str("MatInt([1, 2] [2, 3] [3, 4])", Type::MatInt(3, 2));
        assert!(mat == Some(Val::MatInt(vec![vec![1, 2], vec![2, 3], vec![3, 4]])));

        let mat = Val::from_str("MatInt([1 2 5] [2, 3 5] [3, 4, 5])", Type::MatInt(3, 3));
        assert!(
            mat == Some(Val::MatInt(vec![
                vec![1, 2, 5],
                vec![2, 3, 5],
                vec![3, 4, 5]
            ]))
        );

        let mat = Val::from_str("MatInt([1 2 5] [2, 3; 5] [3, 4, 5])", Type::MatInt(3, 3));
        assert!(mat.is_none());

        let mat = Val::from_str(
            "MatReal([1 2.3 5] [2.1, 3 5] [3, 4.4, 5.4])",
            Type::MatReal(3, 3),
        );
        assert!(
            mat == Some(Val::MatReal(vec![
                vec![1., 2.3, 5.],
                vec![2.1, 3., 5.],
                vec![3., 4.4, 5.4]
            ]))
        );

        let mat = Val::from_str("MatReal([1 2 5] [2, 3; 5] [3, 4, 5])", Type::MatReal(3, 3));
        assert!(mat.is_none())
    }
}
