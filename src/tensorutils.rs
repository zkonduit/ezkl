use halo2_proofs::{arithmetic::FieldExt, plonk::Expression};

pub fn map2<Inner, Func, const I: usize, const J: usize>(mut f: Func) -> Vec<Vec<Inner>>
where
    Func: FnMut(usize, usize) -> Inner,
{
    let mut a: Vec<Vec<Inner>> = Vec::new();
    for i in 0..I {
        let mut b: Vec<Inner> = Vec::new();
        for j in 0..J {
            b.push(f(i, j));
        }
        a.push(b);
    }
    a
}

pub fn map3<Inner, Func, const I: usize, const J: usize, const K: usize>(
    mut f: Func,
) -> Vec<Vec<Vec<Inner>>>
where
    Func: FnMut(usize, usize, usize) -> Inner,
{
    let mut a: Vec<Vec<Vec<Inner>>> = Vec::new();
    for i in 0..I {
        let mut b: Vec<Vec<Inner>> = Vec::new();
        for j in 0..J {
            let mut c: Vec<Inner> = Vec::new();
            for k in 0..K {
                c.push(f(i, j, k));
            }
            b.push(c);
        }
        a.push(b);
    }
    a
}

pub fn map4<Inner, Func, const I: usize, const J: usize, const K: usize, const L: usize>(
    mut f: Func,
) -> Vec<Vec<Vec<Vec<Inner>>>>
where
    Func: FnMut(usize, usize, usize, usize) -> Inner,
{
    let mut a: Vec<Vec<Vec<Vec<Inner>>>> = Vec::new();
    for i in 0..I {
        let mut b: Vec<Vec<Vec<Inner>>> = Vec::new();
        for j in 0..J {
            let mut c: Vec<Vec<Inner>> = Vec::new();
            for k in 0..K {
                let mut d: Vec<Inner> = Vec::new();
                for l in 0..L {
                    d.push(f(i, j, k, l));
                }
                c.push(d);
            }
            b.push(c);
        }
        a.push(b);
    }
    a
}

pub fn map3r<Inner, Func, Err, const I: usize, const J: usize, const K: usize>(
    mut f: Func,
) -> Result<Vec<Vec<Vec<Inner>>>, Err>
where
    Func: FnMut(usize, usize, usize) -> Result<Inner, Err>,
{
    let mut a: Vec<Vec<Vec<Inner>>> = Vec::new();
    for i in 0..I {
        let mut b: Vec<Vec<Inner>> = Vec::new();
        for j in 0..J {
            let mut c: Vec<Inner> = Vec::new();
            for k in 0..K {
                c.push(f(i, j, k)?);
            }
            b.push(c);
        }
        a.push(b);
    }
    Ok(a)
}

pub fn map4r<Inner, Func, Err, const I: usize, const J: usize, const K: usize, const L: usize>(
    mut f: Func,
) -> Result<Vec<Vec<Vec<Vec<Inner>>>>, Err>
where
    Func: FnMut(usize, usize, usize, usize) -> Result<Inner, Err>,
{
    let mut a: Vec<Vec<Vec<Vec<Inner>>>> = Vec::new();
    for i in 0..I {
        let mut b: Vec<Vec<Vec<Inner>>> = Vec::new();
        for j in 0..J {
            let mut c: Vec<Vec<Inner>> = Vec::new();
            for k in 0..K {
                let mut d: Vec<Inner> = Vec::new();
                for l in 0..L {
                    d.push(f(i, j, k, l)?);
                }
                c.push(d);
            }
            b.push(c);
        }
        a.push(b);
    }
    Ok(a)
}

pub fn flatten<T>(nested: Vec<Vec<T>>) -> Vec<T> {
    nested.into_iter().flatten().collect()
}

pub fn flatten3<T>(nested: Vec<Vec<Vec<T>>>) -> Vec<T> {
    flatten(nested).into_iter().flatten().collect()
}

pub fn flatten4<T>(nested: Vec<Vec<Vec<Vec<T>>>>) -> Vec<T> {
    flatten3(nested).into_iter().flatten().collect()
}

pub fn dot3<F: FieldExt>(
    a: &[Vec<Vec<Expression<F>>>],
    b: &[Vec<Vec<Expression<F>>>],
) -> Expression<F> {
    let aflat = a.iter().flatten().flatten(); //flatten3(a);
    let bflat = b.iter().flatten().flatten();
    aflat
        .zip(bflat)
        .map(|(x, y)| x.clone() * y.clone())
        .fold(Expression::Constant(F::zero()), |sum, next| sum + next)
}

pub fn dot3u(a: &[Vec<Vec<u64>>], b: &[Vec<Vec<u64>>]) -> u64 {
    let aflat = a.iter().flatten().flatten(); //flatten3(a);
    let bflat = b.iter().flatten().flatten();
    aflat.zip(bflat).map(|(x, y)| x.clone() * y.clone()).sum()
}

mod test {
    use super::*;

    #[test]
    fn indices4() {
        let t: Vec<Vec<Vec<Vec<(usize, usize, usize, usize)>>>> =
            map4::<_, _, 2, 2, 2, 2>(|i, j, k, l| (i, j, k, l));
        let target = vec![
            vec![
                vec![
                    vec![(0, 0, 0, 0), (0, 0, 0, 1)],
                    vec![(0, 0, 1, 0), (0, 0, 1, 1)],
                ],
                vec![
                    vec![(0, 1, 0, 0), (0, 1, 0, 1)],
                    vec![(0, 1, 1, 0), (0, 1, 1, 1)],
                ],
            ],
            vec![
                vec![
                    vec![(1, 0, 0, 0), (1, 0, 0, 1)],
                    vec![(1, 0, 1, 0), (1, 0, 1, 1)],
                ],
                vec![
                    vec![(1, 1, 0, 0), (1, 1, 0, 1)],
                    vec![(1, 1, 1, 0), (1, 1, 1, 1)],
                ],
            ],
        ];
        assert_eq!(t, target);
        assert_eq!(
            flatten4(t),
            vec![
                (0, 0, 0, 0),
                (0, 0, 0, 1),
                (0, 0, 1, 0),
                (0, 0, 1, 1),
                (0, 1, 0, 0),
                (0, 1, 0, 1),
                (0, 1, 1, 0),
                (0, 1, 1, 1),
                (1, 0, 0, 0),
                (1, 0, 0, 1),
                (1, 0, 1, 0),
                (1, 0, 1, 1),
                (1, 1, 0, 0),
                (1, 1, 0, 1),
                (1, 1, 1, 0),
                (1, 1, 1, 1)
            ]
        );
    }

    #[test]
    fn indices3() {
        let t: Vec<Vec<Vec<(usize, usize, usize)>>> = map3::<_, _, 2, 2, 2>(|i, j, k| (i, j, k));
        let target = vec![
            vec![vec![(0, 0, 0), (0, 0, 1)], vec![(0, 1, 0), (0, 1, 1)]],
            vec![vec![(1, 0, 0), (1, 0, 1)], vec![(1, 1, 0), (1, 1, 1)]],
        ];
        assert_eq!(t, target);
    }

    // #[test]
    // fn dot() {
    //     let input = vec![
    //         vec![
    //             vec![0u64, 1u64, 2u64],
    //             vec![3u64, 4u64, 5u64],
    //             vec![6u64, 7u64, 8u64],
    //         ],
    //         vec![
    //             vec![1u64, 2u64, 3u64],
    //             vec![4u64, 5u64, 6u64],
    //             vec![7u64, 8u64, 9u64],
    //         ],
    //     ];
    //     let kernels = vec![vec![
    //         vec![vec![0u64, 1u64], vec![2u64, 3u64]],
    //         vec![vec![1u64, 2u64], vec![3u64, 4u64]],
    //     ]];
    //     let output = vec![vec![vec![56u64, 72u64], vec![104u64, 120u64]]];

    //     const CHOUT: usize = 1;
    //     const CHIN: usize = 2;
    //     const OH: usize = 2;
    //     const OW: usize = 2;
    //     const KH: usize = 2;
    //     const KW: usize = 2;
    //     for filter in 0..CHOUT {
    //         let kernel = kernels[filter].clone(); //CHIN x KH x KW
    //         for row in 0..OH {
    //             for col in 0..OW {
    //                 //slice input to patch of kernel shape at this location
    //                 let patch = &input[0..CHIN][row..(row + KH)][col..(col + KW)];
    //                 let res = dot3(&patch, &kernel);
    //                 println!("{:?}", res);
    //             }
    //         }
    //     }
    // }
}
