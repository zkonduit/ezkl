fn map3<Inner, Func, const I: usize, const J: usize, const K: usize>(
    f: Func,
) -> Vec<Vec<Vec<Inner>>>
where
    Func: Fn(usize, usize, usize) -> Inner,
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

fn map4<Inner, Func, const I: usize, const J: usize, const K: usize, const L: usize>(
    f: Func,
) -> Vec<Vec<Vec<Vec<Inner>>>>
where
    Func: Fn(usize, usize, usize, usize) -> Inner,
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

fn flatten<T>(nested: Vec<Vec<T>>) -> Vec<T> {
    nested.into_iter().flatten().collect()
}

fn flatten3<T>(nested: Vec<Vec<Vec<T>>>) -> Vec<T> {
    flatten(nested).into_iter().flatten().collect()
}

fn flatten4<T>(nested: Vec<Vec<Vec<Vec<T>>>>) -> Vec<T> {
    flatten3(nested).into_iter().flatten().collect()
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
}
