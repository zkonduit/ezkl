/*
Huge thank you to https://github.com/han0110/halo2-kzg-srs/blob/main/src/util.rs for the code in this file. We replicate the code here to keep the code compatible with latest halo2curves package.

 */

use std::io;

use halo2_proofs::arithmetic::Field;
use halo2_proofs::poly::commitment::CommitmentScheme;
use halo2_proofs::poly::commitment::ParamsProver;
use halo2curves::ff::PrimeField;
use halo2curves::group::Group;
use halo2curves::pairing::MillerLoopResult;
use halo2curves::{pairing::MultiMillerLoop, serde::SerdeObject, CurveAffine};
use rand::rngs::OsRng;

/// for now we use the urls of the powers of tau ceremony from https://github.com/han0110/halo2-kzg-srs
pub const PUBLIC_SRS_URL: &str =
    "https://trusted-setup-halo2kzg.s3.eu-central-1.amazonaws.com/perpetual-powers-of-tau-raw-";

/// URL for the G2 elements in the SRS
pub const PUBLIC_G2_URL: &str = "https://ppot.blob.core.windows.net/public/response_0071_edward";

/// Offset for the G1 elements in the SRS
pub const G1_OFFSET: u64 = 4;

/// Helper function for generating SRS. Only use for testing
pub fn gen_srs<Scheme: CommitmentScheme>(k: u32) -> Scheme::ParamsProver {
    Scheme::ParamsProver::new(k)
}

pub(crate) fn ec_point_raw_size<C: CurveAffine + SerdeObject>() -> usize {
    let mut buf = Vec::new();
    C::default().write_raw(&mut buf).unwrap();
    buf.len()
}

pub(crate) fn ec_point_repr_size<C: CurveAffine>() -> usize {
    C::Repr::default().as_ref().len()
}

pub(crate) fn g2_offset<M: MultiMillerLoop>(k: u32) -> u64 {
    let g1_size = ec_point_repr_size::<M::G1Affine>() as u64;
    G1_OFFSET + g1_size * (2 * (1 << k) - 1)
}

/// This simple utility function will parallelize an operation that is to be
/// performed over a mutable slice.
pub(crate) fn parallelize<T: Send, F: Fn(&mut [T], usize) + Send + Sync + Clone>(
    v: &mut [T],
    f: F,
) {
    let n = v.len();
    let num_threads = rayon::current_num_threads();
    let mut chunk = n / num_threads;
    if chunk < num_threads {
        chunk = n;
    }

    rayon::scope(|scope| {
        for (chunk_num, v) in v.chunks_mut(chunk).enumerate() {
            let f = f.clone();
            scope.spawn(move |_| {
                let start = chunk_num * chunk;
                f(v, start);
            });
        }
    });
}

fn read_ec_points<C: CurveAffine, R: io::Read + io::Seek>(reader: &mut R, n: usize) -> Vec<C> {
    let mut reprs = vec![C::Repr::default(); n];
    for repr in reprs.iter_mut() {
        reader.read_exact(repr.as_mut()).unwrap();
        repr.as_mut().reverse();
    }

    let mut points = vec![C::default(); n];
    parallelize(&mut points, |points, start| {
        for (i, point) in points.iter_mut().enumerate() {
            let candidate = C::from_bytes(&reprs[start + i]).unwrap();
            let minus_candidate = -candidate;

            *point = if (candidate.coordinates().unwrap().y()
                < minus_candidate.coordinates().unwrap().y())
                ^ ((reprs[start + i].as_ref().last().unwrap() & 0b1000_0000) != 0)
            {
                candidate
            } else {
                minus_candidate
            }
        }
    });
    points
}

fn seek_g2_offset<M: MultiMillerLoop, R: io::Read + io::Seek>(reader: &mut R, k: u32) {
    let offset = g2_offset::<M>(k);
    reader.seek(io::SeekFrom::Start(offset)).unwrap();
}

pub(crate) fn read_g2s<M: MultiMillerLoop, R: io::Read + io::Seek, const IN_PLACE: bool>(
    reader: &mut R,
    k: u32,
    n: usize,
) -> Vec<M::G2Affine> {
    if !IN_PLACE {
        seek_g2_offset::<M, _>(reader, k);
    }
    read_ec_points::<M::G2Affine, _>(reader, n)
}

pub(crate) fn read_g1s<M: MultiMillerLoop, R: io::Read + io::Seek>(
    reader: &mut R,
    n: usize,
) -> Vec<M::G1Affine>
where
    M::G1Affine: SerdeObject,
{
    read_ec_points::<_, _>(reader, n)
}

fn multiexp_serial<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C], acc: &mut C::Curve) {
    let coeffs: Vec<_> = coeffs.iter().map(|a| a.to_repr()).collect();

    let c = if bases.len() < 4 {
        1
    } else if bases.len() < 32 {
        3
    } else {
        (f64::from(bases.len() as u32)).ln().ceil() as usize
    };

    fn get_at<F: PrimeField>(segment: usize, c: usize, bytes: &F::Repr) -> usize {
        let skip_bits = segment * c;
        let skip_bytes = skip_bits / 8;

        if skip_bytes >= 32 {
            return 0;
        }

        let mut v = [0; 8];
        for (v, o) in v.iter_mut().zip(bytes.as_ref()[skip_bytes..].iter()) {
            *v = *o;
        }

        let mut tmp = u64::from_le_bytes(v);
        tmp >>= skip_bits - (skip_bytes * 8);
        tmp %= 1 << c;

        tmp as usize
    }

    let segments = (256 / c) + 1;

    for current_segment in (0..segments).rev() {
        for _ in 0..c {
            *acc = acc.double();
        }

        #[derive(Clone, Copy)]
        enum Bucket<C: CurveAffine> {
            None,
            Affine(C),
            Projective(C::Curve),
        }

        impl<C: CurveAffine> Bucket<C> {
            fn add_assign(&mut self, other: &C) {
                *self = match *self {
                    Bucket::None => Bucket::Affine(*other),
                    Bucket::Affine(a) => Bucket::Projective(a + *other),
                    Bucket::Projective(mut a) => {
                        a += *other;
                        Bucket::Projective(a)
                    }
                }
            }

            fn add(self, mut other: C::Curve) -> C::Curve {
                match self {
                    Bucket::None => other,
                    Bucket::Affine(a) => {
                        other += a;
                        other
                    }
                    Bucket::Projective(a) => other + a,
                }
            }
        }

        let mut buckets: Vec<Bucket<C>> = vec![Bucket::None; (1 << c) - 1];

        for (coeff, base) in coeffs.iter().zip(bases.iter()) {
            let coeff = get_at::<C::Scalar>(current_segment, c, coeff);
            if coeff != 0 {
                buckets[coeff - 1].add_assign(base);
            }
        }

        // Summation by parts
        // e.g. 3a + 2b + 1c = a +
        //                    (a) + b +
        //                    ((a) + b) + c
        let mut running_sum = C::Curve::identity();
        for exp in buckets.into_iter().rev() {
            running_sum = exp.add(running_sum);
            *acc += &running_sum;
        }
    }
}

/// Performs a multi-exponentiation operation.
///
/// This function will panic if coeffs and bases have a different length.
///
/// This will use multithreading if beneficial.
pub(crate) fn best_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    assert_eq!(coeffs.len(), bases.len());

    let num_threads = rayon::current_num_threads();
    if coeffs.len() > num_threads {
        let chunk = coeffs.len() / num_threads;
        let num_chunks = coeffs.chunks(chunk).len();
        let mut results = vec![C::Curve::identity(); num_chunks];
        rayon::scope(|scope| {
            let chunk = coeffs.len() / num_threads;

            for ((coeffs, bases), acc) in coeffs
                .chunks(chunk)
                .zip(bases.chunks(chunk))
                .zip(results.iter_mut())
            {
                scope.spawn(move |_| {
                    multiexp_serial(coeffs, bases, acc);
                });
            }
        });
        results.iter().fold(C::Curve::identity(), |a, b| a + b)
    } else {
        let mut acc = C::Curve::identity();
        multiexp_serial(coeffs, bases, &mut acc);
        acc
    }
}

pub(crate) fn same_ratio<M: MultiMillerLoop>(
    g1s: &[M::G1Affine],
    g2: M::G2Affine,
    s_g2: M::G2Affine,
) -> bool {
    let coeffs = std::iter::repeat_with(|| M::Scalar::random(OsRng))
        .take(g1s.len() - 1)
        .collect::<Vec<_>>();

    let lhs = best_multiexp(&coeffs, &g1s[..g1s.len() - 1]);
    let rhs = best_multiexp(&coeffs, &g1s[1..g1s.len()]);

    M::multi_miller_loop(&[(&lhs.into(), &s_g2.into()), (&rhs.into(), &(-g2).into())])
        .final_exponentiation()
        .is_identity()
        .into()
}
