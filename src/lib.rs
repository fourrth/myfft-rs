pub use biquad::{Hertz, ToHertz};
pub use realfft::num_complex;

use biquad::{Biquad, Coefficients, DirectForm1, Errors, Q_BUTTERWORTH_F64};

use realfft::{RealFftPlanner, num_complex::Complex, num_traits};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FFTError {
    #[error("Invalid length of vec, len: {0}")]
    InvalidLength(usize),
    #[error("Alpha must be 0.0..=1.0")]
    InvalidAlpha,
    #[error("{0:?}")]
    BiquadFilterError(Errors),
    #[error("{0:?}")]
    ProcessingError(realfft::FftError),
}
pub trait Float: num_traits::Float + realfft::FftNum + num_traits::ConstZero {}
impl Float for f32 {}
impl Float for f64 {}

fn exp_smooth<T: Float>(mut data: Vec<T>, alpha: T) -> Result<Vec<T>, FFTError> {
    if data.len() < 2 {
        return Err(FFTError::InvalidLength(data.len()));
    }

    if !(T::zero()..T::one()).contains(&alpha) {
        return Err(FFTError::InvalidAlpha);
    }

    let mut prev = data[0];

    for cx in 1..data.len() {
        data[cx - 1] = prev;
        prev = alpha * data[cx] + (T::one() - alpha) * prev;
    }
    Ok(data)
}

// get fft
// scratch can be an empty vec
pub fn spectrum<'a, T: Float>(
    mut data: Vec<T>,
    sample_rate: Hertz<T>,
    scratch: &'a mut Vec<Complex<T>>,
    exp_smoothing: Option<T>,
    low_pass_filter: Option<Hertz<T>>,
) -> Result<(Vec<T>, Vec<T>), FFTError> {
    let n = data.len();

    //TODO: error handling
    if let Some(v) = low_pass_filter {
        let mut filter =
            // always safe
            match Coefficients::from_params(biquad::Type::LowPass, sample_rate, v, unsafe{T::from(Q_BUTTERWORTH_F64).unwrap_unchecked()}) {
                Ok(val) => DirectForm1::new(val),
                Err(e) => {
                    // wish biquad implemented std error, maybe behind a feature or something...
                    return Err(FFTError::BiquadFilterError(e));
                }
            };

        for cx in 0..n {
            data[cx] = filter.run(data[cx]);
        }
    }

    let mut planner = RealFftPlanner::<T>::new();
    let r2c = planner.plan_fft_forward(n);

    let mut spectrum: Vec<Complex<T>> = r2c.make_output_vec();

    scratch.clear();
    scratch.reserve_exact(r2c.get_scratch_len());

    match r2c.process_with_scratch(&mut data, &mut spectrum, scratch) {
        Ok(_) => {}
        Err(e) => return Err(FFTError::ProcessingError(e)),
    };

    // Build frequency and amplitude axes
    let mut freqs: Vec<T> = Vec::with_capacity(spectrum.len());
    let mut amps = Vec::with_capacity(spectrum.len());

    for (k, c) in spectrum.iter().enumerate() {
        // literally impossible for ToPrimitive usize to be None when converting to float
        // though maybe compiler would optimize something like that
        let freq = unsafe {
            T::from(k).unwrap_unchecked() * sample_rate.hz() / T::from(n).unwrap_unchecked()
        };

        let mut amp = c.norm(); // sqrt(re^2 + im^2)
        if k > 0 && !(n % 2 == 0 && k == n / 2) {
            // ToPrimitive float to float is always Some(_)
            amp = amp * unsafe { T::from_f32(2f32).unwrap_unchecked() };
        }

        freqs.push(freq);
        amps.push(amp);
    }
    if let Some(alpha) = exp_smoothing {
        amps = exp_smooth(amps, alpha)?;
    }
    Ok((freqs, amps))
}
