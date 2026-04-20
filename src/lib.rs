use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray1};
use ndarray::{Array2};
use std::f64::consts::PI;
use num_complex::Complex;

/// Compute the Fourier transform of a 2D Gaussian for convolution.
///
/// # Arguments
///
/// * `bmin_in` - Intrinsic psf BMIN (degrees)
/// * `bmaj_in` - Intrinsic psf BMAJ (degrees) 
/// * `bpa_in` - Intrinsic psf BPA (degrees)
/// * `bmin` - Final psf BMIN (degrees)
/// * `bmaj` - Final psf BMAJ (degrees)
/// * `bpa` - Final psf BPA (degrees)
/// * `u` - Fourier coordinates corresponding to image coord x
/// * `v` - Fourier coordinates corresponding to image coord y
///
/// # Returns
///
/// A tuple containing:
/// * `g_final` - Final array to be multiplied to FT(image) for convolution in the FT domain
/// * `g_ratio` - Factor for flux scaling
#[pyfunction]
fn gaussft<'py>(
    py: Python<'py>,
    bmin_in: f64,
    bmaj_in: f64,
    bpa_in: f64,
    bmin: f64,
    bmaj: f64,
    bpa: f64,
    u: PyReadonlyArray1<Complex<f32>>,
    v: PyReadonlyArray1<Complex<f32>>,
) -> PyResult<(Bound<'py, PyArray2<Complex<f32>>>, f64)> {
    let deg2rad = PI / 180.0;
    
    // Convert to radians
    let bmaj_in_rad = bmaj_in * deg2rad;
    let bmin_in_rad = bmin_in * deg2rad;
    let bpa_in_rad = bpa_in * deg2rad;
    let bmaj_rad = bmaj * deg2rad;
    let bmin_rad = bmin * deg2rad;
    let bpa_rad = bpa * deg2rad;
    
    // Calculate standard deviations
    let sqrt_2ln2 = (2.0 * 2.0_f64.ln()).sqrt();
    let sx = bmaj_rad / (2.0 * sqrt_2ln2);
    let sy = bmin_rad / (2.0 * sqrt_2ln2);
    let sx_in = bmaj_in_rad / (2.0 * sqrt_2ln2);
    let sy_in = bmin_in_rad / (2.0 * sqrt_2ln2);
    
    // Get array views
    let u_view = u.as_array();
    let v_view = v.as_array();
    
    // Pre-calculate trigonometric values
    let cos_bpa = bpa_rad.cos();
    let sin_bpa = bpa_rad.sin();
    let cos_bpa_in = bpa_in_rad.cos();
    let sin_bpa_in = bpa_in_rad.sin();
    
    // Calculate amplitudes
    let g_amp = (2.0 * PI * sx * sy).sqrt();
    let dg_amp = (2.0 * PI * sx_in * sy_in).sqrt();
    let g_ratio = g_amp / dg_amp;
    
    // Create output array
    let mut g_final = Array2::<Complex<f32>>::zeros((u_view.len(), v_view.len()));
    
    // Calculate constants
    let pi_sq_2 = -2.0 * PI * PI;
    let sx_sq = sx * sx;
    let sy_sq = sy * sy;
    let sx_in_sq = sx_in * sx_in;
    let sy_in_sq = sy_in * sy_in;

    let sx_sq = sx_sq as f32;
    let sy_sq = sy_sq as f32;
    let sx_in_sq = sx_in_sq as f32; 
    let sy_in_sq = sy_in_sq as f32;
        
    // Vectorized calculation
    for (i, &u_val) in u_view.iter().enumerate() {
        // Pre-calculate u-dependent values
        let u_cosbpa = u_val * (cos_bpa as f32);
        let u_sinbpa = u_val * (sin_bpa as f32);
        let u_cosbpa_in = u_val * (cos_bpa_in as f32);
        let u_sinbpa_in = u_val * (sin_bpa_in as f32);
        
        for (j, &v_val) in v_view.iter().enumerate() {
            // Pre-calculate v-dependent values
            let v_cosbpa = v_val * (cos_bpa as f32);
            let v_sinbpa = v_val * (sin_bpa as f32);
            let v_cosbpa_in = v_val * (cos_bpa_in as f32);
            let v_sinbpa_in = v_val * (sin_bpa_in as f32);
            
            // Calculate rotated coordinates
            let ur = u_cosbpa - v_sinbpa;
            let vr = u_sinbpa + v_cosbpa;
            let ur_in = u_cosbpa_in - v_sinbpa_in;
            let vr_in = u_sinbpa_in + v_cosbpa_in;
            
            // Calculate arguments for exponential
            let g_arg = (pi_sq_2 as f32) * (sx_sq * ur * ur + sy_sq * vr * vr);
            let dg_arg = (pi_sq_2 as f32) * (sx_in_sq * ur_in * ur_in + sy_in_sq * vr_in * vr_in);
            
            // Calculate final value
            g_final[[i, j]] = (g_ratio as f32) * (g_arg - dg_arg).exp();
        }
    }
    
    // Convert to PyArray2 and return
    let g_final_py = PyArray2::from_array(py, &g_final);
    Ok((g_final_py, g_ratio))
}

/// A Python module implemented in Rust.
#[pymodule]
fn gaussft_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gaussft, m)?)?;
    Ok(())
}