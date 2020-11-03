# RACS-tools
Useful scripts for RACS

## Installation
First `numpy` is required before running `pip install`.

```bash
conda install numpy
# or
pip install numpy
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install RACS-tools.


```bash
pip install RACS-tools
```

## Usage

```
$ beamcon_2D -h
usage: beamcon_2D [-h] [-p PREFIX] [-s SUFFIX] [-o OUTDIR]
                  [--conv_mode CONV_MODE] [-v] [-d] [--bmaj BMAJ]
                  [--bmin BMIN] [--bpa BPA] [--log LOG] [-c CUTOFF]
                  [-t TOLERANCE] [-e EPSILON] [-n NSAMPS]
                  [--ncores N_CORES | --mpi]
                  infile [infile ...]

    Smooth a field of 2D images to a common resolution.

    Names of output files are 'infile'.sm.fits

    

positional arguments:
  infile                Input FITS image(s) to smooth (can be a wildcard) - beam info must be in header.

optional arguments:
  -h, --help            show this help message and exit
  -p PREFIX, --prefix PREFIX
                        Add prefix to output filenames.
  -s SUFFIX, --suffix SUFFIX
                        Add suffix to output filenames [...sm.fits].
  -o OUTDIR, --outdir OUTDIR
                        Output directory of smoothed FITS image(s) [same as input file].
  --conv_mode CONV_MODE
                        Which method to use for convolution [robust].
                                'robust' uses the built-in, FFT-based method.
                                Can also be 'scipy', 'astropy', or 'astropy_fft'.
                                Note these other methods cannot cope well with small convolving beams.
                                
  -v, --verbose         verbose output [False].
  -d, --dryrun          Compute common beam and stop [False].
  --bmaj BMAJ           Target BMAJ (arcsec) to convolve to [None].
  --bmin BMIN           Target BMIN (arcsec) to convolve to [None].
  --bpa BPA             Target BPA (deg) to convolve to [None].
  --log LOG             Name of beamlog file. If provided, save beamlog data to a file [None - not saved].
  -c CUTOFF, --cutoff CUTOFF
                        Cutoff BMAJ value (arcsec) -- Blank channels with BMAJ larger than this [None -- no limit]
  -t TOLERANCE, --tolerance TOLERANCE
                        tolerance for radio_beam.commonbeam.
  -e EPSILON, --epsilon EPSILON
                        epsilon for radio_beam.commonbeam.
  -n NSAMPS, --nsamps NSAMPS
                        nsamps for radio_beam.commonbeam.
  --ncores N_CORES      Number of processes (uses multiprocessing).
  --mpi                 Run with MPI.
```

```
$ beamcon_3D -h
usage: beamcon_3D [-h] [--uselogs] [--mode MODE] [--conv_mode CONV_MODE] [-v]
                  [-d] [-p PREFIX] [-s SUFFIX] [-o OUTDIR] [--bmaj BMAJ]
                  [--bmin BMIN] [--bpa BPA] [-c CUTOFF] [-t TOLERANCE]
                  [-e EPSILON] [-n NSAMPS]
                  infile [infile ...]

    Smooth a field of 3D cubes to a common resolution.

    Names of output files are 'infile'.sm.fits

    

positional arguments:
  infile                Input FITS image(s) to smooth (can be a wildcard) 
                                - beam info must be in co-located beamlog files.
                                

optional arguments:
  -h, --help            show this help message and exit
  --uselogs             Get convolving information from previous run [False].
  --mode MODE           Common resolution mode [natural]. 
                                natural  -- allow frequency variation.
                                total -- smooth all plans to a common resolution.
                                
  --conv_mode CONV_MODE
                        Which method to use for convolution [robust].
                                'robust' computes the analytic FT of the convolving Gaussian.
                                Can also be 'scipy', 'astropy', or 'astropy_fft'.
                                Note these other methods cannot cope well with small convolving beams.
                                
  -v, --verbose         verbose output [False].
  -d, --dryrun          Compute common beam and stop [False].
  -p PREFIX, --prefix PREFIX
                        Add prefix to output filenames.
  -s SUFFIX, --suffix SUFFIX
                        Add suffix to output filenames [...{mode}.fits].
  -o OUTDIR, --outdir OUTDIR
                        Output directory of smoothed FITS image(s) [None - same as input].
  --bmaj BMAJ           BMAJ to convolve to [max BMAJ from given image(s)].
  --bmin BMIN           BMIN to convolve to [max BMAJ from given image(s)].
  --bpa BPA             BPA to convolve to [0].
  -c CUTOFF, --cutoff CUTOFF
                        Cutoff BMAJ value (arcsec) -- Blank channels with BMAJ larger than this [None -- no limit]
  -t TOLERANCE, --tolerance TOLERANCE
                        tolerance for radio_beam.commonbeam.
  -e EPSILON, --epsilon EPSILON
                        epsilon for radio_beam.commonbeam.
  -n NSAMPS, --nsamps NSAMPS
                        nsamps for radio_beam.commonbeam.
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause)