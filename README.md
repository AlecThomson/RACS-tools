[![Docker Build and Push](https://github.com/AlecThomson/RACS-tools/actions/workflows/docker.yml/badge.svg)](https://github.com/AlecThomson/RACS-tools/actions/workflows/docker.yml) [![Python package](https://github.com/AlecThomson/RACS-tools/actions/workflows/python-package.yml/badge.svg)](https://github.com/AlecThomson/RACS-tools/actions/workflows/python-package.yml)
# RACS-tools
Useful scripts for RACS

## Installation

### Conda
The recommended way to install. First obtain `conda` from Anaconda or Miniconda. Clone this repo, build the environment, and activate:
```bash
git clone https://github.com/AlecThomson/RACS-tools
cd RACS-tools
conda env create
conda activate racs-tools
```

### Docker / Singularity
A Dockerfile is provided if you wish to build your own container. Otherwise, images are provided on [DockerHub](https://hub.docker.com/r/alecthomson/racstools). You can pull these by running e.g.
```bash
docker pull alecthomson/racstools
```
or
```bash
singularity pull docker://alecthomson/racstools
```

NOTE: These builds are still experimental, and have not been widely tested. In particular, parallelisation may not work as expected.

### Pip
You can also pip install this package into an existing Python environment. You will need both `numpy` and a fortran compiler before running `pip install`.

```bash
conda install numpy
# or
pip install numpy
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install RACS-tools.


```bash
# Stable
pip install RACS-tools
# Latest
pip install git+https://github.com/AlecThomson/RACS-tools
```



## Usage

```
$ beamcon_2D -h
usage: beamcon_2D [-h] [-p PREFIX] [-s SUFFIX] [-o OUTDIR] [--conv_mode {robust,scipy,astropy,astropy_fft}] [-v] [-d] [--bmaj BMAJ] [--bmin BMIN] [--bpa BPA]
                  [--log LOG] [--logfile LOGFILE] [-c CUTOFF] [--circularise] [-t TOLERANCE] [-e EPSILON] [-n NSAMPS] [--ncores N_CORES | --mpi]
                  infile [infile ...]

    Smooth a field of 2D images to a common resolution.

    - Parallelisation can run using multiprocessing or MPI.

    - Default names of output files are /path/to/beamlog{infile//.fits/.{SUFFIX}.fits}

    - By default, the smallest common beam will be automatically computed.
    - Optionally, you can specify a target beam to use.

    

positional arguments:
  infile                Input FITS image(s) to smooth (can be a wildcard) - beam info must be in header.

optional arguments:
  -h, --help            show this help message and exit
  -p PREFIX, --prefix PREFIX
                        Add prefix to output filenames.
  -s SUFFIX, --suffix SUFFIX
                        Add suffix to output filenames [sm].
  -o OUTDIR, --outdir OUTDIR
                        Output directory of smoothed FITS image(s) [same as input file].
  --conv_mode {robust,scipy,astropy,astropy_fft}
                        Which method to use for convolution [robust].
                                'robust' computes the analytic FT of the convolving Gaussian.
                                Note this mode cannot handle NaNs in the data.
                                Can also be 'scipy', 'astropy', or 'astropy_fft'.
                                Note these other methods cannot cope well with small convolving beams.
                                
  -v, --verbosity       Increase output verbosity
  -d, --dryrun          Compute common beam and stop [False].
  --bmaj BMAJ           Target BMAJ (arcsec) to convolve to [None].
  --bmin BMIN           Target BMIN (arcsec) to convolve to [None].
  --bpa BPA             Target BPA (deg) to convolve to [None].
  --log LOG             Name of beamlog file. If provided, save beamlog data to a file [None - not saved].
  --logfile LOGFILE     Save logging output to file
  -c CUTOFF, --cutoff CUTOFF
                        Cutoff BMAJ value (arcsec) -- Blank channels with BMAJ larger than this [None -- no limit]
  --circularise         Circularise the final PSF -- Sets the BMIN = BMAJ, and BPA=0.
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
usage: beamcon_3D [-h] [--uselogs] [--mode MODE] [--conv_mode {robust,scipy,astropy,astropy_fft}] [-v] [--logfile LOGFILE] [-d] [-p PREFIX] [-s SUFFIX]
                  [-o OUTDIR] [--bmaj BMAJ] [--bmin BMIN] [--bpa BPA] [-c CUTOFF] [--circularise] [--ref_chan {first,last,mid}] [-t TOLERANCE] [-e EPSILON]
                  [-n NSAMPS]
                  infile [infile ...]

    Smooth a field of 3D cubes to a common resolution.

    - Parallelisation is done using MPI.

    - Default names of output files are /path/to/beamlog{infile//.fits/.{SUFFIX}.fits}

    - By default, the smallest common beam will be automatically computed.
    - Optionally, you can specify a target beam to use.

    - It is currently assumed that cubes will be 4D with a dummy Stokes axis.
    - Iterating over Stokes axis is not yet supported.

    

positional arguments:
  infile                Input FITS image(s) to smooth (can be a wildcard)
                                - CASA beamtable will be used if present i.e. if CASAMBM = T
                                - Otherwise beam info must be in co-located beamlog files.
                                - beamlog must have the name /path/to/beamlog{infile//.fits/.txt}
                                

optional arguments:
  -h, --help            show this help message and exit
  --uselogs             Get convolving information from previous run [False].
  --mode MODE           Common resolution mode [natural]. 
                                natural -- allow frequency variation.
                                total -- smooth all plans to a common resolution.
                                
  --conv_mode {robust,scipy,astropy,astropy_fft}
                        Which method to use for convolution [robust].
                                'robust' computes the analytic FT of the convolving Gaussian.
                                Note this mode cannot handle NaNs in the data.
                                Can also be 'scipy', 'astropy', or 'astropy_fft'.
                                Note these other methods cannot cope well with small convolving beams.
                                
  -v, --verbosity       Increase output verbosity
  --logfile LOGFILE     Save logging output to file
  -d, --dryrun          Compute common beam and stop [False].
  -p PREFIX, --prefix PREFIX
                        Add prefix to output filenames.
  -s SUFFIX, --suffix SUFFIX
                        Add suffix to output filenames [{MODE}].
  -o OUTDIR, --outdir OUTDIR
                        Output directory of smoothed FITS image(s) [None - same as input].
  --bmaj BMAJ           BMAJ to convolve to [max BMAJ from given image(s)].
  --bmin BMIN           BMIN to convolve to [max BMAJ from given image(s)].
  --bpa BPA             BPA to convolve to [0].
  -c CUTOFF, --cutoff CUTOFF
                        Cutoff BMAJ value (arcsec) -- Blank channels with BMAJ larger than this [None -- no limit]
  --circularise         Circularise the final PSF -- Sets the BMIN = BMAJ, and BPA=0.
  --ref_chan {first,last,mid}
                        Reference psf for header [None]. 
                                    first  -- use psf for first frequency channel.
                                    last -- use psf for the last frequency channel.
                                    mid -- use psf for the centre frequency channel.
                                    Will use the CRPIX channel if not set.
                                    
  -t TOLERANCE, --tolerance TOLERANCE
                        tolerance for radio_beam.commonbeam.
  -e EPSILON, --epsilon EPSILON
                        epsilon for radio_beam.commonbeam.
  -n NSAMPS, --nsamps NSAMPS
                        nsamps for radio_beam.commonbeam.
```

If finding a common beam fails, try tweaking the `tolerance`, `epsilon`, and `nsamps` parameters. See [radio-beam](https://radio-beam.readthedocs.io/en/latest/) for more details.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause)
