[![Docker Build and Push](https://github.com/AlecThomson/RACS-tools/actions/workflows/docker.yml/badge.svg)](https://github.com/AlecThomson/RACS-tools/actions/workflows/docker.yml) ![Tests](https://github.com/AlecThomson/RACS-tools/actions/workflows/pytest.yml/badge.svg) [![Python package](https://github.com/AlecThomson/RACS-tools/actions/workflows/python-package.yml/badge.svg)](https://github.com/AlecThomson/RACS-tools/actions/workflows/python-package.yml) [![PyPi](https://github.com/AlecThomson/RACS-tools/actions/workflows/pypi.yml/badge.svg)](https://github.com/AlecThomson/RACS-tools/actions/workflows/pypi.yml) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/AlecThomson/RACS-tools/master.svg)](https://results.pre-commit.ci/latest/github/AlecThomson/RACS-tools/master)

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
You can also use the package manager [pip](https://pip.pypa.io/en/stable/) to install RACS-tools.


```bash
# Stable
pip install RACS-tools
# Latest
pip install git+https://github.com/AlecThomson/RACS-tools
```



## Usage

```
$ beamcon_2D -h
usage: beamcon_2D [-h] [-p PREFIX] [-s SUFFIX] [-o OUTDIR] [--conv_mode {robust,scipy,astropy,astropy_fft}] [-v] [-d] [--bmaj BMAJ] [--bmin BMIN]
                  [--bpa BPA] [--log LOG] [--logfile LOGFILE] [-c CUTOFF] [--circularise] [-t TOLERANCE] [-e EPSILON] [-n NSAMPS] [--ncores NCORES]
                  [--executor {thread,process,mpi}]
                  infile [infile ...]

Smooth a field of 2D images to a common resolution. - Parallelisation can run using multiprocessing or MPI. - Default names of output files are
/path/to/beamlog{infile//.fits/.{SUFFIX}.fits} - By default, the smallest common beam will be automatically computed. - Optionally, you can specify a
target beam to use.

positional arguments:
  infile                Input FITS image(s) to smooth (can be a wildcard) - beam info must be in header.

options:
  -h, --help            show this help message and exit
  -p PREFIX, --prefix PREFIX
                        Add prefix to output filenames. (default: None)
  -s SUFFIX, --suffix SUFFIX
                        Add suffix to output filenames [sm]. (default: sm)
  -o OUTDIR, --outdir OUTDIR
                        Output directory of smoothed FITS image(s) [same as input file]. (default: None)
  --conv_mode {robust,scipy,astropy,astropy_fft}
                        Which method to use for convolution [robust]. 'robust' computes the analytic FT of the convolving Gaussian. Note this mode can
                        now handle NaNs in the data. Can also be 'scipy', 'astropy', or 'astropy_fft'. Note these other methods cannot cope well with
                        small convolving beams. (default: robust)
  -v, --verbosity       Increase output verbosity (default: 0)
  -d, --dryrun          Compute common beam and stop [False]. (default: False)
  --bmaj BMAJ           Target BMAJ (arcsec) to convolve to [None]. (default: None)
  --bmin BMIN           Target BMIN (arcsec) to convolve to [None]. (default: None)
  --bpa BPA             Target BPA (deg) to convolve to [None]. (default: None)
  --log LOG             Name of beamlog file. If provided, save beamlog data to a file [None - not saved]. (default: None)
  --logfile LOGFILE     Save logging output to file (default: None)
  -c CUTOFF, --cutoff CUTOFF
                        Cutoff BMAJ value (arcsec) -- Blank channels with BMAJ larger than this [None -- no limit] (default: None)
  --circularise         Circularise the final PSF -- Sets the BMIN = BMAJ, and BPA=0. (default: False)
  -t TOLERANCE, --tolerance TOLERANCE
                        tolerance for radio_beam.commonbeam. (default: 0.0001)
  -e EPSILON, --epsilon EPSILON
                        epsilon for radio_beam.commonbeam. (default: 0.0005)
  -n NSAMPS, --nsamps NSAMPS
                        nsamps for radio_beam.commonbeam. (default: 200)
  --ncores NCORES       Number of cores to use for parallelisation. If None, use all available cores. (default: None)
  --executor {thread,process,mpi}
                        Executor to use for parallelisation (default: thread)
```

```
$ beamcon_3D -h
usage: beamcon_3D [-h] [--uselogs] [--mode MODE] [--conv_mode {robust,scipy,astropy,astropy_fft}] [-v] [--logfile LOGFILE] [-d] [-p PREFIX] [-s SUFFIX]
                  [-o OUTDIR] [--bmaj BMAJ] [--bmin BMIN] [--bpa BPA] [-c CUTOFF] [--circularise] [--ref_chan {first,last,mid}] [-t TOLERANCE]
                  [-e EPSILON] [-n NSAMPS] [--ncores NCORES] [--executor_type {thread,process,mpi}]
                  infile [infile ...]

Smooth a field of 3D cubes to a common resolution. - Default names of output files are /path/to/beamlog{infile//.fits/.{SUFFIX}.fits} - By default, the
smallest common beam will be automatically computed. - Optionally, you can specify a target beam to use. - It is currently assumed that cubes will be
4D with a dummy Stokes axis. - Iterating over Stokes axis is not yet supported.

positional arguments:
  infile                Input FITS image(s) to smooth (can be a wildcard) - CASA beamtable will be used if present i.e. if CASAMBM = T - Otherwise beam
                        info must be in co-located beamlog files. - beamlog must have the name /path/to/beamlog{infile//.fits/.txt}

options:
  -h, --help            show this help message and exit
  --uselogs             Get convolving information from previous run [False]. (default: False)
  --mode MODE           Common resolution mode [natural]. natural -- allow frequency variation. total -- smooth all plans to a common resolution.
                        (default: natural)
  --conv_mode {robust,scipy,astropy,astropy_fft}
                        Which method to use for convolution [robust]. 'robust' computes the analytic FT of the convolving Gaussian. Note this mode can
                        now handle NaNs in the data. Can also be 'scipy', 'astropy', or 'astropy_fft'. Note these other methods cannot cope well with
                        small convolving beams. (default: robust)
  -v, --verbosity       Increase output verbosity (default: 0)
  --logfile LOGFILE     Save logging output to file (default: None)
  -d, --dryrun          Compute common beam and stop. (default: False)
  -p PREFIX, --prefix PREFIX
                        Add prefix to output filenames. (default: None)
  -s SUFFIX, --suffix SUFFIX
                        Add suffix to output filenames [{MODE}]. (default: None)
  -o OUTDIR, --outdir OUTDIR
                        Output directory of smoothed FITS image(s) [None - same as input]. (default: None)
  --bmaj BMAJ           BMAJ to convolve to [max BMAJ from given image(s)]. (default: None)
  --bmin BMIN           BMIN to convolve to [max BMAJ from given image(s)]. (default: None)
  --bpa BPA             BPA to convolve to [0]. (default: None)
  -c CUTOFF, --cutoff CUTOFF
                        Cutoff BMAJ value (arcsec) -- Blank channels with BMAJ larger than this [None -- no limit] (default: None)
  --circularise         Circularise the final PSF -- Sets the BMIN = BMAJ, and BPA=0. (default: False)
  --ref_chan {first,last,mid}
                        Reference psf for header [None]. first -- use psf for first frequency channel. last -- use psf for the last frequency channel.
                        mid -- use psf for the centre frequency channel. Will use the CRPIX channel if not set. (default: None)
  -t TOLERANCE, --tolerance TOLERANCE
                        tolerance for radio_beam.commonbeam. (default: 0.0001)
  -e EPSILON, --epsilon EPSILON
                        epsilon for radio_beam.commonbeam. (default: 0.0005)
  -n NSAMPS, --nsamps NSAMPS
                        nsamps for radio_beam.commonbeam. (default: 200)
  --ncores NCORES       Number of cores to use for parallelisation. If None, use all available cores. (default: None)
  --executor_type {thread,process,mpi}
                        Executor type for parallelisation. (default: thread)
```

```
$ getnoise_list -h
usage: getnoise_list [-h] [-s] [-b] [-c CLIPLEV] [-i ITERATE] [-f FILE] qfile ufile

 Find bad channels by checking statistics of each channel image.

positional arguments:
  qfile                 Stokes Q fits file
  ufile                 Stokes U fits file

options:
  -h, --help            show this help message and exit
  -s, --save_noise      Save noise values to disk [default False]
  -b, --blank           Blank bad channels? [default False - just print out bad channels]
  -c CLIPLEV, --cliplev CLIPLEV
                        Clip level in sigma, make this number lower to be more aggressive [default 5]
  -i ITERATE, --iterate ITERATE
                        Iterate flagging check N times [dafult 1 -- one pass only]
  -f FILE, --file FILE  Filename to write bad channel indices to file [None --  do not write]
```

If finding a common beam fails, try tweaking the `tolerance`, `epsilon`, and `nsamps` parameters. See [radio-beam](https://radio-beam.readthedocs.io/en/latest/) for more details.

## Performance

Profiling for `beamcon_3D` suggests this program requires a minimum of ~15X the memory of a data cube slice per process to perform convolution to a common beam. So for a 800 MB slice (e.g. typical POSSUM cube) you would want to allow 15 GB memory per worker (I use 20 GB). Choose `ncores` appropriately given your machine memory availability and this limit to ensure optimal performance with multiprocessing.

An example slurm header for `beamcon_3D`:

```
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=<ncores>
#SBATCH --mem-per-cpu=20G
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause)
