# PhD

## Environment

Clone the multilayer_simulator environment:

`conda create --name phd --clone multilayer_simulator`

Update it with additional dependencies:

`conda env update --name phd --file research/conda_phd.yml`

(This seems to hang forever sometimes, in which case it may be faster to just manually install the additional dependencies.)

Add lumapi to system path:

`conda develop "C:\Program Files\Lumerical\v222\api\python" -n phd`