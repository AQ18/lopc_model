# lopc_model

## Environment

Clone the multilayer_simulator environment from https://github.com/AQ18/multilayer_simulator:

`conda create --name lopc_model --clone multilayer_simulator`

Update it with additional dependencies:

`conda env update --name lopc_model --file research/environment.yml`

(This seems to hang forever sometimes, in which case it may be faster to just manually install the additional dependencies.)

Add lumapi to system path:

`conda develop "C:\Program Files\Lumerical\v222\api\python" -n lopc_model`
