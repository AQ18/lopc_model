# lopc_model

This project contains the essential code for the LOPC model, offered as supporting data for my PhD thesis.
It is available publicly to anyone who is interested in optical multilayers with resonant absorption.

## Disclaimer

This *should* work, but there is always the risk that some hack I did fails on your machine.
The instructions assume you have a Lumerical license to run lumapi - if you don't, then the TMM solver can be substituted fairly easily in the `lopc_data` helper function.
If the process is obtuse, then feel free to get in touch.

## Environment

Clone the multilayer_simulator environment from https://github.com/AQ18/multilayer_simulator:

`conda create --name lopc_model --clone multilayer_simulator`

Update it with additional dependencies:

`conda env update --name lopc_model --file research/environment.yml`

(This seems to hang forever sometimes, in which case it may be faster to just manually install the additional dependencies.)

Add lumapi to system path:

`conda develop "C:\Program Files\Lumerical\v222\api\python" -n lopc_model`

## Data generation
Run the notebooks that aren't in the `for_publication` directory.
You may need to create a new `data` directory to hold the data.

## Recreating thesis figures
Run the notebooks in the `for_publication` directory.
You will probably want to change the target directory to save the figures in.
