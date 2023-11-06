from typing import Hashable, Mapping, Optional, Sequence, Union
from itertools import accumulate, pairwise
import attrs
import numpy as np
from numpy.typing import ArrayLike
import xarray as xr
import lumapi
from . import LOPC

from multilayer_simulator.helpers.xarray import (
    add_absorption_to_xarray_dataset,
    add_unpolarised_to_xarray_dataset,
    DatasetKeys,
)
from multilayer_simulator.helpers.helpers import filter_mapping


Labels = Union[Hashable, list[Hashable]]  # flexibly take a single dim or list of dims
# Note that list is specified rather than the more general Iterable
# This is because tuples are Iterable but break xarray indexing

# Setup tools


def combo_length(combos):
    lengths = [len(val) for val in combos.values()]
    total_length = np.prod(lengths)
    return total_length


def estimate_combo_run_time(run_time, combos):
    total_length = combo_length(combos)
    return run_time * total_length


# Generate the data

ALLOWED_LOPC_KWARGS = set(attrs.fields_dict(LOPC.LOPC))
OTHER_ALLOWED_KWARGS = {
    "keep_lopcs",
    "apply_disorder",
    "delta_mode",
    "disorder_type",
    "correlated",
    "retain_runs",
}


def lopc_data(**kwargs):
    """
    Build an LOPC simulation and return the simulation data.

    If the lumerical_session kwarg is not supplied, start a new instance of
    lumapi.FDTD().
    If keep_lopcs is True, then retain all the LOPC simulations in lopc_list.
    See LOPC.LOPC for other valid kwargs.

    :return: The data output of the simulation.
    :rtype: Variable, provide a formatter to change the type.
    """
    # check the wrong kwargs haven't been passed by mistake
    disallowed_kwargs = set(kwargs) - ALLOWED_LOPC_KWARGS - OTHER_ALLOWED_KWARGS
    if disallowed_kwargs:
        raise ValueError(
            (
                f"Disallowed kwargs {disallowed_kwargs} passed. Allowed kwargs"
                f"are {ALLOWED_LOPC_KWARGS.union(OTHER_ALLOWED_KWARGS)}"
            )
        )

    if "lumerical_session" not in kwargs:
        kwargs["lumerical_session"] = lumapi.FDTD(hide=True)
    filtered_kwargs = filter_mapping(kwargs, filter=ALLOWED_LOPC_KWARGS)
    lopc = LOPC.LOPC(**filtered_kwargs)
    if "keep_lopcs" in kwargs:
        if kwargs["keep_lopcs"]:
            try:
                lopc_data.lopc_list.append(lopc)
            except AttributeError:  # list doesn't exist yet
                lopc_data.lopc_list = [lopc]
    if "apply_disorder" in kwargs:
        # disorder_kwargs = (
        #     {} if "disorder_kwargs" not in kwargs else kwargs["disorder_kwargs"]
        # )
        disorder_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["delta_mode", "disorder_type", "correlated"]
        }
        retain_runs = False if "retain_runs" not in kwargs else kwargs["retain_runs"]
        match kwargs["apply_disorder"]:
            # case bool() as apply:
            #     if apply:  # do nothing at all for Falsey values
            #         lopc.apply_structure_disorder(**disorder_kwargs)
            #         data = lopc.simulate(keep_data=False)
            case False:
                data = lopc.simulate(keep_data=False)
            case int() as N:
                # deciding whether to concat(list) or zeros_like->assign
                # probably makes hardly any difference
                datas = []
                for _ in range(N - 1):
                    lopc.apply_structure_disorder(**disorder_kwargs)
                    datas.append(lopc.simulate(keep_data=False))
                    lopc.reverse_structure_disorder(forget=False)
                else:  # don't reverse the structure for the final loop
                    lopc.apply_structure_disorder(**disorder_kwargs)
                    datas.append(lopc.simulate(keep_data=False))
                if retain_runs:
                    data = xr.concat(datas, dim="run")
                    dxs = xr.DataArray(
                        lopc.dxs,
                        coords=[data.run, ("layer_number", range(lopc.num_periods))],
                    )
                    data["dxs"] = dxs
                else:
                    data = sum(datas) / len(datas)
    else:
        data = lopc.simulate(keep_data=False)
    return data


# def ref_data(**kwargs):

# Intelligent loading


def indexer_from_dataset(ds, drop=None):
    if drop is not None:
        ds = ds.drop_dims(drop)
    idx = ds.reset_coords().coords
    return idx


# Data munging


def assign_period(dataset: xr.Dataset) -> xr.Dataset:
    return dataset.assign_coords(
        period=dataset.passive_layer_thickness + dataset.excitonic_layer_thickness
    )


def assign_total_excitonic_thickness(dataset: xr.Dataset) -> xr.Dataset:
    return dataset.assign_coords(
        total_excitonic_thickness=dataset.excitonic_layer_thickness
        * dataset.num_periods
    )


def assign_total_passive_thickness(dataset: xr.Dataset) -> xr.Dataset:
    return dataset.assign_coords(
        total_passive_thickness=dataset.passive_layer_thickness
        * (dataset.num_periods - dataset.remove_last_layer)
    )


def assign_total_thickness(dataset: xr.Dataset) -> xr.Dataset:
    try:
        modified = dataset.assign_coords(
            total_thickness=dataset.total_excitonic_thickness
            + dataset.total_passive_thickness
        )
    except AttributeError:  # in case those coords haven't been assigned yet
        total_excitonic_thickness = (
            dataset.excitonic_layer_thickness * dataset.num_periods
        )
        total_passive_thickness = dataset.passive_layer_thickness * (
            dataset.num_periods - dataset.remove_last_layer
        )
        modified = dataset.assign_coords(
            total_thickness=total_excitonic_thickness + total_passive_thickness
        )

    return modified


def assign_N_tot(dataset: xr.Dataset) -> xr.Dataset:
    try:
        modified = dataset.assign_coords(
            N_tot=dataset.N * dataset.total_excitonic_thickness
        )
    except AttributeError:  # in case those coords haven't been assigned yet
        total_excitonic_thickness = (
            dataset.excitonic_layer_thickness * dataset.num_periods
        )
        modified = dataset.assign_coords(N_tot=dataset.N * total_excitonic_thickness)

    return modified


def assign_per_oscillator(
    dataset: xr.Dataset,
    var_key: DatasetKeys = ["As", "Ap", "A"],
    oscillator_key="N_tot",
) -> xr.Dataset:
    var_keys = np.atleast_1d(var_key)
    variables = {
        f"{var}_per_oscillator": (lambda x, var=var: x[var] / x[oscillator_key])
        for var in var_keys
    }

    try:
        modified = dataset.assign(variables=variables)
    except KeyError:  # in case N_tot hasn't been assigned yet
        modified = assign_N_tot(dataset)
        modified = modified.assign(
            variables=variables
        )  # will fail if oscillator_key != 'N_tot'
        modified = modified.drop(
            "N_tot"
        )  # assuming there was a reason not to keep this coord

    return modified


def mean_and_std(ds: xr.Dataset, var_key: Optional[DatasetKeys] = None, dim="run", assign_high=False, assign_low=False):
    var_key = (
        list(ds.data_vars.keys()) if var_key is None or var_key is True else var_key
    )
    var_keys = np.atleast_1d(var_key)
    ds = ds[var_keys]

    ds_mean = ds.mean(dim=dim)
    ds_std = ds.std(dim=dim)
    # ds_high = ds_mean + ds_std
    # ds_low = ds_mean - ds_std

    ds_mean = ds_mean.rename({var: f"{var}_mean" for var in var_keys})
    ds_std = ds_std.rename({var: f"{var}_std" for var in var_keys})
    # ds_high = ds_high.rename({var: f"{var}_high" for var in var_keys})
    # ds_low = ds_low.rename({var: f"{var}_low" for var in var_keys})

    stats = xr.merge(
        [
            ds_mean,
            ds_std,
            # ds_high,
            # ds_low,
        ]
    )

    if assign_high:
        stats = assign_high_from_mean_and_std(stats, var_key=var_key)
    if assign_low:
        stats = assign_low_from_mean_and_std(stats, var_key=var_key)

    return stats


def high_from_mean_and_std(ds, var: str):
    return ds[f"{var}_mean"] + ds[f"{var}_std"]


def low_from_mean_and_std(ds, var: str):
    return ds[f"{var}_mean"] - ds[f"{var}_std"]


def assign_high_from_mean_and_std(
    ds,
    var_key: DatasetKeys
    #   var_key: Optional[DatasetKeys] = None
):
    # var_key = (
    #     list(ds.data_vars.keys()) if var_key is None or var_key is True else var_key
    # )  # this won't work because there is a recursion problem
    var_keys = np.atleast_1d(var_key)
    high_dict = {f"{var}_high": high_from_mean_and_std(ds, var) for var in var_keys}
    modified = ds.assign(high_dict)

    return modified


def assign_low_from_mean_and_std(
    ds,
    var_key: DatasetKeys
    #   var_key: Optional[DatasetKeys] = None
):
    # var_key = (
    #     list(ds.data_vars.keys()) if var_key is None or var_key is True else var_key
    # )  # this won't work because there is a recursion problem
    var_key = (
        list(ds.data_vars.keys()) if var_key is None or var_key is True else var_key
    )
    var_keys = np.atleast_1d(var_key)
    low_dict = {f"{var}_low": low_from_mean_and_std(ds, var) for var in var_keys}
    modified = ds.assign(low_dict)

    return modified


def assign_high_and_low(ds, var_key: DatasetKeys):
    ds = assign_high_from_mean_and_std(ds, var_key=var_key)
    ds = assign_low_from_mean_and_std(ds, var_key=var_key)
    return ds


# # This may not be better than just using ds.where(ds == ds.max("wavelength"))
def max_min_pos(
    ds: xr.Dataset, var_key: Optional[DatasetKeys] = None, dim="wavelength"
):
    var_key = (
        list(ds.data_vars.keys()) if var_key is None or var_key is True else var_key
    )
    var_keys = np.atleast_1d(var_key)
    ds = ds[var_keys]

    ds_max = ds.max(dim=dim)
    ds_max_pos = ds.idxmax(dim=dim)
    ds_min = ds.min(dim=dim)
    ds_min_pos = ds.idxmin(dim=dim)

    ds_max = ds_max.rename({var: f"{var}_max" for var in var_keys})
    ds_max_pos = ds_max_pos.rename({var: f"{var}_max_{dim}" for var in var_keys})
    ds_min = ds_min.rename({var: f"{var}_min" for var in var_keys})
    ds_min_pos = ds_min_pos.rename({var: f"{var}_min_{dim}" for var in var_keys})

    stats = xr.merge([ds_max, ds_max_pos, ds_min, ds_min_pos])

    return stats


def assign_mean_and_std(
    ds: xr.Dataset, var_key: Optional[DatasetKeys] = None, dim="run"
):
    stats = mean_and_std(ds, var_key=var_key, dim=dim)
    modified = ds.merge(stats)

    return modified


def assign_derived_attrs(
    dataset: xr.Dataset,
    absorption=True,
    unpolarised=True,
    period=True,
    total_excitonic_thickness=True,
    total_passive_thickness=True,
    total_thickness=True,
    N_tot=True,
    per_oscillator=["As", "Ap", "A"],
    mean_and_std=False,
) -> xr.Dataset:
    ds = dataset
    if absorption:
        ds = add_absorption_to_xarray_dataset(
            ds, ("Rs", "Rp"), ("Ts", "Tp"), ("As", "Ap")
        )
    if unpolarised:
        ds = add_unpolarised_to_xarray_dataset(
            ds, ("Rs", "Ts", "As"), ("Rp", "Tp", "Ap"), ("R", "T", "A")
        )
    if period:
        ds = assign_period(ds)
    if total_excitonic_thickness:
        ds = assign_total_excitonic_thickness(ds)
    if total_passive_thickness:
        ds = assign_total_passive_thickness(ds)
    if total_thickness:
        ds = assign_total_thickness(ds)
    if N_tot:
        ds = assign_N_tot(ds)
    if per_oscillator:
        ds = assign_per_oscillator(ds, var_key=per_oscillator)
    if mean_and_std:
        ds = assign_mean_and_std(ds, var_key=mean_and_std)
    return ds


def restack(
    ds: Union[xr.DataArray, xr.Dataset],
    start_idxs: Sequence[Hashable],
    end_idxs: Sequence[Hashable],
    *,
    dummy_idx: Hashable = "dummy_idx",
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Change a data structure's indexes using the stack->set_index->unstack
    trick. The dummy_idx key is provided in case there is a name clash.

    :param ds: The DataArray or Dataset to restack
    :type ds: Union[xr.DataArray, xr.Dataset]
    :param start_idxs: The indexes to stack
    :type start_idxs: Sequence[Hashable]
    :param end_idxs: The desired final indexes
    :type end_idxs: Sequence[Hashable]
    :param dummy_idx: Temporary index name, defaults to 'dummy_idx'
    :type dummy_idx: Hashable, optional
    :return: Restacked ds
    :rtype: Union[xr.DataArray, xr.Dataset]
    """
    ds = ds.stack(**{dummy_idx: start_idxs})
    ds = ds.set_index(**{dummy_idx: end_idxs})
    ds = ds.unstack()
    return ds


def enhancement_factor(ds, ref, common_dim, method="groupby"):
    match method:
        case "groupby":
            gb = ds.groupby(common_dim)
            ef = gb / ref
        case "vectorized_indexing":
            reidx = ref.sel({common_dim: ds[common_dim]})
            ef = ds / reidx
        case _:
            raise TypeError(
                f"method should be 'groupby' or 'vectorized_indexing', not {method}"
            )
    return ef


# Spectrum


def spectrum(
    func: callable,
    domain: Union[xr.DataArray, ArrayLike],
    *,
    coords: Optional[Mapping] = None,
    domain_name: Optional[Hashable] = None,
    normalisation: Optional[float] = None,
    array_name: Optional[Hashable] = None,
) -> xr.DataArray:
    data = func(domain)
    if coords is None:
        try:
            coords = domain.coords
        except:
            coords = []
            if domain_name is not None:
                coords.append((domain_name, domain))
            else:
                coords.append(domain)
    da = xr.DataArray(data, coords=coords, name=array_name)

    if normalisation is not None:
        integrated = da
        for dim in da.dims:
            integrated = integrated.integrate(dim)
        da = (da / integrated) * normalisation

    return da


def normalise_over_dim(
    da: xr.DataArray,
    dim: Labels,
    normalisation=1,
    *,
    method: int = 1,
    dummy_idx: Hashable = "dummy_idx",
) -> xr.DataArray:
    """
    Normalise a DataArray over one or more dimensions.
    'Normalise' in this case means to multiply by a uniform factor, such that if
    da was integrated over the specified dimensions, then the integral
    would be equal to the normalisation value if da was equal
    to 1 everywhere.
    This is equivalent to multiplying da by the product of the spans of the dims.

    :param da: DataArray to normalise
    :type da: xr.DataArray
    :param dim: Dimension(s) over which to normalise
    :type dim: DimOrDims
    :param normalisation: Value to normalise the integral to, defaults to 1
    :type normalisation: int, optional
    :return: Normalised DataArray
    :rtype: xr.DataArray
    """
    dims = np.atleast_1d(dim)
    match method:
        case 1:
            spans = []
            for d in dims:
                # does funny things if the dims are out of order
                spans.append(da[d][-1] - da[d][0])
            integral = np.prod(spans)
            normaliser = normalisation / integral
        case 2:
            dims = list(dims)  # np array breaks .stack(), cast to list
            # This is a horrible hack to handle multi-dimensional case
            domain = da.stack(**{dummy_idx: dims})[dummy_idx].unstack()

            normaliser = spectrum(np.ones_like, domain, normalisation=normalisation)
        case _:
            raise NotImplementedError(f'argument "method" must be 1 or 2, not {method}')
    return da * normaliser


# Integrate


def integrate_da(
    da: xr.DataArray,
    dim: Labels,
    weighting: Union[xr.DataArray, ArrayLike] = 1,
    normalisation: Optional[float] = None,
) -> xr.DataArray:
    """
    Integrate a DataArray over a given dimension or dimensions. Optional
    weighting and normalisation before integration.

    :param da: The DataArray to be integrated. Slice to integrate between limits
    :type da: xr.DataArray
    :param dim: The name(s) of the dimension(s) over which to integrate
    :type dim: DimOrDims
    :param weighting: An array-like that will be broadcast over da, defaults to 1
    :type weighting: Union[xr.DataArray, ArrayLike], optional
    :param normalisation: Number for which to normalise the integral, defaults to None
    :type normalisation: Optional[float], optional
    :return: _description_
    :rtype: xr.DataArray
    """
    dims = np.atleast_1d(dim)  # wrap a single hashable to enable iteration
    weighted = da * weighting  # xarray's ds.weighted doesn't implement integrals
    if normalisation is not None:
        weighted = normalise_over_dim(weighted, dims, normalisation)
    integrated = weighted.integrate(dim)
    return integrated


def sel_or_integrate(
    da: Union[xr.DataArray, xr.Dataset], dim: Hashable, val, **integrate_kwargs
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Either select a specific value or slice-then-integrate depending on val.
    If val is a sequence, attempt to slice with it and then integrate using
    **integrate_kwargs.
    Otherwise, attempt to select val on dim using method="nearest".

    :param da: The DataArray or Dataset
    :type da: Union[xr.DataArray, xr.Dataset]
    :param dim: The dimension over which to select or integrate
    :type dim: Hashable
    :param val: The value to select or integration limits
    :type val: _type_
    :return: Selected or integrated da
    :rtype: Union[xr.DataArray, xr.Dataset]
    """
    match val:
        case [*vals]:
            da = da.sel(**{dim: slice(*vals)})
            da = integrate_da(da, dim, **integrate_kwargs)
        case _:
            da = da.sel(**{dim: val}, method="nearest")
    return da


# linewidths


@np.vectorize
def linewidth_calculator(x=None, /, centre=0, linewidth=1):
    if x is None:
        return linewidth
    else:
        return centre + x * linewidth


# optimum


def find_optimum(da, dim=..., optimise="max"):
    match optimise:
        case "max":
            optimum_function = xr.DataArray.max
        case "min":
            optimum_function = xr.DataArray.min
        case _:
            raise NotImplementedError(
                f'argument "optimise" must be "max" or "min", not {optimise}'
            )

    da = optimum_function(da, dim=dim)

    return da


def find_optimum_coords(da, dim=..., optimise="max", load=True):
    match optimise:
        case "max":
            optimum_function = xr.DataArray.argmax
        case "min":
            optimum_function = xr.DataArray.argmin
        case _:
            raise NotImplementedError(
                f'argument "optimise" must be "max" or "min", not {optimise}'
            )

    # if dim is a bare string, wrap it in a list (see argmax documentation for why)
    if isinstance(dim, str):
        dim = [dim]

    if load:
        da = da.load()
        optargs = {dim: val for dim, val in optimum_function(da, dim=dim).items()}
    else:
        # have to manually trigger the computation because isel doesn't work with dask arrays
        optargs = {
            dim: val.compute() for dim, val in optimum_function(da, dim=dim).items()
        }
    optimum = da.isel(optargs)

    return optimum


# renormalise


def renormalise_value(val, vmin=0, vmax=1):
    new_val = (val - vmin) / (vmax - vmin)
    return new_val


### Plotting


from bokeh.models import Range1d, LinearAxis
from bokeh.models.renderers import GlyphRenderer
from bokeh.plotting import figure
import holoviews as hv
from multilayer_simulator.helpers.mixins import convert_wavelength_and_frequency
from multilayer_simulator.lumerical_classes import (
    format_stackfield,
)

name_attrs_map = {
    # data vars - note 'long names' are often shorter than 'short names', because long names are the default
    "Rs": {"long_name": r"\(R_s\)", "short_name": "Reflectance (s-polarised)"},
    "Rp": {"long_name": r"\(R_p\)", "short_name": "Reflectance (p-polarised)"},
    "Ts": {"long_name": r"\(T_s\)", "short_name": "Transmittance (s-polarised)"},
    "Tp": {"long_name": r"\(T_p\)", "short_name": "Transmittance (p-polarised)"},
    "As": {"long_name": r"\(A_s\)", "short_name": "Absorptance (s-polarised)"},
    "Ap": {"long_name": r"\(A_p\)", "short_name": "Absorptance (p-polarised)"},
    "R": {"long_name": r"\(R\)", "short_name": "Reflectance"},
    "T": {"long_name": r"\(T\)", "short_name": "Transmittance"},
    "A": {"long_name": r"\(A\)", "short_name": "Absorptance"},
    "Rs_per_oscillator": {
        "long_name": r"\(R_{s,N}\)",
        "short_name": "Reflectance per oscillator (s-polarised)",
    },
    "Rp_per_oscillator": {
        "long_name": r"\(R_{p,N}\)",
        "short_name": "Reflectance per oscillator (p-polarised)",
    },
    "R_per_oscillator": {
        "long_name": r"\(R_{N}\)",
        "short_name": "Reflectance per oscillator",
    },
    "Ts_per_oscillator": {
        "long_name": r"\(T_{s,N}\)",
        "short_name": "Transmittance per oscillator (s-polarised)",
    },
    "Tp_per_oscillator": {
        "long_name": r"\(T_{p,N}\)",
        "short_name": "Transmittance per oscillator (p-polarised)",
    },
    "T_per_oscillator": {
        "long_name": r"\(T_{N}\)",
        "short_name": "Transmittance per oscillator",
    },
    "As_per_oscillator": {
        "long_name": r"\(A_{s,N}\)",
        "short_name": "Absorptance per oscillator (s-polarised)",
    },
    "Ap_per_oscillator": {
        "long_name": r"\(A_{p,N}\)",
        "short_name": "Absorptance per oscillator (p-polarised)",
    },
    "A_per_oscillator": {
        "long_name": r"\(A_{N}\)",
        "short_name": "Absorptance per oscillator",
    },
    "dxs": {
        "long_name": r"\(\delta_x\)",
        "short_name": "Excitonic layer position perturbation",
        "units": "nm",
    },
    # coords
    "frequency": {"long_name": "f", "units": "Hz"},
    "wavelength": {"long_name": "λ", "units": "nm"},
    "theta": {"long_name": "θ", "units": "°"},
    "passive_RI": {"long_name": r"\(n_p\)"},
    "incident_medium_RI": {"long_name": r"\(n_i\)"},
    "exit_medium_RI": {"long_name": r"\(n_o\)"},
    "N": {"long_name": r"\(N_{LO}\)"},
    "permittivity": {"long_name": r"\(\epsilon_\inf)"},
    "lorentz_resonance_wavelength": {"long_name": r"\(\lambda_0\)", "units": "nm"},
    "lorentz_linewidth": {"long_name": r"\(\Gamma\)"},
    "num_periods": {"long_name": "N"},
    "passive_layer_thickness": {"long_name": r"\(d_p\)", "units": "nm"},
    "excitonic_layer_thickness": {"long_name": r"\(d_e\)", "units": "nm"},
    "remove_last_layer": {"long_name": "Last layer removed?"},
    "period": {"long_name": r"\(\Lambda\)", "units": "nm"},
    "total_excitonic_thickness": {"long_name": r"\(d_{e,\, tot}\)", "units": "nm"},
    "total_passive_thickness": {"long_name": r"\(d_{p,\, tot}\)", "units": "nm"},
    "total_thickness": {"long_name": r"\(d_{tot}\)", "units": "nm"},
    "N_tot": {"long_name": r"\(N_{tot}\)"},
    "run": {"long_name": "Run ID"},
    "layer_number": {"long_name": "Excitonic layer number"},
    "delta": {"long_name": "Δ", "short_name": "Degree of disorder"},
}


def assign_attrs_by_name(
    da, name=None, name_attrs_map: Mapping = name_attrs_map, inplace=False, strict=True
):
    name = da.name if name is None else name
    if name in name_attrs_map:
        new_attrs = name_attrs_map[name]
        if inplace:
            da.attrs.update(new_attrs)
        else:
            da = da.assign_attrs(new_attrs)
            return da
    else:
        if strict:
            raise KeyError(f"{name} not in name_attrs_map")


def pre_process_for_plots(ds, strict=True):
    try:
        ds = ds.swap_dims(frequency="wavelength")
        ds = ds.assign(wavelength=(ds.wavelength * 1e9))
    except ValueError:
        pass  # assume I did this already or it was integrated away

    for da in ds.data_vars.values():
        assign_attrs_by_name(da, inplace=True, strict=strict)
    for da in ds.coords.values():
        assign_attrs_by_name(da, inplace=True, strict=strict)

    return ds


# def pre_process_for_plots(dataset):
#     try:
#         dataset = dataset.swap_dims(frequency="wavelength")
#         dataset = dataset.assign(wavelength=(dataset.wavelength * 1e9))
#         dataset.wavelength.attrs["long_name"] = "λ"
#         dataset.wavelength.attrs["units"] = "nm"
#     except:
#         pass

#     try:
#         dataset.theta.attrs["long_name"] = "θ"
#         dataset.theta.attrs["units"] = "degrees"
#     except:
#         pass

#     try:
#         dataset.passive_RI.attrs["long_name"] = "Passive Layer RI"
#     except:
#         pass

#     try:
#         dataset.incident_medium_RI.attrs["long_name"] = "Incident Medium RI"
#     except:
#         pass

#     try:
#         dataset.exit_medium_RI.attrs["long_name"] = "Exit Medium RI"
#     except:
#         pass

#     # try:
#     # dataset.N.attrs["long_name"] = r"$N_O$"
#     # dataset.N_tot.attrs["long_name"] = r"$N_{tot}$"
#     # except:
#     #     pass

#     try:
#         dataset.num_periods.attrs["long_name"] = "Number of periods"
#     except:
#         pass

#     try:
#         dataset.passive_layer_thickness.attrs["long_name"] = "Passive Layer Thickness"
#         dataset.passive_layer_thickness.attrs["units"] = "nm"
#     except:
#         pass

#     try:
#         dataset.excitonic_layer_thickness.attrs[
#             "long_name"
#         ] = "Excitonic Layer Thickness"
#         dataset.excitonic_layer_thickness.attrs["units"] = "nm"
#     except:
#         pass

#     try:
#         dataset.period.attrs["long_name"] = "Λ"
#         dataset.period.attrs["units"] = "nm"
#     except:
#         pass

#     try:
#         dataset.total_excitonic_thickness.attrs[
#             "long_name"
#         ] = "Total Excitonic Layer Thickness"
#         dataset.total_excitonic_thickness.attrs["units"] = "nm"
#     except:
#         pass

#     try:
#         dataset.total_passive_thickness.attrs[
#             "long_name"
#         ] = "Total Passive Layer Thickness"
#         dataset.total_passive_thickness.attrs["units"] = "nm"
#     except:
#         pass

#     try:
#         dataset.total_thickness.attrs["long_name"] = "Total Thickness"
#         dataset.total_thickness.attrs["units"] = "nm"
#     except:
#         pass

#     return dataset


def fix_bin_labels(ds: xr.Dataset, dim: Hashable = "run_bins") -> xr.Dataset:
    """
    Replace the bin labels with hashable alternatives.
    Intended as a quick fix to the problem that the default groupby_bins labels
    are Interval objects which can not be used as legend entries, causing errors.

    :param ds: the dataset with groupby_bins labels
    :type ds: xr.Dataset
    :param dim: name of the bin labels dim, defaults to "run_bins"
    :type dim: Hashable, optional
    :return: relabeled ds
    :rtype: xr.Dataset
    """
    bin_labels = [v.mid for v in ds[dim].values]  # left/mid/right are options here
    ds = ds.assign_coords({dim: bin_labels})
    return ds


def plot_secondary(plot, element):
    """
    Hook to plot data on a secondary (twin) axis on a Holoviews Plot with Bokeh backend.
    More info:
    - http://holoviews.org/user_guide/Customizing_Plots.html#plot-hooks
    - https://docs.bokeh.org/en/latest/docs/user_guide/plotting.html#twin-axes

    """
    fig: figure = plot.state
    glyph_first: GlyphRenderer = fig.renderers[0]  # will be the original plot
    glyph_last: GlyphRenderer = fig.renderers[-1]  # will be the new plot
    right_axis_name = "twiny"
    # Create both axes if right axis does not exist
    if right_axis_name not in fig.extra_y_ranges.keys():
        # Recreate primary axis (left)
        y_first_name = glyph_first.glyph.y
        y_first_min = glyph_first.data_source.data[y_first_name].min()
        y_first_max = glyph_first.data_source.data[y_first_name].max()
        y_first_offset = (y_first_max - y_first_min) * 0.1
        fig.y_range = Range1d(
            start=y_first_min - y_first_offset, end=y_first_max + y_first_offset
        )
        fig.y_range.name = glyph_first.y_range_name
        # Create secondary axis (right)
        y_last_name = glyph_last.glyph.y
        y_last_min = glyph_last.data_source.data[y_last_name].min()
        y_last_max = glyph_last.data_source.data[y_last_name].max()
        y_last_offset = (y_last_max - y_last_min) * 0.1
        fig.extra_y_ranges = {
            right_axis_name: Range1d(
                start=y_last_min - y_last_offset, end=y_last_max + y_last_offset
            )
        }
        fig.add_layout(
            LinearAxis(y_range_name=right_axis_name, axis_label=glyph_last.glyph.y),
            "right",
        )
    # Set right axis for the last glyph added to the figure
    glyph_last.y_range_name = right_axis_name


def vlines(x=0, **kwargs):
    """
    Return either a VLine or Overlay of VLines depending on the input.

    Have to (but can't) check against np.ndarray explicitly because it is not a Sequence:
    https://discuss.python.org/t/relax-requirement-for-sequence-pattern-matches-in-pep-622/22258/6
    """
    match x:
        case [*xs]:
            return hv.Overlay([hv.VLine(x, **kwargs) for x in xs])
        case x:
            return hv.VLine(x, **kwargs)


def coordinate_string(join_str="\n", str_formats=None, **coords):
    coord_strings = []
    for k, v in coords.items():
        try:  # get the desired format if specified
            f = str_formats[k]
        except:  # default is integer format
            f = ".0f"

        try:  # assume v can be formatted as a float
            coord_strings.append(f"{k}: {v:{f}}")
        except TypeError:  # handle sequences
            coord_strings.append(f"{k}: {tuple(f'{val:{f}}' for val in v)}")
    return join_str.join(coord_strings)


def plot_da(
    da: xr.DataArray,
    label_field: Optional[Hashable] = None,
    label_append: Optional[str] = None,
    **hvplot_kwargs,
):
    """
    Plot a DataArray with intelligent labeling based on the DataArray
    attrs or name. (A Dataset can be passed at your own risk.)

    If label_field is passed, it is assumed to be a key in da.attrs
    whose value is the label, e.g. 'short_name' or 'long_name'.
    Otherwise, either da.name is used, or an empty string.

    If label_append is passed, it will be appended directly to the
    label extracted with label_field.

    If the 'label' keyword arg is passed, it will override the
    preceding arguments.

    :param da: The DataArray to plot
    :type da: xr.DataArray
    :param label_field: Key in da.attrs to extract label from, defaults to None
    :type label_field: Optional[Hashable], optional
    :param label_append: String to append to label, defaults to None
    :type label_append: Optional[str], optional
    :return: A Holoviews plot
    :rtype:
    """
    if "label" not in hvplot_kwargs:  # don't override an explicit label
        if label_field is not None:
            hvplot_kwargs["label"] = da.attrs[label_field]
        elif da.name is not None:
            hvplot_kwargs["label"] = da.name
        else:
            hvplot_kwargs["label"] = ""
        label_append = "" if label_append is None else label_append
        hvplot_kwargs["label"] += label_append
    plot = da.hvplot(**hvplot_kwargs)
    return plot


def plot_var(
    variable: Hashable,
    dataset: xr.Dataset,
    label_field: Optional[Hashable] = None,
    label_append: Optional[str] = None,
    **hvplot_kwargs,
):
    """
    Helper function to use plot_da with datasets. Refer to plot_da.

    :param variable: Name of the DataArray(s) to plot
    :type variable: Hashable
    :param dataset: Dataset to extract DataArrays from
    :type dataset: xr.Dataset
    :param label_field: Key in da.attrs to extract label from, defaults to None
    :type label_field: Optional[Hashable], optional
    :param label_append: String to append to label, defaults to None
    :type label_append: Optional[str], optional
    :return: _description_
    :rtype: _type_
    """
    da = dataset[variable]
    plot = plot_da(da, label_field, label_append, **hvplot_kwargs)
    return plot


def plot_optimum_over_dim(
    da,
    dim,
    x,
    y,
    optimise="max",
):
    optimised_da = find_optimum_coords(da, dim=dim, optimise=optimise).squeeze()
    optimum_coords = find_optimum_coords(optimised_da, optimise=optimise)
    x_point = float(optimum_coords[x])
    y_point = float(optimum_coords[y])

    plot = optimised_da.hvplot.quadmesh(x=x, y=y) * hv.Points((x_point, y_point))

    return plot, optimum_coords


def plot_field(wavelength, ri_lower=1, ri_upper=2, z_scale=1, **kwargs):
    # assume wavelength in nm
    frequency = convert_wavelength_and_frequency(wavelength * 1e-9)
    lopc_kwargs = {
        "frequencies": frequency,
        "keep_lopcs": True,
        "simulation_mode": "stackfield",
        "formatter": format_stackfield(
            **{"output_format": "xarray_dataset", "add_norms": True}
        ),
        **kwargs,  # will throw an error if repeated kwargs, which is good
    }

    data = lopc_data(**lopc_kwargs)
    data = data.assign_coords(z=data.z / z_scale)
    lopc = lopc_data.lopc_list[-1]  # get the most recent LOPC
    # indexes = lopc.structure.index(frequencies=frequency)
    # # extra 0 used for finding layer boundaries
    # thicknesses = np.concatenate([[0], lopc.structure.thickness])

    data_plot = data["|Es|^2"].squeeze().hvplot()
    # TODO: replace with function calls
    # vspans = [
    #     hv.VSpan(start, stop).opts(
    #         alpha=renormalise_value(
    #             float(np.abs(index)), ri_lower, ri_upper
    #         )  # vary transparency within given bounds
    #     )
    #     for (start, stop), index in zip(
    #         pairwise(accumulate(thicknesses)), indexes
    #     )  # pairwise/accumulate returns layer boundaries
    # ]

    vspans = visualise_multilayer(
        multilayer=lopc.structure,
        frequency=frequency,
        ri_lower=ri_lower,
        ri_upper=ri_upper,
        scale=z_scale,
    )

    overlay = hv.Overlay(vspans) * data_plot

    return overlay


def thicknesses_to_vspans(
    thicknesses: Sequence,
    alphas: Optional[Sequence] = None,
    prepend_zero_layer: bool = True,
) -> hv.Overlay:
    """
    Turn a sequence of thicknesses into an overlay of VSpans.
    If the first VSpan should start at 0 and the first element of thicknesses is
    not 0, set preprend_zero_layer to True.
    If alphas is set, it must be a sequence of valid alpha values (real numbers
    between 0 and 1) with length equal to the desired number of VSpans.
    If alphas is not set, it will automatically be set to alternatve between
    0 and 1 for each VSpan.

    :param thicknesses: thicknesses of VSpans
    :type thicknesses: Sequence
    :param alphas: alphas of VSpans, defaults to None
    :type alphas: Optional[Sequence], optional
    :param prepend_zero_layer: defaults to True
    :type prepend_zero_layer: bool, optional
    :return: overlay of VSpans
    :rtype: hv.Overlay
    """
    if prepend_zero_layer:
        thicknesses = np.concatenate([[0], thicknesses])

    if alphas is None:
        alphas = np.zeros(len(thicknesses) - 1)
        alphas[1::2] = 1  # make an alternating array of 1s and 0s

    vspans = [
        hv.VSpan(start, stop).opts(alpha=alpha)  # vary transparency within given bounds
        for (start, stop), alpha in zip(
            pairwise(accumulate(thicknesses)), alphas
        )  # pairwise/accumulate returns layer boundaries
    ]

    overlay = hv.Overlay(vspans)
    return overlay


def visualise_multilayer(
    multilayer: "multilayer_simulator.Multilayer",
    frequency: Optional[float] = None,
    ri_lower: Optional[float] = None,
    ri_upper: Optional[float] = None,
    scale=1,
) -> hv.Overlay:
    """
    Visualise a multilayer structure as an overlay of VSpans.
    frequency is used to extract the multilayer refractive index as alpha values
    for VSpans. If set to None then alpha values will alternate between 0 and 1.
    ri_lower and ri_upper can be used to set the normalisation bounds for the
    conversion of index values to alpha values. Default behaviour is to set the
    max and min index to alpha=0 and 1, respectively.

    :param multilayer: multilayer structure to visualise
    :type multilayer: multilayer_simulator.Multilayer
    :param frequency: frequency to extract index for alphas, defaults to None
    :type frequency: Optional[float], optional
    :param ri_lower: refractive index mapping to alpha=0, defaults to None
    :type ri_lower: Optional[float], optional
    :param ri_upper: refractive index mapping to alpha=1, defaults to None
    :type ri_upper: Optional[float], optional
    :return: overlay of VSpans
    :rtype: hv.Overlay
    """
    thicknesses = multilayer.thickness / scale
    if frequency is None:
        alphas = None
    else:
        # cast complex to real numbers
        alphas = np.abs(multilayer.index(frequencies=frequency)).flatten()
        # renormalise to [0, 1] to be valid alpha values
        ri_lower = alphas.min() if ri_lower is None else ri_lower
        ri_upper = alphas.max() if ri_upper is None else ri_upper
        alphas = renormalise_value(alphas, vmin=ri_lower, vmax=ri_upper)

    overlay = thicknesses_to_vspans(
        thicknesses=thicknesses, alphas=alphas, prepend_zero_layer=True
    )

    return overlay


def complex_elements(*args, element=hv.Curve, auto_label=None, **kwargs):
    """
    A convenience function to create and label Holoviews-type Elements for the
    real and imaginary parts of a complex dataset ys defined over a real domain xs.
    Returns a dictionary containing the Elements.

    The returned dict can be directly passed to hv.NdLayout or another dimensioned
    container. To use a non-dimensioned container such as hv.Layout, use the
    .values() method.

    args: the first positional argument MUST be a two-element sequence that would
        be a valid input to a Holoviews-type Element, with the exception that the
        second item in the sequence may be complex. All other args are passed
        unchanged to the Element class.
    element: Holoviews-type Element class, default hv.Curve.
    auto_label: Optional. If a valid key is supplied, the corresponding keyword
        argument with be overridden by 'Real' and 'Imaginary' for the real and
        imaginary parts of ys, respectively. For Holoviews Elements, this can be
        either 'group' or 'label'. Default None.
    **kwargs: passed to the element unchanged except as noted above.
    """
    xs, ys = args[0]
    args_real = ((xs, np.real(ys)),) + args[1:]
    args_imag = ((xs, np.imag(ys)),) + args[1:]

    kwargs_real = kwargs.copy()
    kwargs_imag = kwargs.copy()
    if auto_label is not None:
        kwargs_real |= {auto_label: "Real"}
        kwargs_imag |= {auto_label: "Imaginary"}

    real_curve = element(*args_real, **kwargs_real)
    imag_curve = element(*args_imag, **kwargs_imag)
    return {"Real": real_curve, "Imaginary": imag_curve}
