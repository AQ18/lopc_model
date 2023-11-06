from typing import Optional, SupportsFloat
import warnings
import attrs
from attrs import define, mutable, frozen, field, setters
import lumapi

import multilayer_simulator as ml
from multilayer_simulator.helpers.lorentz_oscillator import plasma_frequency_squared
from multilayer_simulator.helpers.mixins import convert_wavelength_and_frequency
from multilayer_simulator.lumerical_classes import (
    STACKRT,
    STACKFIELD,
    LumericalOscillator,
    format_stackrt,
)
import numpy as np


def label_layer(string):
    return string + " Layer"


def build_LOPC_structure(
    passive_RI,
    incident_medium_RI,
    exit_medium_RI,
    permittivity,
    lorentz_resonance,
    lorentz_permittivity,
    lorentz_linewidth,
    passive_layer_thickness,
    excitonic_layer_thickness,
    num_periods,
    oscillator=None,
    lumerical_session=None,
    copy_layers=False,
    add_first_layer=False,
    remove_last_layer=True,
):
    incident_medium = ml.material.ConstantIndex(incident_medium_RI, name="Incident")
    exit_medium = ml.material.ConstantIndex(exit_medium_RI, name="Exit")
    passive = ml.material.ConstantIndex(passive_RI, name="Passive")

    if oscillator is None:
        oscillator = LumericalOscillator(session=lumerical_session, name="Excitonic")
    oscillator.set_property(
        permittivity=permittivity,
        lorentz_resonance=lorentz_resonance,
        lorentz_permittivity=lorentz_permittivity,
        lorentz_linewidth=lorentz_linewidth,
    )
    oscillator.sync_backwards()

    incident_halfspace = ml.Layer.from_material(incident_medium, name=label_layer)
    exit_halfspace = ml.Layer.from_material(exit_medium, name=label_layer)
    passive_layer = ml.Layer.from_material(
        passive, thickness=passive_layer_thickness, name=label_layer
    )
    excitonic_layer = ml.Layer.from_material(
        oscillator, thickness=excitonic_layer_thickness, name=label_layer
    )

    unit_cell = [excitonic_layer, passive_layer]
    multilayer = ml.Multilayer.from_given_unit_cell(
        unit_cell=unit_cell,
        incident_layer=incident_halfspace,
        exit_layer=exit_halfspace,
        num_periods=num_periods,
        copy_layers=copy_layers,
    )
    if add_first_layer:
        # shouldn't need deepcopy because this layer doesn't already exist in layers
        multilayer.layers.insert(1, passive_layer)
    if remove_last_layer:
        del multilayer.layers[-2]
    return multilayer


def adjust_layer_thickness(layer: ml.Layer, dd: float = 0):
    """
    Adjust the thickness of a Layer by dd.
    Warn if the new layer thickness is negative.

    :param layer: layer to be adjusted
    :type layer: ml.Layer
    :param dd: amount to adjust thickness by, defaults to 0
    :type dd: float, optional
    """
    new_thickness = layer.thickness + dd
    if new_thickness < 0:
        warnings.warn("Setting negative layer thickness")
    layer.thickness = new_thickness


def shift_layer_position(
    multilayer: ml.Multilayer,
    layer_index: int,
    dx: float = 0,
    compensate: bool = False,
):
    """
    Emulate a positional shift of a layer by adjusting the thickness of the
    preceding layer. If compensate is True, also adjust the thickness of the
    subsequent layer so that the position of other layers in the structure is
    unaffected.

    layer_index = 0 or 1 are special cases.
    layer_index = 0 will raise an error.
    # layer_index = 1 will adjust the thickness of the subsequent rather than
    # preceding layer, because the input medium layer should not be adjusted.
    # This may cause unexpected behaviour if the input medium layer has a non-zero
    # imaginary refractive index component.

    :param multilayer: multilayer the layer belongs to
    :type multilayer: ml.Multilayer
    :param layer_index: index of the layer in multilayer.layers
    :type layer_index: int
    :param dx: amount to adjust position by, defaults to 0
    :type dx: float, optional
    :param compensate: adjust subsequent layer thickness, defaults to False
    :type compensate: bool, optional
    """
    layers = multilayer.layers
    match layer_index:
        case 0:
            raise ValueError("The input medium layer can't be shifted")
        # case 1:
        #     post_layer = layers[2]
        #     adjust_layer_thickness(layer=post_layer, dd=(-dx))
        #     if compensate:
        #         warnings.warn("Do not compensate for shift of first layer")
        #     compensate = False  # do not compensate for shift of first layer
        case _:
            pre_layer = layers[layer_index - 1]
            adjust_layer_thickness(layer=pre_layer, dd=dx)

    if compensate:
        try:
            post_layer = layers[layer_index + 1]
        except IndexError:
            warnings.warn("Cannot compensate for shifting output medium layer")
        else:
            adjust_layer_thickness(layer=post_layer, dd=(-dx))


@mutable(kw_only=True)
class LOPC(ml.Simulation):
    lumerical_session: lumapi.FDTD = field(factory=lambda: lumapi.FDTD())
    simulation_mode: str = "stackrt"  # convert to Literal type

    # Refractive indices
    passive_RI: float = 1.35
    incident_medium_RI: float = 1.35
    exit_medium_RI: float = field()

    # Lorentz oscillator parameters
    N: int = 1e26
    permittivity: float = 2.2
    lorentz_resonance_wavelength: float = 680
    # lorentz_resonance: float = field()
    # lorentz_permittivity: float = field()
    lorentz_linewidth: float = 7.5e13

    # Structure
    num_periods: int = 10
    passive_layer_thickness: float = 0
    excitonic_layer_thickness: float = 0

    # Disorder
    rng: np.random.Generator = field(
        converter=np.random.default_rng,  # can accept any valid arg to np.random.default_rng()
        on_setattr=setters.convert,
    )
    delta: float = 0  # warning! interpreted different by different methods
    dxs: np.ndarray = field()  # record of excitonic layer displacements

    # Options
    length_scale: float = 1e-9
    oscillator: LumericalOscillator = None
    copy_layers: bool = False
    add_first_layer: bool = False  # only needed for disorder simulations
    remove_last_layer: bool = True

    @exit_medium_RI.default
    def _exit_medium_RI_factory(self):
        return self.incident_medium_RI

    # @lorentz_resonance.default
    # def _lorentz_resonance_factory(self):
    # return (
    #     2
    #     * np.pi
    #     * convert_wavelength_and_frequency(
    #         self.lorentz_resonance_wavelength * 1e-9
    #     )
    # )

    @rng.default
    def _rng_factory(self):
        return None

    @dxs.default
    def _dxs_factory(self):
        return np.empty(shape=(0, self.num_periods))

    def scale_length(self, length: SupportsFloat):
        return length * self.length_scale

    @property
    def lorentz_resonance(self):
        return (
            2
            * np.pi
            * convert_wavelength_and_frequency(
                self.scale_length(self.lorentz_resonance_wavelength)
            )
        )

    # @lorentz_permittivity.default
    # def _lorentz_permittivity_factory(self):
    # """This formula removes the frequency dependency of the oscillator strength in Lumerical's formula."""
    # return plasma_frequency_squared(self.N) / (self.lorentz_resonance**2)

    @property
    def lorentz_permittivity(self):
        """This formula removes the frequency dependency of the oscillator strength in Lumerical's formula."""
        return plasma_frequency_squared(self.N) / (self.lorentz_resonance**2)

    def __attrs_post_init__(self):
        self.set_structure()
        self.set_engine()
        # self.set_formatter() # first add formatter to simulation and write special LOPC formatter

    def set_structure(self):
        self.structure = self.build_structure()

    def set_engine(self):
        match self.simulation_mode:
            case "stackrt":
                engine = STACKRT(self.lumerical_session)
            case "stackfield":
                engine = STACKFIELD(self.lumerical_session)
        self.engine = engine

    def build_structure(self):
        return build_LOPC_structure(
            passive_RI=self.passive_RI,
            incident_medium_RI=self.incident_medium_RI,
            exit_medium_RI=self.exit_medium_RI,
            permittivity=self.permittivity,
            lorentz_resonance=self.lorentz_resonance,
            lorentz_permittivity=self.lorentz_permittivity,
            lorentz_linewidth=self.lorentz_linewidth,
            passive_layer_thickness=self.scale_length(self.passive_layer_thickness),
            excitonic_layer_thickness=self.scale_length(self.excitonic_layer_thickness),
            num_periods=self.num_periods,
            oscillator=self.oscillator,
            lumerical_session=self.lumerical_session,
            copy_layers=self.copy_layers,
            add_first_layer=self.add_first_layer,
            remove_last_layer=self.remove_last_layer,
        )

    @staticmethod
    def parameter_names():
        return [
            "passive_RI",
            "incident_medium_RI",
            "exit_medium_RI",
            "N",
            "permittivity",
            "lorentz_resonance_wavelength",
            "lorentz_linewidth",
            "num_periods",
            "passive_layer_thickness",
            "excitonic_layer_thickness",
        ]

    @classmethod
    def parameter_attributes(cls):
        fields_dict = attrs.fields_dict(cls)
        attributes_list = [
            fields_dict[parameter] for parameter in cls.parameter_names()
        ]
        return attributes_list

    def parameters(self):
        included = attrs.filters.include(*self.parameter_attributes())
        return attrs.asdict(self, filter=included)

    def expand_data_dims(self):
        dim = {k: np.atleast_1d(v) for k, v in self.parameters().items()}
        expanded = self.data.expand_dims(dim)
        return expanded

    def get_excitonic_layer_indices(self):
        """
        Return the indices of self.structure.layers which match the excitonic
        layers that can safely be shifted. (Accounts for add_first_layer)

        This only works if we don't start deleting layers in the middle. If we
        do then there needs to be a more complicated algorithm, probably
        involving tracking by name or id.

        :return: _description_
        :rtype: _type_
        """
        first_excitonic_layer = 1 + int(self.add_first_layer)
        last_structure_layer = len(self.structure.layers) - 1
        period = 2

        return list(range(first_excitonic_layer, last_structure_layer, period))

    def record_dxs(self, dxs):
        """
        Record dxs as a new column in self.dxs.

        :param dxs: an array of same length as self.dxs
        :type dxs: np.ndarray
        """
        self.dxs = np.r_["0,2", self.dxs, dxs]

    def _apply_displacements_to_excitonic_layers(self, dxs):
        if not self.copy_layers:
            warnings.warn(
                "Structure layers have not been copied;"
                " applying displacements may have unintended affects"
            )
        excitonic_layer_indices = self.get_excitonic_layer_indices()
        # assume dxs scaled like other lengths
        dxs = self.scale_length(np.atleast_1d(dxs))

        for index, dx in zip(excitonic_layer_indices, dxs):
            shift_layer_position(
                multilayer=self.structure,
                layer_index=index,
                dx=dx,
                compensate=False,  # compensation is assumed done at a higher level if needed
            )

    def _reset_dxs(self):
        self.dxs = self._dxs_factory()

    def reverse_structure_disorder(self, index=-1, forget=True):
        dxs = -self.dxs[index]  # negative
        self._apply_displacements_to_excitonic_layers(dxs)
        if forget:  # remove the displacements from the record
            self.dxs = np.delete(self.dxs, obj=index, axis=0)

    def apply_structure_disorder(
        self,
        delta: Optional[float] = None,
        delta_mode: str = "exact",
        disorder_type: str = "uniform",
        correlated: bool = False,
        dxs=None,
    ):
        delta = self.delta if delta is None else delta

        match delta_mode:
            case "abs":  # absolute
                Delta = np.abs(delta)
            case "pplt":  # proportional to passive layer thickness
                Delta = delta * self.passive_layer_thickness
            case "exact":  # use exactly this value
                Delta = delta
            case _:
                raise NotImplementedError(
                    f'argument "delta_mode" must be "exact", "abs" or "pplt", not {delta_mode}'
                )

        match disorder_type:
            case "uniform":
                if correlated:
                    if Delta > self.passive_layer_thickness:
                        warnings.warn(
                            "Delta is greater than the passive layer thickness;"
                            " positional displacements may result in negative layer thickness"
                        )
                else:
                    if Delta > self.passive_layer_thickness / 2:
                        warnings.warn(
                            "Delta is greater than half the passive layer thickness;"
                            " positional displacements may result in negative layer thickness"
                        )
                dxs = self.rng.uniform(low=(-Delta), high=Delta, size=self.num_periods)
            case "dxs":
                dxs = self.dxs[-1] if dxs is None else dxs  # fallback to self.dxs
                dxs *= Delta
            case _:
                raise NotImplementedError(
                    f'argument "disorder_type" must be "uniform" or "dxs", not {disorder_type}'
                )

        if correlated:
            pass  # explictly noting that the positional disorder is correlated by default
        else:
            # subtract each layer thickness perturbation from the subsequent layer thickness
            # effectively implementing the 'compensate' heuristic in shift_layer_thickness()
            # which decorrelates the positional disorder, but more efficiently
            dxs[1:] -= dxs[0:-1]

        self.record_dxs(dxs)  # record the displacements
        self._apply_displacements_to_excitonic_layers(dxs=dxs)


### IDEA: modified simulate method (or modify in lopc_data?)
# simulate with spoof structure 'DisorderedMask'
# DisorderedMask is just a wrapper round the true structure
# except the .thickness() method is adapted to add in disorder
# should be much faster to just modify the values in the array
# rather than actually looping through all the layers
# and changing their thicknesses in memory
# this could enable much faster aggregation of disordered runs!
