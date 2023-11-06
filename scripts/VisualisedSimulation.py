import numpy as np
import param
import panel as pn
import holoviews as hv
import xarray as xr

import multilayer_simulator as ml
from multilayer_simulator.helpers.lorentz_oscillator import plasma_frequency_squared
from multilayer_simulator.helpers.mixins import convert_wavelength_and_frequency
from multilayer_simulator.lumerical_classes import (
    STACKRT,
    LumericalMaterial,
    LumericalOscillator,
    format_stackrt,
)


class RefractiveIndex(param.Number):
    def __init__(
        self,
        default=1,
        bounds=(0, None),
        softbounds=(1, 2),
        inclusive_bounds=(False, True),
        step=None,
        **params
    ):
        super().__init__(default, bounds, softbounds, inclusive_bounds, step, **params)


def append_layer(string):
    return string + " Layer"


def RI_curve(wavelengths, index, label):
    curve = hv.Curve(
        (wavelengths, index), "Wavelength", "Refractive Index", label=label
    )
    return curve


def layer_RI_curve(wavelengths, layer, label=None):
    label = layer.name if label is None else label
    index = layer.index(convert_wavelength_and_frequency(wavelengths))
    label_real = label + " (n)"
    label_imag = label + " (κ)"
    real_curve = RI_curve(wavelengths, index.real, label_real)
    imag_curve = RI_curve(wavelengths, index.imag, label_imag)
    return real_curve, imag_curve


class VisualisedSimulation(param.Parameterized):
    wavelength_range = param.Range(
        (380e-9, 980e-9), bounds=(0, None), inclusive_bounds=(False, True)
    )
    wavelength_res = param.Integer(1000, bounds=(1, None), softbounds=(50, 1000))
    angle_range = param.Range(
        (0, 26), bounds=(0, 90), inclusive_bounds=(True, False), step=5
    )
    angle = param.Number(0, bounds=(0, 90), inclusive_bounds=(True, False))
    passive_RI = RefractiveIndex(1)
    incident_medium_RI = RefractiveIndex(1)
    exit_medium_RI = RefractiveIndex(None)
    N = param.Number(1e26, bounds=(0, None))
    permittivity = param.Number(10.2)
    lorentz_resonance_wavelength = param.Number(
        680e-9, bounds=(0, None), inclusive_bounds=(False, True)
    )
    lorentz_linewidth = param.Number(
        7.5e13, bounds=(0, None), inclusive_bounds=(False, True)
    )
    excitonic_layer_thickness = param.Number(0, bounds=(0, None))
    passive_layer_thickness = param.Number(0, bounds=(0, None))
    num_periods = param.Integer(1, bounds=(0, None))
    lumerical_session = param.Parameter(per_instance=False)

    rta_data = param.ClassSelector(class_=xr.Dataset)
    field_data = param.ClassSelector(class_=xr.Dataset)

    def __init__(self, **params):
        super().__init__(**params)
        self.exit_medium_RI = (
            self.exit_medium_RI
            if self.exit_medium_RI is not None
            else self.incident_medium_RI
        )
        # self.incident_medium = ml.material.ConstantIndex(
        #     self.incident_medium_RI, name="Incident"
        # )
        # self.exit_medium = ml.material.ConstantIndex(self.exit_medium_RI, name="Exit")
        # self.passive = ml.material.ConstantIndex(self.passive_RI, name="Passive")
        # self.oscillator = LumericalOscillator(
        #     session=lumerical_session,
        #     name="Lorentz Oscillator",
        #     permittivity=self.permittivity,
        #     lorentz_resonance=self.lorentz_resonance_angular_frequency,
        #     lorentz_permittivity=self.lorentz_permittivity,
        #     lorentz_linewidth=self.lorentz_linewidth,
        # )
        # self.incident_halfspace = ml.Layer.from_material(
        #     self.incident_medium, name=append_layer
        # )
        # self.exit_halfspace = ml.Layer.from_material(
        #     self.exit_medium, name=append_layer
        # )
        # self.passive_layer = ml.Layer.from_material(
        #     self.passive, thickness=self.passive_layer_thickness, name=append_layer
        # )
        # self.excitonic_layer = ml.Layer.from_material(
        #     self.oscillator, thickness=self.excitonic_layer_thickness, name=append_layer
        # )
        # self.unit_cell = [self.excitonic_layer, self.passive_layer]

    @property
    def wavelengths(self):
        return np.linspace(*self.wavelength_range, self.wavelength_res)

    @property
    def frequencies(self):
        return convert_wavelength_and_frequency(self.wavelengths)

    @property
    def lorentz_resonance_angular_frequency(self):
        return (
            2
            * np.pi
            * convert_wavelength_and_frequency(self.lorentz_resonance_wavelength)
        )

    @property
    def lorentz_permittivity(self):
        return plasma_frequency_squared(self.N) / (
            self.lorentz_resonance_angular_frequency**2
        )

    @property
    def incident_medium(self):
        return ml.material.ConstantIndex(self.incident_medium_RI, name="Incident")

    @property
    def exit_medium(self):
        return ml.material.ConstantIndex(self.exit_medium_RI, name="Exit")

    @property
    def passive_medium(self):
        return ml.material.ConstantIndex(self.passive_RI, name="Passive")

    @property
    def oscillator(self):
        return LumericalOscillator(
            session=self.lumerical_session,
            permittivity=self.permittivity,
            lorentz_resonance=self.lorentz_resonance_angular_frequency,
            lorentz_permittivity=self.lorentz_permittivity,
            lorentz_linewidth=self.lorentz_linewidth,
        )

    @property
    def incident_layer(self):
        return ml.Layer.from_material(self.incident_medium, name=append_layer)

    @property
    def exit_layer(self):
        return ml.Layer.from_material(self.exit_medium, name=append_layer)

    @property
    def passive_layer(self):
        return ml.Layer.from_material(
            self.passive_medium, thickness=self.passive_layer_thickness, name=append_layer
        )

    @property
    def excitonic_layer(self):
        return ml.Layer.from_material(
            self.oscillator, thickness=self.excitonic_layer_thickness, name=append_layer
        )

    @property
    def unit_cell(self):
        return (self.excitonic_layer, self.passive_layer)

    @param.depends(
        "passive_RI", "wavelength_range", "wavelength_res"
    )
    def passive_layer_RI_curve(self):
        real_curve, imag_curve = layer_RI_curve(self.wavelengths, self.passive_layer)
        return real_curve, imag_curve

    @param.depends("incident_medium_RI", "wavelength_range", "wavelength_res")
    def incident_layer_RI_curve(self):
        real_curve, imag_curve = layer_RI_curve(
            self.wavelengths, self.incident_halfspace
        )
        return real_curve, imag_curve

    @param.depends("exit_medium_RI", "wavelength_range", "wavelength_res")
    def exit_layer_RI_curve(self):
        real_curve, imag_curve = layer_RI_curve(self.wavelengths, self.exit_halfspace)
        return real_curve, imag_curve

    @param.depends(
        "N",
        "permittivity",
        "lorentz_resonance_wavelength",
        "lorentz_linewidth",
        "wavelength_range",
        "wavelength_res",
    )
    def excitonic_layer_RI_curve(self):
        real_curve, imag_curve = layer_RI_curve(
            self.wavelengths, self.excitonic_layer, label="Lorentz Oscillator layer"
        )
        return real_curve, imag_curve

    @param.depends()
    def calculate_rta(self):
        pass
