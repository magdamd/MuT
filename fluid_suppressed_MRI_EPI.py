# /// script
# requires-python = ">=3.9"
# dependencies = ["pypulseq==1.5.0.post1"]
#
# [tool.anyfield]
# micropip_no_deps = ["pypulseq"]
# ///

# --- Notebook setup (Colab / Jupyter / JupyterLab / VS Code) ---
_ipython = globals().get('get_ipython', lambda: None)()  # detect nb
if _ipython is not None:
    _ipython.run_line_magic('pip', 'install -q pypulseq==1.5.0.post1')
# --- Notebook setup end ---

import numpy as np
import pypulseq as pp
import torch
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['figure.dpi'] = 100

def seq_EPI_2D(
    fov=(220e-3, 220e-3, 8e-3),
    Nread=64,
    Nphase=64,
    Npart=1,
    FA=torch.tensor(90 * np.pi / 180),
    fluid_suppression=True,
    TI = 0.7e-3,
    slice_thickness=8e-3,
    experiment_id='EPI_2D',
    system=None,
    # EPI-specific parameters
    rf_duration=1e-3,
    rf_apodization=0.5,
    rf_time_bw_product=4,
    adc_duration_OG=0.25e-3,
    eddy_currents=True,
    eddy_currents_induced_delay=0.0000015,
    blip_duration=0.1e-3
):
    """
    2D EPI sequence function following MRzero standard.
    Args:
        fov: tuple of floats (x, y, z) in meters
        Nread: int - frequency encoding steps
        Nphase: int - phase encoding steps
        Npart: int - number of partitions
        FA: tensor - flip angle
        fluid_suppression: bool - whether to apply fluid suppression
        TI: float - inversion time for fluid suppression
        slice_thickness: float - slice thickness
        experiment_id: string - experiment identifier
        system: optional scanner system limits
        rf_duration: float - RF pulse duration
        rf_apodization: float - RF apodization
        rf_time_bw_product: float - RF time-bandwidth product
        adc_duration_OG: float - ADC duration
        eddy_currents: bool - enable eddy current compensation
        eddy_currents_induced_delay: float - eddy current delay
        blip_duration: float - blip gradient duration
    Returns:
        pp.Sequence: PyPulseq sequence object
    """
    # Choose the scanner limits
    if system is None:
        system = pp.Opts(
            max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
            rf_ringdown_time=20e-6, rf_dead_time=100e-6,
            adc_dead_time=20e-6, grad_raster_time=10e-6
        )
    # Define the sequence
    seq = pp.Sequence()
    # Define RF events
    rf1, _, _ = pp.make_sinc_pulse(
        flip_angle=FA.item(), duration=rf_duration,
        slice_thickness=slice_thickness, apodization=rf_apodization,
        time_bw_product=rf_time_bw_product, system=system, return_gz=True
    )
    # Define other gradients and ADC events
    a = int(system.adc_raster_time * Nread * 10**7)
    b = int(system.grad_raster_time * 10**7)
    c = int(adc_duration_OG * 10**7)
    lcm_ab = abs(a * b) // np.gcd(a, b)
    adc_raster_duration = (lcm_ab if round(c / lcm_ab) == 0 else round(c / lcm_ab) * lcm_ab) / 10**7
    eddy_currents_induced_delay *= eddy_currents
    gx = pp.make_trapezoid(channel='x', flat_area=Nread / fov[0], flat_time=adc_raster_duration, system=system)
    gx_ = pp.make_trapezoid(channel='x', flat_area=-Nread / fov[0], flat_time=adc_raster_duration, system=system)
    adc = pp.make_adc(num_samples=Nread, duration=adc_raster_duration, phase_offset=0 * np.pi / 180,
                     delay=gx.rise_time + eddy_currents_induced_delay, system=system)
    gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)
    # Construct sequence
    if fluid_suppression:
        rf_inv = pp.make_block_pulse(
            flip_angle=np.pi, 
            duration=2e-3, 
            system=system,
            delay=system.rf_dead_time, 
            use='inversion'
        )
        seq.add_block(rf_inv)
        seq.add_block(pp.make_delay(TI))
    gp_blip = pp.make_trapezoid(channel='y', area=1 / fov[1], duration=blip_duration, system=system)
    seq.add_block(rf1)
    gp = pp.make_trapezoid(channel='y', area=-Nphase//2 / fov[1], duration=1e-3, system=system)
    seq.add_block(gx_pre, gp)
    for ii in range(0, Nphase//2):
        seq.add_block(gx, adc)
        seq.add_block(gp_blip)
        seq.add_block(gx_, adc)
        seq.add_block(gp_blip)
    # Required sequence definitions
    seq.set_definition('name', experiment_id)
    seq.set_definition('fov', [fov[0], fov[1], fov[2]])
    seq.set_definition('matrix', [Nread, Nphase, Npart])
    return seq

#@title quick 2D brain phantom sim and plot
# Define parameters as plain variables
experiment_id = 'EPI_2D'
fov = 240e-3 # @param {type:"number"} # Define FOV
slice_thickness = 8e-3
Nread = 64  # @param {type:"integer"} # frequency encoding steps/samples
Nphase = 64 # @param {type:"integer"} # phase encoding steps/samples
Npart = 1
FA = torch.tensor(90 * np.pi / 180)
slice_thickness = 8e-3
# EPI-specific parameters
rf_duration = 1e-3
rf_apodization = 0.5
rf_time_bw_product = 4
adc_duration_OG = 0.25e-3  # @param {type: "slider", min: 0.25e-3, max: 10e-3, step: 0.05e-3}
# Moved to function parameters
eddy_currents = True       # @param {type:"boolean"}
eddy_currents_induced_delay = 0.0000015 # @param {type: "slider", min: -1e-4, max: 1e-4, step: 1e-8}
# Moved to function parameters
blip_duration = 0.1e-3
# Generate sequence using standard parameters
seq = seq_EPI_2D(
    fov=(fov, fov, slice_thickness),
    Nread=Nread,
    Nphase=Nphase,
    Npart=Npart,
    FA=FA,
    slice_thickness=slice_thickness,
    experiment_id=experiment_id,
    rf_duration=rf_duration,
    rf_apodization=rf_apodization,
    rf_time_bw_product=rf_time_bw_product,
    adc_duration_OG=adc_duration_OG,
    eddy_currents=eddy_currents,
    eddy_currents_induced_delay=eddy_currents_induced_delay,
    blip_duration=blip_duration
)
# Quick simulation and plot
