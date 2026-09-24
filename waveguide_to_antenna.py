import os
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
import matplotlib.pyplot as plt

# ---- Design parameters (WR-90, 8.3 GHz) ----
a = 22.86   # waveguide width (mm), x-direction
b = 10.16   # waveguide height (mm), y-direction
wg_length = 150  # mm, ~2.5 lambda_g
free_space_pad = 40  # mm, radiating region beyond open end
pml_cells = 8  # standard PML thickness

f0 = 8.3e9   # center frequency
fc = 2e9     # excitation bandwidth (Gaussian pulse half-bandwidth)

# ---- FDTD setup ----
FDTD = openEMS(EndCriteria=1e-4)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8', 'PML_8'])

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1e-3)  # mm

# ---- Domain sizing ----
x_max = a/2 + free_space_pad
y_max = b/2 + free_space_pad
z_min = 0
z_max = wg_length + free_space_pad

wall_thickness = 3

mesh.AddLine('x', np.linspace(-x_max, x_max, 41))
mesh.AddLine('y', np.linspace(-y_max, y_max, 41))
mesh.AddLine('z', np.linspace(z_min, z_max, 81))


# ---- Waveguide walls: hollow PEC duct, zero-thickness (idealized) ----
metal = CSX.AddMetal('WG_walls')

metal.AddBox(priority=10, start=[-a/2, b/2, 0], stop=[a/2, b/2 + wall_thickness, wg_length])
metal.AddBox(priority=10, start=[-a/2, -b/2 - wall_thickness, 0], stop=[a/2, -b/2, wg_length])
metal.AddBox(priority=10, start=[a/2, -b/2, 0], stop=[a/2 + wall_thickness, b/2, wg_length])
metal.AddBox(priority=10, start=[-a/2 - wall_thickness, -b/2, 0], stop=[-a/2, b/2, wg_length])

# ---- Waveguide port: TE10 excitation at z=0 (feed end) ----
port = FDTD.AddRectWaveGuidePort(
    0, [-a/2, -b/2, 0], [a/2, b/2, 10], 'z',
    a*1e-3, b*1e-3, 'TE10', excite=1
)

# ---- Nf2ff recording box ----
nf2ff = FDTD.CreateNF2FFBox()

# ---- Export & run ----
sim_path = os.path.abspath('sim_output')
os.makedirs(sim_path, exist_ok=True)
xml_file = os.path.join(sim_path, 'wg_transition_geometry.xml')
CSX.Write2XML(xml_file)
print(f"Geometry written to: {xml_file}")

FDTD.Run(sim_path, cleanup=False)   # FIX: cleanup=False so port files survive for CalcPort

# ---- Frequency range for post-processing ----
f_start = 5e9
f_stop = 12e9
freq = np.linspace(f_start, f_stop, 601)

# FIX: removed the redundant relative-path reassignment — reuse the absolute sim_path already set above
port.CalcPort(sim_path, freq)

# ---- S11 ----
s11 = port.uf_ref / port.uf_inc
s11_dB = 20 * np.log10(np.abs(s11))

plt.figure()
plt.plot(freq / 1e9, s11_dB)
plt.axvline(8.3, color='r', linestyle='--', label='8.3 GHz design freq')
plt.xlabel('Frequency (GHz)')
plt.ylabel('S11 (dB)')
plt.title('Return Loss — Waveguide-to-Antenna Transition')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(sim_path, 'S11_plot.png'), dpi=150)
plt.show()

try:
    idx = np.argmin(np.abs(freq - 8.3e9))
    print(f"S11 at 8.3 GHz: {s11_dB[idx]:.2f} dB")
except Exception as e:
    print(f"ERROR computing S11 at 8.3GHz: {e}")

# ---- Task 5: Far-field radiation pattern ----
theta = np.arange(-180, 180, 2)
phi = [0, 90]  # E-plane and H-plane cuts

f_res = 8.3e9  # evaluate pattern at your design frequency
nf2ff_result = nf2ff.CalcNF2FF(sim_path, f_res, theta, phi, center=[0, 0, 0], read_cached=False)

# ---- -3dB Beamwidth calculation (E-plane and H-plane) ----
def calc_beamwidth(theta_deg, gain_dB):
    peak_idx = np.argmax(gain_dB)
    peak_val = gain_dB[peak_idx]
    half_power = peak_val - 3.0

    # Search left of peak
    left_idx = peak_idx
    while left_idx > 0 and gain_dB[left_idx] > half_power:
        left_idx -= 1
    # Search right of peak
    right_idx = peak_idx
    while right_idx < len(gain_dB) - 1 and gain_dB[right_idx] > half_power:
        right_idx += 1

    # Linear interpolation for more accurate crossing points
    if left_idx < peak_idx:
        t1, t2 = theta_deg[left_idx], theta_deg[left_idx+1]
        g1, g2 = gain_dB[left_idx], gain_dB[left_idx+1]
        theta_left = t1 + (half_power - g1) * (t2 - t1) / (g2 - g1)
    else:
        theta_left = theta_deg[left_idx]

    if right_idx > peak_idx:
        t1, t2 = theta_deg[right_idx-1], theta_deg[right_idx]
        g1, g2 = gain_dB[right_idx-1], gain_dB[right_idx]
        theta_right = t1 + (half_power - g1) * (t2 - t1) / (g2 - g1)
    else:
        theta_right = theta_deg[right_idx]

    return theta_right - theta_left

E_plane_dB = 20*np.log10(nf2ff_result.E_norm[0][:, 0] / np.max(nf2ff_result.E_norm[0][:, 0]))
H_plane_dB = 20*np.log10(nf2ff_result.E_norm[0][:, 1] / np.max(nf2ff_result.E_norm[0][:, 1]))

bw_E = calc_beamwidth(theta, E_plane_dB)
bw_H = calc_beamwidth(theta, H_plane_dB)

print(f"E-plane -3dB beamwidth: {bw_E:.1f} degrees")
print(f"H-plane -3dB beamwidth: {bw_H:.1f} degrees")

# 2D pattern plots (E-plane, H-plane)
plt.figure()
plt.plot(theta, 20*np.log10(nf2ff_result.E_norm[0][:, 0] / np.max(nf2ff_result.E_norm[0][:, 0])), label='phi=0 (E-plane)')
plt.plot(theta, 20*np.log10(nf2ff_result.E_norm[0][:, 1] / np.max(nf2ff_result.E_norm[0][:, 1])), label='phi=90 (H-plane)')
plt.xlabel('Theta (deg)')
plt.ylabel('Normalized Gain (dB)')
plt.title(f'Radiation Pattern at {f_res/1e9:.1f} GHz')
plt.legend()
plt.grid(True)
plt.ylim(-40, 0)
plt.savefig(os.path.join(sim_path, 'radiation_pattern.png'), dpi=150)
plt.show()

print(f"Max directivity: {nf2ff_result.Dmax[0]:.2f} ({10*np.log10(nf2ff_result.Dmax[0]):.2f} dBi)")