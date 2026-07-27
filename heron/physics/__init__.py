"""Layer B — the physics engines (one per Instrument).

Each engine turns the Radiance Graph into a physical signal field. Phase 0 ships
thermography; later phases add fluoroscope, kirlian, schlieren, photon, spectral,
biolum — all sharing this layer's plumbing.
"""

from heron.physics.thermal import ThermalParams, ThermalField, synthesize_thermal

__all__ = ["ThermalParams", "ThermalField", "synthesize_thermal"]
