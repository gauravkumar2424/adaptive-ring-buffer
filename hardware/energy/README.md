# Energy measurement, N=4096

STM32F446RE @ 168 MHz, release build (-O2), CAP=4096.
Current measured in series at JP6 (IDD), DM-1150B on the 20 A range.
Zero offset 0.3 mA characterised with leads disconnected; subtracted
from every reading in energy_measurements.csv.

Cycles from DWT_CYCCNT over 2e4 iterations after buffer warm-up
(MODE 6 in energy_measure.c); values are cycles x1000 as read from
g_cyc_x1000.

E = I * 3.3 V * cycles / 168e6, per mode. MODE 4 (signal only) is the
subtracted baseline for modes 1-3; MODE 0 (idle) is a system reference
only and is NOT the energy baseline.

Reported: heap 1.50 +/- 0.25 uJ, cached scan 48.9 +/- 1.8 uJ per
eviction. Intervals propagate +/-1.4 mA current repeatability (n=4).
