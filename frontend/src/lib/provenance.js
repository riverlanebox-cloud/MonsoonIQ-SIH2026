/**
 * Where the numbers on screen come from, in one place.
 *
 * The API reports `provenance` from the archive's own metadata:
 *   REAL_IMD_GFS                    IMD 0.25° observations + NOAA GFS forecasts
 *   SYNTHETIC_PHYSICALLY_PLAUSIBLE  the repository's seeded simulator
 */
export const isReal = (p) => typeof p === 'string' && p.toUpperCase().startsWith('REAL');

export function provenanceChip(p) {
  if (isReal(p)) return 'Data: IMD observations + NOAA GFS forecasts';
  if (p === 'SYNTHETIC_PHYSICALLY_PLAUSIBLE') return 'Provenance: synthetic research archive';
  return `Provenance: ${p || 'unknown'}`;
}

export function provenanceFooter(p) {
  if (isReal(p)) {
    return 'Research prototype. Observed rainfall is IMD 0.25° gridded data (Pai et al. 2014); raw '
      + 'forecasts are NOAA GFS, a public stand-in for NCMRWF NCUM. Not an official IMD/NCMRWF '
      + 'product — do not use for public warnings.';
  }
  return 'Research prototype. Forecast fields are produced by a synthetic physically-plausible archive '
    + 'generated inside this repository and are not official IMD/NCMRWF products; do not use for '
    + 'public warnings. Verification figures are internal comparisons on that archive.';
}
