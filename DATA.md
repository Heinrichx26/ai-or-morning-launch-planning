# Data Availability and Local Regeneration Notes

This repository does not store raw public-data downloads, processed benchmark tables, or result tables. Reproduction uses public source data downloaded by the user into local folders.

## Public Data Sources

The airline benchmark uses Bureau of Transportation Statistics On-Time Performance records. The relevant source is the United States Department of Transportation Bureau of Transportation Statistics On-Time Performance database:

https://www.transtats.bts.gov/

The weather layer uses Iowa Environmental Mesonet airport observations, including Automated Surface Observing System, Automated Weather Observing System, and Meteorological Aerodrome Report fields:

https://mesonet.agron.iastate.edu/request/download.phtml

Taxi fuel and carbon dioxide conversion uses public landing-and-take-off factors:

- ICAO Doc 9889 *Airport Air Quality Manual* (default taxi-out time).
- EMEP/EEA Air Pollutant Emission Inventory Guidebook, 1.A.3.a Aviation.
- ICAO Aircraft Engine Emissions Databank (idle fuel flow): https://www.easa.europa.eu/en/domains/environment/icao-aircraft-engine-emissions-databank
- U.S. EPA GHG Emission Factors Hub (Jet A-1 CO2 index): https://www.epa.gov/climateleadership/ghg-emission-factors-hub

The conversion script is `python -m src.trc_launch.environmental_eval --mode convert`.

The public transfer audit uses General Transit Feed Specification feeds from Bay Area Rapid Transit, the Massachusetts Bay Transportation Authority, and TriMet:

https://www.bart.gov/dev/schedules/google_transit.zip

https://cdn.mbta.com/MBTA_GTFS.zip

https://developer.trimet.org/schedule/gtfs.zip

## Expected Local Layout

Local regenerated data should be placed under `data/`, which is ignored by version control:

```text
data/
  raw_us_bts/
  weather_asos/
  trc_open_smoke/
```

Generated outputs should be placed under `results/`, which is also ignored by version control:

```text
results/
  trc_smoke/
  trc_ai_strengthening/
```

## Regeneration Workflow

Use `src/trc_launch/download_bts_2025.py` to download Bureau of Transportation Statistics monthly records into `data/raw_us_bts/`. Weather observations are fetched and cached by the weather-enabled benchmark and audit scripts through `src/trc_launch/weather_asos.py`.

Run a small smoke experiment first. After confirming local data access and solver behavior, use the commands in `experiment_manifest.md` to regenerate benchmark selections, audit tables, and result summaries.
