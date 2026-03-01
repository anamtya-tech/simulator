# Contributing

## Running locally

```bash
cd /home/azureuser/simulator
streamlit run app.py
```

## Branch & PR flow

- Work in a `dev/<feature>` branch
- Open a PR targeting `main`
- PRs must pass the **Python CI** check (lint + import smoke-test)

## Commit style

```
<type>(<scope>): short description
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`  
Scopes: `analyzer`, `renderer`, `simulator`, `configurator`, `dataset`, `ui`, `ci`

## Adding a new audio source

1. Add WAV files to `~/audio_cache/forest/audio/`
2. Add a row to `~/config/sources.csv`:
   ```
   /home/azureuser/audio_cache/forest/audio/<file>.wav,directional,<Label>
   ```
3. Re-launch the app — the new source appears in the Scene Configurator automatically.

## Hard-coded paths

All absolute paths are isolated to the top of `app.py`:

```python
SOURCES_CSV_PATH = "/home/azureuser/config/sources.csv"
SCENES_DIR       = "/home/azureuser/config/scenes"
OUTPUT_DIR       = "/home/azureuser/simulator/outputs"
ODAS_LOGS_DIR    = "/home/azureuser/z_odas_newbeamform/build/ClassifierLogs"
```

If you move the workspace, update these four constants.

## Related repos

- ODAS firmware: [anamtya-tech/chatak-odas](https://github.com/anamtya-tech/chatak-odas)
- YAMNet model: [anamtya-tech/yamnet](https://github.com/anamtya-tech/yamnet)
