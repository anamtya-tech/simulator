"""
YAMNet Fine-Tuner — bridge between simulator gt_datasets and the yamnet training repo.

Responsibilities
----------------
  1. Discover gt_datasets and prepare labels.csv for the data_loader.
  2. Merge multiple datasets into a single staging directory (symlinks, no copies).
  3. Launch training (train_yamnet.py) and export (export_finetuned.py) as
     non-blocking subprocesses, streaming output to a log file.
  4. Manage model_store/registry.json — track checkpoints, set active model.
  5. Provide active model paths to the analyzer for seamless inference switching.

Pipeline
--------
  gt_datasets/*/manifest.csv
        │  prepare_training_dir()
        ▼
  labels.csv  (filename relative to audio/, label, fold)
        │  start_training()   →  subprocess  →  log file
        ▼
  model_store/checkpoints/<run>/model.keras
        │  start_export()     →  subprocess  →  log file
        ▼
  model_store/releases/<version>/chatak_yamnet_<v>.tflite
                                 custom_class_map.csv
        │  set_active_model() / deploy_to_odas()
        ▼
  ODAS uses the TFLite — analyzer uses YAMNetSpectrumClassifier with it
"""

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Paths inside the yamnet repo ──────────────────────────────────────────────
YAMNET_REPO      = Path('/home/azureuser/yamnet')
SAVEDMODEL_PATH  = YAMNET_REPO / 'integration' / 'yamnet_core'
TRAIN_SCRIPT     = YAMNET_REPO / 'training' / 'train_yamnet.py'
EXPORT_SCRIPT    = YAMNET_REPO / 'training' / 'export_finetuned.py'
REGISTRY_PATH    = YAMNET_REPO / 'model_store' / 'registry.json'
CHECKPOINTS_DIR  = YAMNET_REPO / 'model_store' / 'checkpoints'
RELEASES_DIR     = YAMNET_REPO / 'model_store' / 'releases'

# ODAS deployment target
ODAS_MODELS_DIR  = Path('/home/azureuser/z_odas_newbeamform/models')

# PYTHONPATH additions for yamnet layer defs (train_yamnet.py also sets these,
# but we add them to the subprocess env as well for belt-and-suspenders)
_YAMNET_PY_PATH = ':'.join([
    str(YAMNET_REPO / 'models' / 'research' / 'audioset' / 'yamnet'),
    str(YAMNET_REPO / 'integration'),
    str(YAMNET_REPO / 'training'),
])


class YAMNetFinetuner:
    """Orchestrates the full fine-tuning pipeline from the simulator side."""

    def __init__(self, output_dir: str):
        self.output_dir           = Path(output_dir)
        self.gt_dir               = self.output_dir / 'gt_datasets'
        self.staging_root         = self.gt_dir / '_staged'
        self.yamnet_datasets_dir  = self.output_dir / 'yamnet_datasets'
        self.odas_staging_root    = self.yamnet_datasets_dir / '_staged'

    # ── Dataset discovery ─────────────────────────────────────────────────────

    def list_gt_datasets(self) -> list[dict]:
        """Return metadata dicts for every valid gt_dataset directory."""
        datasets = []
        if not self.gt_dir.exists():
            return datasets
        for info_path in sorted(self.gt_dir.glob('*/dataset_info.json')):
            ds_dir = info_path.parent
            if ds_dir.name.startswith('_'):   # skip staging dirs
                continue
            try:
                info = json.loads(info_path.read_text())
                datasets.append({
                    'path'           : str(ds_dir),
                    'name'           : ds_dir.name,
                    'n_clips'        : info.get('n_clips', 0),
                    'labels'         : info.get('labels', []),
                    'clips_per_label': info.get('clips_per_label', {}),
                    'fold_counts'    : info.get('fold_counts', {}),
                    'created_at'     : info.get('created_at', ''),
                    'sample_rate'    : info.get('sample_rate', 16000),
                })
            except Exception:
                continue
        return datasets

    def list_odas_datasets(self) -> list[dict]:
        """Return metadata dicts for post-ODAS curator datasets in yamnet_datasets/.

        Reads curator_config.json (if present) to skip any directory tagged as
        unknown_dataset (the rejected-detections pile).
        """
        datasets = []
        if not self.yamnet_datasets_dir.exists():
            return datasets

        # Identify unknown-reject datasets so we can skip them
        unknown_names: set[str] = set()
        cfg_path = self.yamnet_datasets_dir / 'curator_config.json'
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
                ud  = cfg.get('unknown_dataset')
                if ud:
                    unknown_names.add(ud)
            except Exception:
                pass

        for ds_dir in sorted(self.yamnet_datasets_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            if ds_dir.name.startswith('_'):          # skip _staged
                continue
            if ds_dir.name in unknown_names:         # skip rejected-detection pile
                continue
            labels_path = ds_dir / 'labels.csv'
            if not labels_path.exists():
                continue
            if (ds_dir / 'dataset_info.json').exists():  # skip any accidental gt_datasets
                continue
            try:
                df = pd.read_csv(labels_path)
                if 'label' not in df.columns or 'filename' not in df.columns:
                    continue
                fold_col        = df['fold'] if 'fold' in df.columns \
                                  else pd.Series(['train'] * len(df))
                clips_per_label = df['label'].value_counts().to_dict()
                fold_counts     = fold_col.value_counts().to_dict()
                datasets.append({
                    'path'           : str(ds_dir),
                    'name'           : ds_dir.name,
                    'n_clips'        : len(df),
                    'labels'         : sorted(clips_per_label.keys()),
                    'clips_per_label': {str(k): int(v) for k, v in clips_per_label.items()},
                    'fold_counts'    : {str(k): int(v) for k, v in fold_counts.items()},
                    'created_at'     : '',
                    'sample_rate'    : 16000,
                    'dataset_source' : 'odas_curator',
                })
            except Exception:
                continue
        return datasets

    # ── Dataset preparation ───────────────────────────────────────────────────

    def prepare_training_dir(
        self,
        dataset_paths    : list[str],
        inject_bg_clips  : int = 0,
    ) -> tuple[Path, int]:
        """
        Prepare a training-ready directory that data_loader.py can consume.

        Single dataset  → writes labels.csv in-place (no copies/links).
        Multiple datasets → creates a merged staging dir with symlinked wavs.

        inject_bg_clips > 0 will pull that many background clips from GT
        datasets and inject them as a 'background' class.  Injection is
        skipped when the selected datasets already contain a background class.

        Returns (training_dir, n_samples).
        """
        if len(dataset_paths) == 1:
            ds_dir = Path(dataset_paths[0])
            if self._is_odas_dataset(ds_dir):
                return self._prepare_odas_single(ds_dir, inject_bg_clips=inject_bg_clips)
            return self._prepare_single(ds_dir)
        return self._prepare_merged(dataset_paths, inject_bg_clips=inject_bg_clips)

    def _prepare_single(self, ds_dir: Path) -> tuple[Path, int]:
        manifest   = pd.read_csv(ds_dir / 'manifest.csv')
        audio_base = ds_dir / 'audio'
        rows = []
        for _, row in manifest.iterrows():
            wav_abs = Path(row['wav_path'])
            try:
                rel = wav_abs.relative_to(audio_base)
            except ValueError:
                # Fallback: put under label subdir
                rel = Path(str(row['label'])) / wav_abs.name
            rows.append({
                'filename': str(rel),
                'label'   : row['label'],
                'fold'    : row.get('fold', 'train'),
            })
        df = pd.DataFrame(rows)
        df = self._rebalance_folds(df)
        labels_csv = ds_dir / 'labels.csv'
        df.to_csv(labels_csv, index=False)
        return ds_dir, len(df)

    @staticmethod
    def _is_odas_dataset(ds_dir: Path) -> bool:
        """True for post-ODAS curator datasets: no manifest.csv, has labels.csv."""
        return not (ds_dir / 'manifest.csv').exists() and (ds_dir / 'labels.csv').exists()

    def _collect_gt_background(self, max_clips: int, seed: int = 42) -> list[dict]:
        """Return up to max_clips background-class rows from GT manifests.

        Prefers datasets that already have a train fold for background (skips
        datasets where background only landed in val/test).  Within the eligible
        pool, clips are sampled proportionally per fold so the fold distribution
        is preserved.

        Returns a list of dicts: {wav_path, fold}.
        """
        import numpy as np
        rng = np.random.default_rng(seed)

        all_bg: list[dict] = []
        for info_path in sorted(self.gt_dir.glob('*/dataset_info.json')):
            ds_dir = info_path.parent
            if ds_dir.name.startswith('_'):
                continue
            manifest_path = ds_dir / 'manifest.csv'
            if not manifest_path.exists():
                continue
            try:
                manifest = pd.read_csv(manifest_path)
            except Exception:
                continue
            bg = manifest[manifest['label'] == 'background']
            if bg.empty:
                continue
            # Skip datasets where ALL background clips are in val (unusable for train)
            if set(bg['fold'].unique()) <= {'val', 'test'}:
                continue
            for _, row in bg.iterrows():
                all_bg.append({
                    'wav_path': row['wav_path'],
                    'fold'    : row.get('fold', 'train'),
                })

        if not all_bg:
            return []

        if len(all_bg) <= max_clips:
            return all_bg

        # Stratified subsample: keep fold proportions intact
        chosen: list[dict] = []
        by_fold: dict[str, list] = {}
        for r in all_bg:
            by_fold.setdefault(r['fold'], []).append(r)
        for fold, rows in by_fold.items():
            n_take = max(1, int(round(len(rows) / len(all_bg) * max_clips)))
            idx    = rng.choice(len(rows), min(n_take, len(rows)), replace=False)
            chosen.extend(rows[i] for i in idx)

        return chosen[:max_clips]

    def _prepare_odas_single(
        self,
        ds_dir          : Path,
        inject_bg_clips : int = 0,
    ) -> tuple[Path, int]:
        """Prepare a post-ODAS curator dataset (flat audio/, no manifest.csv) for training.

        Creates a staging dir with:
          - a real audio/ directory containing per-file symlinks (allows injection)
          - a clean 3-column labels.csv (filename, label, fold)

        The original labels.csv (25+ columns) is never touched.
        """
        df   = pd.read_csv(ds_dir / 'labels.csv')
        slim = df[['filename', 'label', 'fold']].copy() if 'fold' in df.columns \
               else df[['filename', 'label']].assign(fold='train')
        slim = slim.copy()

        # Stratified train/val/test split when everything is still 'train'
        if set(slim['fold'].unique()) <= {'train'}:
            slim = self._stratified_split(slim)
        else:
            slim = self._rebalance_folds(slim)

        # Staging dir — keeps the original dataset pristine
        key   = hashlib.md5(str(ds_dir).encode()).hexdigest()[:8]
        ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
        stage = self.odas_staging_root / f'{ds_dir.name}_{ts}_{key}'
        stage.mkdir(parents=True, exist_ok=True)

        # Real audio/ dir with per-file symlinks so extra files can be injected
        audio_dir = stage / 'audio'
        audio_dir.mkdir(exist_ok=True)
        src_audio = (ds_dir / 'audio').resolve()
        for wav in src_audio.iterdir():
            if wav.suffix.lower() == '.wav':
                link = audio_dir / wav.name
                if not link.exists():
                    link.symlink_to(wav)

        # Optionally inject background clips from GT datasets
        already_has_bg = 'background' in slim['label'].values
        if inject_bg_clips > 0 and not already_has_bg:
            bg_rows = self._collect_gt_background(inject_bg_clips)
            if bg_rows:
                for row in bg_rows:
                    src_wav   = Path(row['wav_path']).resolve()
                    link_name = f'gt_bg_{src_wav.name}'
                    link_path = audio_dir / link_name
                    if not link_path.exists():
                        link_path.symlink_to(src_wav)
                bg_df = pd.DataFrame([{
                    'filename': f'gt_bg_{Path(r["wav_path"]).name}',
                    'label'   : 'background',
                    'fold'    : r['fold'],
                } for r in bg_rows])
                slim = pd.concat([slim, bg_df], ignore_index=True)

        slim.to_csv(stage / 'labels.csv', index=False)
        return stage, len(slim)

    def _prepare_merged(
        self,
        dataset_paths   : list[str],
        inject_bg_clips : int = 0,
    ) -> tuple[Path, int]:
        """Create a staging directory that merges N datasets via symlinks.

        Handles both GT datasets (read from manifest.csv) and post-ODAS curator
        datasets (read directly from labels.csv, flat audio/) in the same merge.
        """
        key   = hashlib.md5('|'.join(sorted(dataset_paths)).encode()).hexdigest()[:8]
        ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
        stage = self.staging_root / f'merged_{ts}_{key}'
        audio = stage / 'audio'
        audio.mkdir(parents=True, exist_ok=True)

        all_rows: list[dict] = []
        for ds_path in dataset_paths:
            ds_dir = Path(ds_path)
            # Short prefix derived from dataset name (e.g. gt_forest → forest)
            prefix = ds_dir.name.replace('gt_', '').replace('-', '_')[:12]

            if self._is_odas_dataset(ds_dir):
                # ── Post-ODAS curator dataset ──────────────────────────────
                df   = pd.read_csv(ds_dir / 'labels.csv')
                slim = df[['filename', 'label', 'fold']].copy() if 'fold' in df.columns \
                       else df[['filename', 'label']].assign(fold='train')
                slim = slim.copy()
                if set(slim['fold'].unique()) <= {'train'}:
                    slim = self._stratified_split(slim)

                audio_base = ds_dir / 'audio'
                for _, row in slim.iterrows():
                    wav_abs   = (audio_base / str(row['filename'])).resolve()
                    label     = str(row['label'])
                    label_dir = audio / label
                    label_dir.mkdir(exist_ok=True)
                    link_name = f'{prefix}_{Path(str(row["filename"])).name}'
                    link_path = label_dir / link_name
                    if not link_path.exists():
                        link_path.symlink_to(wav_abs)
                    all_rows.append({
                        'filename': f'{label}/{link_name}',
                        'label'   : label,
                        'fold'    : row['fold'],
                    })
            else:
                # ── GT dataset ─────────────────────────────────────────────
                manifest   = pd.read_csv(ds_dir / 'manifest.csv')
                audio_base = ds_dir / 'audio'

                for _, row in manifest.iterrows():
                    wav_abs   = Path(row['wav_path'])
                    label     = str(row['label'])
                    label_dir = audio / label
                    label_dir.mkdir(exist_ok=True)

                    link_name = f'{prefix}_{wav_abs.name}'
                    link_path = label_dir / link_name
                    if not link_path.exists():
                        link_path.symlink_to(wav_abs)

                    all_rows.append({
                        'filename': f'{label}/{link_name}',
                        'label'   : label,
                        'fold'    : row.get('fold', 'train'),
                    })

        labels_csv = stage / 'labels.csv'
        merged_df = pd.DataFrame(all_rows)
        merged_df = self._rebalance_folds(merged_df)

        # Inject background from GT if requested and not already present
        already_has_bg = 'background' in merged_df['label'].values
        if inject_bg_clips > 0 and not already_has_bg:
            bg_rows = self._collect_gt_background(inject_bg_clips)
            for row in bg_rows:
                src_wav   = Path(row['wav_path']).resolve()
                label_dir = audio / 'background'
                label_dir.mkdir(exist_ok=True)
                link_name = f'gt_bg_{src_wav.name}'
                link_path = label_dir / link_name
                if not link_path.exists():
                    link_path.symlink_to(src_wav)
                all_rows.append({
                    'filename': f'background/{link_name}',
                    'label'   : 'background',
                    'fold'    : row['fold'],
                })
            merged_df = pd.DataFrame(all_rows)
            merged_df = self._rebalance_folds(merged_df)

        merged_df.to_csv(labels_csv, index=False)
        return stage, len(merged_df)

    # ── Fold rebalancing ──────────────────────────────────────────────────────

    @staticmethod
    def _rebalance_folds(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
        """
        Ensure every class has at least some training samples.

        When a class's clips all ended up in val/test (e.g. background with
        source_idx=-1 in older datasets), move 70% of its non-train clips into
        train so the model can actually learn it.

        This is a safety net for already-built datasets.  New datasets built
        after the gt_dataset_builder fix won't need this.
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        df  = df.copy()

        for label in df['label'].unique():
            mask       = df['label'] == label
            train_rows = df.index[mask & (df['fold'] == 'train')]
            if len(train_rows) > 0:
                continue  # already has training samples — nothing to do

            # All clips are in val/test — redistribute 70% → train
            all_idx  = df.index[mask].tolist()
            shuffled = rng.permutation(all_idx)
            n_train  = max(1, int(round(len(shuffled) * 0.70)))
            df.loc[shuffled[:n_train], 'fold'] = 'train'
            # Remaining 30% stay in val (already labelled that way)

        return df

    @staticmethod
    def _stratified_split(
        df        : pd.DataFrame,
        val_frac  : float = 0.15,
        test_frac : float = 0.15,
        seed      : int   = 42,
    ) -> pd.DataFrame:
        """Assign val/test folds via per-class stratified sampling (70/15/15).

        Classes with fewer than 3 clips stay entirely in train.
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        df  = df.copy()
        df['fold'] = 'train'

        for label in df['label'].unique():
            idx = df.index[df['label'] == label].tolist()
            n   = len(idx)
            if n < 3:
                continue                          # too few — all in train
            shuffled = rng.permutation(idx)
            n_val  = max(1, int(round(n * val_frac)))
            n_test = max(1, int(round(n * test_frac)))
            df.loc[shuffled[:n_val],              'fold'] = 'val'
            df.loc[shuffled[n_val:n_val + n_test], 'fold'] = 'test'
            # rest remain 'train'

        return df

    # ── Training subprocess ───────────────────────────────────────────────────

    def start_training(
        self,
        training_dir          : str,
        phase1_epochs         : int  = 20,
        phase2_epochs         : int  = 20,
        batch_size            : int  = 32,
        unfreeze_top          : int  = 4,
        run_name              : Optional[str] = None,
        warm_start_checkpoint : Optional[str] = None,
        nickname              : Optional[str] = None,
    ) -> tuple[subprocess.Popen, Path, str]:
        """
        Launch train_yamnet.py as a non-blocking subprocess.

        warm_start_checkpoint: path to a .keras model to warm-start backbone
        weights from instead of loading straight from the base SavedModel.
        nickname: short human-readable label stored in meta.json alongside the
        checkpoint so you can remember what each run was trained for.

        Returns (process, log_path, actual_run_name).
        """
        if not run_name:
            run_name = f'chatak_yamnet_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_dir = CHECKPOINTS_DIR / run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Write human-readable metadata immediately so it's available even if
        # training is interrupted before training_log.json is written.
        meta = {'nickname': nickname or '', 'run_name': run_name,
                'created_at': datetime.now().isoformat()}
        (ckpt_dir / 'meta.json').write_text(json.dumps(meta, indent=2))

        log_path = CHECKPOINTS_DIR / f'{run_name}_train.log'

        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            '--dataset',       str(training_dir),
            '--savedmodel',    str(SAVEDMODEL_PATH),
            '--phase1-epochs', str(phase1_epochs),
            '--phase2-epochs', str(phase2_epochs),
            '--batch-size',    str(batch_size),
            '--unfreeze-top',  str(unfreeze_top),
            '--output-dir',    str(CHECKPOINTS_DIR),
            '--run-name',      run_name,
        ]
        if warm_start_checkpoint:
            cmd += ['--warm-start', str(warm_start_checkpoint)]

        env = {
            **os.environ,
            'PYTHONPATH'            : _YAMNET_PY_PATH,
            'TF_CPP_MIN_LOG_LEVEL'  : '2',
            'TF_USE_LEGACY_KERAS'   : '1',     # keep Keras 2 / tf_keras behaviour
        }

        log_fh   = open(log_path, 'w')
        proc     = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
        return proc, log_path, run_name

    # ── Export subprocess ─────────────────────────────────────────────────────

    def start_export(
        self,
        checkpoint_dir: str,
        version       : str = 'v1.0.0',
    ) -> tuple[subprocess.Popen, Path]:
        """
        Launch export_finetuned.py as a non-blocking subprocess.

        Returns (process, log_path).
        """
        ckpt_dir = Path(checkpoint_dir)
        log_path = ckpt_dir / f'export_{version}.log'
        RELEASES_DIR.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(EXPORT_SCRIPT),
            '--checkpoint', str(ckpt_dir),
            '--version',    version,
            '--output-dir', str(RELEASES_DIR),
        ]

        env = {
            **os.environ,
            'PYTHONPATH'           : _YAMNET_PY_PATH,
            'TF_CPP_MIN_LOG_LEVEL' : '2',
            'TF_USE_LEGACY_KERAS'  : '1',
        }

        log_fh = open(log_path, 'w')
        proc   = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
        return proc, log_path

    # ── Registry management ───────────────────────────────────────────────────

    def get_registry(self) -> dict:
        if not REGISTRY_PATH.exists():
            return {'schema_version': '1', 'models': [], 'active_model': None}
        return json.loads(REGISTRY_PATH.read_text())

    def _save_registry(self, reg: dict) -> None:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY_PATH.write_text(json.dumps(reg, indent=2))

    def set_active_model(self, run_name: str) -> None:
        reg = self.get_registry()
        reg['active_model'] = run_name
        self._save_registry(reg)

    def set_nickname(self, run_name: str, nickname: str) -> None:
        """Update the nickname for a model in the registry and meta.json."""
        nickname = nickname.strip()
        # Update registry
        reg = self.get_registry()
        for m in reg.get('models', []):
            if m.get('run_name') == run_name:
                m['nickname'] = nickname
                break
        self._save_registry(reg)
        # Update / create meta.json in checkpoint dir
        ckpt_dir = CHECKPOINTS_DIR / run_name
        if ckpt_dir.exists():
            meta_path = ckpt_dir / 'meta.json'
            existing = {}
            if meta_path.exists():
                try:
                    existing = json.loads(meta_path.read_text())
                except Exception:
                    pass
            existing.update({'nickname': nickname, 'run_name': run_name})
            meta_path.write_text(json.dumps(existing, indent=2))

    # ── Active model for inference ────────────────────────────────────────────

    def get_active_model_paths(self) -> tuple[Optional[str], Optional[str]]:
        """
        Return (tflite_path, class_map_path) for the currently active model.

        The paths point to:
          model_store/releases/<version>/chatak_yamnet_<v>.tflite
          model_store/releases/<version>/custom_class_map.csv

        Returns (None, None) when no model is active or files are missing.
        """
        reg    = self.get_registry()
        active = reg.get('active_model')
        if not active:
            return None, None

        for entry in reg.get('models', []):
            if entry.get('run_name') != active:
                continue
            tflite = entry.get('tflite_path')
            if not tflite or not Path(tflite).exists():
                return None, None
            # class map lives in the same release directory
            class_map = Path(tflite).parent / 'custom_class_map.csv'
            if class_map.exists():
                return tflite, str(class_map)
            # fallback: training checkpoint class_map.csv
            ckpt_class_map = CHECKPOINTS_DIR / active / 'class_map.csv'
            if ckpt_class_map.exists():
                return tflite, str(ckpt_class_map)

        return None, None

    def get_active_model_info(self) -> Optional[dict]:
        """Return the registry entry for the active model, or None."""
        reg    = self.get_registry()
        active = reg.get('active_model')
        if not active:
            return None
        for entry in reg.get('models', []):
            if entry.get('run_name') == active:
                return entry
        return None

    # ── Checkpoint listing ────────────────────────────────────────────────────

    def list_checkpoints(self) -> list[dict]:
        """Return training_log.json dicts for all completed training runs."""
        if not CHECKPOINTS_DIR.exists():
            return []
        results = []
        for log_path in sorted(CHECKPOINTS_DIR.glob('*/training_log.json')):
            try:
                entry = json.loads(log_path.read_text())
                # Merge meta.json if present (nickname etc.)
                meta_path = log_path.parent / 'meta.json'
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                        entry.setdefault('nickname', meta.get('nickname', ''))
                    except Exception:
                        pass
                results.append(entry)
            except Exception:
                continue
        return results

    def checkpoint_dir_for(self, run_name: str) -> Path:
        return CHECKPOINTS_DIR / run_name

    # ── Log tailing ───────────────────────────────────────────────────────────

    def read_log_tail(self, log_path, n_lines: int = 60) -> str:
        try:
            lines = Path(log_path).read_text().splitlines()
            return '\n'.join(lines[-n_lines:])
        except Exception:
            return '(log not yet available)'

    # ── ODAS deployment ───────────────────────────────────────────────────────

    def deploy_to_odas(self, tflite_path: str, class_map_path: str) -> tuple[bool, str]:
        """
        Copy fine-tuned TFLite + class map to the ODAS models directory.

        The ODAS firmware reads yamnet_core.tflite and yamnet_class_map.csv
        from this directory at startup — no config change needed.

        Returns (success, message).
        """
        try:
            ODAS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            dst_tflite    = ODAS_MODELS_DIR / 'yamnet_core.tflite'
            dst_class_map = ODAS_MODELS_DIR / 'yamnet_class_map.csv'

            # Back up originals if this is the first deploy
            orig_tflite = ODAS_MODELS_DIR / 'yamnet_core_base.tflite'
            if dst_tflite.exists() and not orig_tflite.exists():
                shutil.copy2(dst_tflite, orig_tflite)

            orig_csv = ODAS_MODELS_DIR / 'yamnet_class_map_base.csv'
            if dst_class_map.exists() and not orig_csv.exists():
                shutil.copy2(dst_class_map, orig_csv)

            shutil.copy2(tflite_path, dst_tflite)
            shutil.copy2(class_map_path, dst_class_map)
            return True, f'Deployed to {ODAS_MODELS_DIR}'
        except Exception as exc:
            return False, str(exc)

    def restore_base_odas_model(self) -> tuple[bool, str]:
        """Restore the original base YAMNet model in ODAS models dir."""
        try:
            orig_tflite   = ODAS_MODELS_DIR / 'yamnet_core_base.tflite'
            orig_class_map = ODAS_MODELS_DIR / 'yamnet_class_map_base.csv'
            if not orig_tflite.exists():
                return False, 'No backup found — base model was never overwritten'
            shutil.copy2(orig_tflite,    ODAS_MODELS_DIR / 'yamnet_core.tflite')
            shutil.copy2(orig_class_map, ODAS_MODELS_DIR / 'yamnet_class_map.csv')
            return True, 'Base model restored'
        except Exception as exc:
            return False, str(exc)
