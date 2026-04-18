"""
YAMNet Fine-Tuner UI — Streamlit interface for the full fine-tuning pipeline.

Tabs
----
  📁 Dataset   — pick gt_datasets, preview class counts, prepare labels.csv
  🏋️ Train     — configure epochs/batch/unfreeze, start/stop training, live log
  📊 Results   — browse checkpoints, accuracy metrics, class maps
  🚀 Deploy    — export to TFLite, set active model, push to ODAS

Integration with the analyzer
------------------------------
  When the user sets an active model here, the analyzer's "Fine-tuned model"
  strategy automatically picks it up via YAMNetFinetuner.get_active_model_paths().
  No restart needed — the classifier is cached in st.session_state and
  invalidated whenever the active model changes.
"""

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from yamnet_finetuner import (
    YAMNetFinetuner,
    CHECKPOINTS_DIR,
    RELEASES_DIR,
    YAMNET_REPO,
)


class YAMNetFinetunerUI:
    """Renders the fine-tuning pipeline UI inside the Streamlit app."""

    # session_state keys
    _K_DATASETS    = 'ft_selected_datasets'
    _K_TRAIN_DIR   = 'ft_training_dir'
    _K_PROC        = 'ft_train_proc'
    _K_LOG         = 'ft_train_log'
    _K_RUN_NAME    = 'ft_run_name'

    def __init__(self, output_dir: str):
        self.ft = YAMNetFinetuner(output_dir)

    # ── Public entry point ────────────────────────────────────────────────────

    def render(self):
        tab1, tab2, tab3, tab4 = st.tabs([
            "📁 Dataset",
            "🏋️ Train",
            "📊 Results",
            "🚀 Deploy",
        ])
        with tab1:
            self._dataset_tab()
        with tab2:
            self._train_tab()
        with tab3:
            self._results_tab()
        with tab4:
            self._deploy_tab()

    # ── Tab 1 — Dataset ───────────────────────────────────────────────────────

    def _dataset_tab(self):
        st.markdown(
            "Select one or more datasets to train on, then click **Prepare Training Directory**. "
            "You can mix GT datasets and post-ODAS curator datasets in a single training run."
        )

        gt_datasets   = self.ft.list_gt_datasets()
        odas_datasets = self.ft.list_odas_datasets()

        if not gt_datasets and not odas_datasets:
            st.warning(
                "No datasets found. Build a **GT dataset** with the 🏷️ GT Dataset Builder "
                "or run the **🔬 Curator** to produce a post-ODAS training set."
            )
            return

        selected_paths: list[str] = []
        all_datasets: list[dict]  = []

        # ── GT Datasets ───────────────────────────────────────────────────────
        if gt_datasets:
            st.markdown("#### 🏷️ GT Datasets")
            for ds in gt_datasets:
                cols    = st.columns([0.05, 0.95])
                checked = cols[0].checkbox(ds['name'], key=f'ft_ds_{ds["name"]}', value=True,
                                           label_visibility='collapsed')
                if checked:
                    selected_paths.append(ds['path'])
                all_datasets.append(ds)
                with cols[1].expander(
                    f"**{ds['name']}** — {ds['n_clips']} clips · {len(ds['labels'])} classes",
                    expanded=False,
                ):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.caption("Clips per class")
                        for lbl, n in sorted(ds['clips_per_label'].items(), key=lambda x: -x[1]):
                            st.write(f"• {lbl}: **{n}**")
                    with c2:
                        st.caption("Fold distribution")
                        for fold, n in ds['fold_counts'].items():
                            st.write(f"• {fold}: **{n}**")
                    with c3:
                        st.caption("Info")
                        st.write(f"Sample rate: {ds['sample_rate']} Hz")
                        if ds['created_at']:
                            st.write(f"Created: {ds['created_at'][:10]}")

        # ── Post-ODAS Curator Datasets ────────────────────────────────────────
        if odas_datasets:
            st.markdown("#### 🔬 Post-ODAS Curator Datasets")
            st.caption(
                "These were produced by the YAMNet Curator from live ODAS runs. "
                "All folds default to `train`; a stratified 70/15/15 split is applied "
                "automatically when you click **Prepare Training Directory**."
            )
            for ds in odas_datasets:
                cols    = st.columns([0.05, 0.95])
                checked = cols[0].checkbox(ds['name'], key=f'ft_ds_{ds["name"]}', value=True,
                                           label_visibility='collapsed')
                if checked:
                    selected_paths.append(ds['path'])
                all_datasets.append(ds)
                fold_note = (
                    "⚠️ all train — split on prepare"
                    if set(ds['fold_counts'].keys()) <= {'train'}
                    else "/".join(f"{k}:{v}" for k, v in ds['fold_counts'].items())
                )
                with cols[1].expander(
                    f"**{ds['name']}** `[ODAS]` — {ds['n_clips']} clips · "
                    f"{len(ds['labels'])} classes",
                    expanded=False,
                ):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.caption("Clips per class")
                        for lbl, n in sorted(ds['clips_per_label'].items(), key=lambda x: -x[1]):
                            st.write(f"• {lbl}: **{n}**")
                    with c2:
                        st.caption("Fold distribution")
                        for fold, n in ds['fold_counts'].items():
                            st.write(f"• {fold}: **{n}**")
                        if set(ds['fold_counts'].keys()) <= {'train'}:
                            st.caption("↳ auto-split at prepare time")
                    with c3:
                        st.caption("Info")
                        st.write(f"Sample rate: {ds['sample_rate']} Hz")
                        st.write("Source: ODAS curator")

        st.session_state[self._K_DATASETS] = selected_paths

        if not selected_paths:
            st.info("Select at least one dataset above.")
            return

        # ── Summary across selected ───────────────────────────────────────────
        all_labels: set[str] = set()
        total_clips = 0
        for ds in all_datasets:
            if ds['path'] in selected_paths:
                all_labels.update(ds['labels'])
                total_clips += ds['n_clips']

        st.info(
            f"**{len(selected_paths)} dataset(s)** selected — "
            f"**{total_clips}** total clips — "
            f"**{len(all_labels)}** unique classes: `{', '.join(sorted(all_labels))}`"
        )

        # ── Background injection ──────────────────────────────────────────────
        has_bg_already = 'background' in all_labels
        gt_bg_available = sum(
            ds['clips_per_label'].get('background', 0)
            for ds in self.ft.list_gt_datasets()
            # only count datasets where bg has a train fold
            if ds['clips_per_label'].get('background', 0) > 0
               and ds['fold_counts'].get('train', 0) > 0
        )
        if has_bg_already:
            inject_bg = 0
            st.caption("ℹ️ Background class already present in selected datasets — no injection needed.")
        elif gt_bg_available == 0:
            inject_bg = 0
            st.caption("⚠️ No GT background clips available for injection.")
        else:
            inject_bg = st.number_input(
                f"🌿 Background clips to inject from GT ({gt_bg_available} available)",
                min_value=0,
                max_value=gt_bg_available,
                value=min(gt_bg_available, 150),
                step=25,
                help=(
                    "ODAS only tracks localised sources — background silence is never "
                    "captured. Injecting GT background clips teaches the model to output "
                    "low confidence on ambient audio instead of misclassifying it. "
                    "~100–200 clips is a good starting point; clips are stratified by fold."
                ),
            )

        st.divider()

        train_dir = st.session_state.get(self._K_TRAIN_DIR)
        if train_dir:
            st.success(f"✅ Training directory ready: `{Path(train_dir).name}`")

        if st.button("📋 Prepare Training Directory", type="primary", use_container_width=True):
            with st.spinner("Preparing labels.csv …"):
                try:
                    training_dir, n = self.ft.prepare_training_dir(
                        selected_paths,
                        inject_bg_clips=inject_bg,
                    )
                    st.session_state[self._K_TRAIN_DIR] = str(training_dir)
                    # reset any cached classifier so the analyzer picks up the new model
                    st.session_state.pop('_ft_yamnet_obj', None)
                    st.success(f"✅ {n} samples ready in `{training_dir.name}` — now go to **🏋️ Train**")
                except Exception as exc:
                    st.error(f"Preparation failed: {exc}")

    # ── Tab 2 — Train ─────────────────────────────────────────────────────────

    def _train_tab(self):
        train_dir = st.session_state.get(self._K_TRAIN_DIR)
        if not train_dir:
            st.info("👆 Select and prepare a dataset in the **📁 Dataset** tab first.")
            return

        st.info(f"Training from: `{Path(train_dir).name}`")

        # ── Starting weights ─────────────────────────────────────────────────
        checkpoints  = self.ft.list_checkpoints()
        ckpt_options = {c['run_name']: c['model_path'] for c in checkpoints
                        if c.get('model_path') and Path(c['model_path']).exists()}

        weight_mode = st.radio(
            "Starting weights",
            options=["🧬 Pretrained YAMNet (base)"] + list(ckpt_options.keys()),
            index=0,
            horizontal=False,
            help=(
                "**Pretrained YAMNet** — backbone loaded from the base SavedModel, "
                "head trained from scratch. Best for a clean first experiment.\n\n"
                "**Checkpoint** — backbone weights from a previously fine-tuned run. "
                "Useful when re-training on a superset or different dataset split; "
                "backbone + head_fc transfer, custom_predictions reset if class count differs."
            ),
        )
        warm_start_path = (
            None if weight_mode == "🧬 Pretrained YAMNet (base)"
            else ckpt_options[weight_mode]
        )
        if warm_start_path:
            ckpt_meta = next(c for c in checkpoints if c['run_name'] == weight_mode)
            st.caption(
                f"↳ warm-starting from **{weight_mode}** — "
                f"classes: `{', '.join(ckpt_meta.get('classes', []))}` — "
                f"test acc: {ckpt_meta.get('test_accuracy', 0):.1%}"
            )

        # ── Hyperparameter controls ──────────────────────────────────────────
        with st.expander("⚙️ Hyperparameters", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                phase1 = st.slider(
                    "Phase 1 epochs — head only", 5, 60, 20,
                    help="Train Dense head with backbone frozen. Faster, safer for small datasets."
                )
                phase2 = st.slider(
                    "Phase 2 epochs — backbone unfreeze", 0, 60, 20,
                    help="Fine-tune top backbone layers at low LR. Set to 0 to skip."
                )
            with c2:
                batch_size = st.select_slider(
                    "Batch size", options=[8, 16, 32, 64], value=32,
                    help="Smaller = less RAM, more noise in gradients."
                )
                unfreeze_top = st.slider(
                    "Unfreeze top N backbone layers (Phase 2)", 1, 8, 4,
                    help="How many MobileNet layers to unfreeze. 4 is a safe default."
                )
            with c3:
                run_name_input = st.text_input(
                    "Run name (optional)",
                    placeholder="auto-generated timestamp",
                    help="Leave blank for auto: chatak_yamnet_<timestamp>"
                )
                nickname_input = st.text_input(
                    "Model nickname",
                    placeholder="e.g. forest v2 + drones, 17 classes",
                    help="Short human label — stored alongside the model so you remember what it is."
                )
                st.caption(
                    f"Outputs → `{CHECKPOINTS_DIR.relative_to(YAMNET_REPO)}/`"
                )

        proc     = st.session_state.get(self._K_PROC)
        log_path = st.session_state.get(self._K_LOG)

        # ── Running state ────────────────────────────────────────────────────
        if proc is not None:
            rc = proc.poll()

            if rc is None:
                st.warning("⏳ Training in progress …")
                c_stop, c_refresh = st.columns([1, 3])
                with c_stop:
                    if st.button("🛑 Stop", use_container_width=True):
                        proc.terminate()
                        st.session_state[self._K_PROC] = None
                        st.rerun()
                with c_refresh:
                    st.caption("Log auto-refreshes every 3 s")

            elif rc == 0:
                st.success("✅ Training complete! Go to **📊 Results** or **🚀 Deploy**.")
                st.session_state[self._K_PROC] = None

            else:
                st.error(f"❌ Training failed (exit code {rc}) — check log below.")
                st.session_state[self._K_PROC] = None

            # Always show log tail
            if log_path:
                st.subheader("📋 Live training log")
                tail = self.ft.read_log_tail(log_path, n_lines=80)
                st.code(tail, language='')
                # Auto-refresh while still running
                if proc is not None and proc.poll() is None:
                    time.sleep(3)
                    st.rerun()

        # ── Idle state ───────────────────────────────────────────────────────
        else:
            if log_path:
                with st.expander("📋 Last training log"):
                    st.code(self.ft.read_log_tail(log_path), language='')

            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                rn = run_name_input.strip() or None
                nick = nickname_input.strip() or None
                try:
                    proc, log_path, actual_rn = self.ft.start_training(
                        train_dir, phase1, phase2, batch_size, unfreeze_top, rn,
                        warm_start_checkpoint=warm_start_path,
                        nickname=nick,
                    )
                    st.session_state[self._K_PROC]     = proc
                    st.session_state[self._K_LOG]      = str(log_path)
                    st.session_state[self._K_RUN_NAME] = actual_rn
                    # Bust cached classifier
                    st.session_state.pop('_ft_yamnet_obj', None)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to start training: {exc}")

    # ── Tab 3 — Results ───────────────────────────────────────────────────────

    def _results_tab(self):
        checkpoints = self.ft.list_checkpoints()
        if not checkpoints:
            st.info("No completed training runs yet. Train a model first.")
            return

        # Summary table
        rows = []
        reg     = self.ft.get_registry()
        active  = reg.get('active_model')
        for ck in sorted(checkpoints, key=lambda x: x.get('timestamp', ''), reverse=True):
            rn = ck.get('run_name', '')
            rows.append({
                'Active'    : '⭐' if rn == active else '',
                'Run'       : rn,
                'Nickname'  : ck.get('nickname', ''),
                'Date'      : ck.get('timestamp', '')[:10],
                'Dataset'   : Path(ck.get('dataset', '')).name,
                'Classes'   : ', '.join(ck.get('classes', [])),
                'N classes' : ck.get('num_classes', 0),
                'Test Acc'  : f"{ck.get('test_accuracy', 0):.1%}",
                'P1 ep'     : ck.get('phase1_epochs', 0),
                'P2 ep'     : ck.get('phase2_epochs', 0),
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Detail view
        run_names   = [c['run_name'] for c in checkpoints]
        selected_rn = st.selectbox("Inspect run", run_names)
        ck = next((c for c in checkpoints if c['run_name'] == selected_rn), None)
        if ck is None:
            return

        st.divider()
        _nick_key = f'_nick_edit_{selected_rn}'
        _nick_col, _nick_btn = st.columns([4, 1])
        with _nick_col:
            new_nick = st.text_input(
                "🏷️ Nickname",
                value=ck.get('nickname', ''),
                placeholder="Add a nickname…",
                key=_nick_key,
                label_visibility='collapsed',
            )
        with _nick_btn:
            if st.button("💾 Save", key=f'_nick_save_{selected_rn}', use_container_width=True):
                self.ft.set_nickname(selected_rn, new_nick)
                st.rerun()
        m1, m2, m3 = st.columns(3)
        m1.metric("Test Accuracy", f"{ck.get('test_accuracy', 0):.1%}")
        m2.metric("Phase 1 epochs run", ck.get('phase1_epochs', 0))
        m3.metric("Phase 2 epochs run", ck.get('phase2_epochs', 0))

        st.markdown(f"**Classes ({ck.get('num_classes', 0)}):** `{', '.join(ck.get('classes', []))}`")
        st.markdown(f"**Dataset:** `{ck.get('dataset', '')}`")

        # Show class_map.csv if present
        class_map_path = CHECKPOINTS_DIR / selected_rn / 'class_map.csv'
        if class_map_path.exists():
            with st.expander("📋 Class map"):
                st.dataframe(pd.read_csv(class_map_path), hide_index=True)

        # Show training log if present
        log_path = CHECKPOINTS_DIR / f'{selected_rn}_train.log'
        if log_path.exists():
            with st.expander("📋 Training log"):
                st.code(self.ft.read_log_tail(log_path, n_lines=100), language='')

    # ── Tab 4 — Deploy ────────────────────────────────────────────────────────

    def _deploy_tab(self):
        reg    = self.ft.get_registry()
        models = reg.get('models', [])
        active = reg.get('active_model')

        if not models:
            st.info("No models in the registry. Complete a training run first.")
            # Check if there are checkpoints without registry entries
            ckpts = self.ft.list_checkpoints()
            if ckpts:
                st.caption(
                    f"Found {len(ckpts)} checkpoint(s) not yet in registry "
                    "(training may not have finished writing to registry.json)."
                )
            return

        # Current active model banner
        if active:
            active_info = self.ft.get_active_model_info()
            if active_info:
                tflite_ok = bool(active_info.get('tflite_path')) and Path(active_info['tflite_path']).exists()
                st.success(
                    f"⭐ **Active model:** `{active}` — "
                    f"classes: `{', '.join(active_info.get('classes', []))}` — "
                    f"TFLite: {'✅ ready' if tflite_ok else '⚠️ not exported yet'}"
                )

        st.divider()

        for m in reversed(models):
            run   = m.get('run_name', '')
            is_act = run == active
            tflite = m.get('tflite_path')
            tflite_ok = bool(tflite) and Path(tflite).exists()

            nick_suffix = f" · _{m['nickname']}_" if m.get('nickname') else ''
            label = f"{'⭐ ' if is_act else ''}**{run}**{nick_suffix} — acc {m.get('val_accuracy', 0):.1%} — {', '.join(m.get('classes', []))}"
            with st.expander(label, expanded=is_act):

                # ── Inline rename ────────────────────────────────────────────
                _dn_col, _dn_btn = st.columns([4, 1])
                with _dn_col:
                    _new_nick = st.text_input(
                        "Nickname",
                        value=m.get('nickname', ''),
                        placeholder="Add a nickname…",
                        key=f'_dep_nick_{run}',
                        label_visibility='collapsed',
                    )
                with _dn_btn:
                    if st.button("💾", key=f'_dep_nick_save_{run}',
                                 use_container_width=True, help="Save nickname"):
                        self.ft.set_nickname(run, _new_nick)
                        st.rerun()

                c_info, c_actions = st.columns([2, 1])

                with c_info:
                    st.markdown(
                        f"**Classes ({m.get('num_classes', 0)}):** "
                        f"`{', '.join(m.get('classes', []))}`"
                    )
                    st.markdown(f"**Dataset:** `{Path(m.get('dataset', '')).name}`")
                    if tflite_ok:
                        st.markdown(f"**TFLite:** `{Path(tflite).name}`")
                        release_dir = Path(tflite).parent
                        class_map   = release_dir / 'custom_class_map.csv'
                        if class_map.exists():
                            st.markdown(f"**Class map:** `{class_map.name}`")
                    else:
                        st.warning("TFLite not exported yet — export below.")

                with c_actions:
                    # ── Set active ──────────────────────────────────────────
                    if not is_act:
                        if st.button("⭐ Set Active", key=f'act_{run}', use_container_width=True):
                            self.ft.set_active_model(run)
                            st.session_state.pop('_ft_yamnet_obj', None)
                            st.rerun()
                    else:
                        st.success("✅ Active model")

                    # ── Export TFLite ────────────────────────────────────────
                    exp_proc_key = f'_ft_exp_proc_{run}'
                    exp_log_key  = f'_ft_exp_log_{run}'
                    exp_proc = st.session_state.get(exp_proc_key)

                    if not tflite_ok:
                        ver = st.text_input("Version tag", "v1.0.0", key=f'ver_{run}')
                        if exp_proc is None:
                            if st.button("📦 Export TFLite", key=f'exp_{run}', use_container_width=True):
                                ckpt_dir = str(CHECKPOINTS_DIR / run)
                                ep, elp  = self.ft.start_export(ckpt_dir, ver)
                                st.session_state[exp_proc_key] = ep
                                st.session_state[exp_log_key]  = str(elp)
                                st.rerun()
                        else:
                            rc = exp_proc.poll()
                            if rc is None:
                                st.warning("⏳ Exporting…")
                                # show tail
                                elp = st.session_state.get(exp_log_key)
                                if elp:
                                    st.code(self.ft.read_log_tail(elp, 20), language='')
                                time.sleep(3)
                                st.rerun()
                            elif rc == 0:
                                st.success("✅ Export done!")
                                st.session_state.pop(exp_proc_key, None)
                                st.rerun()
                            else:
                                st.error(f"❌ Export failed (code {rc})")
                                st.session_state.pop(exp_proc_key, None)

                    # ── Deploy to ODAS ───────────────────────────────────────
                    if tflite_ok:
                        release_dir = Path(tflite).parent
                        class_map   = str(release_dir / 'custom_class_map.csv')
                        if st.button("🚀 Deploy to ODAS", key=f'dep_{run}', use_container_width=True,
                                     help="Overwrites yamnet_core.tflite in the ODAS models directory. "
                                          "Originals are backed up as *_base.*"):
                            ok, msg = self.ft.deploy_to_odas(tflite, class_map)
                            if ok:
                                st.success(f"✅ {msg}")
                                # mark deployed in registry
                                m['deployed'] = True
                                self.ft._save_registry(reg)
                            else:
                                st.error(f"❌ {msg}")

        st.divider()

        # ── Restore base model ───────────────────────────────────────────────
        with st.expander("🔄 Restore base YAMNet model in ODAS"):
            st.markdown(
                "Restores the original `yamnet_core.tflite` and `yamnet_class_map.csv` "
                "backed up during the first deploy."
            )
            if st.button("🔄 Restore Base Model", use_container_width=True):
                ok, msg = self.ft.restore_base_odas_model()
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")

        # ── Quick-reference inference note ───────────────────────────────────
        with st.expander("ℹ️ How inference works after deploy"):
            st.markdown("""
**In the Analyzer** — select *Fine-tuned model* from the Label Strategy dropdown.
The analyzer loads the active model's `.tflite` + `custom_class_map.csv` via
`YAMNetSpectrumClassifier`, which runs the **same 96 × 257 → mel → TFLite** pipeline
as the standard YAMNet, but outputs your **custom class labels** instead of 521 AudioSet classes.

**In live ODAS** — after clicking *Deploy to ODAS*, the ODAS firmware automatically
uses the new model on the next restart (it reads `yamnet_core.tflite` at launch).
No config changes are needed — the input shape is identical (96 × 64 mel patches).
            """)
