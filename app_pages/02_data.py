import streamlit as st
import pandas as pd

from returnwise.data.kaggle import EXPECTED_FILES, cli_version, list_competition_files, read_metadata, sync_competition
from returnwise.data.profile import data_dictionary, quality_report
from returnwise.storage import load_active_dataset, save_manual_csv, use_kaggle_training_data

st.caption("WORKSPACE / DATA QUALITY")
st.title("Select and inspect a dataset")
st.write("Use the Kaggle competition connector or import a CSV. ReturnWise keeps the selected dataset locally and reports the source and validation status.")

mode = st.segmented_control("Dataset mode", ["Kaggle connected", "Manual CSV"], default="Kaggle connected", key="dataset_mode")
if mode == "Kaggle connected":
    if not st.session_state.get("kaggle_auto_sync_attempted", False):
        with st.spinner("Checking the validated Kaggle cache and refreshing it if due…"):
            auto_result = sync_competition(force=False)
        st.session_state["kaggle_auto_sync_attempted"] = True
        st.session_state["kaggle_auto_sync_result"] = {
            "ok": auto_result.ok,
            "message": auto_result.message,
            "details": auto_result.details,
        }
    auto_status = st.session_state.get("kaggle_auto_sync_result", {})
    if auto_status.get("ok"):
        st.caption(auto_status.get("message", "Kaggle cache ready."))
    elif auto_status:
        st.warning(auto_status.get("message", "Kaggle cache could not be checked."))
        if auto_status.get("details"):
            st.info(auto_status["details"])
    st.subheader("Kaggle competition")
    st.write("Competition: [E-Commerce Returns: Who Will Send It Back?](https://www.kaggle.com/competitions/retail-return-risk-modeling)")
    st.caption("Required files: " + ", ".join(EXPECTED_FILES))
    meta = read_metadata()
    if meta:
        last_verified = meta.get("last_synced_at") or meta.get("cache_validated_at") or meta.get("last_checked_at", "unknown")
        st.success(f"Validated cache · version {meta.get('dataset_version', 'unknown')} · last verified {last_verified}")
        st.caption(f"Fingerprint (SHA-256): {meta.get('fingerprint_sha256', 'not recorded')}")
        archive = meta.get("archive_validation") or {}
        if archive.get("present") and not archive.get("valid"):
            st.warning("The cached ZIP is incomplete. The required CSV files were validated independently; the ZIP was left untouched.")
    check_col, sync_col = st.columns(2, vertical_alignment="bottom")
    with check_col:
        check_access = st.button("Check Kaggle access", icon=":material/verified_user:", width="stretch")
    with sync_col:
        sync_now = st.button("Synchronize dataset", type="primary", icon=":material/sync:", width="stretch")
    if check_access:
        with st.spinner("Checking Kaggle CLI, authentication, and competition access…"):
            cli = cli_version()
            result = list_competition_files() if cli.ok else cli
        if cli.ok and result.ok:
            st.caption(f"Kaggle CLI: {cli.message}")
            st.success(result.message)
            st.dataframe({"competition files": result.files or []}, hide_index=True, alt="Competition files available through Kaggle CLI")
        else:
            st.error(result.message, icon=":material/error:")
            if result.details:
                st.info(result.details)
    if sync_now:
        with st.spinner("Checking the validated cache and downloading only if needed…"):
            result = sync_competition(force=False)
        if result.ok:
            st.success(result.message)
            st.json(result.metadata or {})
        else:
            st.error(result.message, icon=":material/error:")
            if result.details:
                st.info(result.details)
    st.code("kaggle auth login\nkaggle competitions files -c retail-return-risk-modeling\nkaggle competitions download -c retail-return-risk-modeling -p data/kaggle", language="powershell")
    st.caption("Authentication is handled by Kaggle CLI in your local terminal. Never enter API tokens in this app or in chat.")
    if st.button("Use validated Kaggle train.csv", disabled=not (meta and (meta.get("integrity") or {}))):
        try:
            manifest = use_kaggle_training_data()
            st.session_state["active_manifest_refresh"] = manifest
            st.success("Kaggle training data selected as the active dataset.")
        except (OSError, ValueError, FileNotFoundError) as exc:
            st.error(str(exc))
else:
    st.subheader("Manual CSV import")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], help="A labeled file can be profiled and evaluated. An unlabeled file can be explored and later scored when a model artifact exists.")
    if uploaded and st.button("Import CSV", type="primary", icon=":material/upload_file:"):
        try:
            manifest = save_manual_csv(uploaded.name, uploaded.getvalue())
            st.success(f"Imported {manifest['rows']:,} rows and {len(manifest['columns'])} columns from {manifest['original_name']}.")
        except (OSError, ValueError, UnicodeDecodeError, Exception) as exc:
            st.error(f"CSV import failed: {exc}")

frame, manifest = load_active_dataset()
st.divider()
st.subheader("Active dataset")
if frame is None:
    if manifest and manifest.get("load_error"):
        st.error(manifest["load_error"])
    else:
        st.info("No dataset selected yet.")
else:
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(frame):,}")
    b.metric("Columns", f"{len(frame.columns):,}")
    c.metric("Target available", "Yes" if "returned" in frame.columns else "No")
    st.caption(f"Source: {manifest.get('source', 'unknown')} · Imported or selected: {manifest.get('imported_at', 'unknown')}")
    with st.expander("Preview rows", expanded=True):
        st.dataframe(frame.head(50), hide_index=True, height=360, alt="First 50 rows of the active dataset")
    st.subheader("Automatic data dictionary")
    st.dataframe(data_dictionary(frame), hide_index=True, height=360, alt="Observed schema, role, missingness, examples, and known field descriptions")
    st.subheader("Data quality summary")
    quality = quality_report(frame)
    x, y, z = st.columns(3)
    x.metric("Exact duplicate rows", f"{quality['duplicate_rows']:,}")
    y.metric("Missing cells", f"{int(frame.isna().sum().sum()):,}")
    z.metric("Identifier candidates", f"{len(quality['identifier_columns'])}")
    if not quality["missing"].empty:
        st.dataframe(quality["missing"], hide_index=True, alt="Columns with missing rows and their percentages")
    if quality["duplicate_identifiers"]:
        st.write("Duplicate identifier candidates:")
        st.json(quality["duplicate_identifiers"])
    if quality["invalid_values"]:
        st.write("Domain and target value flags")
        st.dataframe(
            pd.DataFrame([{"column": key, **value} for key, value in quality["invalid_values"].items()]),
            hide_index=True,
            alt="Values that violate documented binary-target or basic domain screening rules",
        )
    if quality["numeric_text_candidates"]:
        st.write("Text fields that look numeric")
        st.json(quality["numeric_text_candidates"])
    if quality["categorical_inconsistencies"]:
        st.write("Potential whitespace or case inconsistencies")
        st.json(quality["categorical_inconsistencies"])
    st.write("Potential leakage candidates for pre-shipment use")
    st.dataframe(quality["leakage_candidates"], hide_index=True, alt="Feature timing risks flagged by name and intended pre-shipment availability")
    if quality["target_counts"] is not None:
        st.write("Target balance")
        st.bar_chart(quality["target_counts"], x="class", y="rows", alt="Observed count for each returned target class")
    st.caption("Outliers are flagged using the 1.5×IQR rule for review; rows are never removed automatically.")
