"""Explicit prepare/launch/status/collect commands for the private GPU run."""

from __future__ import annotations

import argparse
import fcntl
import json
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    get_token,
    hf_hub_download,
    snapshot_download,
)
from huggingface_hub.errors import HfHubHTTPError

from box2boxcam.data.build_mobile_dataset import build, sha256
from box2boxcam.data.reserved_sessions import assert_no_reserved_training

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    return json.loads(path.read_text())


def submit_prepared(config, record, record_path, *, retry_rejected=False):
    """Submit pinned uploads; only an explicit HTTP 402 rejection is retryable."""
    with record_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = json.loads(record_path.read_text())
        if current != record or current["config"] != config or current.get("job_id"):
            raise ValueError("Submission record changed or a job already exists")
        if retry_rejected:
            if (current.get("status") != "rejected_insufficient_credits"
                    or current.get("submission_http_status") != 402):
                raise ValueError("Never retry a submitted or ambiguous job automatically")
        elif current["status"] != "submitting":
            raise ValueError("Unexpected initial submission state")
        # Reservations can be added after uploads were prepared, including before
        # an explicitly retried submission. Check again before any paid job call.
        if config.get("dataset"):
            manifest = ROOT / config["dataset"] / "manifest.json"
            assert_no_reserved_training(ROOT, json.loads(manifest.read_text()))
        token = get_token()
        if not token:
            raise RuntimeError("Missing Hugging Face login")
        api = HfApi()
        account = api.whoami()["name"]
        for key, kind, revision_key in (("dataset_repo", "dataset", "dataset_revision"),
                                        ("model_repo", "model", "code_revision")):
            if config[key].split("/")[0] != account:
                raise ValueError("Unexpected repository owner")
            if not api.repo_info(config[key], repo_type=kind, revision=record[revision_key]).private:
                raise ValueError("Prepared repositories must remain private")
        remote_config = hf_hub_download(repo_id=config["model_repo"],
                                       filename="code/config.json", revision=record["code_revision"])
        if json.loads(Path(remote_config).read_text()) != config:
            raise ValueError("Pinned worker configuration differs from requested configuration")
        env = {
            "BOX2BOXCAM_MODEL_REPO": config["model_repo"], "BOX2BOXCAM_DATASET_REPO": config["dataset_repo"],
            "BOX2BOXCAM_CODE_REVISION": record["code_revision"],
            "BOX2BOXCAM_DATASET_REVISION": record["dataset_revision"],
            "BOX2BOXCAM_ARCHIVE_SHA256": record["archive_sha256"],
            "MPLBACKEND": "Agg", "YOLO_AUTOINSTALL": "false",
        }
        bootstrap = (
            "import os,runpy,subprocess,sys; "
            "subprocess.check_call([sys.executable,'-m','pip','install','--no-cache-dir',"
            "'ultralytics==8.4.153','huggingface-hub>=1.4,<2','numpy<2','Pillow<13']); "
            "subprocess.check_call([sys.executable,'-m','pip','install','--no-cache-dir','--no-deps',"
            "'opencv-python-headless==4.11.0.86']); "
            "from huggingface_hub import hf_hub_download; "
            "p=hf_hub_download(repo_id=os.environ['BOX2BOXCAM_MODEL_REPO'],"
            "filename='code/train_mobile_hf.py',revision=os.environ['BOX2BOXCAM_CODE_REVISION']); "
            "runpy.run_path(p,run_name='__main__')"
        )
        record["status"] = "submitting"
        record.pop("submission_http_status", None)
        record.pop("message", None)
        # Persist before the call; ambiguous network failures remain non-retryable.
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        try:
            job = api.run_job(image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime",
                              command=["python", "-c", bootstrap], env=env, secrets={"HF_TOKEN": token},
                              flavor=config["hardware"], timeout=config["job_timeout"], name=config["experiment_id"])
        except HfHubHTTPError as error:
            if error.response is not None and error.response.status_code == 402:
                record.update(status="rejected_insufficient_credits", submission_http_status=402,
                              message="HF Jobs rejected the submission: insufficient prepaid credits")
                record_path.write_text(json.dumps(record, indent=2) + "\n")
            raise
        record.update(job_id=job.id, job_url=job.url, status=job.status.stage)
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        return record


def launch(config, record_path):
    if record_path.exists():
        raise FileExistsError("Job already recorded: use status/collect, never duplicate a paid run")
    if not config["repositories_private"]:
        raise ValueError("This experiment requires private repositories")
    token = get_token()
    if not token:
        raise RuntimeError("Missing Hugging Face login")
    api = HfApi()
    account = api.whoami()["name"]
    for key in ("dataset_repo", "model_repo"):
        if config[key].split("/")[0] != account:
            raise ValueError("Unexpected repository owner")
    dataset = ROOT / config["dataset"]
    if not dataset.exists():
        build(ROOT, dataset, include_tv=config.get("include_tv", False),
              extra_train_sources=config.get("extra_train_sources", ()),
              baseline_dataset=config.get("baseline_dataset", "mobile_v001"))
    assert_no_reserved_training(ROOT, json.loads((dataset / "manifest.json").read_text()))
    transfer_preflight = None
    if config.get("sports_ball_transfer"):
        if (config["sports_ball_transfer"] not in {"retain_full_classifier", "continue_retained_classifier"}
                or config.get("class_name") != "sports ball"):
            raise ValueError("Unexpected sports ball transfer protocol")
        continuing = config["sports_ball_transfer"] == "continue_retained_classifier"
        if continuing != (config.get("initialization") == "previous_checkpoint"):
            raise ValueError("Continuation protocol and initialization disagree")
        import torch
        from ultralytics import YOLO

        from box2boxcam.training.sports_ball_transfer import (
            retain_sports_ball,
            verify_dense_predictions,
        )
        torch.set_num_threads(2)
        base_model = config.get("base_model", "yolo26n.pt")
        source_path = ROOT / (config["previous_weights"] if continuing else f"models/checkpoints/{base_model}")
        source = YOLO(str(source_path)).model
        target, transfer_preflight = retain_sports_ball(source)
        generator = torch.Generator().manual_seed(42)
        transfer_preflight["dense_prediction_check"] = verify_dense_predictions(
            source, target, torch.rand(1, 3, 320, 320, generator=generator))
        expected_class, expected_count = (0, 1) if continuing else (32, 80)
        if (transfer_preflight["source_class_id"] != expected_class
                or transfer_preflight["source_classes"] != expected_count
                or transfer_preflight["classification_hidden_widths"] != config.get("classification_hidden_widths", [80] * 6)):
            raise ValueError("Unexpected sports-ball head; refusing information-losing rebuild")
        transfer_preflight["protocol"] = config["sports_ball_transfer"]
        transfer_preflight["input_weights_sha256"] = sha256(source_path)
        transfer_preflight["dataset_manifest_sha256"] = sha256(dataset / "manifest.json")
        print("Local preflight passed: " + ("fine-tuned checkpoint preserved" if continuing else "COCO sports-ball transfer preserved"), flush=True)
    reused = None
    if config.get("reuse_dataset_record"):
        reused = json.loads((ROOT / config["reuse_dataset_record"]).read_text())
        if (reused["config"]["dataset"] != config["dataset"]
                or reused["config"]["dataset_repo"] != config["dataset_repo"]
                or config.get("include_tv", False)):
            raise ValueError("Reused dataset is not the exact modern-only reference")
        # Validate the unchanged local frozen export before reusing the same cloud revision.
        for row in json.loads((dataset / "manifest.json").read_text()):
            for folder, suffix, key in (("images", ".jpg", "image_sha256"), ("labels", ".txt", "label_sha256")):
                if sha256(dataset / row["split"] / folder / (row["file_id"] + suffix)) != row[key]:
                    raise ValueError("Frozen reference images or annotations were changed")
        if not api.repo_info(config["dataset_repo"], repo_type="dataset", revision=reused["dataset_revision"]).private:
            raise ValueError("The reused dataset must remain private")
    repositories = [(config["model_repo"], "model")]
    if not reused:
        repositories.insert(0, (config["dataset_repo"], "dataset"))
    for repo, kind in repositories:
        api.create_repo(repo_id=repo, repo_type=kind, private=True, exist_ok=False)
        if not api.repo_info(repo_id=repo, repo_type=kind).private:
            raise RuntimeError("Repository is not private; stopping before uploading images")
    if reused:
        data_revision, digest = reused["dataset_revision"], reused["archive_sha256"]
        print("Reusing the exact frozen modern-only HF dataset revision; no image upload", flush=True)
    else:
        # Temporary transport archive only: JPEG data is not permanently duplicated locally.
        with tempfile.TemporaryDirectory(prefix="box2boxcam-mobile-upload-") as temporary:
            archive = Path(temporary) / "dataset.tar"
            with tarfile.open(archive, "w") as bundle:
                for path in sorted(dataset.rglob("*")):
                    if path.is_file() and path.suffix != ".cache":
                        # Each image gets its own regular archive member despite local hardlinks.
                        bundle.add(path, arcname=str(path.relative_to(dataset)), recursive=False)
            digest = sha256(archive)
            print(f"Private dataset upload: {archive.stat().st_size / 1024**2:.1f} MiB", flush=True)
            data_commit = api.upload_file(path_or_fileobj=archive, path_in_repo="dataset.tar",
                                          repo_id=config["dataset_repo"], repo_type="dataset",
                                          commit_message="Frozen modern training and validation; no test images")
            data_revision = data_commit.oid
    files = {
        "code/train_mobile_hf.py": ROOT / "scripts/train_mobile_hf.py",
        "code/evaluate.py": ROOT / "src/box2boxcam/training/evaluate.py",
        f"inputs/{config.get('base_model', 'yolo26n.pt')}": ROOT / "models/checkpoints" / config.get("base_model", "yolo26n.pt"),
        config.get("reference_filename", "inputs/previous_luna_best.pt"): ROOT / config["previous_weights"],
    }
    if config.get("sports_ball_transfer") in {"retain_full_classifier", "continue_retained_classifier"}:
        files["code/sports_ball_transfer.py"] = ROOT / "src/box2boxcam/training/sports_ball_transfer.py"
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=path) for name, path in files.items()]
    operations.append(CommitOperationAdd(path_in_repo="code/config.json", path_or_fileobj=(json.dumps(config, indent=2) + "\n").encode()))
    if transfer_preflight:
        operations.append(CommitOperationAdd(path_in_repo="code/transfer_preflight.json",
                                            path_or_fileobj=(json.dumps(transfer_preflight, indent=2) + "\n").encode()))
    code_commit = api.create_commit(repo_id=config["model_repo"], operations=operations,
                                    commit_message="Reproducible mobile experiment code and baseline weights")
    hardware = next(h for h in api.list_jobs_hardware() if h.name == config["hardware"])
    record = {"config": config, "dataset_revision": data_revision, "code_revision": code_commit.oid,
              "archive_sha256": digest, "status": "submitting", "repositories_private": True}
    if transfer_preflight:
        record["transfer_preflight"] = transfer_preflight
    record["hardware_price_usd_per_hour"] = hardware.unit_cost_usd * 60
    record_path.parent.mkdir(parents=True, exist_ok=True)
    # Guard against accidental retries after an ambiguous API/network failure.
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return submit_prepared(config, record, record_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "launch", "resubmit", "status", "collect"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mobile_nano_960.json")
    args = parser.parse_args()
    config = load_config(args.config)
    record_path = ROOT / "models/reports" / (config["experiment_id"] + "_job.json")
    if args.action == "prepare":
        print(json.dumps(build(ROOT, ROOT / config["dataset"], include_tv=config.get("include_tv", False),
                               extra_train_sources=config.get("extra_train_sources", ()),
                               baseline_dataset=config.get("baseline_dataset", "mobile_v001")), ensure_ascii=False, indent=2))
    elif args.action == "launch":
        print(json.dumps(launch(config, record_path), indent=2))
    elif args.action == "resubmit":
        record = json.loads(record_path.read_text())
        print(json.dumps(submit_prepared(config, record, record_path, retry_rejected=True), indent=2))
    else:
        record = json.loads(record_path.read_text())
        if not record.get("job_id"):
            print(json.dumps({"status": record["status"], "message": record.get("message"),
                              "training_started": False}, indent=2))
            if args.action == "collect":
                raise RuntimeError("No GPU job exists; nothing to collect")
            return
        api = HfApi()
        job = api.inspect_job(job_id=record["job_id"])
        record["status"] = job.status.stage
        record["message"] = job.status.message
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({key: record.get(key) for key in ("job_id", "job_url", "status", "message")}, indent=2))
        if args.action == "status":
            lines = []
            for line in api.fetch_job_logs(job_id=record["job_id"], follow=False, tail=15):
                lines.extend(str(line).splitlines())
            # A single HF log chunk can contain many progress updates.
            for line in lines[-15:]:
                print(line)
        elif job.status.stage != "COMPLETED":
            raise RuntimeError("Job is not complete: do not treat progress checkpoints as final")
        else:
            destination = ROOT / "models/checkpoints/baseline_runs" / config["experiment_id"]
            revision = api.model_info(config["model_repo"]).sha
            snapshot_download(repo_id=config["model_repo"], revision=revision, local_dir=destination,
                              allow_patterns=["artifacts/*", "code/config.json"])
            report = json.loads((destination / "artifacts/training_report.json").read_text())
            if report["status"] != "complete":
                raise RuntimeError("Incomplete training report")
            record.update(result_revision=revision, local_result=str(destination), validation=report["validation"])
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            print(f"Collected: {destination}")


if __name__ == "__main__":
    main()
