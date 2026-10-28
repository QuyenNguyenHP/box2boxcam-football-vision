"""Submission guards must never create duplicate paid jobs."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from huggingface_hub.errors import HfHubHTTPError

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/mobile_training_hf.py"
spec = importlib.util.spec_from_file_location("mobile_job_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "job.json"
        self.config = dict(dataset_repo="owner/data", model_repo="owner/model",
                           hardware="a10g-small", job_timeout="90m", experiment_id="experiment")
        self.record = dict(config=self.config, dataset_revision="data_sha", code_revision="code_sha",
                           archive_sha256="archive_sha", status="submitting")
        self.remote_config = self.root / "config.json"
        self.remote_config.write_text(json.dumps(self.config))
        self.save()
        self.api_patch = patch.object(launcher, "HfApi")
        self.api = self.api_patch.start().return_value
        self.addCleanup(self.api_patch.stop)
        self.api.whoami.return_value = {"name": "owner"}
        self.api.repo_info.return_value = SimpleNamespace(private=True)
        self.api.run_job.return_value = SimpleNamespace(id="job1", url="https://example.com/job1",
                                                        status=SimpleNamespace(stage="SCHEDULING"))
        for name, value in (("get_token", "test-token"), ("hf_hub_download", str(self.remote_config))):
            mock = patch.object(launcher, name, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)

    def save(self):
        self.path.write_text(json.dumps(self.record))

    def error(self, code):
        response = httpx.Response(code, request=httpx.Request("POST", "https://example.com/jobs"))
        return HfHubHTTPError("rejected", response=response)

    def test_success_records_job_and_pinned_revisions(self):
        result = launcher.submit_prepared(self.config, self.record, self.path)
        self.assertEqual(result["job_id"], "job1")
        self.assertEqual(json.loads(self.path.read_text())["status"], "SCHEDULING")
        self.api.run_job.assert_called_once()
        kwargs = self.api.run_job.call_args.kwargs
        self.assertEqual(kwargs["env"]["PITCHLENS_CODE_REVISION"], "code_sha")
        self.assertEqual(kwargs["timeout"], "90m")

    def test_402_is_recorded_but_not_automatically_retried(self):
        self.api.run_job.side_effect = self.error(402)
        with self.assertRaises(HfHubHTTPError):
            launcher.submit_prepared(self.config, self.record, self.path)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["status"], "rejected_insufficient_credits")
        self.assertEqual(saved["submission_http_status"], 402)
        self.assertNotIn("job_id", saved)
        self.api.run_job.assert_called_once()

    def test_explicit_retry_only_after_a_confirmed_402(self):
        self.record.update(status="rejected_insufficient_credits", submission_http_status=402)
        self.save()
        launcher.submit_prepared(self.config, self.record, self.path, retry_rejected=True)
        self.api.run_job.assert_called_once()
        with self.assertRaises(ValueError):
            launcher.submit_prepared(self.config, self.record, self.path, retry_rejected=True)
        self.api.run_job.assert_called_once()

    def test_ambiguous_failure_remains_non_retryable(self):
        self.api.run_job.side_effect = self.error(500)
        with self.assertRaises(HfHubHTTPError):
            launcher.submit_prepared(self.config, self.record, self.path)
        self.record = json.loads(self.path.read_text())
        with self.assertRaises(ValueError):
            launcher.submit_prepared(self.config, self.record, self.path, retry_rejected=True)
        self.api.run_job.assert_called_once()

    def test_changed_config_or_existing_job_cannot_be_submitted(self):
        with self.assertRaises(ValueError):
            launcher.submit_prepared({**self.config, "job_timeout": "3h"}, self.record, self.path)
        self.record["job_id"] = "existing"
        self.save()
        with self.assertRaises(ValueError):
            launcher.submit_prepared(self.config, self.record, self.path)
        self.api.run_job.assert_not_called()

    def test_new_reservation_also_blocks_an_already_prepared_job(self):
        self.config["dataset"] = "dataset"
        self.save()
        (self.root / "dataset").mkdir()
        (self.root / "dataset/manifest.json").write_text(json.dumps([
            {"split": "train", "match_id": "reserved_match"}]))
        (self.root / "configs").mkdir()
        (self.root / "configs/reserved_sessions.json").write_text(json.dumps({
            "schema_version": 1, "sessions": [{"match_id": "reserved_match",
            "videos": [{"source_video_sha256": "abc", "source_video": "clip.mov"}]}]}))
        with patch.object(launcher, "ROOT", self.root), self.assertRaisesRegex(ValueError, "Reserved field session"):
            launcher.submit_prepared(self.config, self.record, self.path)
        self.api.run_job.assert_not_called()


if __name__ == "__main__":
    unittest.main()
