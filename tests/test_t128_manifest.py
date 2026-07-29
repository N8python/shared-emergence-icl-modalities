from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
T128_MANIFEST = REPO_ROOT / "configs" / "paper_runs_t128.json"
LEGACY_MANIFEST = REPO_ROOT / "configs" / "legacy_t8_principal_runs.json"
ARTIFACT_MANIFEST = REPO_ROOT / "data" / "t128" / "artifact_manifest.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def trial_count(run: dict) -> int:
    args = run["base_args"]
    index = args.index("--trials-per-program")
    return int(args[index + 1])


def cell_count(manifest: dict) -> int:
    return sum(
        len(condition["shots"])
        for run in manifest["runs"].values()
        for condition in run["conditions"].values()
    )


class T128ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_json(T128_MANIFEST)
        cls.artifact = load_json(ARTIFACT_MANIFEST)

    def test_complete_paper_grid(self) -> None:
        experiment = self.manifest["experiment"]
        groups = (
            experiment["canonical_run_keys"],
            experiment["scale_run_keys"],
            experiment["paper_control_run_keys"],
        )
        flattened = [key for group in groups for key in group]

        self.assertEqual(list(map(len, groups)), [6, 13, 2])
        self.assertEqual(len(flattened), 21)
        self.assertEqual(len(set(flattened)), 21)
        self.assertEqual(set(flattened), set(self.manifest["runs"]))
        self.assertEqual(cell_count(self.manifest), 281)

    def test_every_cell_uses_128_trials(self) -> None:
        self.assertEqual(self.manifest["experiment"]["trials_per_program"], 128)
        for run_key, run in self.manifest["runs"].items():
            with self.subTest(run_key=run_key):
                self.assertEqual(trial_count(run), 128)

    def test_clean_and_deranged_shots_are_paired_where_defined(self) -> None:
        for run_key, run in self.manifest["runs"].items():
            with self.subTest(run_key=run_key):
                conditions = run["conditions"]
                clean = set(conditions["clean"]["shots"])
                deranged = set(conditions["deranged"]["shots"])
                self.assertTrue(deranged <= clean)
                unpaired = clean - deranged
                expected_unpaired = {1} if run_key == "musicroll" else set()
                self.assertEqual(unpaired, expected_unpaired)

    def test_paths_are_checkout_portable(self) -> None:
        default_root = Path(self.manifest["default_output_root"])
        self.assertFalse(default_root.is_absolute())
        for run_key, run in self.manifest["runs"].items():
            with self.subTest(run_key=run_key):
                self.assertFalse(Path(run["default_model"]).is_absolute())
                for condition in run["conditions"].values():
                    self.assertFalse(
                        Path(condition["output_template"]).is_absolute()
                    )

    def test_artifact_totals_match_manifest(self) -> None:
        predictions = (
            cell_count(self.manifest)
            * self.artifact["tasks_per_cell"]
            * self.artifact["trials_per_task"]
        )
        self.assertEqual(self.artifact["run_keys"], len(self.manifest["runs"]))
        self.assertEqual(
            self.artifact["expected_experiment_cells"],
            cell_count(self.manifest),
        )
        self.assertEqual(predictions, 3_596_800)
        self.assertEqual(self.artifact["expected_predictions"], predictions)
        self.assertEqual(self.artifact["trials_per_task"], 128)

    def test_legacy_manifest_remains_explicitly_t8(self) -> None:
        legacy = load_json(LEGACY_MANIFEST)
        for run_key, run in legacy["runs"].items():
            with self.subTest(run_key=run_key):
                self.assertEqual(trial_count(run), 8)

    def test_executable_defaults_are_128_trials(self) -> None:
        sources = (
            REPO_ROOT / "src" / "program_synth.py",
            REPO_ROOT / "src" / "program_synth_evo_smartbatch.py",
            REPO_ROOT / "src" / "run_evo2_h100_clean_deranged.py",
        )
        pattern = re.compile(
            r"--trials-per-program.{0,160}?default\s*=\s*128",
            re.DOTALL,
        )
        for source in sources:
            with self.subTest(source=source.name):
                self.assertRegex(source.read_text(), pattern)

    def test_campaign_mlx_versions_are_pinned(self) -> None:
        requirements = (REPO_ROOT / "requirements.txt").read_text().splitlines()
        self.assertIn("mlx==0.29.4", requirements)
        self.assertIn("mlx-lm==0.29.1", requirements)
        self.assertIn("transformers==4.57.6", requirements)


if __name__ == "__main__":
    unittest.main()
