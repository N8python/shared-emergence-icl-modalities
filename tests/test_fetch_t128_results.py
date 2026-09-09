from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.fetch_t128_results import extract_archive


class T128ExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "results.tar.gz"
        self.destination = self.root / "extracted"

    def write_archive(self, names: list[str], link_type: bytes | None = None) -> None:
        with tarfile.open(self.archive, "w:gz") as archive:
            for name in names:
                info = tarfile.TarInfo(name)
                payload = b'{"correct": true}\n'
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            if link_type is not None:
                info = tarfile.TarInfo("results_128/link")
                info.type = link_type
                info.linkname = "../../outside"
                archive.addfile(info)

    def extract(self) -> None:
        extract_archive(self.archive, self.destination, "results_128")

    def test_published_layout_extracts_only_raw_results(self) -> None:
        self.write_archive([
            "analysis_128_full/summary.csv",
            "analyze_canonical_128_now.sh",
            "./results_128/qwen3/clean/trials.json",
            "results_128_backup/unrelated.json",
        ])
        self.extract()
        self.assertEqual(
            (self.destination / "results_128/qwen3/clean/trials.json").read_bytes(),
            b'{"correct": true}\n',
        )
        self.assertEqual([p.name for p in self.destination.iterdir()], ["results_128"])

    def test_single_root_archive_still_extracts(self) -> None:
        self.write_archive(["results_128/trials.json"])
        self.extract()
        self.assertTrue((self.destination / "results_128/trials.json").is_file())

    def test_missing_results_fail_before_extracting_other_files(self) -> None:
        self.write_archive(["analysis_128_full/summary.csv"])
        with self.assertRaisesRegex(ValueError, "no .* subtree"):
            self.extract()
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_paths_cannot_escape_destination_or_selected_subtree(self) -> None:
        for name in ["../outside", "results_128/../outside", str(self.root / "outside")]:
            with self.subTest(name=name):
                self.write_archive(["results_128/trials.json", name])
                with self.assertRaisesRegex(ValueError, "escapes"):
                    self.extract()
                self.assertEqual(list(self.destination.iterdir()), [])
                self.assertFalse((self.root / "outside").exists())

    def test_links_are_rejected_before_extraction(self) -> None:
        for link_type in [tarfile.SYMTYPE, tarfile.LNKTYPE]:
            with self.subTest(link_type=link_type):
                self.write_archive(["results_128/trials.json"], link_type)
                with self.assertRaisesRegex(ValueError, "unsupported member type"):
                    self.extract()
                self.assertEqual(list(self.destination.iterdir()), [])

    def test_result_root_must_be_a_directory(self) -> None:
        self.write_archive(["results_128"])
        with self.assertRaisesRegex(ValueError, "root is not a directory"):
            self.extract()
        self.assertEqual(list(self.destination.iterdir()), [])

    def test_existing_results_are_preserved(self) -> None:
        target = self.destination / "results_128/trials.json"
        target.parent.mkdir(parents=True)
        target.write_text("existing result")
        self.write_archive(["results_128/trials.json"])
        self.extract()
        self.assertEqual(target.read_text(), "existing result")


if __name__ == "__main__":
    unittest.main()
