import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

ROOT = Path(__file__).resolve().parent


def load_converter(dataset):
    spec = importlib.util.spec_from_file_location(
        dataset.replace("-", "_") + "_convert", ROOT / dataset / "convert.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = load_converter("ml-1m")
converter_100k = load_converter("ml-100k")


class MovieLens1MTests(unittest.TestCase):
    def test_stop_before_embedding_generates_resumable_descriptions(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            directory = Path("lib/ml-1m")
            directory.mkdir(parents=True)
            (directory / "movies.dat").write_bytes(
                "1::Toy Story (1995)::Animation|Children's|Comedy\n"
                "2::Amélie (2001)::Comedy|Romance\n".encode("ISO-8859-1")
            )
            (directory / "movies.description").write_text(
                "1|Existing description\n", encoding="utf-8"
            )
            with patch.object(converter, "get_description", return_value="A film.\nSecond line.") as describe, patch.object(
                converter, "get_embedding", side_effect=AssertionError("embedding must not run")
            ):
                result = runner.invoke(converter.convert, ["--stop-before-embedding"])
                self.assertEqual(result.exit_code, 0, result.output + repr(result.exception))
                describe.assert_called_once_with("Amélie (2001)")
                self.assertEqual(
                    (directory / "movies.description").read_text(encoding="utf-8"),
                    "1|Existing description\n2|A film. Second line.\n",
                )
                self.assertFalse((directory / "movies.embedding").exists())
                self.assertFalse((directory / "ml-1m.bin").exists())
                result = runner.invoke(converter.convert, ["--stop-before-embedding"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(describe.call_count, 1)

    def test_dump_preserves_ml1m_fields(self):
        import json
        import protocol_pb2

        runner = CliRunner()
        with runner.isolated_filesystem():
            directory = Path("lib/ml-1m")
            directory.mkdir(parents=True)
            (directory / "movies.dat").write_text("1::Toy Story (1995)::Animation|Children's|Comedy\n")
            (directory / "users.dat").write_text("1::F::1::10::48067\n")
            (directory / "ratings.dat").write_text("1::1::5::978300760\n")
            (directory / "movies.description").write_text("1|A toy adventure.\n")
            (directory / "movies.embedding").write_text("1|0.1|0.2\n")
            with patch.object(converter, "get_description", side_effect=AssertionError("cached")), patch.object(
                converter, "get_embedding", side_effect=AssertionError("cached")
            ):
                result = runner.invoke(converter.convert, [])
            self.assertEqual(result.exit_code, 0, result.output + repr(result.exception))
            with (directory / "ml-1m.bin").open("rb") as dump:
                records = []
                for marker, cls in [(-1, protocol_pb2.User), (-2, protocol_pb2.Item), (-3, protocol_pb2.Feedback)]:
                    self.assertEqual(int.from_bytes(dump.read(8), "little", signed=True), marker)
                    size = int.from_bytes(dump.read(8), "little")
                    records.append(cls.FromString(dump.read(size)))
                self.assertEqual(dump.read(8), bytes(8))
                self.assertEqual(dump.read(), b"")
            user, item, feedback = records
            self.assertEqual(json.loads(user.labels), {"gender": "F", "age": 1, "occupation": "K-12 student", "zip_code": "48067"})
            self.assertEqual(list(item.categories), ["Animation", "Children's", "Comedy"])
            self.assertEqual(item.comment, "Toy Story (1995)")
            self.assertEqual(item.timestamp.ToDatetime().year, 1995)
            self.assertEqual(json.loads(item.labels)["embedding"], [0.1, 0.2])
            self.assertEqual(feedback.value, 5)
            self.assertEqual(feedback.timestamp.seconds, 978300760)


class StandaloneScriptTests(unittest.TestCase):
    def test_scripts_run_directly(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            for dataset in ("ml-100k", "ml-1m"):
                with self.subTest(dataset=dataset):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / dataset / "convert.py"), "--help"],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("--stop-before-embedding", result.stdout)
                    self.assertNotIn("DATASET", result.stdout)

    def test_ml100k_stops_after_descriptions(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            directory = Path("lib/ml-100k")
            directory.mkdir(parents=True)
            Path("lib/ml-100k.zip").touch()
            (directory / "u.item").write_text("1|Toy Story (1995)|\n")
            with patch.object(converter_100k, "get_description", return_value="A toy adventure."), patch.object(
                converter_100k, "get_embedding", side_effect=AssertionError("embedding must not run")
            ):
                result = runner.invoke(converter_100k.convert, ["--stop-before-embedding"])
            self.assertEqual(result.exit_code, 0, result.output + repr(result.exception))
            self.assertEqual((directory / "u.description").read_text(), "1|A toy adventure.\n")
            self.assertFalse((directory / "u.embedding").exists())


if __name__ == "__main__":
    unittest.main()
