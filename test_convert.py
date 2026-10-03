import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

import convert as converter


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
                result = runner.invoke(converter.convert, ["ml-1m", "--stop-before-embedding"])
                self.assertEqual(result.exit_code, 0, result.output + repr(result.exception))
                describe.assert_called_once_with("Amélie (2001)")
                self.assertEqual(
                    (directory / "movies.description").read_text(encoding="utf-8"),
                    "1|Existing description\n2|A film. Second line.\n",
                )
                self.assertFalse((directory / "movies.embedding").exists())
                self.assertFalse((directory / "ml-1m.bin").exists())
                result = runner.invoke(converter.convert, ["ml-1m", "--stop-before-embedding"])
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
                result = runner.invoke(converter.convert, ["ml-1m"])
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


if __name__ == "__main__":
    unittest.main()
