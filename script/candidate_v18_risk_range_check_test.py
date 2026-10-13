"""Temporary-root governance tests; original scalar synthesis is mocked."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import candidate_v18_risk_range_check as checker


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


class ArchiveBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name).resolve()
        archive=self.root/checker.ARCHIVE;archive.parent.mkdir(parents=True,exist_ok=True)
        archive.write_bytes(b"frozen manufactured feedback\n")
        self.digest=checker.sha(archive)
        self.handoff_patch=patch.object(checker.original,"HANDOFF_SHA",self.digest)
        self.handoff_patch.start()
        sources={}
        for name in checker.source_names():
            if name=="HANDOFF.md":
                (self.root/name).write_bytes(b"new live feedback deliberately different\n")
                sources[name]=self.digest
            else:
                p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(("manufactured original source: "+name).encode("utf-8"))
                sources[name]=checker.sha(p)
        for name in checker.NEW_SOURCES:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"manufactured supplemental source\n")
        frozen=dict(schema="candidate_v18_risk_range_freeze_v1",precision_bits=192,label_count=256,
            reviewed_commit=checker.original.REVIEWED_COMMIT,handoff_sha256=self.digest,
            no_new_grid_or_integral=True,probability_mass_renormalization=False,sources=sources)
        put(self.root/checker.FREEZE,frozen)
        freeze_sha=checker.sha(self.root/checker.FREEZE)
        start=dict(source_sha256=sources,freeze_sha256=freeze_sha)
        put(self.root/checker.START,start)
        self.expected=dict(manufactured_scalar="unchanged",new_integrals=0)
        saved=dict(**self.expected,precision_bits=192,label_count=256,
            source_sha256=sources,freeze_sha256=freeze_sha,
            run_start_sha256=checker.sha(self.root/checker.START),
            reviewed_commit=checker.original.REVIEWED_COMMIT,handoff_sha256=self.digest)
        put(self.root/checker.RESULT,saved)
        self.synthesis_patch=patch.object(checker.original,"synthesis",return_value=self.expected)
        self.synthesis=self.synthesis_patch.start()

    def tearDown(self):
        self.synthesis_patch.stop();self.handoff_patch.stop();self.temp.cleanup()

    def test_changed_live_handoff_matching_archive_passes(self):
        before=checker.original_file_hashes(self.root)
        metadata=checker.adaptation_record(self.root)
        put(self.root/checker.ADAPTATION,metadata)
        first=checker.check(self.root)
        (self.root/"HANDOFF.md").write_bytes(b"another later feedback revision\n")
        second=checker.check(self.root)
        self.assertEqual(first,second)
        self.assertEqual(checker.original_file_hashes(self.root),before)
        self.assertFalse(first["original_freeze_rewritten"])
        self.assertEqual(first["new_integrals"],0)

    def test_tampered_archive_rejected_before_scalar_rebuild(self):
        (self.root/checker.ARCHIVE).write_bytes(b"tampered historical feedback\n")
        with self.assertRaises(ValueError):checker.core_check(self.root)
        self.synthesis.assert_not_called()

    def test_other_input_drift_rejected_before_scalar_rebuild(self):
        source=checker.original.SOURCE_PATHS[0]
        (self.root/source).write_bytes(b"changed original input\n")
        with self.assertRaises(ValueError):checker.core_check(self.root)
        self.synthesis.assert_not_called()

    def test_result_and_runstart_bindings_strict(self):
        start=checker.read(self.root/checker.START);start["source_sha256"]={}
        put(self.root/checker.START,start)
        with self.assertRaises(ValueError):checker.core_check(self.root)
        self.synthesis.assert_not_called()

    def test_scalar_mismatch_rejected(self):
        self.synthesis.return_value={"manufactured_scalar":"changed","new_integrals":0}
        with self.assertRaises(ValueError):checker.core_check(self.root)

    def test_adaptation_locks_original_bytes(self):
        metadata=checker.adaptation_record(self.root);put(self.root/checker.ADAPTATION,metadata)
        # A metadata-only change to result still must fail the added byte seal,
        # even if the original scalar fields would continue to match.
        saved=checker.read(self.root/checker.RESULT);saved["created_utc"]="different"
        put(self.root/checker.RESULT,saved)
        with self.assertRaises(ValueError):checker.check(self.root)

    def test_repository_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as outside_dir:
            outside=Path(outside_dir).resolve()/"outside-source"
            outside.write_bytes(b"outside")
            relative_escape="../"+outside.parent.name+"/outside-source"
            for name in (relative_escape,str(outside),"C:relative-drive-escape"):
                with self.assertRaises(ValueError):checker.safe_path(self.root,name)


if __name__=="__main__":unittest.main(verbosity=2)
