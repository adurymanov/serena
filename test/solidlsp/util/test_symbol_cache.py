from __future__ import annotations

import pickle
from pathlib import Path

from solidlsp.util.cache import save_cache
from solidlsp.util.symbol_cache import LazyFileCache, import_pickle_cache


def _cache(tmp_path: Path, version: object = 1, **kwargs: int) -> LazyFileCache[str]:
    return LazyFileCache(tmp_path / "symbols.db", version, **kwargs)


class TestPersistence:
    def test_entries_survive_reopening(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        cache["a.py"] = "alpha"
        cache["b.py"] = "beta"
        cache.save()
        cache.close()

        reopened = _cache(tmp_path)
        assert reopened.get("a.py") == "alpha"
        assert reopened.get("b.py") == "beta"
        assert len(reopened) == 2

    def test_unsaved_entries_are_readable_before_flushing(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        cache["a.py"] = "alpha"
        assert cache.get("a.py") == "alpha"
        assert "a.py" in cache
        assert len(cache) == 1

    def test_missing_key_yields_none(self, tmp_path: Path) -> None:
        assert _cache(tmp_path).get("nope.py") is None

    def test_overwriting_an_entry(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        cache["a.py"] = "alpha"
        cache.save()
        cache["a.py"] = "alpha2"
        cache.save()
        cache.close()
        assert _cache(tmp_path).get("a.py") == "alpha2"
        assert len(_cache(tmp_path)) == 1

    def test_close_flushes_pending_writes(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        cache["a.py"] = "alpha"
        cache.close()
        assert _cache(tmp_path).get("a.py") == "alpha"


class TestVersioning:
    def test_mismatching_version_discards_entries(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path, version=1)
        cache["a.py"] = "alpha"
        cache.close()

        assert _cache(tmp_path, version=2).get("a.py") is None

    def test_matching_composite_version_keeps_entries(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path, version=(4, "swift", 2))
        cache["a.py"] = "alpha"
        cache.close()

        assert _cache(tmp_path, version=(4, "swift", 2)).get("a.py") == "alpha"


class TestBoundedMemory:
    def test_only_the_configured_number_of_entries_is_retained(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path, retained_entries=2, max_pending_entries=1)
        for i in range(10):
            cache[f"f{i}.py"] = f"value{i}"
        cache.save()

        # all values remain readable, but they are re-read from disk rather than held in memory
        assert len(cache._retained) <= 2
        assert [cache.get(f"f{i}.py") for i in range(10)] == [f"value{i}" for i in range(10)]
        assert len(cache._retained) <= 2

    def test_writes_are_flushed_once_the_buffer_is_full(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path, max_pending_entries=3)
        for i in range(3):
            cache[f"f{i}.py"] = f"value{i}"

        assert cache._pending == {}, "buffer should have been flushed without an explicit save"
        assert _cache(tmp_path).get("f0.py") == "value0"


class TestDictCompatibility:
    def test_comparison_with_a_dict(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        assert cache == {}
        assert not cache
        cache["a.py"] = "alpha"
        assert cache
        assert cache == {"a.py": "alpha"}
        assert cache != {}
        assert cache != {"a.py": "beta"}

    def test_keys_and_items(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path, max_pending_entries=1)
        cache["a.py"] = "alpha"
        cache["b.py"] = "beta"
        assert sorted(cache.keys()) == ["a.py", "b.py"]
        assert sorted(cache.items()) == [("a.py", "alpha"), ("b.py", "beta")]


class TestResilience:
    def test_a_corrupt_database_does_not_break_the_session(self, tmp_path: Path) -> None:
        path = tmp_path / "symbols.db"
        path.write_bytes(b"this is not a database")

        cache: LazyFileCache[str] = LazyFileCache(path, 1)
        cache["a.py"] = "alpha"
        cache.save()
        assert cache.get("a.py") == "alpha", "cache should still work as an in-memory cache"

    def test_a_corrupt_entry_is_discarded(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        cache["a.py"] = "alpha"
        cache["b.py"] = "beta"
        cache.close()

        cache = _cache(tmp_path)
        assert cache._conn is not None
        cache._conn.execute("UPDATE entries SET value = ? WHERE key = ?", (b"not a pickle", "a.py"))
        cache._conn.commit()

        assert cache.get("a.py") is None
        assert cache.get("b.py") == "beta"
        assert len(cache) == 1, "the unreadable entry should have been deleted"


class TestLegacyImport:
    def test_pickled_dict_cache_is_imported_and_deleted(self, tmp_path: Path) -> None:
        legacy_path = tmp_path / "symbols.pkl"
        save_cache(str(legacy_path), 1, {"a.py": "alpha", "b.py": "beta"})

        cache = _cache(tmp_path)
        assert import_pickle_cache(legacy_path, cache, 1)

        assert cache.get("a.py") == "alpha"
        assert cache.get("b.py") == "beta"
        assert not legacy_path.exists()

    def test_outdated_pickled_dict_cache_is_discarded(self, tmp_path: Path) -> None:
        legacy_path = tmp_path / "symbols.pkl"
        save_cache(str(legacy_path), 1, {"a.py": "alpha"})

        cache = _cache(tmp_path, version=2)
        assert import_pickle_cache(legacy_path, cache, 2)

        assert cache.get("a.py") is None
        assert not legacy_path.exists()

    def test_unreadable_pickled_dict_cache_is_discarded(self, tmp_path: Path) -> None:
        legacy_path = tmp_path / "symbols.pkl"
        legacy_path.write_bytes(b"not a pickle")

        cache = _cache(tmp_path)
        assert import_pickle_cache(legacy_path, cache, 1)
        assert not legacy_path.exists()


class TestConcurrentProcesses:
    def test_a_second_instance_sees_committed_entries(self, tmp_path: Path) -> None:
        writer = _cache(tmp_path)
        reader = _cache(tmp_path)

        writer["a.py"] = "alpha"
        writer.save()

        assert reader.get("a.py") == "alpha"

    def test_values_are_independent_copies(self, tmp_path: Path) -> None:
        # a value read back from disk is a fresh object; mutating it must not affect the stored entry
        cache: LazyFileCache[list[str]] = LazyFileCache(tmp_path / "symbols.db", 1, retained_entries=1)
        cache["a.py"] = ["alpha"]
        cache["b.py"] = ["beta"]  # evicts "a.py" from the retention window
        cache.save()

        value = cache.get("a.py")
        assert value == ["alpha"]
        assert value is not None
        value.append("mutated")
        assert cache.get("b.py") == ["beta"]

    def test_pickle_protocol_roundtrip_of_nested_values(self, tmp_path: Path) -> None:
        payload = {"hash": "abc", "symbols": [{"name": "Foo", "children": [{"name": "bar"}]}]}
        cache: LazyFileCache[object] = LazyFileCache(tmp_path / "symbols.db", 1)
        cache["a.py"] = payload
        cache.close()

        assert _cache(tmp_path).get("a.py") == pickle.loads(pickle.dumps(payload))


class TestLegacyImportSizeGuard:
    def test_an_oversized_legacy_cache_is_discarded_rather_than_loaded(self, tmp_path: Path) -> None:
        legacy_path = tmp_path / "symbols.pkl"
        save_cache(str(legacy_path), 1, {"a.py": "alpha" * 1000})

        cache = _cache(tmp_path)
        assert not import_pickle_cache(legacy_path, cache, 1, max_size_mb=0.001)

        assert cache.get("a.py") is None
        assert not legacy_path.exists(), "the legacy file should not be left behind, taking up disk space"


class TestReuseAfterClosing:
    def test_a_closed_cache_remains_usable(self, tmp_path: Path) -> None:
        # a language server may be stopped and started again, and its caches must survive that
        cache = _cache(tmp_path)
        cache["a.py"] = "alpha"
        cache.close()

        assert cache.get("a.py") == "alpha"
        cache["b.py"] = "beta"
        cache.close()
        assert cache.get("b.py") == "beta"

    def test_closing_releases_retained_values(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        cache["a.py"] = "alpha"
        cache.close()
        assert cache._retained == {}
        assert cache._pending == {}
