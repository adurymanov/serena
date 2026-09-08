"""
On-disk, lazily loaded caches for per-file symbol data.

The document symbol caches are keyed by relative file path, so their combined size grows
with the repository: on a large monorepo, the pickled high-level cache reaches several
gigabytes. Materializing such a cache as a plain :class:`dict` means that every Serena
process holds all of it as resident memory for the lifetime of the session, even though a
session typically touches only a small fraction of the files.

:class:`LazyFileCache` keeps the access surface of the dict it replaces
(:meth:`~LazyFileCache.get`, ``__setitem__``, ``__len__``) but stores one pickled record
per key in an SQLite database, unpickling records on access and retaining only a bounded
number of recently used ones. Resident memory therefore scales with the working set
instead of with the repository.
"""

import logging
import pickle
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Hashable, Iterator, Mapping
from pathlib import Path
from typing import Any, Generic, TypeVar

from solidlsp.util.cache import load_cache

log = logging.getLogger(__name__)

V = TypeVar("V")

DEFAULT_RETAINED_ENTRIES = 256
"""number of unpickled entries kept in memory (least recently used ones are dropped)"""

DEFAULT_MAX_PENDING_ENTRIES = 128
"""number of written entries buffered before they are flushed to disk"""


class LazyFileCache(Generic[V]):
    """
    A dict-like, persistent mapping from string keys to pickleable values which keeps only a
    bounded number of values in memory.

    Values are stored individually, so writing a single entry does not require rewriting the
    whole cache, and reading one entry does not require loading the whole cache.

    The mapping is safe to use from multiple threads. Multiple processes may use the same
    cache file concurrently; writes are last-writer-wins per key.
    """

    _SQLITE_TIMEOUT_SECS = 10.0
    _VERSION_KEY = "version"

    def __init__(
        self,
        path: Path | str,
        version: Hashable,
        retained_entries: int = DEFAULT_RETAINED_ENTRIES,
        max_pending_entries: int = DEFAULT_MAX_PENDING_ENTRIES,
    ) -> None:
        """
        :param path: the path of the SQLite database file backing this cache; created if it does not exist.
        :param version: the expected cache version; if the persisted cache was written with a different
            version, its content is discarded (mirroring the semantics of :func:`solidlsp.util.cache.load_cache`).
        :param retained_entries: the maximum number of unpickled values to keep in memory.
        :param max_pending_entries: the maximum number of written values to buffer before flushing them to disk.
        """
        self._path = Path(path)
        self._version = version
        self._retained_entries = max(1, retained_entries)
        self._max_pending_entries = max(1, max_pending_entries)
        self._lock = threading.RLock()
        self._retained: OrderedDict[str, V] = OrderedDict()
        self._pending: dict[str, V] = {}
        self._conn: sqlite3.Connection | None = None
        self._open_failed = False
        self._connection()

    @property
    def path(self) -> Path:
        return self._path

    def _connection(self) -> sqlite3.Connection | None:
        """
        :return: the database connection, establishing it if necessary (which is the case initially as well
            as after :meth:`close`), or None if the database is unusable, in which case this cache degrades
            to a memory-only cache rather than failing the session.
        """
        with self._lock:
            if self._conn is not None or self._open_failed:
                return self._conn
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = self._connect()
            except Exception as e:
                self._open_failed = True
                log.warning("Could not open symbol cache at %s (%s); continuing without an on-disk cache", self._path, e)
            return self._conn

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False, timeout=self._SQLITE_TIMEOUT_SECS)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA busy_timeout={int(self._SQLITE_TIMEOUT_SECS * 1000)}")
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
        conn.commit()
        self._check_version(conn)
        return conn

    def _check_version(self, conn: sqlite3.Connection) -> None:
        """
        Discards all entries if the persisted version does not match the expected one, and persists
        the expected version.
        """
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (self._VERSION_KEY,)).fetchone()
        persisted_version: Any = None
        if row is not None:
            try:
                persisted_version = pickle.loads(row[0])
            except Exception:
                persisted_version = object()  # sentinel which cannot match the expected version
        if row is not None and persisted_version == self._version:
            return
        if row is not None:
            log.info("Cache at %s is outdated (expected version %s, got %s); discarding it", self._path, self._version, persisted_version)
            conn.execute("DELETE FROM entries")
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (self._VERSION_KEY, pickle.dumps(self._version, protocol=pickle.HIGHEST_PROTOCOL)),
        )
        conn.commit()

    def get(self, key: str) -> V | None:
        """
        :param key: the key to look up.
        :return: the value stored for the given key or None if there is none.
        """
        with self._lock:
            pending = self._pending.get(key)
            if pending is not None:
                return pending
            if key in self._retained:
                self._retained.move_to_end(key)
                return self._retained[key]
            conn = self._connection()
            if conn is None:
                return None
            try:
                row = conn.execute("SELECT value FROM entries WHERE key = ?", (key,)).fetchone()
            except Exception as e:
                log.warning("Failed to read entry '%s' from cache %s: %s", key, self._path, e)
                return None
            if row is None:
                return None
            try:
                value: V = pickle.loads(row[0])
            except Exception as e:
                # a single unreadable record must not poison the whole cache
                log.warning("Failed to unpickle entry '%s' from cache %s (%s); discarding the entry", key, self._path, e)
                self._delete(key)
                return None
            self._retain(key, value)
            return value

    def __setitem__(self, key: str, value: V) -> None:
        with self._lock:
            self._pending[key] = value
            self._retained.pop(key, None)  # while pending, the entry is served from the write buffer
            if len(self._pending) >= self._max_pending_entries:
                self._flush()

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with self._lock:
            return key in self._pending or key in self._retained or self._is_persisted(key)

    def __len__(self) -> int:
        with self._lock:
            conn = self._connection()
            if conn is None:
                return len(self._pending)
            try:
                row = conn.execute("SELECT COUNT(*) FROM entries").fetchone()
            except Exception:
                return len(self._pending)
            persisted = int(row[0]) if row is not None else 0
            # entries which are still buffered may or may not already have a persisted counterpart
            unpersisted = sum(1 for key in self._pending if not self._is_persisted(key))
            return persisted + unpersisted

    def _is_persisted(self, key: str) -> bool:
        conn = self._connection()
        if conn is None:
            return False
        try:
            return conn.execute("SELECT 1 FROM entries WHERE key = ?", (key,)).fetchone() is not None
        except Exception:
            return False

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LazyFileCache):
            return self is other
        if isinstance(other, Mapping):
            if len(self) != len(other):
                return False
            return all(isinstance(key, str) and self.get(key) == value for key, value in other.items())
        return NotImplemented

    __hash__ = None

    def keys(self) -> Iterator[str]:
        with self._lock:
            pending_keys = list(self._pending)
            persisted_keys: list[str] = []
            conn = self._connection()
            if conn is not None:
                try:
                    persisted_keys = [row[0] for row in conn.execute("SELECT key FROM entries")]
                except Exception as e:
                    log.warning("Failed to enumerate keys of cache %s: %s", self._path, e)
        yield from pending_keys
        for key in persisted_keys:
            if key not in pending_keys:
                yield key

    def items(self) -> Iterator[tuple[str, V]]:
        for key in list(self.keys()):
            value = self.get(key)
            if value is not None:
                yield key, value

    def update(self, entries: Mapping[str, V]) -> None:
        for key, value in entries.items():
            self[key] = value

    def save(self) -> None:
        """Persists all buffered writes."""
        with self._lock:
            self._flush()

    def close(self) -> None:
        """
        Persists all buffered writes and releases the underlying database connection along with all
        values retained in memory. The cache remains usable; the connection is re-established on the
        next access.
        """
        with self._lock:
            self._flush()
            self._retained.clear()
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception as e:
                    log.warning("Failed to close cache %s: %s", self._path, e)
                self._conn = None

    def _retain(self, key: str, value: V) -> None:
        self._retained[key] = value
        self._retained.move_to_end(key)
        while len(self._retained) > self._retained_entries:
            self._retained.popitem(last=False)

    def _delete(self, key: str) -> None:
        conn = self._connection()
        if conn is None:
            return
        try:
            conn.execute("DELETE FROM entries WHERE key = ?", (key,))
            conn.commit()
        except Exception as e:
            log.warning("Failed to delete entry '%s' from cache %s: %s", key, self._path, e)

    def _flush(self) -> None:
        """Writes all buffered entries in a single transaction; must be called while holding the lock."""
        if not self._pending:
            return
        conn = self._connection()
        if conn is None:
            # memory-only mode: retain what we can and drop the rest
            for key, value in self._pending.items():
                self._retain(key, value)
            self._pending.clear()
            return
        try:
            rows = [(key, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)) for key, value in self._pending.items()]
            with conn:
                conn.executemany("INSERT OR REPLACE INTO entries (key, value) VALUES (?, ?)", rows)
        except Exception as e:
            log.error("Failed to write %d entries to cache %s: %s", len(self._pending), self._path, e)
            return
        for key, value in self._pending.items():
            self._retain(key, value)
        self._pending.clear()


DEFAULT_MAX_IMPORT_SIZE_MB = 256.0
"""maximum size of a legacy cache file that is imported rather than discarded"""


def import_pickle_cache(
    pickle_path: Path,
    cache: LazyFileCache,
    version: Hashable,
    max_size_mb: float = DEFAULT_MAX_IMPORT_SIZE_MB,
) -> bool:
    """
    Imports a legacy cache, which holds all entries in a single pickled dict, into the given cache.
    The legacy file is deleted afterwards, whether or not its content was imported.

    Note that the legacy format requires the entire cache to be loaded into memory at once, which is
    precisely what :class:`LazyFileCache` avoids: unpickling a multi-gigabyte cache file can exceed the
    memory available on the machine. Legacy files larger than `max_size_mb` are therefore discarded
    instead of being imported, and the cache is repopulated on demand.

    :param pickle_path: the path of the legacy pickled cache.
    :param cache: the cache to import the entries into.
    :param version: the expected version of the legacy cache; entries are discarded if it does not match.
    :param max_size_mb: the maximum size, in MB, of a legacy cache file to import; larger files are discarded.
    :return: True if entries were imported, False otherwise.
    """
    size_mb = pickle_path.stat().st_size / 1024**2
    if size_mb > max_size_mb:
        log.info(
            "Legacy cache %s is too large to be imported (%.1f MB > %.1f MB); discarding it. "
            "The cache at %s will be repopulated on demand.",
            pickle_path,
            size_mb,
            max_size_mb,
            cache.path,
        )
        _delete_quietly(pickle_path)
        return False
    log.info("Importing legacy cache %s (%.1f MB) into %s; this happens only once", pickle_path, size_mb, cache.path)
    try:
        entries = load_cache(str(pickle_path), version)
    except Exception as e:
        log.warning("Failed to read legacy cache %s (%s); ignoring it", pickle_path, e)
        entries = None
    if isinstance(entries, dict):
        try:
            cache.update(entries)
            cache.save()
            log.info("Imported %d entries from legacy cache %s", len(entries), pickle_path)
        except Exception as e:
            log.error("Failed to import legacy cache %s: %s", pickle_path, e)
            return False
        finally:
            del entries
    _delete_quietly(pickle_path)
    return True


def _delete_quietly(path: Path) -> None:
    try:
        path.unlink()
    except Exception as e:
        log.warning("Failed to delete %s: %s", path, e)
