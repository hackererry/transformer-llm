# -*- coding: utf-8 -*-
"""
存储模块
Storage modules
"""
from .database import LegacyDatabase as Database, create_sqlite_db
from .file_storage import FileStorage
from .redis_cache import RedisCache, create_redis_cache

__all__ = ["Database", "create_sqlite_db", "FileStorage", "RedisCache", "create_redis_cache"]
