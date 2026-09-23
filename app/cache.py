from cachetools import TTLCache

response_cache = TTLCache(
    maxsize=100,
    ttl=300
)