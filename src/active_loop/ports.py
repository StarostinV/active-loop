"""Ports for the used local servers"""

from dataclasses import dataclass

__all__ = [
    'FETCH_SERVER_PORT',
    'PUSH_SERVER_PORT',
    'PROD_REDIS_CONFIG',
    'TEST_REDIS_CONFIG',
]

@dataclass
class RedisConfig:
    control_url: str
    info_url: str


PROD_REDIS_CONFIG = RedisConfig(
    control_url="tcp://haspp08bliss:60615",
    info_url="tcp://haspp08bliss:60625"
)

TEST_REDIS_CONFIG = RedisConfig(
    control_url="tcp://haszvmp:60615",
    info_url="tcp://haszvmp:60625"
)


FETCH_SERVER_PORT: int = 8080
PUSH_SERVER_PORT: int = 8082
