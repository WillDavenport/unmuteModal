"""Service discovery for the TTS, ASR, VLLM etc. Based on Redis."""

import asyncio
import logging
import random
import socket
import time
import typing as tp
from collections import defaultdict
from collections.abc import Awaitable
from functools import partial, wraps

from unmute import metrics as mt
from unmute.exceptions import MissingServiceAtCapacity, MissingServiceTimeout
from unmute.kyutai_constants import LLM_SERVER, STT_SERVER, TTS_SERVER
from unmute.timer import Stopwatch

logger = logging.getLogger(__name__)
SERVICES = {
    "tts": TTS_SERVER,
    "stt": STT_SERVER,
    "llm": LLM_SERVER,
}
K = tp.TypeVar("K", bound=tp.Hashable)
V = tp.TypeVar("V")
S = tp.TypeVar("S", bound="ServiceWithStartup")


def async_ttl_cached(func: tp.Callable[[K], Awaitable[V]], ttl_sec: float = 0.1):
    """Cache an async function with some TTL for the cached values."""
    cache: dict[K, tuple[float, V]] = {}
    locks: dict[K, asyncio.Lock] = defaultdict(asyncio.Lock)

    @wraps(func)
    async def cached(key: K):
        async with locks[key]:
            now = time.time()
            try:
                key_time, value = cache[key]
            except KeyError:
                pass
            else:
                if now - key_time < ttl_sec:
                    return value
            value = await func(key)
            cache[key] = (now, value)
            return value

    return cached


@partial(async_ttl_cached, ttl_sec=0.5)
async def _resolve(hostname: str) -> list[str]:
    *_, ipaddrlist = await asyncio.to_thread(socket.gethostbyname_ex, hostname)
    return ipaddrlist


async def get_instances(service_name: str) -> list[str]:
    url = SERVICES[service_name]
    print(f"=== SERVICE_DISCOVERY: Getting instances for {service_name}, base URL: {url} ===")
    protocol, remaining = url.split("://", 1)
    
    # For secure protocols (wss, https) OR Modal services, don't resolve to IP addresses
    # to avoid SSL certificate validation issues and Modal routing problems
    if protocol in ("wss", "https") or "modal.run" in url:
        print(f"=== SERVICE_DISCOVERY: Using secure protocol {protocol} or Modal service, returning original URL ===")
        return [url]
    
    # Handle URLs with and without explicit ports for non-secure protocols
    if ":" in remaining:
        hostname, port = remaining.split(":", 1)
        print(f"=== SERVICE_DISCOVERY: Resolving hostname {hostname} with port {port} ===")
        ips = list(await _resolve(hostname))
        random.shuffle(ips)
        result = [f"{protocol}://{ip}:{port}" for ip in ips]
        print(f"=== SERVICE_DISCOVERY: Resolved to {len(ips)} IPs: {result} ===")
        return result
    else:
        # No explicit port in URL (e.g., local services)
        hostname = remaining
        print(f"=== SERVICE_DISCOVERY: Resolving hostname {hostname} (no port) ===")
        ips = list(await _resolve(hostname))
        random.shuffle(ips)
        result = [f"{protocol}://{ip}" for ip in ips]
        print(f"=== SERVICE_DISCOVERY: Resolved to {len(ips)} IPs: {result} ===")
        return result


class ServiceWithStartup(tp.Protocol):
    async def start_up(self) -> None:
        """Initiate connection. Should raise an exception if the instance is not ready."""
        ...


async def find_instance(
    service_name: str,
    client_factory: tp.Callable[[str], S],
    timeout_sec: float | None = None,  # Use get_service_discovery_timeout() by default
    max_trials: int = 3,
) -> S:
    print(f"=== SERVICE_DISCOVERY: Finding instance for {service_name} ===")
    # Use the environment-configurable timeout if not specified
    if timeout_sec is None:
        from unmute.kyutai_constants import get_service_discovery_timeout
        timeout_sec = get_service_discovery_timeout()
    
    print(f"=== SERVICE_DISCOVERY: Using timeout {timeout_sec}s for {service_name} ===")
    stopwatch = Stopwatch()
    instances = await get_instances(service_name)
    print(f"=== SERVICE_DISCOVERY: Found {len(instances)} instances for {service_name}: {instances} ===")
    max_trials = min(len(instances), max_trials)
    for instance in instances:
        client = client_factory(instance)
        print(f"=== SERVICE_DISCOVERY: [{service_name}] Trying to connect to {instance} ===")
        logger.debug(f"[{service_name}]Trying to connect to {instance}")
        pingwatch = Stopwatch()
        try:
            async with asyncio.timeout(timeout_sec):
                print(f"=== SERVICE_DISCOVERY: [{service_name}] Starting up client for {instance} ===")
                await client.start_up()
                print(f"=== SERVICE_DISCOVERY: [{service_name}] Successfully connected to {instance} ===")
        except Exception as exc:
            elapsed = pingwatch.time()
            print(f"=== SERVICE_DISCOVERY: [{service_name}] Failed to connect to {instance}: {exc} (took {elapsed*1000:.1f}ms) ===")
            max_trials -= 1
            
            # Enhanced TTS-specific logging for debugging server shutdowns
            if service_name == "tts":
                logger.error(f"=== TTS CONNECTION FAILURE ===")
                logger.error(f"TTS instance: {instance}")
                logger.error(f"Error type: {type(exc).__name__}")
                logger.error(f"Error message: {exc}")
                logger.error(f"Connection attempt took: {elapsed*1000:.1f}ms")
                logger.error(f"Remaining trials: {max_trials}")
                
                # Check if this looks like a server restart scenario
                if elapsed < 1.0:  # Very quick failure suggests server not running
                    logger.error("=== QUICK FAILURE - TTS SERVER MAY NOT BE RUNNING ===")
                elif elapsed > 30.0:  # Long timeout suggests server hanging/overloaded
                    logger.error("=== TIMEOUT - TTS SERVER MAY BE HANGING OR OVERLOADED ===")
            
            if isinstance(exc, MissingServiceAtCapacity):
                logger.info(
                    f"[{service_name}] Instance {instance} took {elapsed * 1000:.1f}ms to reject us."
                )
                if service_name == "tts":
                    mt.TTS_PING_TIME.observe(elapsed)
                elif service_name == "stt":
                    mt.STT_PING_TIME.observe(elapsed)
            else:
                mt.HARD_SERVICE_MISSES.inc()
                if service_name == "tts":
                    mt.TTS_HARD_MISSES.inc()
                elif service_name == "stt":
                    mt.STT_HARD_MISSES.inc()
                if isinstance(exc, TimeoutError):
                    logger.warning(
                        f"[{service_name}] Instance {instance} did not reply in time."
                    )
                else:
                    logger.error(
                        f"[{service_name}] Unexpected error connecting to {instance}: {exc}."
                    )
            if max_trials > 0:
                continue
            else:
                mt.SERVICE_MISSES.inc()
                if service_name == "tts":
                    mt.TTS_MISSES.inc()
                elif service_name == "stt":
                    mt.STT_MISSES.inc()
                if isinstance(exc, MissingServiceAtCapacity):
                    raise
                else:
                    if isinstance(exc, TimeoutError):
                        raise MissingServiceTimeout(service_name) from exc
                    else:
                        raise  # Let internal errors propagate
        elapsed = pingwatch.time()
        logger.info(
            f"[{service_name}] Instance {instance} took {elapsed * 1000:.1f}ms to accept us."
        )
        
        # Enhanced TTS connection success logging
        if service_name == "tts":
            logger.info(f"=== TTS CONNECTION SUCCESSFUL ===")
            logger.info(f"TTS instance: {instance}")
            logger.info(f"Connection time: {elapsed * 1000:.1f}ms")
            logger.info(f"Total discovery time: {stopwatch.time() * 1000:.1f}ms")
            mt.TTS_PING_TIME.observe(elapsed)
        elif service_name == "stt":
            mt.STT_PING_TIME.observe(elapsed)
        elapsed = stopwatch.time()
        if service_name == "tts":
            mt.TTS_FIND_TIME.observe(elapsed)
        elif service_name == "stt":
            mt.STT_FIND_TIME.observe(elapsed)
        logger.info(
            f"[{service_name}] Connection to {instance} took {1000 * elapsed:.1f}ms."
        )
        return client
    raise AssertionError("Should not be reached.")
