import asyncio
import heapq
from dataclasses import dataclass, field
from typing import AsyncIterable, Callable, Iterable, TypeVar

T = TypeVar("T")


@dataclass(order=True)
class TimedItem[T]:
    time: float
    item: T = field(compare=False)

    def as_tuple(self) -> tuple[float, T]:
        return self.time, self.item


class RealtimeQueue[T]:
    """A data structure that accumulates timestamped items and releases them at the given times.

    Implemented as a heap, so it doesn't have to be FIFO.
    """

    def __init__(self, get_time: Callable[[], float] | None = None, timeout_sec: float = 5.0):
        self.queue: list[TimedItem] = []
        self.start_time: float | None = None
        self.timeout_sec = timeout_sec  # Safety timeout to prevent messages from getting stuck
        self._last_release_time: float | None = None

        if get_time is None:
            self.get_time = lambda: asyncio.get_event_loop().time()
        else:
            # Use an external time function to support use cases where "real time"
            # means something different
            self.get_time = get_time

    def start_if_not_started(self):
        if self.start_time is None:
            self.start_time = self.get_time()
            self._last_release_time = 0.0  # Initialize to start of timing

    def put(self, item: T, time: float):
        heapq.heappush(self.queue, TimedItem(time, item))

    async def get(self) -> AsyncIterable[tuple[float, T]]:
        """Get all items that are past due. If none is, wait for the next one."""

        if self.start_time is None:
            return
        if not self.queue:
            return

        time_since_start = self.get_time() - self.start_time
        while self.queue:
            delta = self.queue[0].time - time_since_start

            if delta > 0:
                await asyncio.sleep(delta)

            yield heapq.heappop(self.queue).as_tuple()

    def get_nowait(self) -> Iterable[tuple[float, T]]:
        if self.start_time is None:
            return None

        current_time = self.get_time()
        time_since_start = current_time - self.start_time

        # Debug logging for timing issue investigation
        if self.queue:
            next_message_time = self.queue[0].time
            queue_size = len(self.queue)
            
            # Log every few seconds or when there are stuck messages
            if hasattr(self, '_last_debug_log_time'):
                time_since_last_log = time_since_start - self._last_debug_log_time
                should_log = time_since_last_log > 2.0  # Log every 2 seconds
            else:
                should_log = True
                
            # Always log if we have messages that should have been released
            if next_message_time <= time_since_start - 1.0:  # Message is 1+ seconds overdue
                should_log = True
                
            if should_log:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"RealtimeQueue timing: current_time={current_time:.3f}, start_time={self.start_time:.3f}, time_since_start={time_since_start:.3f}, next_message_time={next_message_time:.3f}, queue_size={queue_size}, message_overdue_by={time_since_start - next_message_time:.3f}s")
                self._last_debug_log_time = time_since_start

        released_count = 0
        timeout_released_count = 0
        
        # Normal release: messages whose time has come
        while self.queue and self.queue[0].time <= time_since_start:
            released_count += 1
            yield heapq.heappop(self.queue).as_tuple()
            
        # Safety timeout mechanism: if no messages have been released for too long,
        # force-release the next few messages to prevent indefinite blocking
        if self.queue and self._last_release_time is not None:
            time_since_last_release = time_since_start - self._last_release_time
            if time_since_last_release > self.timeout_sec:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"RealtimeQueue timeout triggered! No messages released for {time_since_last_release:.1f}s (timeout={self.timeout_sec}s). Force-releasing next messages.")
                
                # Force-release up to 3 messages to prevent a complete freeze
                max_timeout_release = min(3, len(self.queue))
                for _ in range(max_timeout_release):
                    if self.queue:
                        timeout_released_count += 1
                        item = heapq.heappop(self.queue)
                        logger.warning(f"Timeout release: message scheduled for {item.time:.3f}s (current time_since_start={time_since_start:.3f}s)")
                        yield item.as_tuple()
        
        # Update last release time
        if released_count > 0 or timeout_released_count > 0:
            self._last_release_time = time_since_start
            
        # Log if we released messages or if we have stuck messages
        if released_count > 0 or timeout_released_count > 0 or (self.queue and self.queue[0].time <= time_since_start - 0.5):
            import logging
            logger = logging.getLogger(__name__)
            if released_count > 0:
                logger.info(f"RealtimeQueue: Released {released_count} messages normally, {len(self.queue)} remaining")
            if timeout_released_count > 0:
                logger.warning(f"RealtimeQueue: Force-released {timeout_released_count} messages due to timeout, {len(self.queue)} remaining")
            if released_count == 0 and timeout_released_count == 0 and self.queue:
                logger.warning(f"RealtimeQueue: No messages released, but {len(self.queue)} messages waiting (next due at {self.queue[0].time:.3f}s, current time_since_start={time_since_start:.3f}s)")

    async def __aiter__(self):
        if self.start_time is None or not self.queue:
            return

        while self.queue:
            time_since_start = self.get_time() - self.start_time
            delta = self.queue[0].time - time_since_start

            if delta > 0:
                await asyncio.sleep(delta)

            yield heapq.heappop(self.queue).as_tuple()

    def empty(self):
        return not self.queue
