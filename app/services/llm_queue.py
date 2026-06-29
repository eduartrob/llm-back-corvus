import asyncio
import logging
from typing import Callable, Any, Coroutine

logger = logging.getLogger(__name__)

class LLMQueue:
    def __init__(self):
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._worker_task: asyncio.Task | None = None
        self._is_running: bool = False

    def start(self):
        if not self._is_running:
            self._is_running = True
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("LLM Priority Queue Worker started.")

    def stop(self):
        self._is_running = False
        if self._worker_task:
            self._worker_task.cancel()
            logger.info("LLM Priority Queue Worker stopped.")

    async def enqueue(self, priority: int, coro: Coroutine) -> Any:
        future = asyncio.get_running_loop().create_future()
        # Item format: (priority, counter, coro, future)
        # Using id(future) as counter to prevent comparing coroutines if priorities are equal
        await self.queue.put((priority, id(future), coro, future))
        return await future

    async def _worker_loop(self):
        while self._is_running:
            try:
                priority, _, coro, future = await self.queue.get()
                logger.info(f"Procesando tarea de cola con prioridad: {priority}")
                
                try:
                    result = await coro
                    if not future.done():
                        future.set_result(result)
                except Exception as e:
                    logger.error(f"Error procesando tarea en la cola LLM: {e}")
                    if not future.done():
                        future.set_exception(e)
                finally:
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error crítico en LLM Queue Worker: {e}")
                await asyncio.sleep(5)

llm_queue = LLMQueue()
