"""
Backward compatibility wrapper for UnmuteHandler.
This file now provides a simplified interface that wraps the new Conversation class.
"""

import time
from typing import Any, Literal
from unmute.conversation import Conversation, GradioUpdate


class UnmuteHandler(Conversation):
    """
    Backward compatibility wrapper for UnmuteHandler.
    This now inherits from Conversation and provides the same interface.
    """

    def __init__(self) -> None:
        # Generate a unique conversation ID for backward compatibility
        conversation_id = f"unmute_handler_{int(time.time())}"
        super().__init__(conversation_id)

    async def start_up(self) -> None:
        """Start up the conversation services (backward compatibility method)."""
        await self.start_services()

    async def cleanup(self) -> None:
        """Clean up method for backward compatibility."""
        await super().cleanup()