"""Response guards for swarm MCP tools.

Provides validation and error handling to ensure consistent, reliable responses
from swarm admin/status tools (rlm_swarm_members, rlm_agent_status, etc.)
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class SwarmToolError:
    """Standardized swarm tool error."""

    error: str
    error_code: str
    correlation_id: str
    recoverable: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for ToolResult.data."""
        return {
            "success": False,
            "error": self.error,
            "error_code": self.error_code,
            "correlation_id": self.correlation_id,
            "recoverable": self.recoverable,
        }


# Error codes for classification
ERROR_CODES = {
    "SWARM_NOT_FOUND": "swarm_not_found",
    "AGENT_NOT_FOUND": "agent_not_found",
    "DB_ERROR": "database_error",
    "TIMEOUT": "timeout_error",
    "VALIDATION": "validation_error",
    "INTERNAL": "internal_error",
}


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for request tracing."""
    return f"swarm-{uuid.uuid4().hex[:12]}"


async def safe_swarm_call(
    func: Callable,
    *args,
    error_context: str = "swarm operation",
    **kwargs,
) -> tuple[dict[str, Any] | None, SwarmToolError | None]:
    """Safely execute a swarm service function with error handling.

    Args:
        func: The async function to call
        *args: Positional arguments for the function
        error_context: Context string for error messages
        **kwargs: Keyword arguments for the function

    Returns:
        Tuple of (result, error) - one will be None
    """
    correlation_id = generate_correlation_id()

    try:
        result = await func(*args, **kwargs)
        return result, None

    except TimeoutError as e:
        logger.error(f"[{correlation_id}] Timeout in {error_context}: {e}")
        return None, SwarmToolError(
            error=f"Timeout executing {error_context}",
            error_code=ERROR_CODES["TIMEOUT"],
            correlation_id=correlation_id,
            recoverable=True,
        )

    except ConnectionError as e:
        logger.error(f"[{correlation_id}] Connection error in {error_context}: {e}")
        return None, SwarmToolError(
            error=f"Database connection error in {error_context}",
            error_code=ERROR_CODES["DB_ERROR"],
            correlation_id=correlation_id,
            recoverable=True,
        )

    except Exception as e:
        logger.exception(f"[{correlation_id}] Unexpected error in {error_context}: {e}")
        return None, SwarmToolError(
            error=f"Internal error in {error_context}: {type(e).__name__}",
            error_code=ERROR_CODES["INTERNAL"],
            correlation_id=correlation_id,
            recoverable=False,
        )


def validate_swarm_members_response(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize rlm_swarm_members response.

    Ensures required fields exist with correct types.
    """
    if not isinstance(data, dict):
        return {
            "success": False,
            "error": "Invalid response format",
            "error_code": ERROR_CODES["INTERNAL"],
        }

    # Check for error response
    if "error" in data:
        return {
            "success": False,
            "error": data.get("error", "Unknown error"),
            "swarm_id": data.get("swarm_id"),
        }

    # Validate and normalize successful response
    return {
        "success": data.get("success", True),
        "swarm_id": data.get("swarm_id") or data.get("id", ""),
        "swarm_name": data.get("swarm_name") or data.get("name", ""),
        "agent_count": data.get("agent_count", len(data.get("agents", []))),
        "max_agents": data.get("max_agents") or data.get("maxAgents", 10),
        "agents": _normalize_agents_list(data.get("agents", [])),
    }


def validate_agent_status_response(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize rlm_agent_status response.

    Ensures required fields exist with correct types.
    """
    if not isinstance(data, dict):
        return {
            "success": False,
            "error": "Invalid response format",
            "error_code": ERROR_CODES["INTERNAL"],
        }

    # Check for error response
    if "error" in data and not data.get("success", True):
        return {
            "success": False,
            "error": data.get("error", "Unknown error"),
        }

    # Validate and normalize successful response
    swarm = data.get("swarm", {})
    return {
        "success": True,
        "swarm": {
            "id": swarm.get("id", ""),
            "name": swarm.get("name", ""),
            "description": swarm.get("description"),
        },
        "agent_id": data.get("agent_id", ""),
        "pending_tasks": data.get("pending_tasks", []),
        "pending_count": data.get("pending_count", len(data.get("pending_tasks", []))),
        "current_task": data.get("current_task"),
        "has_work": data.get("has_work", False),
        "instructions": data.get("instructions", ""),
    }


def validate_swarm_info_response(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize get_swarm_info response."""
    if not isinstance(data, dict):
        return {"success": False, "error": "Invalid response format"}

    # Check for error
    if data.get("success") is False or "error" in data:
        return {
            "success": False,
            "error": data.get("error", "Unknown error"),
        }

    # Normalize response
    return {
        "success": True,
        "id": data.get("swarm_id") or data.get("id", ""),
        "name": data.get("name", ""),
        "description": data.get("description"),
        "maxAgents": data.get("max_agents") or data.get("maxAgents", 10),
        "isActive": data.get("is_active", True),
        "agent_count": data.get("agent_count", len(data.get("agents", []))),
        "agents": _normalize_agents_list(data.get("agents", [])),
    }


def _normalize_agents_list(agents: list | None) -> list[dict[str, Any]]:
    """Normalize agents list ensuring consistent structure."""
    if not agents:
        return []

    normalized = []
    for agent in agents:
        if isinstance(agent, dict):
            normalized.append({
                "agent_id": agent.get("agent_id") or agent.get("agentId", ""),
                "name": agent.get("name", ""),
                "is_active": agent.get("is_active", True),
                "last_heartbeat": agent.get("last_heartbeat"),
                "tasks_completed": agent.get("tasks_completed", 0),
                "tasks_failed": agent.get("tasks_failed", 0),
                "joined_at": agent.get("joined_at"),
            })
    return normalized


def validate_leave_swarm_response(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize rlm_swarm_leave response."""
    if not isinstance(data, dict):
        return {"success": False, "error": "Invalid response format"}

    if data.get("success") is False or "error" in data:
        return {
            "success": False,
            "error": data.get("error", "Unknown error"),
        }

    return {
        "success": True,
        "message": data.get("message", "Left swarm successfully"),
    }
