"""Agent Limits Service for Phase 8-9.

Enforces plan limits for memories, swarms, and agents.
"""

import logging
from typing import Any

from ..db import get_db

logger = logging.getLogger(__name__)

# Plan limits by tier
PLAN_LIMITS = {
    "STARTER": {
        "memories": 1000,
        "retention_days": 7,
        "swarms": 1,
        "agents_per_swarm": 2,
        "cache_ttl_seconds": 300,  # 5 min
        "real_time_events": False,
    },
    "PRO": {
        "memories": 5000,
        "retention_days": 30,
        "swarms": 5,
        "agents_per_swarm": 5,
        "cache_ttl_seconds": 1800,  # 30 min
        "real_time_events": False,
    },
    "TEAM": {
        "memories": 25000,
        "retention_days": 90,
        "swarms": 20,
        "agents_per_swarm": 15,
        "cache_ttl_seconds": 7200,  # 2 hr
        "real_time_events": True,
    },
    "ENTERPRISE": {
        "memories": -1,  # Unlimited
        "retention_days": -1,  # Unlimited
        "swarms": -1,  # Unlimited
        "agents_per_swarm": 50,
        "cache_ttl_seconds": 86400,  # 24 hr
        "real_time_events": True,
    },
}

# Default limits if no subscription
DEFAULT_LIMITS = PLAN_LIMITS["STARTER"]


async def get_agents_subscription(project_id: str) -> dict[str, Any] | None:
    """Get the agents subscription for a project.

    First tries to find by team (if project belongs to team),
    then falls back to project owner's personal subscription.

    Args:
        project_id: The project ID

    Returns:
        Subscription dict or None if no subscription
    """
    db = await get_db()

    # Get project with team info
    project = await db.project.find_unique(
        where={"id": project_id},
        include={"team": True},
    )

    if not project:
        logger.warning(f"Project not found: {project_id}")
        return None

    # Try team subscription first
    if project.teamId:
        team_sub = await db.agentssubscription.find_first(
            where={
                "teamId": project.teamId,
                "status": "active",
            },
        )
        if team_sub:
            return _subscription_to_dict(team_sub)

    # Fall back to project owner's subscription
    # Get project owner from team members or API key owner
    team_member = await db.teammember.find_first(
        where={
            "teamId": project.teamId,
            "role": "OWNER",
        },
    ) if project.teamId else None

    user_id = team_member.userId if team_member else None

    if user_id:
        user_sub = await db.agentssubscription.find_first(
            where={
                "userId": user_id,
                "teamId": None,  # Personal subscription
                "status": "active",
            },
        )
        if user_sub:
            return _subscription_to_dict(user_sub)

    return None


def _subscription_to_dict(sub: Any) -> dict[str, Any]:
    """Convert subscription model to dict."""
    return {
        "id": sub.id,
        "plan": sub.plan,
        "status": sub.status,
        "memory_limit": sub.memoryLimit,
        "swarm_limit": sub.swarmLimit,
        "agents_per_swarm_limit": sub.agentsPerSwarmLimit,
        "cache_ttl_seconds": sub.cacheTtlSeconds,
    }


def get_plan_limits(plan: str) -> dict[str, Any]:
    """Get limits for a plan.

    Args:
        plan: Plan name (STARTER, PRO, TEAM, ENTERPRISE)

    Returns:
        Dict with plan limits
    """
    return PLAN_LIMITS.get(plan.upper(), DEFAULT_LIMITS)


async def check_memory_limits(project_id: str) -> tuple[bool, str | None]:
    """Check if project can create new memories.

    Args:
        project_id: The project ID

    Returns:
        Tuple of (allowed, error_message)
    """
    db = await get_db()
    subscription = await get_agents_subscription(project_id)

    # Get limit from subscription or use default
    if subscription:
        memory_limit = subscription["memory_limit"]
    else:
        memory_limit = DEFAULT_LIMITS["memories"]

    # -1 means unlimited
    if memory_limit == -1:
        return True, None

    # Count current memories
    current_count = await db.agentmemory.count(
        where={"projectId": project_id}
    )

    if current_count >= memory_limit:
        return False, f"Memory limit reached ({current_count}/{memory_limit}). Upgrade your plan for more memories."

    return True, None


async def check_swarm_limits(project_id: str) -> tuple[bool, str | None]:
    """Check if project can create new swarms.

    Args:
        project_id: The project ID

    Returns:
        Tuple of (allowed, error_message)
    """
    db = await get_db()
    subscription = await get_agents_subscription(project_id)

    # Get limit from subscription or use default
    if subscription:
        swarm_limit = subscription["swarm_limit"]
    else:
        swarm_limit = DEFAULT_LIMITS["swarms"]

    # -1 means unlimited
    if swarm_limit == -1:
        return True, None

    # Count current active swarms
    current_count = await db.agentswarm.count(
        where={
            "projectId": project_id,
            "isActive": True,
        }
    )

    if current_count >= swarm_limit:
        return False, f"Swarm limit reached ({current_count}/{swarm_limit}). Upgrade your plan for more swarms."

    return True, None


async def check_swarm_agent_limits(swarm_id: str) -> tuple[bool, str | None]:
    """Check if swarm can accept more agents.

    Args:
        swarm_id: The swarm ID

    Returns:
        Tuple of (allowed, error_message)
    """
    db = await get_db()

    # Get swarm with project
    swarm = await db.agentswarm.find_unique(
        where={"id": swarm_id},
        include={"project": True},
    )

    if not swarm:
        return False, "Swarm not found"

    subscription = await get_agents_subscription(swarm.projectId)

    # Get limit from subscription or use default
    if subscription:
        agents_limit = subscription["agents_per_swarm_limit"]
    else:
        agents_limit = DEFAULT_LIMITS["agents_per_swarm"]

    # Also check swarm's own maxAgents setting
    swarm_max = swarm.maxAgents
    effective_limit = min(agents_limit, swarm_max)

    # Count current agents
    current_count = await db.swarmagent.count(
        where={
            "swarmId": swarm_id,
            "isActive": True,
        }
    )

    if current_count >= effective_limit:
        return False, f"Agent limit reached ({current_count}/{effective_limit}). Upgrade your plan or increase swarm capacity."

    return True, None


async def get_memory_retention_limit(project_id: str) -> int:
    """Get max retention days for memories.

    Args:
        project_id: The project ID

    Returns:
        Max retention days (-1 = unlimited)
    """
    subscription = await get_agents_subscription(project_id)

    if subscription:
        plan = subscription["plan"]
        limits = get_plan_limits(plan)
        return limits["retention_days"]

    return DEFAULT_LIMITS["retention_days"]


async def get_cache_ttl(project_id: str) -> int:
    """Get cache TTL for project.

    Args:
        project_id: The project ID

    Returns:
        Cache TTL in seconds
    """
    subscription = await get_agents_subscription(project_id)

    if subscription:
        return subscription["cache_ttl_seconds"]

    return DEFAULT_LIMITS["cache_ttl_seconds"]


async def has_real_time_events(project_id: str) -> bool:
    """Check if project has real-time events enabled.

    Args:
        project_id: The project ID

    Returns:
        True if real-time events are enabled
    """
    subscription = await get_agents_subscription(project_id)

    if subscription:
        plan = subscription["plan"]
        limits = get_plan_limits(plan)
        return limits["real_time_events"]

    return DEFAULT_LIMITS["real_time_events"]


async def get_usage_stats(project_id: str) -> dict[str, Any]:
    """Get current usage statistics for a project.

    Args:
        project_id: The project ID

    Returns:
        Dict with current usage and limits
    """
    db = await get_db()
    subscription = await get_agents_subscription(project_id)

    # Get current counts
    memory_count = await db.agentmemory.count(where={"projectId": project_id})
    swarm_count = await db.agentswarm.count(
        where={"projectId": project_id, "isActive": True}
    )

    # Get limits
    if subscription:
        plan = subscription["plan"]
        memory_limit = subscription["memory_limit"]
        swarm_limit = subscription["swarm_limit"]
    else:
        plan = "STARTER"
        memory_limit = DEFAULT_LIMITS["memories"]
        swarm_limit = DEFAULT_LIMITS["swarms"]

    return {
        "plan": plan,
        "subscription_active": subscription is not None,
        "memories": {
            "current": memory_count,
            "limit": memory_limit,
            "unlimited": memory_limit == -1,
        },
        "swarms": {
            "current": swarm_count,
            "limit": swarm_limit,
            "unlimited": swarm_limit == -1,
        },
    }
