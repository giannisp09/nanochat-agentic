"""
verl multi-turn tool: a stateful Python interpreter for agentic coding rollouts.

TEMPLATE — verl's BaseTool API is version-specific (method names/async signatures
and the tool-schema format change across releases). Confirm against the verl
version pinned by posttrain/setup.sh, then adjust the method bodies. The actual
sandbox logic (PersistentPythonSession) is real and reused from nanochat.

verl wires tools to the SGLang rollout via a YAML tool config that points at this
class. Per-trajectory state is keyed by an instance id so concurrent rollouts get
isolated interpreters (exactly what PersistentPythonSession provides per row).
"""

from nanochat.execution import PersistentPythonSession, format_execution_output


OPENAI_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Execute Python code in a stateful sandbox. State (variables, "
                       "imports, functions) persists across calls within a trajectory. "
                       "Returns stdout and any error/traceback.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."}
            },
            "required": ["code"],
        },
    },
}


class PythonInterpreterTool:
    """One PersistentPythonSession per active trajectory (keyed by instance id)."""

    def __init__(self, config=None, tool_schema=None, timeout=8.0, max_output_chars=1500):
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self._sessions = {}  # instance_id -> PersistentPythonSession

    def get_openai_tool_schema(self):
        return OPENAI_TOOL_SCHEMA

    # --- verl lifecycle (confirm exact names/async-ness for your verl version) ---

    async def create(self, instance_id, **kwargs):
        self._sessions[instance_id] = PersistentPythonSession(timeout=self.timeout)
        return instance_id

    async def execute(self, instance_id, parameters, **kwargs):
        session = self._sessions.get(instance_id)
        if session is None:  # defensive: create on first use
            session = self._sessions[instance_id] = PersistentPythonSession(timeout=self.timeout)
        code = parameters.get("code", "") if isinstance(parameters, dict) else str(parameters)
        result = session.run(code, timeout=self.timeout)
        tool_response = format_execution_output(result, max_chars=self.max_output_chars)
        step_reward = 0.0          # outcome reward is computed by code_reward.py at trajectory end
        metadata = {"success": result.success, "timeout": result.timeout}
        return tool_response, step_reward, metadata

    async def release(self, instance_id, **kwargs):
        session = self._sessions.pop(instance_id, None)
        if session is not None:
            session.close()
