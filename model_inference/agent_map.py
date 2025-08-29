from .model_agent import Qwen3AgentFCInference, Qwen3AgentInference

AGENT_NAME_MAP = {
    "Qwen2_5-7B-FC-Think": Qwen3AgentFCInference,
    "Qwen2_5-7B-Think": Qwen3AgentInference,
    "Qwen2_5-14B-FC-Think": Qwen3AgentFCInference,
    "Qwen2_5-14B-Think": Qwen3AgentInference,
    "Qwen2_5-32B-FC-Think": Qwen3AgentFCInference,
    "Qwen2_5-32B-Think": Qwen3AgentInference,
    "Qwen3-8B-FC": Qwen3AgentFCInference,
    "Qwen3-8B": Qwen3AgentInference,
    "Qwen3-14B-FC": Qwen3AgentFCInference,
    "Qwen3-14B": Qwen3AgentInference,
    "Qwen3-32B-FC": Qwen3AgentFCInference,
    "Qwen3-32B": Qwen3AgentInference,
}
