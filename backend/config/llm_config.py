"""
Configuration classes for LLM models.
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator

class LLMConfig(BaseModel):
    """Configuration for LLM models"""
    provider: str = Field(
        default="openai",
        description="LLM provider (openai, anthropic, cohere, etc.)"
    )
    model: str = Field(
        default="gpt-4",
        description="Model name to use"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key (if not provided, will use environment variable)"
    )
    api_url: Optional[str] = Field(
        default=None,
        description="Custom API URL endpoint (for self-hosted models or proxies)"
    )
    temperature: float = Field(
        default=0.7,
        description="Temperature for generation (0-1)"
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum tokens to generate"
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Top-p sampling parameter"
    )
    frequency_penalty: Optional[float] = Field(
        default=None,
        description="Frequency penalty parameter"
    )
    presence_penalty: Optional[float] = Field(
        default=None,
        description="Presence penalty parameter"
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None,
        description="Sequences that will stop generation"
    )
    additional_params: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional provider-specific parameters"
    )

    @validator('temperature')
    def validate_temperature(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, filtering out None values"""
        result = self.dict(exclude_none=True)
        return result

    class Config:
        """Pydantic configuration"""
        extra = "allow"  # Allow extra fields


def get_provider_specific_config(config: LLMConfig) -> Dict[str, Any]:
    """
    Get provider-specific configuration dictionary.

    Args:
        config: LLM configuration

    Returns:
        Dictionary with provider-specific configuration
    """
    base_config = config.to_dict()
    provider = base_config.pop("provider", "openai")

    # Map generic parameters to provider-specific ones
    if provider == "anthropic":
        # Anthropic uses different parameter names
        if "max_tokens" in base_config:
            base_config["max_tokens_to_sample"] = base_config.pop("max_tokens")
        if "stop_sequences" in base_config:
            base_config["stop_sequences"] = base_config.pop("stop_sequences")
    elif provider == "cohere":
        # Cohere uses different parameter names
        if "max_tokens" in base_config:
            base_config["max_tokens"] = base_config.pop("max_tokens")
        if "temperature" in base_config:
            base_config["temperature"] = base_config.pop("temperature")

    # Add any additional provider-specific parameters
    additional_params = base_config.pop("additional_params", {})
    base_config.update(additional_params)

    return base_config
