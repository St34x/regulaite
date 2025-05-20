import logging
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel
import json # For parsing LLM string output to JSON
import os

logger = logging.getLogger(__name__)

# Import actual LLM libraries
from openai import OpenAI, AsyncOpenAI
try:
    import anthropic
except ImportError:
    logger.warning("Anthropic library not found. Anthropic models will not be available.")
    anthropic = None

class LLMClient:
    """Client for interacting with a Large Language Model service."""

    def __init__(self, api_key: Optional[str] = None, default_model_name: str = "gpt-3.5-turbo", config: Optional[Dict[str, Any]] = None):
        self.api_key = api_key
        self.config = config if config else {}
        self.default_model_name = default_model_name
        self.provider = self.config.get("provider", "openai")
        
        # Initialize appropriate client based on provider
        if self.provider == "openai":
            if not self.api_key:
                self.api_key = os.environ.get("OPENAI_API_KEY")
                if not self.api_key:
                    logger.warning("No API key provided for OpenAI client, and OPENAI_API_KEY not set.")
            
            self.client = OpenAI(api_key=self.api_key)
            self.async_client = AsyncOpenAI(api_key=self.api_key)
            
            # Configure custom API URL if provided
            if self.config.get("api_url"):
                self.client.base_url = self.config["api_url"]
                self.async_client.base_url = self.config["api_url"]
        
        elif self.provider == "anthropic" and anthropic:
            if not self.api_key:
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not self.api_key:
                    logger.warning("No API key provided for Anthropic client, and ANTHROPIC_API_KEY not set.")
            
            self.client = anthropic.Anthropic(api_key=self.api_key)
            # Anthropic also has async client
            if hasattr(anthropic, "AsyncAnthropic"):
                self.async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
            else:
                self.async_client = None
                logger.warning("AsyncAnthropic not available. Will use synchronous client.")
        
        else:
            logger.warning(f"Provider {self.provider} not supported or required libraries not installed.")
            self.client = None
            self.async_client = None
            
        logger.info(f"LLMClient initialized. Provider: {self.provider}, Default Model: {self.default_model_name}, Configuration: {self.config}")

    async def generate_text(self, prompt: str, model_name: Optional[str] = None, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generates text based on a given prompt."""
        current_model = model_name if model_name else self.default_model_name
        logger.info(f"Generating text using {current_model} for prompt (first 50 chars): {prompt[:50]}...")
        
        try:
            if self.provider == "openai":
                # Format the prompt for chat models
                messages = [{"role": "user", "content": prompt}]
                
                if self.async_client:
                    response = await self.async_client.chat.completions.create(
                        model=current_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    generated_text = response.choices[0].message.content.strip()
                else:
                    response = self.client.chat.completions.create(
                        model=current_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    generated_text = response.choices[0].message.content.strip()
                    
            elif self.provider == "anthropic" and anthropic:
                if self.async_client:
                    response = await self.async_client.messages.create(
                        model=current_model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                else:
                    response = self.client.messages.create(
                        model=current_model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                generated_text = response.content[0].text
            else:
                logger.warning(f"Provider {self.provider} not supported or client not initialized.")
                return f"Error: Provider {self.provider} not supported or client not initialized."
                
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during LLM text generation with {current_model}: {e}", exc_info=True)
            return f"Error: Could not generate text due to: {e}"

    async def generate_structured_output(self, prompt: str, output_model: Type[BaseModel], model_name: Optional[str] = None, max_tokens: int = 1000, temperature: float = 0.3) -> Optional[BaseModel]:
        """Generates structured output (e.g., JSON) matching a Pydantic model from a prompt."""
        current_model = model_name if model_name else self.default_model_name
        logger.info(f"Generating structured output for model {output_model.__name__} using {current_model} from prompt (first 50 chars): {prompt[:50]}...")

        # Prompt Engineering for JSON output
        enhanced_prompt = f"""
{prompt}

Please provide the response as a JSON object that strictly conforms to the following Pydantic schema:
{output_model.schema_json(indent=2)}

JSON Output:
"""
        
        try:
            raw_json_output = None
            
            if self.provider == "openai":
                # OpenAI supports JSON mode in many models
                if self.async_client:
                    response = await self.async_client.chat.completions.create(
                        model=current_model,
                        messages=[{"role": "user", "content": enhanced_prompt}],
                        response_format={"type": "json_object"},  # Use JSON mode if available
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    raw_json_output = response.choices[0].message.content.strip()
                else:
                    response = self.client.chat.completions.create(
                        model=current_model,
                        messages=[{"role": "user", "content": enhanced_prompt}],
                        response_format={"type": "json_object"},  # Use JSON mode if available
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    raw_json_output = response.choices[0].message.content.strip()
                    
            elif self.provider == "anthropic" and anthropic:
                # Anthropic doesn't have JSON mode, so we rely on prompting
                if self.async_client:
                    response = await self.async_client.messages.create(
                        model=current_model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=[
                            {"role": "user", "content": enhanced_prompt}
                        ]
                    )
                else:
                    response = self.client.messages.create(
                        model=current_model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=[
                            {"role": "user", "content": enhanced_prompt}
                        ]
                    )
                raw_json_output = response.content[0].text
                
                # Anthropic might include extra text around the JSON, so extract it
                json_start = raw_json_output.find('{')
                json_end = raw_json_output.rfind('}')
                if json_start != -1 and json_end != -1 and json_start < json_end:
                    raw_json_output = raw_json_output[json_start:json_end+1]
                else:
                    raise ValueError("No JSON object found in LLM response")
            else:
                # Fallback: use generic text generation
                raw_response_text = await self.generate_text(
                    prompt=enhanced_prompt, 
                    model_name=current_model,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Attempt to extract JSON from the response
                json_start = raw_response_text.find('{')
                json_end = raw_response_text.rfind('}')
                if json_start != -1 and json_end != -1 and json_start < json_end:
                    raw_json_output = raw_response_text[json_start:json_end+1]
                else:
                    raise ValueError("No JSON object found in LLM response")

            # Validate and parse the JSON
            try:
                # Parse using Pydantic model
                return output_model.model_validate_json(raw_json_output)
            except AttributeError:
                # Fallback for older Pydantic versions
                return output_model.parse_raw(raw_json_output)
        
        except json.JSONDecodeError as jde:
            logger.error(f"Failed to decode LLM JSON output: {jde}. Raw output: {raw_json_output if 'raw_json_output' in locals() else '[not captured]'}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error during LLM structured output generation with {current_model}: {e}", exc_info=True)
            return None

    # Add other methods as needed, e.g., for embeddings, specific tasks, etc.

# Example basic usage (for testing this client directly)
# async def main():
#     client = LLMClient()
#     text_response = await client.generate_text("Tell me a joke.")
#     print(f"Text response: {text_response}")

    # from backend.autonomous_agent.processing_nodes.input_processor import LLMParsedInput # For testing
    # structured_response = await client.generate_structured_output(
    #     "User prompt: What is the capital of France and list key export products?", 
    #     LLMParsedInput
    # )
    # if structured_response:
    #     print(f"Structured response: Core Question - {structured_response.core_question}")
    #     print(f"Structured response: Key Terms - {structured_response.key_terms}")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main()) 