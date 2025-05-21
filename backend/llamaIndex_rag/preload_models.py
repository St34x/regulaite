"""
Utility script to preload embedding models for improved RAG performance.
"""
import logging
import os
from typing import List, Dict, Optional, Union, Callable
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configure model caching
os.environ["TRANSFORMERS_CACHE"] = "/app/model_cache"
os.environ["HF_HOME"] = "/app/model_cache"

# Dictionary mapping language codes to model names
language_models = {
    'fr': 'all-MiniLM-L6-v2',  # Primary language
    'en': 'all-MiniLM-L6-v2',
    'de': 'all-MiniLM-L6-v2',  # Using same model as fallback
    'es': 'all-MiniLM-L6-v2',  # Using same model as fallback
    'it': 'all-MiniLM-L6-v2',  # Using same model as fallback
    'multi': 'all-MiniLM-L6-v2'  # Using same model as fallback
}

# Global cache for loaded models
loaded_models = {}

def preload_models(languages=['en', 'it', 'de', 'multi'], force_reload=False, progress_callback: Optional[Callable[[str, float, bool, Optional[str]], None]] = None):
    """
    Preload the models for the given languages
    
    Args:
        languages: List of language codes to preload models for
        force_reload: Whether to force reload models even if they are already loaded
        progress_callback: Optional callback function that receives (language_code, progress_percentage, is_complete, error_message)
        
    Returns:
        A dictionary mapping language codes to Boolean success indicators
    """
    if not languages:
        languages = ['en', 'it', 'de', 'multi']
    
    # Log preloading start with count
    logger.info(f"Preloading {len(languages)} embedding models: {', '.join(languages)}")
    
    # Track success per language
    results = {}
    
    # Check if FastEmbedEmbedding is available
    try:
        from llama_index.embeddings.fastembed import FastEmbedEmbedding
        logger.info("Using FastEmbedEmbedding for embedding generation")
        
        # Create a single instance to use for all languages
        # This is more efficient than creating separate models
        try:
            # Initialize with default parameters
            embed_model = FastEmbedEmbedding()
            model_name = embed_model.model_name
            logger.info(f"Successfully initialized FastEmbedEmbedding with model: {model_name}")
            
            # Store the model for all languages
            for language in languages:
                loaded_models[language] = embed_model
                results[language] = True
                
                # Notify about completion for each language
                if progress_callback:
                    progress_callback(language, 100.0, True, None)
                    
            logger.info(f"Successfully loaded FastEmbedEmbedding model for all languages")
            return results
            
        except Exception as e:
            logger.error(f"Error initializing FastEmbedEmbedding: {str(e)}")
            # Continue with fallback
    except ImportError:
        logger.warning("FastEmbedEmbedding not available, using fallback methods")
        
    # Fallback to trying individual models if the shared approach fails
    for language in languages:
        start_time = time.time()
        logger.info(f"Loading model for language '{language}' (model: {language_models.get(language, 'default')})")
        
        # Skip if already loaded and not forcing reload
        if language in loaded_models and not force_reload:
            logger.info(f"Model for '{language}' already loaded, skipping")
            results[language] = True
            
            # Notify about completion
            if progress_callback:
                progress_callback(language, 100.0, True, None)
            continue
        
        # Initialize progress at 0%
        if progress_callback:
            progress_callback(language, 0.0, False, None)
            
        try:
            # Get model name from mapping, default to English model
            model_name = language_models.get(language, language_models.get('en', 'all-MiniLM-L6-v2'))
            
            # Update progress to 25% - model name identified
            if progress_callback:
                progress_callback(language, 25.0, False, None)
                
            # Use FastEmbedEmbedding as primary approach
            try:
                # Update progress to 50% - downloading complete, initializing model
                if progress_callback:
                    progress_callback(language, 50.0, False, None)
                    
                model = FastEmbedEmbedding()
                
                # Update progress to 75% - model initialized, warming up
                if progress_callback:
                    progress_callback(language, 75.0, False, None)
                
                # Warm up the model with a sample input
                _ = model.get_text_embedding("Warm up text to initialize the model")
                
                loaded_models[language] = model
                results[language] = True
                
                elapsed = time.time() - start_time
                logger.info(f"Successfully loaded FastEmbedEmbedding model for '{language}' in {elapsed:.2f}s")
                
                # Notify about completion
                if progress_callback:
                    progress_callback(language, 100.0, True, None)
                    
            except Exception as e:
                logger.error(f"Failed to load FastEmbedEmbedding model: {str(e)}")
                # Mark as failed
                results[language] = False
                
                # Notify about failure
                if progress_callback:
                    progress_callback(language, 100.0, False, f"Error loading model: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error loading model for '{language}': {str(e)}")
            results[language] = False
            
            # Notify about failure
            if progress_callback:
                progress_callback(language, 100.0, False, f"Error: {str(e)}")
                
    return results

def get_model(language: str) -> Optional[Union[object, tuple]]:
    """
    Get a preloaded model for the specified language.
    
    Args:
        language: Language code to get model for
        
    Returns:
        Preloaded model or None if not available
    """
    if language in loaded_models:
        return loaded_models[language]
    
    # Try to initialize a new FastEmbedEmbedding instance
    try:
        from llama_index.embeddings.fastembed import FastEmbedEmbedding
        logger.info(f"Creating new FastEmbedEmbedding instance for language: {language}")
        
        model = FastEmbedEmbedding()
        # Cache it for future use
        loaded_models[language] = model
        
        return model
    except Exception as e:
        logger.error(f"Failed to create FastEmbedEmbedding for language {language}: {str(e)}")
    
    # If we get here, creation failed - try to load the model on demand as fallback
    preload_models([language])
    return loaded_models.get(language)

# If this script is run directly, preload all models
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preload_models() 