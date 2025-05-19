import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Fetches the current LLM configuration from the backend.
 * @returns {Promise<Object>} The LLM configuration.
 */
const getLlmConfig = async () => {
  try {
    const response = await axios.get(`${API_URL}/settings/llm`);
    return response.data;
  } catch (error) {
    console.error('Error fetching LLM config:', error.response?.data || error.message);
    throw error.response?.data || new Error('Failed to fetch LLM configuration');
  }
};

/**
 * Updates the LLM configuration on the backend.
 * @param {Object} llmSettings - The LLM settings to update.
 * @returns {Promise<Object>} The updated LLM configuration.
 */
const updateLlmConfig = async (llmSettings) => {
  try {
    const response = await axios.post(`${API_URL}/config/llm`, llmSettings);
    return response.data;
  } catch (error) {
    console.error('Error updating LLM config:', error.response?.data || error.message);
    throw error.response?.data || new Error('Failed to update LLM configuration');
  }
};

const configService = {
  getLlmConfig,
  updateLlmConfig,
};

export default configService; 