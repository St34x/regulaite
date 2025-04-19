import axios from 'axios';
import authService from './authService';
import { jwtDecode } from 'jwt-decode';

// Base URL for API calls
const API_URL = process.env.AI_BACKEND_API_URL || 'http://ai-backend:8080';

// Create axios instance with base URL
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add interceptor to include auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
      
      // Add user ID from auth token if available
      const userData = authService.getCurrentUserData();
      if (userData) {
        // Try to find user ID in various possible locations
        const userId = userData.user_id || userData.sub || userData.id || userData.userId;
        
        if (userId) {
          config.headers['X-User-ID'] = userId;
        } else {
          // If no explicit user ID, try to extract from the token subject
          try {
            const decoded = jwtDecode(token);
            if (decoded && decoded.sub) {
              config.headers['X-User-ID'] = decoded.sub;
            }
          } catch (e) {
            console.warn('Could not extract user ID from token', e);
          }
        }
      }
    }
    return config;
  },
  (error) => Promise.reject(error)
);

/**
 * Service to handle interactions with the chat API
 */
const chatService = {
  /**
   * Get chat sessions for the current user
   * @param {string} userId - Optional user ID
   * @param {number} limit - Maximum number of sessions to return
   * @param {number} offset - Offset for pagination
   * @returns {Promise<Array>} List of chat sessions
   */
  getChatSessions: async (userId = null, limit = 20, offset = 0) => {
    try {
      // If userId not provided, get from auth service
      if (!userId) {
        const userData = authService.getCurrentUserData();
        userId = userData?.user_id;
        
        // Additional logging to help diagnose issues
        if (!userId) {
          console.warn('No user ID available from getCurrentUserData()', userData);
        } else {
          console.log('Fetching chat sessions for user ID:', userId);
        }
      }
      
      const response = await api.get(`/chat/sessions`, {
        params: { user_id: userId, limit, offset },
        // Add timeout to prevent long-hanging requests
        timeout: 10000
      });
      
      // Log response status and data shape for debugging
      console.log(`Sessions response status: ${response.status}, found ${response.data.sessions?.length || 0} sessions`);
      
      // Check if we have valid sessions in the response
      if (!response.data.sessions) {
        console.warn('API response missing sessions array:', response.data);
        return [];
      }
      
      return response.data.sessions || [];
    } catch (error) {
      // Enhanced error logging with more details
      console.error('Error fetching chat sessions:', error);
      
      // Check if this is a 404 error, which could mean no sessions yet
      if (error.response && error.response.status === 404) {
        console.log('No sessions found for user - likely a new user');
        // Return empty array instead of throwing for new users
        return [];
      }
      
      // Check for network errors that might be transient
      if (!error.response && error.request) {
        console.error('Network error when fetching sessions - no response received');
      }
      
      // Check for timeout
      if (error.code === 'ECONNABORTED') {
        console.error('Request timeout when fetching sessions');
      }
      
      // Re-throw the error to be handled by the calling component
      throw error;
    }
  },

  /**
   * Send a message to the chat API
   * @param {string} sessionId - ID of the chat session
   * @param {string} message - Message content
   * @param {Object} options - Additional options for the request
   * @param {Array} allMessages - Optional array of all messages in the conversation history
   * @returns {Promise<Object>} Chat response
   */
  sendMessage: async (sessionId, message, options = {}, allMessages = null) => {
    try {
      // Log message length for debugging
      const messageLength = message.trim().length;
      console.log(`Sending message with length: ${messageLength}`);
      
      // Prepare LLM parameters
      const llmParams = {
        model: options.model || 'gpt-4',
        temperature: options.temperature !== undefined ? options.temperature : 0.7,
        max_tokens: options.max_tokens || 2048,
        top_p: options.top_p !== undefined ? options.top_p : 1.0,
        frequency_penalty: options.frequency_penalty !== undefined ? options.frequency_penalty : 0.0,
        presence_penalty: options.presence_penalty !== undefined ? options.presence_penalty : 0.0,
      };
      
      // Prepare agent parameters if agent is enabled
      const agentParams = options.agent ? {
        use_agent: true,
        agent_type: options.agent.agent_type,
        use_tree_reasoning: options.agent.use_tree_reasoning,
        tree_template: options.agent.use_tree_reasoning ? options.agent.tree_template : null,
      } : {
        use_agent: false
      };
      
      // Prepare messages array - either use full history or just the current message
      const messages = allMessages || [{ role: 'user', content: message }];
      
      // For very short messages, ensure we're sending context
      if (messageLength <= 20 && !allMessages) {
        console.warn('Short message detected but no context provided. For better results, pass the conversation history.');
      }

      // Combine all parameters
      const requestParams = {
        session_id: sessionId,
        messages: messages,
        stream: false,
        include_context: options.includeContext !== undefined ? options.includeContext : true,
        context_query: options.contextQuery || null,
        retrieval_type: options.retrievalType || 'auto',
        ...llmParams,
        ...agentParams
      };

      const response = await api.post(`/chat`, requestParams);
      return response.data;
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  },

  /**
   * Send a message with streaming response
   * @param {string} sessionId - ID of the chat session
   * @param {string} message - Message content
   * @param {Function} onChunk - Callback for each chunk of the streamed response
   * @param {Object} options - Additional options for the request
   * @param {Array} allMessages - Optional array of all messages in the conversation history
   * @returns {Promise<Object>} Result info including execution IDs if applicable
   */
  sendMessageStreaming: async (sessionId, message, onChunk, options = {}, allMessages = null) => {
    try {
      // Log message length for debugging
      const messageLength = message.trim().length;
      console.log(`Streaming message with length: ${messageLength}`);
      
      // For very short messages, ensure we're sending context
      if (messageLength <= 20 && !allMessages) {
        console.warn('Short message detected but no context provided. For better results, pass the conversation history.');
      }
      
      // For streaming, we use fetch API with proper auth headers
      const token = localStorage.getItem('token');
      const headers = {
        'Content-Type': 'application/json',
      };
      
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
        
        // Add user ID from auth token if available
        const userData = authService.getCurrentUserData();
        if (userData) {
          // Try to find user ID in various possible locations
          const userId = userData.user_id || userData.sub || userData.id || userData.userId;
          
          if (userId) {
            headers['X-User-ID'] = userId;
          } else {
            // If no explicit user ID, try to extract from the token subject
            try {
              const decoded = jwtDecode(token);
              if (decoded && decoded.sub) {
                headers['X-User-ID'] = decoded.sub;
              }
            } catch (e) {
              console.warn('Could not extract user ID from token', e);
            }
          }
        }
      }
      
      // Prepare LLM parameters
      const llmParams = {
        model: options.model || 'gpt-4',
        temperature: options.temperature !== undefined ? options.temperature : 0.7,
        max_tokens: options.max_tokens || 2048,
        top_p: options.top_p !== undefined ? options.top_p : 1.0,
        frequency_penalty: options.frequency_penalty !== undefined ? options.frequency_penalty : 0.0,
        presence_penalty: options.presence_penalty !== undefined ? options.presence_penalty : 0.0,
      };
      
      // Prepare agent parameters if agent is enabled
      const agentParams = options.agent ? {
        use_agent: true,
        agent_type: options.agent.agent_type,
        use_tree_reasoning: options.agent.use_tree_reasoning,
        tree_template: options.agent.use_tree_reasoning ? options.agent.tree_template : null,
      } : {
        use_agent: false
      };
      
      // Prepare messages array - either use full history or just the current message
      const messages = allMessages || [{ role: 'user', content: message }];
      
      // Combine all parameters
      const requestParams = {
        session_id: sessionId,
        messages: messages,
        stream: true,
        include_context: options.includeContext !== undefined ? options.includeContext : true,
        context_query: options.contextQuery || null,
        retrieval_type: options.retrievalType || 'auto',
        ...llmParams,
        ...agentParams
      };
      
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(requestParams),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Error ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let fullResponse = '';
      let responseMetadata = {};

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        buffer += text;

        // Process SSE events (Server-Sent Events)
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim() === '') continue;
          if (!line.startsWith('data:')) continue;

          try {
            const eventData = JSON.parse(line.slice(5).trim());
            
            // Handle different event types
            if (eventData.event === 'start') {
              // Start of the streaming response
              responseMetadata.session_id = eventData.session_id;
            } else if (eventData.event === 'chunk') {
              // Content chunk
              if (eventData.content) {
                const chunk = eventData.content;
                fullResponse += chunk;
                onChunk(chunk);
              }
            } else if (eventData.event === 'end') {
              // End of response
              continue;
            } else if (eventData.type === 'metadata') {
              // Store metadata about the response
              responseMetadata = {
                ...responseMetadata,
                ...eventData.data
              };
            } else if (eventData.type === 'agent_progress') {
              // Update agent progress information
              responseMetadata.agent_execution_id = eventData.data.execution_id;
              responseMetadata.current_tree_node = eventData.data.current_node;
              responseMetadata.agent_status = eventData.data.status;
            } else if (eventData.type === 'content') {
              // Handle regular content chunks
              if (eventData.data && eventData.data.content) {
                const chunk = eventData.data.content;
                fullResponse += chunk;
                onChunk(chunk);
              }
            } else if (eventData.content) {
              // Direct content in event data (for compatibility)
              const chunk = eventData.content;
              fullResponse += chunk;
              onChunk(chunk);
            }
          } catch (e) {
            console.error('Error parsing server event:', e, line);
          }
        }
      }

      return {
        message: fullResponse,
        ...responseMetadata
      };
    } catch (error) {
      console.error('Error in streaming chat:', error);
      throw error;
    }
  },

  /**
   * Create a new chat session
   * @param {string} userId - Optional user ID
   * @returns {Promise<Object>} New session information
   */
  createSession: async (userId = null) => {
    try {
      // If userId not provided, get from auth service
      if (!userId) {
        const userData = authService.getCurrentUserData();
        if (userData) {
          userId = userData.user_id || userData.sub || userData.id || userData.userId;
          console.log('Creating session for user ID:', userId);
        } else {
          console.warn('No user data available when creating session');
        }

        // If still no userId, try to get it from the token directly
        if (!userId) {
          const token = localStorage.getItem('token');
          if (token) {
            try {
              const decoded = jwtDecode(token);
              userId = decoded.sub || decoded.user_id || decoded.id || decoded.userId;
              console.log('Extracted user ID from token:', userId);
            } catch (e) {
              console.warn('Could not extract user ID from token', e);
            }
          } else {
            console.warn('No auth token found when creating session');
          }
        }
      }
      
      // If we still don't have a user ID, generate a temporary one
      // This ensures the API call doesn't fail just because of missing user ID
      if (!userId) {
        userId = `temp-${Date.now()}`;
        console.warn('Using temporary user ID for session creation:', userId);
      }
      
      const response = await api.post(`/chat/sessions`, {
        user_id: userId,
      }, {
        // Add timeout to prevent long-hanging requests
        timeout: 10000
      });
      
      // Validate the response contains a session ID
      if (!response.data || !response.data.session_id) {
        console.error('Invalid response from create session API:', response.data);
        throw new Error('Server returned invalid session data');
      }
      
      return response.data;
    } catch (error) {
      console.error('Error creating chat session:', error);
      
      // Check for specific error types for better handling
      if (error.response) {
        const status = error.response.status;
        // For 401/403 errors, we should clear the token and redirect to login
        if (status === 401 || status === 403) {
          console.error('Authentication error when creating session:', status);
        } else if (status === 429) {
          console.error('Rate limit exceeded when creating session');
        } else if (status === 500) {
          console.error('Server error when creating session:', error.response.data);
        }
        
        // Add more specific error message to the error object
        error.sessionCreationFailed = true;
        error.detailedMessage = error.response.data?.detail || `Server returned ${status}`;
      } else if (error.request) {
        console.error('Network error when creating session - no response received');
        // Add network-specific error info
        error.sessionCreationFailed = true;
        error.isNetworkError = true;
        error.detailedMessage = 'Network error: Could not reach the server';
      } else if (error.code === 'ECONNABORTED') {
        console.error('Request timeout when creating session');
        error.sessionCreationFailed = true;
        error.isTimeoutError = true;
        error.detailedMessage = 'Request timed out: Server took too long to respond';
      }
      
      throw error;
    }
  },
  
  /**
   * Track agent execution progress
   * @param {string} executionId - ID of the execution to track
   * @returns {Promise<Object>} Progress information
   */
  getAgentProgress: async (executionId) => {
    if (!executionId) return null;
    
    try {
      const response = await api.get(`/chat/progress/${executionId}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching agent progress:', error);
      return null;
    }
  },
  
  /**
   * Get available agent types
   * @returns {Promise<Object>} Map of agent types to descriptions
   */
  getAgentTypes: async () => {
    try {
      const response = await api.get(`/agents/types`);
      return response.data;
    } catch (error) {
      console.error('Error fetching agent types:', error);
      return {};
    }
  },
  
  /**
   * Get available decision trees
   * @returns {Promise<Object>} Map of tree IDs to tree info
   */
  getDecisionTrees: async () => {
    try {
      const response = await api.get(`/agents/trees`);
      return response.data;
    } catch (error) {
      console.error('Error fetching decision trees:', error);
      return {};
    }
  },
  
  /**
   * Get details about a specific decision tree
   * @param {string} treeId - ID of the tree to get
   * @returns {Promise<Object>} Tree details
   */
  getDecisionTree: async (treeId) => {
    try {
      const response = await api.get(`/agents/trees/${treeId}`);
      return response.data;
    } catch (error) {
      console.error('Error fetching decision tree:', error);
      return null;
    }
  },
  
  /**
   * Delete a chat session
   * @param {string} sessionId - ID of the session to delete
   * @returns {Promise<Object>} Result of the deletion
   */
  deleteSession: async (sessionId) => {
    try {
      const response = await api.delete(`/chat/sessions/${sessionId}`);
      return response.data;
    } catch (error) {
      console.error('Error deleting chat session:', error);
      throw error;
    }
  },
  
  /**
   * Get messages for a specific chat session
   * @param {string} sessionId - ID of the chat session
   * @param {number} limit - Maximum number of messages to return
   * @param {number} offset - Offset for pagination
   * @returns {Promise<Array>} List of messages in the session
   */
  getSessionMessages: async (sessionId, limit = 50, offset = 0) => {
    try {
      const response = await api.get(`/chat/sessions/${sessionId}/messages`, {
        params: { limit, offset },
      });
      return response.data.messages || [];
    } catch (error) {
      console.error('Error fetching session messages:', error);
      throw error;
    }
  },
  
  /**
   * Get LLM configuration
   * @returns {Promise<Object>} LLM configuration
   */
  getLLMConfig: async () => {
    try {
      const response = await api.get(`/config`);
      return response.data.llm || {};
    } catch (error) {
      console.error('Error fetching LLM config:', error);
      return {};
    }
  }
};

export default chatService; 