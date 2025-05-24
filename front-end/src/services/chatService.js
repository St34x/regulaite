import axios from 'axios';
import authService from './authService';
import { jwtDecode } from 'jwt-decode';

// Base URL for API calls
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8090';

// Create axios instance with base URL and timeout
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 600000, // 10 minutes timeout (increased from 5 minutes)
});

// Utility function to ensure we always have proper Error objects
const createSafeError = (err, defaultMessage = 'An unexpected error occurred') => {
  if (!err) {
    console.warn('createSafeError: Received null/undefined error, using default message');
    return new Error(defaultMessage);
  }
  if (err instanceof Error) return err;
  if (typeof err === 'string') {
    console.log('createSafeError: Converting string error to Error object');
    return new Error(err);
  }
  if (err.message) {
    console.log('createSafeError: Converting object with message property to Error object');
    return new Error(err.message);
  }
  if (err.toString && typeof err.toString === 'function') {
    console.log('createSafeError: Converting object using toString() to Error object');
    return new Error(err.toString());
  }
  console.warn('createSafeError: Unknown error type, using default message:', typeof err);
  return new Error(defaultMessage);
};

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
        console.log('🔐 Current user data from auth service:', userData);
        userId = userData?.user_id;
        
        // Additional logging to help diagnose issues
        if (!userId) {
          console.warn('❌ No user ID available from getCurrentUserData()', userData);
        } else {
          console.log('✅ Fetching chat sessions for user ID:', userId);
        }
      }
      
      console.log('📡 Making request to /chat/sessions with params:', { user_id: userId, limit, offset });
      
      const response = await api.get(`/chat/sessions`, {
        params: { user_id: userId, limit, offset },
        // Add timeout to prevent long-hanging requests
        timeout: 10000
      });
      
      // Log complete response for debugging
      console.log('📋 Complete response from /chat/sessions:', response);
      console.log('📋 Response data:', response.data);
      console.log('📋 Response data keys:', Object.keys(response.data || {}));
      
      // Log response status and data shape for debugging
      console.log(`Sessions response status: ${response.status}, found ${response.data.sessions?.length || 0} sessions`);
      
      // Check if we have valid sessions in the response
      // Try different possible response structures
      let sessions = null;
      if (response.data.sessions) {
        sessions = response.data.sessions;
        console.log('✅ Found sessions in response.data.sessions');
      } else if (Array.isArray(response.data)) {
        sessions = response.data;
        console.log('✅ Found sessions directly in response.data (array)');
      } else if (response.data.data && Array.isArray(response.data.data)) {
        sessions = response.data.data;
        console.log('✅ Found sessions in response.data.data');
      } else {
        console.warn('❌ API response missing sessions array or unexpected structure:', response.data);
        return [];
      }
      
      console.log('📊 Returning sessions:', sessions);
      return sessions || [];
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
      
      // Prepare agent parameters - autonomous agent enabled by default
      const agentParams = {
        use_agent: true
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
   * @param {string} requestId - Optional request ID for cancellation
   * @returns {Promise<Object>} Result info including execution IDs if applicable
   */
  sendMessageStreaming: async (sessionId, message, onChunk, options = {}, allMessages = null, requestId = null) => {
    // Generate request ID if not provided
    const reqId = requestId || `stream_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    let reader = null;
    let heartbeatInterval = null;
    let connectionCheckInterval = null;
    let timeoutId = null;
    
    try {
      // Log message length for debugging
      const messageLength = message.trim().length;
      console.log(`Streaming message with length: ${messageLength}, Request ID: ${reqId}`);
      
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
      
      // Prepare agent parameters - autonomous agent disabled by default for streaming
      const agentParams = {
        use_agent: options.use_agent || false  // Disable by default to prevent hanging
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
      
      // Enhanced timeout handling - longer timeout to prevent issues
      const timeoutMs = options.timeout || 600000; // 10 minutes default
      timeoutId = setTimeout(() => {
        console.warn(`Request ${reqId} timed out after ${timeoutMs}ms`);
      }, timeoutMs);
      
      const response = await fetch(`${API_URL}/chat/rag`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(requestParams),
      });

      // Clear timeout since we got a response
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Error ${response.status}: ${response.statusText}`);
      }

      reader = response.body.getReader();
      let fullResponse = '';
      let responseMetadata = {};
      
      // Enhanced progress tracking
      let lastProgressTime = Date.now();
      const progressTimeout = 60000; // 60 seconds without progress (increased)
      
      // Set up heartbeat to detect if streaming stalls
      heartbeatInterval = setInterval(() => {
        const now = Date.now();
        if (now - lastProgressTime > progressTimeout) {
          console.warn(`No progress in ${progressTimeout}ms, request may be stalled`);
          // Notify the UI about potential stalling
          try {
            onChunk({
              type: 'processing',
              state: 'Request taking longer than expected... Please wait.',
              step: 'timeout_warning',
              timestamp: new Date().toISOString()
            });
          } catch (chunkError) {
            console.warn('Error sending timeout warning chunk:', chunkError);
          }
        }
      }, 30000); // Check every 30 seconds (increased)
      
      // Connection health monitoring
      let connectionHealthy = true;
      connectionCheckInterval = setInterval(() => {
        const now = Date.now();
        const timeSinceLastProgress = now - lastProgressTime;
        
        if (timeSinceLastProgress > 90000 && connectionHealthy) {
          // Connection may be unhealthy
          connectionHealthy = false;
          try {
            onChunk({
              type: 'processing',
              state: 'Connection may be unstable. Monitoring...',
              step: 'connection_check',
              timestamp: new Date().toISOString()
            });
          } catch (chunkError) {
            console.warn('Error sending connection check chunk:', chunkError);
          }
        } else if (timeSinceLastProgress < 15000 && !connectionHealthy) {
          // Connection recovered
          connectionHealthy = true;
          try {
            onChunk({
              type: 'processing',
              state: 'Connection restored, continuing...',
              step: 'connection_restored',
              timestamp: new Date().toISOString()
            });
          } catch (chunkError) {
            console.warn('Error sending connection restored chunk:', chunkError);
          }
        }
      }, 45000); // Check every 45 seconds (increased)
      
      try {
        while (true) {
          const { done, value } = await reader.read();
          
          if (done) {
            console.log('Stream completed');
            break;
          }
          
          // Update progress time
          lastProgressTime = Date.now();
          
          const chunk = new TextDecoder().decode(value);
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (line.trim() === '') continue;
            
            try {
              const eventData = JSON.parse(line);
              
              // Enhanced event handling with better error recovery
              if (eventData.type === 'start') {
                console.log('Stream started:', eventData);
                responseMetadata.request_id = eventData.request_id;
                // Notify UI that streaming has started
                onChunk({
                  type: 'start',
                  timestamp: eventData.timestamp,
                  request_id: eventData.request_id
                });
              } else if (eventData.type === 'processing') {
                // Forward processing updates to UI
                onChunk({
                  type: 'processing',
                  state: eventData.state,
                  step: eventData.step,
                  step_number: eventData.step_number,
                  total_steps: eventData.total_steps,
                  details: eventData.details,
                  context_metadata: eventData.context_metadata,
                  internal_thoughts: eventData.internal_thoughts,
                  timestamp: eventData.timestamp
                });
              } else if (eventData.type === 'token') {
                // Handle response tokens
                if (eventData.content) {
                  const tokenContent = eventData.content;
                  fullResponse += tokenContent;
                  onChunk({
                    type: 'token',
                    content: tokenContent
                  });
                }
              } else if (eventData.type === 'end') {
                // Handle completion
                console.log('Stream ended with final data:', eventData);
                fullResponse = eventData.message || fullResponse;
                responseMetadata = {
                  ...responseMetadata,
                  model: eventData.model,
                  context_used: eventData.context_used,
                  session_id: eventData.session_id,
                  sources: eventData.sources,
                  context_quality: eventData.context_quality,
                  hallucination_risk: eventData.hallucination_risk,
                  internal_thoughts: eventData.internal_thoughts
                };
                
                // Notify UI of completion
                onChunk({
                  type: 'end',
                  message: fullResponse,
                  metadata: responseMetadata
                });
              } else if (eventData.type === 'error') {
                // Handle streaming errors
                console.error('Stream error:', eventData);
                const errorMessage = eventData.message || 'An error occurred during streaming';
                
                // Notify UI of error
                onChunk({
                  type: 'error',
                  message: errorMessage,
                  error_code: eventData.error_code,
                  request_id: eventData.request_id
                });
                
                // Throw error to exit the loop
                throw new Error(errorMessage);
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
                
                // Forward agent progress to UI
                onChunk({
                  type: 'agent_progress',
                  data: eventData.data
                });
              } else if (eventData.content) {
                // Direct content in event data (for compatibility)
                const chunk = eventData.content;
                fullResponse += chunk;
                onChunk({
                  type: 'token',
                  content: chunk
                });
              }
            } catch (e) {
              console.error('Error parsing server event:', e, 'Line:', line);
              // Continue processing other lines instead of failing completely
              continue;
            }
          }
        }
      } finally {
        // Always cleanup intervals
        if (heartbeatInterval) {
          clearInterval(heartbeatInterval);
          heartbeatInterval = null;
        }
        if (connectionCheckInterval) {
          clearInterval(connectionCheckInterval);
          connectionCheckInterval = null;
        }
        
        // Clean up the reader safely
        if (reader) {
          try {
            await reader.cancel();
          } catch (cancelError) {
            // Ignore cancel errors
            console.log('Reader cancel completed');
          }
          
          try {
            reader.releaseLock();
          } catch (lockError) {
            // Ignore lock release errors
            console.log('Reader lock released');
          }
          reader = null;
        }
      }

      return {
        message: fullResponse,
        requestId: reqId,
        ...responseMetadata
      };
    } catch (error) {
      console.error('Error in streaming chat:', error);
      
      // Simplified error handling - no more cancellation detection
      const safeError = createSafeError(error, 'An error occurred during streaming');
      const errorMessage = `Streaming error: ${safeError.message}`;
      
      try {
        onChunk({
          type: 'error',
          message: errorMessage,
          error_code: 'STREAMING_ERROR',
          request_id: reqId
        });
      } catch (chunkError) {
        console.warn('Error sending error chunk:', chunkError);
      }
      
      throw safeError;
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
   * Send a chat message with RAG enabled by default
   * @param {Object} payload - The message payload
   * @returns {Promise<Object>} Chat response
   */
  sendChatMessage: async (payload) => {
    try {
      // Ensure RAG is enabled by default
      const requestPayload = {
        ...payload,
        use_rag: true
      };
      
      const response = await api.post(`/chat/rag`, requestPayload);
      return response.data;
    } catch (error) {
      console.error('Error sending RAG-enabled message:', error);
      throw error;
    }
  },
  
  /**
   * Delete a chat session
   * @param {string} sessionId - ID of the session to delete
   * @returns {Promise<Object>} Result of the deletion
   */
  deleteSession: async (sessionId) => {
    try {
      console.log(`Preparing to delete session: ${sessionId}`);
      
      // Get auth headers and ensure we have a token
      const token = localStorage.getItem('token');
      if (!token) {
        console.error('No authentication token found for session deletion');
        throw new Error('Authentication required to delete sessions');
      }
      
      // Extract user ID from token to include in headers
      let userId = null;
      try {
        const decoded = jwtDecode(token);
        userId = decoded.sub || decoded.user_id || decoded.id;
        console.log(`Extracted user ID from token for deletion: ${userId}`);
      } catch (e) {
        console.warn('Could not extract user ID from token', e);
      }
      
      // Build headers with authentication information
      const headers = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      };
      
      // Add user ID to headers if available
      if (userId) {
        headers['X-User-ID'] = userId;
      }
      
      console.log(`Sending DELETE request to /chat/sessions/${sessionId}`);
      console.log('Using authentication headers:', JSON.stringify(headers));
      
      // Use fetch with the DELETE method explicitly
      const response = await fetch(`${API_URL}/chat/sessions/${sessionId}`, {
        method: 'DELETE',
        headers: headers,
        // Don't include body for DELETE requests to avoid issues with some servers
      });
      
      console.log(`Received response status: ${response.status} for session deletion: ${sessionId}`);
      
      // Handle error responses with more detailed logging
      if (!response.ok) {
        let errorText = '';
        try {
          // Try to parse error response as JSON
          const errorData = await response.json();
          errorText = errorData.detail || JSON.stringify(errorData);
        } catch (e) {
          // If not JSON, get as text
          errorText = await response.text();
        }
        
        console.error(`Error deleting session (${response.status}):`, errorText);
        throw new Error(`Failed to delete session: ${response.status} ${response.statusText} - ${errorText}`);
      }
      
      // Parse the response
      const data = await response.json();
      console.log('Delete session response:', data);
      
      // Log deletion results
      if (data.messages_deleted !== undefined) {
        console.log(`Successfully deleted session ${sessionId} with ${data.messages_deleted} messages`);
      } else {
        console.log(`Successfully deleted session ${sessionId}`);
      }
      
      return data;
    } catch (error) {
      console.error('Error deleting chat session:', error);
      
      // Add more specific error information
      if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
        console.error('Network error: Could not connect to the server');
      } else if (error.name === 'AbortError') {
        console.error('Request was aborted');
      } else if (error.message.includes('401') || error.message.includes('403')) {
        console.error('Authentication or authorization error when deleting session');
      }
      
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
  },

  /**
   * Stream a chat message to get real-time responses including internal thoughts
   * @param {Object} payload - Chat message payload
   * @param {Object} callbacks - Callback functions for streaming events
   * @param {string} requestId - Optional request ID for cancellation
   * @returns {Promise} - Promise that resolves when streaming is complete
   */
  streamChatMessage: async (payload, callbacks = {}, requestId = null) => {
    // Generate request ID if not provided
    const reqId = requestId || `stream_chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    console.log('🚀 Starting streamChatMessage with payload:', payload);
    console.log('🌐 API_URL:', API_URL);
    console.log('🔑 Request ID:', reqId);
    
    try {
      // Always ensure streaming is enabled
      payload.stream = true;
      
      // Make sure the callbacks are defined
      const {
        onToken = () => {},
        onProcessing = () => {},
        onComplete = () => {},
        onError = () => {}
      } = callbacks;
      
      console.log('📡 Making fetch request to:', `${API_URL}/chat/rag`);
      
      // Make the streaming request
      const response = await fetch(`${API_URL}/chat/rag`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
          'Cache-Control': 'no-cache'
        },
        body: JSON.stringify(payload),
      });
      
      console.log('📨 Response received:', {
        status: response.status,
        statusText: response.statusText,
        headers: Object.fromEntries(response.headers.entries())
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Response not ok:', errorText);
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }
      
      if (!response.body) {
        console.error('❌ No response body');
        throw new Error('No response body received');
      }
      
      // Check if response is actually a stream
      const contentType = response.headers.get('content-type');
      console.log('📋 Content-Type:', contentType);
      
      if (!contentType || !contentType.includes('text/event-stream')) {
        console.warn('⚠️ Unexpected content type, might not be streaming properly');
      }
      
      // Process the stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let messageBuffer = '';
      let isProcessing = false;
      
      console.log('🔄 Starting to read stream...');
      
      while (true) {
        try {
          const { done, value } = await reader.read();
          
          if (done) {
            console.log('✅ Stream reading completed');
            break;
          }
          
          // Decode the chunk
          const chunk = decoder.decode(value, { stream: true });
          buffer += chunk;
          
          console.log('📦 Received chunk:', chunk.length, 'bytes');
          
          // Process complete lines
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep incomplete line in buffer
          
          for (const line of lines) {
            if (line.trim() === '') continue;
            
            console.log('📝 Processing line:', line.substring(0, 100) + (line.length > 100 ? '...' : ''));
            
            try {
              const data = JSON.parse(line);
              console.log('🔍 Parsed data type:', data.type);
              
              switch (data.type) {
                case 'start':
                  console.log('🎬 Stream started');
                  isProcessing = true;
                  onProcessing('Starting to process your query...');
                  break;
                  
                case 'processing':
                  console.log('⚙️ Processing update:', data.state);
                  if (data.internal_thoughts) {
                    console.log('💭 Internal thoughts:', data.internal_thoughts.substring(0, 50) + '...');
                  }
                  onProcessing(data.state, data);
                  break;
                  
                case 'token':
                  console.log('🎯 Token received:', data.content);
                  messageBuffer += data.content;
                  onToken(data.content);
                  break;
                  
                case 'end':
                  console.log('🏁 Stream ended with message length:', data.message?.length || 0);
                  isProcessing = false;
                  onComplete({
                    message: data.message || messageBuffer,
                    model: data.model,
                    context_used: data.context_used,
                    session_id: data.session_id,
                    sources: data.sources,
                    internal_thoughts: data.internal_thoughts,
                    context_quality: data.context_quality,
                    hallucination_risk: data.hallucination_risk
                  });
                  return; // Exit the function
                  
                case 'error':
                  console.error('❌ Stream error:', data.message);
                  isProcessing = false;
                  onError(new Error(data.message));
                  return;
                  
                default:
                  console.warn('⚠️ Unknown data type:', data.type);
              }
            } catch (parseError) {
              console.error('❌ Error parsing JSON line:', parseError, 'Line:', line);
            }
          }
        } catch (readError) {
          console.error('❌ Error reading stream:', readError);
          throw readError;
        }
      }
      
      // If we get here without an 'end' event, something went wrong
      console.warn('⚠️ Stream ended without completion event');
      if (isProcessing) {
        onComplete({
          message: messageBuffer || 'Response was incomplete',
          model: payload.model,
          context_used: false,
          session_id: payload.session_id
        });
      }
      
    } catch (error) {
      console.error('❌ streamChatMessage error:', error);
      
      // Call error callback for unexpected errors only
      const { onError = () => {} } = callbacks;
      onError(error);
      
      throw error;
    }
  }
};

export default chatService; 