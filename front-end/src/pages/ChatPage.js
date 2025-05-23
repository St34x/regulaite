import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Shield, PanelLeft, X, RefreshCw } from 'lucide-react';
import ChatMessage from '../components/chat/ChatMessage';
import ChatHistory from '../components/chat/ChatHistory';
import LoadingOverlay from '../components/ui/LoadingOverlay';
import useMediaQuery from '../hooks/useMediaQuery';
import chatService from '../services/chatService';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import ChatControls from '../components/chat/ChatControls';
import { Box, Flex, Heading, Button, IconButton, Text, Spinner, VStack, HStack, useToast, useColorModeValue } from '@chakra-ui/react';

// Initial message for new chats
const initialMessage = {
  role: "assistant",
  content:
    "Hello! I'm your GRC AI Assistant. I can help you with governance, risk, and compliance questions. How can I assist you today?",
};

// Sample suggested questions
const suggestedQuestions = [
  "What are the key requirements for SOC 2 compliance?",
  "How do I conduct a risk assessment for my organization?",
  "Explain the main components of a governance framework",
  "What are the GDPR requirements for data processing?",
];

/**
 * Chat page component
 */
const ChatPage = () => {
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState([initialMessage]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [error, setError] = useState(null);
  const [reasoningNodeId, setReasoningNodeId] = useState(null);
  const [agentProgress, setAgentProgress] = useState(null);
  const [currentRequestId, setCurrentRequestId] = useState(null);
  
  const messagesEndRef = useRef(null);
  const isMobile = useMediaQuery('(max-width: 768px)');
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const toast = useToast();

  // Theme colors
  const accentColor = '#4415b6';
  const accentHoverColor = '#3a1296';
  const sidebarBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const headerBg = useColorModeValue('white', 'gray.800');
  const chatBg = useColorModeValue('gray.50', 'gray.900');
  const errorBg = useColorModeValue('red.100', 'red.900');
  const errorColor = useColorModeValue('red.800', 'red.200');
  const errorBorderColor = useColorModeValue('red.200', 'red.700');
  const textColor = useColorModeValue('gray.800', 'gray.200');
  const secondaryTextColor = useColorModeValue('gray.500', 'gray.400');
  const buttonHoverBg = useColorModeValue('purple.50', 'purple.900');
  const buttonHoverBorderColor = useColorModeValue('purple.300', 'purple.600');
  const questionButtonBg = useColorModeValue('white', 'gray.700');
  const questionButtonBorder = useColorModeValue('gray.200', 'gray.600');
  const loadingBg = useColorModeValue('blue.50', 'blue.900');
  const loadingBorderColor = useColorModeValue('blue.200', 'blue.700');
  const loadingTextColor = useColorModeValue('blue.700', 'blue.200');

  // Add ref for tracking accumulated content to prevent stale closures
  const messageContentRef = useRef(new Map());

  // Close sidebar on mobile by default
  useEffect(() => {
    if (isMobile) {
      setIsSidebarOpen(false);
    } else {
      setIsSidebarOpen(true);
    }
  }, [isMobile]);

  // Redirect to login page if user is not authenticated
  useEffect(() => {
    if (!isAuthenticated()) {
      navigate('/login');
    }
  }, [isAuthenticated, navigate]);

  // Fetch chat sessions on component mount
  useEffect(() => {
    fetchSessions();
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  // Track agent progress if there's an active execution
  useEffect(() => {
    let timer = null;
    
    if (agentProgress && agentProgress.execution_id && agentProgress.status !== 'completed' && agentProgress.status !== 'failed') {
      timer = setInterval(async () => {
        try {
          const progress = await chatService.getAgentProgress(agentProgress.execution_id);
          if (progress) {
            setAgentProgress(progress);
            
            // If there's a current tree node, update the reasoning node ID
            if (progress.current_tree_node) {
              setReasoningNodeId(progress.current_tree_node);
            }
            
            // If completed or failed, clear the interval
            if (progress.status === 'completed' || progress.status === 'failed') {
              clearInterval(timer);
            }
          }
        } catch (err) {
          console.error('Error fetching agent progress:', err);
        }
      }, 1000);
    }
    
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [agentProgress]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      console.log('🧹 ChatPage unmounting, cleaning up');
    };
  }, []);

  const fetchSessions = async () => {
    setError(null);
    try {
      const fetchedSessions = await chatService.getChatSessions();
      
      if (fetchedSessions && Array.isArray(fetchedSessions) && fetchedSessions.length > 0) {
        // Process and deduplicate sessions
        const uniqueSessions = [];
        const sessionIds = new Set();
        
        fetchedSessions.forEach(session => {
          if (!session) return; // Skip null or undefined sessions
          
          const sessionId = session.session_id || session.id;
          if (!sessionId) return; // Skip sessions without ID
          
          if (!sessionIds.has(sessionId)) {
            sessionIds.add(sessionId);
            
            // Transform the backend format to our frontend format
            uniqueSessions.push({
              id: sessionId,
              title: session.title || "New Conversation",
              date: session.last_message_time ? new Date(session.last_message_time).toLocaleString() : "Just now",
              preview: session.preview || "",
              message_count: session.message_count || 0
            });
          }
        });
        
        setSessions(uniqueSessions);
        
        // Select the first session if we have any
        if (uniqueSessions.length > 0) {
          handleSelectSession(uniqueSessions[0].id);
        } else {
          await handleNewSession();
        }
      } else {
        // If no sessions, create a new one
        console.log('No existing sessions found, creating a new session');
        await handleNewSession();
      }
    } catch (err) {
      console.error('Failed to fetch chat sessions:', err);
      let errorMessage = 'Failed to load chat sessions. Creating a new session.';
      let errorType = 'session_load_error';
      let shouldCreateFallback = true;
      let shouldDisplayError = false;
      
      if (err.response) {
        const status = err.response.status;
        if (status === 401) {
          errorMessage = 'Authentication required to access chat sessions.';
          errorType = 'auth_error';
          shouldDisplayError = true;
          // Redirect to login page if unauthorized
          navigate('/login');
          return;
        } else if (status === 403) {
          errorMessage = 'You do not have permission to access these chat sessions.';
          errorType = 'permission_error';
          shouldDisplayError = true;
        } else if (status === 404) {
          errorMessage = 'User chat history not found. Starting a new session.';
          errorType = 'not_found_error';
        } else if (status === 500) {
          errorMessage = 'Server error while loading sessions. Creating a new session.';
          errorType = 'server_error';
        } else if (status === 400) {
          // Check if this is a specific error related to missing chat history
          const errorDetail = err.response.data?.detail || '';
          if (errorDetail.includes('history') || errorDetail.includes('sessions')) {
            errorMessage = 'No chat history found. Starting a new conversation.';
            errorType = 'no_history_error';
          }
        }
      } else if (err.request) {
        errorMessage = 'Network error. Unable to connect to the chat server.';
        errorType = 'network_error';
      }
      
      // Only set the error in state if we should display it
      if (shouldDisplayError) {
        setError(errorMessage);
        
        // Show toast notification for better user feedback
        toast({
          title: 'Chat History',
          description: errorMessage,
          status: 'info',
          duration: 5000,
          isClosable: true,
        });
      } else {
        // Clear any existing error
        setError(null);
      }
      
      // Only create fallback sessions if not redirecting due to auth error
      // and if we should create a fallback
      if (shouldCreateFallback) {
        console.log('Creating fallback session due to error:', errorType);
        const fallbackSessionId = Date.now().toString();
        setSessions([{
          id: fallbackSessionId,
          title: "New Conversation",
          date: "Just now",
          preview: "",
          messages: [initialMessage],
          is_fallback: true
        }]);
        
        setActiveSessionId(fallbackSessionId);
        setMessages([initialMessage]);
      }
    }
  };

  const handleSendMessage = async (content) => {
    console.log('🎯 handleSendMessage called with content:', content);
    console.log('🔄 Current isLoading state:', isLoading);
    
    if (!content.trim() || isLoading) {
      console.log('❌ Message rejected - empty content or already loading');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    // Generate request ID for this message
    const requestId = `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setCurrentRequestId(requestId);
    
    console.log('✅ Starting message processing with requestId:', requestId);
    
    try {
      // Create a new user message
      const userMessage = {
        role: "user",
        content: content
      };
      
      // Add to UI immediately
      const updatedMessages = [...messages, userMessage];
      
      // Create a temporary assistant message that will show processing state
      const tempAssistantMessage = {
        role: "assistant",
        content: "",
        isGenerating: true,
        processingState: "Initializing request...",
        requestId: requestId,
        metadata: {
          internal_thoughts: "",
          processingSteps: [],
          currentStep: 0,
          totalSteps: 6,
          startTime: Date.now(),
          requestId: requestId,
          isConnected: true
        }
      };
      
      // Add both messages to the UI
      setMessages([...updatedMessages, tempAssistantMessage]);
      
      // Update the session with the user message
      if (activeSessionId) {
        updateSessionWithMessages(activeSessionId, updatedMessages);
      }
      
      // Prepare messages for the API call
      const messagesForAPI = updatedMessages.map(msg => ({
        role: msg.role,
        content: msg.content
      }));
      
      console.log('📞 Starting streaming request to /chat/rag');
      
      // Use the enhanced streaming API
      const result = await chatService.sendMessageStreaming(
        activeSessionId,
        content,
        (chunkData) => {
          // Handle different types of streaming data
          setMessages(currentMessages => {
            const updatedMessages = [...currentMessages];
            const lastMessage = updatedMessages[updatedMessages.length - 1];
            
            if (lastMessage.role === "assistant" && lastMessage.isGenerating) {
              // Create a unique key for this message
              const messageKey = `${lastMessage.id || 'temp'}_${lastMessage.timestamp}`;
              
              if (chunkData.type === 'start') {
                // Streaming started - initialize the content accumulator
                lastMessage.processingState = "Connection established, starting processing...";
                lastMessage.metadata.streamStarted = true;
                lastMessage.metadata.backendRequestId = chunkData.request_id;
                
                // Initialize content accumulator for this message
                if (!messageContentRef.current.has(messageKey)) {
                  messageContentRef.current.set(messageKey, '');
                }
              } else if (chunkData.type === 'processing') {
                // Processing update
                lastMessage.processingState = chunkData.state || "Processing...";
                
                if (chunkData.internal_thoughts) {
                  lastMessage.metadata.internal_thoughts = chunkData.internal_thoughts;
                }
                
                // Update step information
                if (chunkData.step_number && chunkData.total_steps) {
                  lastMessage.metadata.currentStep = chunkData.step_number;
                  lastMessage.metadata.totalSteps = chunkData.total_steps;
                  
                  // Initialize processingSteps if not exists
                  if (!lastMessage.metadata.processingSteps) {
                    lastMessage.metadata.processingSteps = [];
                  }
                  
                  // Update or add the current step
                  const stepData = {
                    step: chunkData.step,
                    stepNumber: chunkData.step_number,
                    totalSteps: chunkData.total_steps,
                    message: chunkData.state,
                    details: chunkData.details,
                    contextMetadata: chunkData.context_metadata,
                    status: 'in_progress',
                    timestamp: chunkData.timestamp
                  };
                  
                  const existingStepIndex = lastMessage.metadata.processingSteps.findIndex(
                    step => step.step === chunkData.step
                  );
                  
                  if (existingStepIndex >= 0) {
                    lastMessage.metadata.processingSteps[existingStepIndex] = stepData;
                  } else {
                    lastMessage.metadata.processingSteps.push(stepData);
                  }
                  
                  // Mark previous steps as completed
                  lastMessage.metadata.processingSteps.forEach((step) => {
                    if (step.stepNumber < chunkData.step_number) {
                      step.status = 'completed';
                    }
                  });
                }
                
                // Store context metadata
                if (chunkData.context_metadata) {
                  lastMessage.metadata.contextMetadata = chunkData.context_metadata;
                }
                
                // Handle special processing steps
                if (chunkData.step === 'generation_active') {
                  // This is a heartbeat during AI generation - show activity indicator
                  lastMessage.metadata.isGeneratingActive = true;
                  lastMessage.metadata.lastGenerationHeartbeat = Date.now();
                } else if (chunkData.step === 'generation_delay') {
                  // Generation is taking longer than expected
                  lastMessage.metadata.generationDelayed = true;
                } else if (chunkData.step === 'timeout_warning') {
                  // Request may be stalled
                  lastMessage.metadata.timeoutWarning = true;
                }
              } else if (chunkData.type === 'token' && chunkData.content) {
                // Response token - FIXED ACCUMULATION TO PREVENT DUPLICATION
                const currentContent = messageContentRef.current.get(messageKey) || '';
                const newContent = currentContent + chunkData.content;
                
                // Update the ref with accumulated content
                messageContentRef.current.set(messageKey, newContent);
                
                // Update the message content from ref to prevent stale closures
                lastMessage.content = newContent;
                lastMessage.processingState = "Generating response...";
              } else if (chunkData.type === 'end') {
                // Streaming completed
                lastMessage.isGenerating = false;
                lastMessage.processingState = "Complete";
                
                // Use the final content from the backend or the accumulated content
                const accumulatedContent = messageContentRef.current.get(messageKey) || lastMessage.content;
                const finalContent = chunkData.message || accumulatedContent;
                
                // Update final message content if provided from backend is longer
                if (finalContent && finalContent.length >= lastMessage.content.length) {
                  lastMessage.content = finalContent;
                  messageContentRef.current.set(messageKey, finalContent);
                }
                
                // Store final metadata
                if (chunkData.metadata) {
                  lastMessage.metadata = {
                    ...lastMessage.metadata,
                    ...chunkData.metadata,
                    completed: true,
                    endTime: Date.now()
                  };
                }
                
                // Mark all steps as completed
                if (lastMessage.metadata.processingSteps) {
                  lastMessage.metadata.processingSteps.forEach(step => {
                    if (step.status === 'in_progress') {
                      step.status = 'completed';
                    }
                  });
                }
                
                // Clean up the content accumulator for this message
                messageContentRef.current.delete(messageKey);
              } else if (chunkData.type === 'error') {
                // Handle streaming errors
                lastMessage.isGenerating = false;
                lastMessage.processingState = "Error occurred";
                lastMessage.metadata.error = {
                  message: chunkData.message,
                  error_code: chunkData.error_code,
                  request_id: chunkData.request_id
                };
                
                // Show error message in content if no content was generated
                if (!lastMessage.content.trim()) {
                  lastMessage.content = `❌ ${chunkData.message}`;
                }
                
                // Mark current step as failed
                if (lastMessage.metadata.processingSteps && lastMessage.metadata.processingSteps.length > 0) {
                  const currentStepIndex = lastMessage.metadata.processingSteps.findIndex(
                    step => step.status === 'in_progress'
                  );
                  if (currentStepIndex >= 0) {
                    lastMessage.metadata.processingSteps[currentStepIndex].status = 'failed';
                  }
                }
                
                // Clean up content accumulator on error
                messageContentRef.current.delete(messageKey);
              } else if (chunkData.type === 'agent_progress') {
                // Agent progress update
                lastMessage.metadata.agentProgress = chunkData.data;
              }
              
              // Update connection status based on activity
              lastMessage.metadata.isConnected = true;
              lastMessage.metadata.lastActivity = Date.now();
            }
            
            return updatedMessages;
          });
        },
        {
          model: 'gpt-4',
          temperature: 0.7,
          max_tokens: 2048,
          includeContext: true,
          use_agent: false, // Disable agent by default to prevent hanging
          timeout: 300000 // 5 minutes
        },
        messagesForAPI,
        requestId
      );
      
      console.log('✅ Streaming completed successfully:', result);
      
      // Final cleanup and session update
      if (activeSessionId) {
        setMessages(currentMessages => {
          updateSessionWithMessages(activeSessionId, currentMessages);
          return currentMessages;
        });
      }
      
    } catch (error) {
      console.error('❌ Error in handleSendMessage:', error);
      
      // Safely extract error message - handle cases where error.message might be undefined
      const getErrorMessage = (err) => {
        if (!err) return 'Unknown error occurred';
        if (typeof err === 'string') return err;
        if (err.message) return err.message;
        if (err.toString && typeof err.toString === 'function') return err.toString();
        return 'Unknown error occurred';
      };
      
      const errorMessage = getErrorMessage(error);
      
      // Update the last message to show the error
      setMessages(currentMessages => {
        const updatedMessages = [...currentMessages];
        const lastMessage = updatedMessages[updatedMessages.length - 1];
        
        if (lastMessage && lastMessage.role === "assistant" && lastMessage.isGenerating) {
          lastMessage.isGenerating = false;
          lastMessage.processingState = "Error occurred";
          lastMessage.metadata.error = {
            message: errorMessage,
            timestamp: new Date().toISOString(),
            originalError: error
          };
          
          // Show user-friendly error message
          if (errorMessage.includes('timeout')) {
            lastMessage.content = "⏱️ The request timed out. This may be due to high server load or a complex query. Please try again with a simpler question or check your connection.";
          } else if (errorMessage.includes('network') || errorMessage.includes('Failed to fetch')) {
            lastMessage.content = "🌐 Network error. Please check your internet connection and try again.";
          } else {
            lastMessage.content = `❌ ${errorMessage}`;
          }
        }
        
        return updatedMessages;
      });
      
      // Show toast notification for better user feedback
      toast({
        title: 'Chat Error',
        description: errorMessage.includes('timeout') 
          ? 'The request timed out. Please try again.'
          : errorMessage.includes('network')
          ? 'Network error. Please check your connection.'
          : 'An error occurred while processing your message.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      // Always clean up loading state
      setIsLoading(false);
      setCurrentRequestId(null);
      
      console.log('🧹 handleSendMessage cleanup completed');
    }
  };

  const handleSelectSession = async (sessionId) => {
    setError(null);
    setIsLoading(true);
    
    // Clean up any accumulated content from previous conversations
    messageContentRef.current.clear();
    
    try {
      // Fetch messages for this session
      const sessionMessages = await chatService.getSessionMessages(sessionId);
      
      if (sessionMessages && Array.isArray(sessionMessages) && sessionMessages.length > 0) {
        // Transform messages to the correct format if needed
        const formattedMessages = sessionMessages.map(msg => ({
          role: msg.role || msg.message_role,
          content: msg.content || msg.message_text,
        }));
        
        setMessages(formattedMessages);
      } else {
        // If no messages, add the initial welcome message
        setMessages([initialMessage]);
      }
      
      // Update the active session ID
      setActiveSessionId(sessionId);
    } catch (err) {
      console.error('Failed to load chat messages:', err);
      let errorMessage = 'Failed to load messages for this conversation.';
      
      if (err.response) {
        const status = err.response.status;
        if (status === 401) {
          errorMessage = 'Authentication required to access messages.';
          // Redirect to login page if unauthorized
          navigate('/login');
          return;
        } else if (status === 404) {
          errorMessage = 'Chat session not found. It may have been deleted.';
        } else if (status === 500) {
          errorMessage = 'Server error while loading messages.';
        }
      } else if (err.request) {
        errorMessage = 'Network error. Unable to connect to the chat server.';
      }
      
      setError(errorMessage);
      
      // Set empty messages with welcome message
      setMessages([initialMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewSession = async (force = false) => {
    setError(null);
    setIsLoading(true);
    
    // Clean up any accumulated content from previous conversations
    messageContentRef.current.clear();
    
    // Check if there's already an empty conversation we can reuse
    if (!force) {
      const currentSession = sessions.find(s => s.id === activeSessionId);
      const hasNoUserMessages = messages.every(m => m.role !== 'user');
      
      // If we already have an active session with no user messages, just reset it
      if (currentSession && hasNoUserMessages) {
        setMessages([initialMessage]);
        setIsLoading(false);
        return activeSessionId;
      }
    }
    
    try {
      console.log('Creating new chat session via API...');
      // Create a new session
      const response = await chatService.createSession();
      const newSessionId = response.session_id;
      
      if (!newSessionId) {
        console.error('No session ID returned from createSession API call');
        throw new Error('Invalid session ID returned from server');
      }
      
      console.log('New session created successfully:', newSessionId);
      
      // Add the new session to our list
      const newSession = {
        id: newSessionId,
        title: "New Conversation",
        date: new Date().toLocaleString(),
        preview: "",
        message_count: 0
      };
      
      setSessions([newSession, ...sessions]);
      setActiveSessionId(newSessionId);
      setMessages([initialMessage]);
      
      return newSessionId;
    } catch (err) {
      console.error('Failed to create new session:', err);
      let errorMessage = 'Failed to create a new conversation.';
      let retryAttempted = false;
      let shouldDisplayError = false;
      
      if (err.response) {
        const status = err.response.status;
        if (status === 401) {
          errorMessage = 'Authentication required to create a new session.';
          shouldDisplayError = true;
          // Redirect to login page if unauthorized
          navigate('/login');
          return null;
        } else if (status === 429) {
          errorMessage = 'Too many requests. Please try again in a moment.';
          shouldDisplayError = true;
        } else if (status === 500) {
          errorMessage = 'Server error while creating a new session.';
          
          // Try once more if server error
          if (!retryAttempted) {
            retryAttempted = true;
            console.log('Retrying session creation after server error...');
            try {
              // Short delay before retry
              await new Promise(resolve => setTimeout(resolve, 500));
              const retryResponse = await chatService.createSession();
              const newSessionId = retryResponse.session_id;
              
              if (newSessionId) {
                // Success on retry
                const newSession = {
                  id: newSessionId,
                  title: "New Conversation",
                  date: new Date().toLocaleString(),
                  preview: "",
                  message_count: 0
                };
                
                setSessions([newSession, ...sessions]);
                setActiveSessionId(newSessionId);
                setMessages([initialMessage]);
                
                setIsLoading(false);
                return newSessionId;
              }
            } catch (retryErr) {
              console.error('Retry also failed:', retryErr);
            }
          }
        }
      } else if (err.request) {
        errorMessage = 'Network error. Unable to connect to the chat server.';
      }
      
      // Only set the error if we should display it
      if (shouldDisplayError) {
        setError(errorMessage);
        
        // Show toast notification for better user feedback
        toast({
          title: 'New Chat Session',
          description: errorMessage,
          status: 'warning',
          duration: 5000,
          isClosable: true,
        });
      } else {
        // Clear any existing error
        setError(null);
      }
      
      // Create a fallback session ID with a consistent format
      const fallbackSessionId = `fallback-${Date.now()}`;
      console.log('Creating fallback session with ID:', fallbackSessionId);
      
      const fallbackSession = {
        id: fallbackSessionId,
        title: "New Conversation",
        date: new Date().toLocaleString(),
        preview: "",
        is_fallback: true, // Mark this as a fallback session
      };
      
      setSessions([fallbackSession, ...sessions]);
      setActiveSessionId(fallbackSessionId);
      setMessages([initialMessage]);
      
      return fallbackSessionId;
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteSession = async (sessionId) => {
    setError(null);
    
    // For debugging purposes
    console.log('Attempting to delete session:', sessionId);
    
    try {
      // Find the session we want to delete
      const sessionToDelete = sessions.find(session => session.id === sessionId);
      
      if (!sessionToDelete) {
        console.error('Could not find session to delete:', sessionId);
        toast({
          title: 'Delete Error',
          description: 'Could not find the conversation to delete.',
          status: 'error',
          duration: 3000,
          isClosable: true,
        });
        return;
      }
      
      console.log('Found session to delete:', sessionToDelete);
      
      // Show loading state to indicate deletion in progress
      setIsLoading(true);
      
      // First, remove from local state to update UI immediately (for better responsiveness)
      const updatedSessions = sessions.filter(session => session.id !== sessionId);
      setSessions(updatedSessions);
      
      // If we deleted the active session, switch to another one immediately for better UX
      if (sessionId === activeSessionId) {
        if (updatedSessions.length > 0) {
          handleSelectSession(updatedSessions[0].id);
        } else {
          handleNewSession();
        }
      }
      
      // If the session was a fallback session, we don't need to delete from server
      const isFallbackSession = sessionToDelete.is_fallback === true || 
                                sessionId.startsWith('fallback-') || 
                                !sessionId.includes('-');
      
      if (isFallbackSession) {
        console.log('Skipping server delete for fallback session:', sessionId);
        
        // Short delay to ensure UI updates are visible
        await new Promise(resolve => setTimeout(resolve, 300));
        setIsLoading(false);
        
        toast({
          title: 'Conversation Deleted',
          description: 'The conversation has been removed.',
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
        
        return;
      }
      
      // For regular sessions, attempt to delete from server
      let serverDeletionSuccess = false;
      let errorDetails = null;
      
      try {
        // Add a small delay to ensure UI updates first
        await new Promise(resolve => setTimeout(resolve, 300));
        
        console.log('Deleting session from server:', sessionId);
        const result = await chatService.deleteSession(sessionId);
        console.log('Server deletion response:', result);
        
        if (result && result.messages_deleted !== undefined) {
          console.log(`Deleted ${result.messages_deleted} messages from session ${sessionId}`);
        }
        
        serverDeletionSuccess = true;
      } catch (serverError) {
        console.error('Server deletion failed:', serverError);
        errorDetails = serverError?.message || 'Unknown server error';
        
        // Try one more time with a slight delay in case it was a temporary issue
        try {
          console.log('Retrying session deletion after failure...');
          await new Promise(resolve => setTimeout(resolve, 500)); // 500ms delay
          const result = await chatService.deleteSession(sessionId);
          console.log('Server deletion retry response:', result);
          serverDeletionSuccess = true;
        } catch (retryError) {
          console.error('Server deletion retry also failed:', retryError);
          // We'll continue with UI updates even though server deletion failed
        }
      } finally {
        // Always turn off loading state
        setIsLoading(false);
      }
      
      // Show appropriate toast based on server deletion success
      if (serverDeletionSuccess) {
        toast({
          title: 'Conversation Deleted',
          description: 'The conversation has been deleted.',
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
      } else {
        toast({
          title: 'Partial Success',
          description: `The conversation was removed from your view, but the server reported an error: ${errorDetails}`,
          status: 'warning',
          duration: 5000,
          isClosable: true,
        });
      }
      
    } catch (error) {
      console.error('Error in handleDeleteSession:', error);
      setIsLoading(false);
      
      // Make sure we still update the UI even if there was an error
      const updatedSessions = sessions.filter(session => session.id !== sessionId);
      setSessions(updatedSessions);
      
      // If we deleted the active session, switch to another one
      if (sessionId === activeSessionId) {
        if (updatedSessions.length > 0) {
          handleSelectSession(updatedSessions[0].id);
        } else {
          handleNewSession();
        }
      }
      
      toast({
        title: 'Note',
        description: 'The conversation was removed from your history, but there might have been an issue with the server.',
        status: 'info',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const updateSessionWithMessages = (sessionId, updatedMessages) => {
    // Update the sessions list with new message information
    const updatedSessions = sessions.map(session => {
      if (session.id === sessionId) {
        // Get the last message for preview (from either user or assistant)
        const lastMessage = updatedMessages.length > 0 ? updatedMessages[updatedMessages.length - 1] : null;
        
        // Create preview text from the last message
        const preview = lastMessage 
          ? lastMessage.content.substring(0, 60) + (lastMessage.content.length > 60 ? '...' : '') 
          : '';
        
        return {
          ...session,
          preview,
          message_count: updatedMessages.length,
          date: new Date().toLocaleString()
        };
      }
      return session;
    });
    
    setSessions(updatedSessions);
  };

  const generateSessionTitle = (message) => {
    // Create a title from the user's message
    // Truncate long messages, capitalize first letter
    if (!message) return "New Conversation";
    
    // Clean up message - remove excess whitespace
    const cleanMessage = message.trim().replace(/\s+/g, ' ');
    
    // Truncate if too long
    const truncated = cleanMessage.length > 50 
      ? cleanMessage.substring(0, 47) + '...'
      : cleanMessage;
      
    // Capitalize first letter
    return truncated.charAt(0).toUpperCase() + truncated.slice(1);
  };
  
  const updateSessionTitle = (sessionId, title) => {
    // Update the title of a session
    setSessions(prevSessions => 
      prevSessions.map(session => 
        session.id === sessionId 
          ? { ...session, title }
          : session
      )
    );
    
    // If this is a real session (not a fallback), update on the server too
    const session = sessions.find(s => s.id === sessionId);
    if (session && !session.is_fallback) {
      try {
        // This could be implemented with a real API call if backend supports it
        console.log(`Updating session title on server: ${sessionId} => "${title}"`);
        // Async function to update title on server could go here
        // e.g., chatService.updateSessionTitle(sessionId, title);
      } catch (err) {
        console.error('Failed to update session title on server:', err);
        // Continue anyway as this is not critical
      }
    }
  };

  const handleSuggestedQuestion = (question) => {
    handleSendMessage(question);
  };

  return (
    <Box h="100vh" flexDirection="column">
      {/* Header - Simplified */}
      <Box 
        py={4} 
        px={6} 
        borderBottomWidth="1px" 
        borderColor={borderColor}
        bg={headerBg}
        zIndex="1"
      >
        <Flex justify="space-between" align="center">
          <Heading 
            size="lg" 
            fontWeight="600"
            color={accentColor}
          >
            RegulAIte Chat
          </Heading>
          
          {/* Mobile sidebar toggle */}
          {isMobile && (
            <IconButton
              icon={isSidebarOpen ? <X size={18} /> : <PanelLeft size={18} />}
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              variant="ghost"
              aria-label="Toggle Sidebar"
              size="sm"
            />
          )}
        </Flex>
      </Box>
      
      <Flex flex="1" overflow="hidden">
        {/* Chat history sidebar */}
        <Box
          w={isSidebarOpen ? { base: "full", md: "300px" } : "0px"}
          h="full"
          bg={sidebarBg}
          borderRightWidth="1px"
          borderRightColor={borderColor}
          position={{ base: isSidebarOpen ? "absolute" : "static", md: "static" }}
          zIndex="2"
          transition="width 0.3s"
          overflow="hidden"
          display={isSidebarOpen ? "block" : "none"}
          boxShadow={isMobile && isSidebarOpen ? "0 0 15px rgba(0,0,0,0.2)" : "none"}
        >
          <ChatHistory
            sessions={sessions}
            activeSessionId={activeSessionId}
            onSelectSession={handleSelectSession}
            onNewSession={handleNewSession}
            onDeleteSession={handleDeleteSession}
          />
        </Box>
        
        {/* Main chat area */}
        <Flex 
          flex="1" 
          flexDirection="column" 
          h="100%" 
          overflow="hidden"
          bg={chatBg}
        >
          {/* Simplified Loading Indicator */}
          {isLoading && (
            <Box 
              bg={loadingBg} 
              borderBottom="1px solid" 
              borderBottomColor={loadingBorderColor} 
              px={4} 
              py={2}
            >
              <HStack spacing={2}>
                <Spinner size="sm" color={accentColor} />
                <Text fontSize="sm" color={loadingTextColor}>
                  Processing...
                </Text>
              </HStack>
            </Box>
          )}
          
          {/* Error notification - Simplified */}
          {error && (
            <Box 
              bg={errorBg} 
              color={errorColor} 
              p={3} 
              borderBottomWidth="1px" 
              borderBottomColor={errorBorderColor}
            >
              <Flex align="center" justify="space-between">
                <Text fontSize="sm">{error}</Text>
                <Button 
                  size="xs" 
                  onClick={() => setError(null)}
                  variant="ghost"
                  color={errorColor}
                >
                  ×
                </Button>
              </Flex>
            </Box>
          )}
          
          {/* Message area */}
          <VStack 
            spacing={4} 
            flex="1" 
            overflowY="auto" 
            p={6} 
            align="stretch"
          >
            {/* Welcome header for new chats - Simplified */}
            {messages.length <= 1 && (
              <Box textAlign="center" my={8}>
                <Heading as="h2" size="lg" mb={4} color={textColor}>
                  How can I help you today?
                </Heading>
                <Text color={secondaryTextColor} mb={6}>
                  Ask me about governance, risk, and compliance
                </Text>
                
                {/* Suggested questions - Simplified */}
                <VStack spacing={2} maxW="md" mx="auto">
                  {suggestedQuestions.slice(0, 3).map((question, index) => (
                    <Button
                      key={index}
                      size="sm"
                      variant="outline"
                      width="full"
                      onClick={() => handleSuggestedQuestion(question)}
                      textAlign="left"
                      justifyContent="flex-start"
                    >
                      {question}
                    </Button>
                  ))}
                </VStack>
              </Box>
            )}
            
            {/* Messages */}
            {messages.map((message, index) => (
              <ChatMessage 
                key={index} 
                message={message} 
                isLoading={isLoading && index === messages.length - 1}
              />
            ))}
            
            {/* Invisible element to scroll to */}
            <Box ref={messagesEndRef} />
          </VStack>
          
          {/* Chat input area */}
          <ChatControls 
            onSendMessage={handleSendMessage}
            disabled={isLoading}
            reasoningNodeId={reasoningNodeId}
          />
        </Flex>
      </Flex>
      
      {/* Loading Overlay for critical operations */}
      <LoadingOverlay
        isVisible={isLoading && !currentRequestId} // Show for session operations, not chat messages
        message="Setting up your session..."
        subMessage="Please wait while we prepare your chat environment."
      />
    </Box>
  );
};

export default ChatPage; 