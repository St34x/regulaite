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
  console.log('🚀 ChatPage component is mounting...');
  console.log('🌐 Current URL:', window.location.href);
  console.log('🌐 Current pathname:', window.location.pathname);
  
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState([initialMessage]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [error, setError] = useState(null);
  const [reasoningNodeId, setReasoningNodeId] = useState(null);
  const [agentProgress, setAgentProgress] = useState(null);
  const [currentRequestId, setCurrentRequestId] = useState(null);
  
  console.log('📊 ChatPage state initialized:', {
    sessionsCount: sessions.length,
    activeSessionId,
    messagesCount: messages.length
  });
  
  console.log('🔍 RENDER: ChatPage is rendering with sessions:', sessions);
  console.log('🔍 RENDER: Current sessions array:', JSON.stringify(sessions, null, 2));
  
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
  const mobileToggleBg = useColorModeValue('white', 'gray.800');
  const fixedInputBg = useColorModeValue('white', 'gray.800');

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
    console.log('🔐 Checking authentication...');
    const authResult = isAuthenticated();
    console.log('🔐 isAuthenticated() result:', authResult);
    
    if (!authResult) {
      console.log('❌ User not authenticated, redirecting to /login');
      navigate('/login');
    } else {
      console.log('✅ User is authenticated, staying on chat page');
    }
  }, [isAuthenticated, navigate]);

  // Fetch chat sessions on component mount
  useEffect(() => {
    console.log('🔄 useEffect for fetchSessions is running...');
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
    console.log('🔍 fetchSessions called');
    console.log('🔍 IMMEDIATE LOG - fetchSessions function started');
    setError(null);
    try {
      console.log('📡 Calling chatService.getChatSessions()...');
      const fetchedSessions = await chatService.getChatSessions();
      console.log('📦 Received sessions from API:', fetchedSessions);
      
      if (fetchedSessions && Array.isArray(fetchedSessions) && fetchedSessions.length > 0) {
        console.log('✅ Processing', fetchedSessions.length, 'sessions');
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
        
        console.log('🎯 Setting', uniqueSessions.length, 'unique sessions to state');
        setSessions(uniqueSessions);
        
        // Select the first session if we have any
        if (uniqueSessions.length > 0) {
          console.log('📌 Selecting first session:', uniqueSessions[0].id);
          handleSelectSession(uniqueSessions[0].id);
        } else {
          console.log('🆕 No sessions found, creating new session');
          await handleNewSession();
        }
      } else {
        // If no sessions, create a new one
        console.log('📭 No existing sessions found, creating a new session');
        await handleNewSession();
      }
    } catch (err) {
      console.error('❌ Failed to fetch chat sessions:', err);
      console.error('❌ Error details:', {
        status: err.response?.status,
        statusText: err.response?.statusText,
        data: err.response?.data,
        message: err.message
      });
      
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
        console.log('🔄 Creating fallback session due to error:', errorType);
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
                  lastMessage.metadata.generationDelay = chunkData.details;
                } else if (chunkData.step === 'reasoning_agent') {
                  // Reasoning agent progress
                  if (chunkData.reasoning_node_id) {
                    setReasoningNodeId(chunkData.reasoning_node_id);
                  }
                  
                  if (chunkData.details?.execution_id) {
                    // Update agent progress
                    setAgentProgress({
                      execution_id: chunkData.details.execution_id,
                      status: chunkData.details.status || 'running',
                      current_tree_node: chunkData.reasoning_node_id
                    });
                  }
                }
              } else if (chunkData.type === 'content') {
                // Get the accumulated content for this message
                const currentAccumulatedContent = messageContentRef.current.get(messageKey) || '';
                const newAccumulatedContent = currentAccumulatedContent + (chunkData.content || '');
                
                // Update the accumulated content
                messageContentRef.current.set(messageKey, newAccumulatedContent);
                
                // Update the message with the new accumulated content
                lastMessage.content = newAccumulatedContent;
                lastMessage.processingState = "Generating response...";
              } else if (chunkData.type === 'complete') {
                // Final completion
                lastMessage.isGenerating = false;
                lastMessage.processingState = null;
                
                // Final content update if provided
                if (chunkData.final_content) {
                  lastMessage.content = chunkData.final_content;
                  // Update the accumulator for consistency
                  messageContentRef.current.set(messageKey, chunkData.final_content);
                }
                
                // Clean up the accumulated content for this message
                messageContentRef.current.delete(messageKey);
              } else if (chunkData.type === 'error') {
                // Error during streaming
                lastMessage.isGenerating = false;
                lastMessage.content = lastMessage.content || "I apologize, but there was an error processing your request. Please try again.";
                lastMessage.error = chunkData.error;
                lastMessage.processingState = null;
                
                // Clean up
                messageContentRef.current.delete(messageKey);
              }
            }
            
            return updatedMessages;
          });
        }
      );
      
      console.log('✅ Streaming completed successfully');
      
    } catch (err) {
      console.error('❌ Error in handleSendMessage:', err);
      
      // Get detailed error message
      const getErrorMessage = (err) => {
        if (err.response) {
          const status = err.response.status;
          const detail = err.response.data?.detail;
          
          if (status === 401) {
            navigate('/login'); 
            return "Authentication required. Please log in again.";
          } else if (status === 403) {
            return "You don't have permission to send messages.";
          } else if (status === 429) {
            return "Rate limit exceeded. Please wait a moment before sending another message.";
          } else if (status === 500) {
            return "Server error occurred. Please try again in a few moments.";
          } else if (detail) {
            return detail;
          }
        } else if (err.request) {
          return "Network error. Please check your connection and try again.";
        }
        return "An unexpected error occurred. Please try again.";
      };

      const errorMessage = getErrorMessage(err);
      setError(errorMessage);

      // Remove the temporary processing message and replace with error
      setMessages(currentMessages => {
        const updatedMessages = [...currentMessages];
        const lastMessage = updatedMessages[updatedMessages.length - 1];
        
        if (lastMessage && lastMessage.role === "assistant" && lastMessage.isGenerating) {
          lastMessage.isGenerating = false;
          lastMessage.content = "I apologize, but I encountered an error while processing your request. Please try again.";
          lastMessage.error = errorMessage;
          lastMessage.processingState = null;
        }
        
        return updatedMessages;
      });

      // Show error toast
      toast({
        title: "Message Error",
        description: errorMessage,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
      setCurrentRequestId(null);
    }
  };

  const handleSelectSession = async (sessionId) => {
    console.log('🎯 handleSelectSession called with sessionId:', sessionId);
    
    if (!sessionId || sessionId === activeSessionId) {
      console.log('❌ Session selection rejected - no ID or already active');
      return;
    }
    
    setError(null);
    
    try {
      // Don't show loading overlay for session switches - it's distracting
      // setIsLoading(true);
      
      const session = sessions.find(s => s.id === sessionId);
      if (session && session.is_fallback && session.messages) {
        // This is a fallback session, use the stored messages
        console.log('📝 Loading fallback session messages');
        setActiveSessionId(sessionId);
        setMessages(session.messages);
        return;
      }
      
      // Fetch the conversation history for this session
      const conversationHistory = await chatService.getSessionMessages(sessionId);
      
      if (conversationHistory && conversationHistory.length > 0) {
        console.log('✅ Successfully loaded conversation history:', conversationHistory.length, 'messages');
        setActiveSessionId(sessionId);
        setMessages(conversationHistory);
      } else {
        console.log('📝 No conversation history found, using initial message');
        setActiveSessionId(sessionId);
        setMessages([initialMessage]);
      }
      
    } catch (err) {
      console.error('❌ Failed to load session:', err);
      let errorMessage = 'Failed to load conversation history';
      
      if (err.response?.status === 404) {
        errorMessage = 'Conversation not found. It may have been deleted.';
        // Remove this session from the list
        setSessions(prevSessions => prevSessions.filter(s => s.id !== sessionId));
      } else if (err.response?.status === 401) {
        errorMessage = 'Authentication required';
        navigate('/login');
        return;
      }
      
      setError(errorMessage);
      toast({
        title: "Session Error", 
        description: errorMessage,
        status: "warning",
        duration: 3000,
        isClosable: true,
      });
      
      // Fallback to initial message
      setActiveSessionId(sessionId);
      setMessages([initialMessage]);
    } finally {
      // setIsLoading(false);
    }
  };

  const handleNewSession = async (force = false) => {
    console.log('🎯 handleNewSession called, force:', force);
    
    // Don't create a new session if we already have an empty one (unless forced)
    if (!force && activeSessionId && messages.length <= 1) {
      console.log('❌ New session rejected - already have empty session');
      return;
    }
    
    try {
      setError(null);
      setIsLoading(true);
      
      console.log('🔄 Creating new session...');
      
      // Create session using the service
      const newSession = await chatService.createSession();
      
      console.log('✅ New session created:', newSession);
      
      const sessionId = newSession.session_id || newSession.id;
      
      if (!sessionId) {
        throw new Error('Failed to get session ID from response');
      }
      
      // Create a new session object for the UI
      const newSessionObj = {
        id: sessionId,
        title: "New Conversation",
        date: "Just now", 
        preview: "",
        message_count: 0
      };
      
      // Add to sessions list
      setSessions(prevSessions => [newSessionObj, ...prevSessions]);
      
      // Set as active session
      setActiveSessionId(sessionId);
      setMessages([initialMessage]);
      
      console.log('✅ New session setup complete');
      
    } catch (err) {
      console.error('❌ Failed to create new session:', err);
      
      let errorMessage = 'Failed to create new conversation';
      let shouldCreateFallback = true;
      
      if (err.response) {
        const status = err.response.status;
        if (status === 401) {
          errorMessage = 'Authentication required to create a new conversation';
          navigate('/login');
          return;
        } else if (status === 403) {
          errorMessage = 'You do not have permission to create conversations';
          shouldCreateFallback = false;
        } else if (status === 429) {
          errorMessage = 'Rate limit exceeded. Please wait before creating a new conversation';
          shouldCreateFallback = false;
        }
      } else if (err.request) {
        errorMessage = 'Network error. Unable to create new conversation';
      }
      
      setError(errorMessage);
      toast({
        title: "New Conversation Error",
        description: errorMessage,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
      
      // Create a fallback local session if appropriate
      if (shouldCreateFallback) {
        console.log('🔄 Creating fallback session due to error');
        const fallbackSessionId = `fallback_${Date.now()}`;
        const fallbackSession = {
          id: fallbackSessionId,
          title: "New Conversation",
          date: "Just now",
          preview: "",
          messages: [initialMessage],
          is_fallback: true
        };
        
        setSessions(prevSessions => [fallbackSession, ...prevSessions]);
        setActiveSessionId(fallbackSessionId);
        setMessages([initialMessage]);
        
        console.log('✅ Fallback session created');
      }
      
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteSession = async (sessionId) => {
    if (!sessionId) return;
    
    try {
      // Don't delete if it's the only session or the active session
      if (sessions.length <= 1) {
        toast({
          title: "Cannot Delete",
          description: "Cannot delete the last remaining conversation.",
          status: "warning",
          duration: 3000,
          isClosable: true,
        });
        return;
      }
      
      // Check if this is a fallback session
      const session = sessions.find(s => s.id === sessionId);
      if (session && !session.is_fallback) {
        // Delete from server for real sessions
        await chatService.deleteSession(sessionId);
      }
      
      // Remove from local state
      setSessions(prevSessions => prevSessions.filter(s => s.id !== sessionId));
      
      // If we deleted the active session, switch to another one
      if (activeSessionId === sessionId) {
        const remainingSessions = sessions.filter(s => s.id !== sessionId);
        if (remainingSessions.length > 0) {
          await handleSelectSession(remainingSessions[0].id);
        } else {
          // Create a new session if no sessions remain
          await handleNewSession();
        }
      }
      
      toast({
        title: "Conversation Deleted",
        description: "The conversation has been successfully deleted.",
        status: "success",
        duration: 2000,
        isClosable: true,
      });
      
    } catch (err) {
      console.error('Failed to delete session:', err);
      
      let errorMessage = 'Failed to delete conversation';
      if (err.response?.status === 404) {
        errorMessage = 'Conversation not found. It may have already been deleted.';
        // Remove from local state anyway
        setSessions(prevSessions => prevSessions.filter(s => s.id !== sessionId));
      } else if (err.response?.status === 401) {
        errorMessage = 'Authentication required';
        navigate('/login');
        return;
      } else if (err.response?.status === 403) {
        errorMessage = 'You do not have permission to delete this conversation';
      }
      
      toast({
        title: "Delete Error",
        description: errorMessage,
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    }
  };

  const updateSessionWithMessages = (sessionId, updatedMessages) => {
    setSessions(prevSessions =>
      prevSessions.map(session => {
        if (session.id === sessionId) {
          const lastUserMessage = [...updatedMessages].reverse().find(msg => msg.role === 'user');
          return {
            ...session,
            preview: lastUserMessage ? lastUserMessage.content.substring(0, 100) : session.preview,
            message_count: updatedMessages.length,
            date: "Just now"
          };
        }
        return session;
      })
    );
  };

  const generateSessionTitle = (message) => {
    // Extract first 50 chars and clean up
    return message.substring(0, 50).replace(/[^\w\s]/g, '').trim() + (message.length > 50 ? '...' : '');
  };

  const updateSessionTitle = (sessionId, title) => {
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
    <Box h="100vh" display="flex" overflow="hidden">
      {/* Chat history sidebar */}
      <Box
        w={isSidebarOpen ? { base: "full", md: "300px" } : "0px"}
        h="100vh"
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
        {console.log('🎨 Rendering ChatHistory with sessions:', sessions, 'activeSessionId:', activeSessionId)}
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
        h="100vh" 
        overflow="hidden"
        bg={chatBg}
        position="relative"
      >
        {/* Mobile sidebar toggle - positioned absolutely in top left */}
        {isMobile && (
          <IconButton
            icon={isSidebarOpen ? <X size={18} /> : <PanelLeft size={18} />}
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            variant="ghost"
            aria-label="Toggle Sidebar"
            size="sm"
            position="absolute"
            top="4"
            left="4"
            zIndex="3"
            bg={mobileToggleBg}
            boxShadow="sm"
          />
        )}

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
        
        {/* Chat messages area - scrollable */}
        <VStack 
          spacing={4} 
          flex="1" 
          overflowY="auto" 
          p={6} 
          align="stretch"
          pb="120px" // Add padding bottom to account for fixed input
        >
          {/* Welcome header for new chats */}
          {messages.length <= 1 && (
            <Box textAlign="center" my={8}>
              <Heading as="h2" size="xl" mb={4} color={textColor}>
                How can I help with GRC today?
              </Heading>
              <Text color={secondaryTextColor} mb={6} fontSize="lg">
                Ask me anything about governance, risk, and compliance
              </Text>
              
              {/* Suggested questions */}
              <VStack spacing={3} maxW="2xl" mx="auto">
                {suggestedQuestions.slice(0, 4).map((question, index) => (
                  <Button
                    key={index}
                    size="md"
                    variant="outline"
                    width="full"
                    onClick={() => handleSuggestedQuestion(question)}
                    textAlign="left"
                    justifyContent="flex-start"
                    py={6}
                    px={6}
                    fontSize="sm"
                    color={accentColor}
                    borderColor={accentColor}
                    bg={questionButtonBg}
                    _hover={{
                      bg: buttonHoverBg,
                      borderColor: buttonHoverBorderColor,
                      transform: 'translateY(-1px)',
                      boxShadow: '0 4px 12px rgba(68, 21, 182, 0.15)'
                    }}
                    transition="all 0.2s ease"
                    borderRadius="lg"
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
        
        {/* Fixed chat input area at bottom */}
        <Box
          position="fixed"
          bottom="0"
          left={isSidebarOpen && !isMobile ? { base: "0", md: "300px" } : "0"}
          right="0"
          bg={fixedInputBg}
          borderTopWidth="1px"
          borderTopColor={borderColor}
          zIndex="1"
          transition="left 0.3s"
        >
          <ChatControls 
            onSendMessage={handleSendMessage}
            disabled={isLoading}
            reasoningNodeId={reasoningNodeId}
          />
        </Box>
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