import React, { useState, useEffect, useRef } from 'react';
import { Shield, PanelLeft, X, RefreshCw } from 'lucide-react';
import ChatMessage from '../components/chat/ChatMessage';
import ChatHistory from '../components/chat/ChatHistory';
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
    if (!content.trim() || isLoading) return;
    
    setIsLoading(true);
    setError(null);
    
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
        processingState: "Starting to process your query...",
        metadata: {
          internal_thoughts: ""
        }
      };
      
      // Add both messages to the UI
      setMessages([...updatedMessages, tempAssistantMessage]);
      
      // Update the session with the user message
      if (activeSessionId) {
        updateSessionWithMessages(activeSessionId, updatedMessages);
      }
      
      // Prepare request payload - Always use RAG
      const payload = {
        messages: updatedMessages.map(msg => ({
          role: msg.role,
          content: msg.content
        })),
        use_rag: true,
        stream: true, // Enable streaming for internal thoughts
        session_id: activeSessionId
      };
      
      // Send request to API with streaming
      const response = await chatService.streamChatMessage(payload, {
        onToken: (token) => {
          // Update the assistant message with each new token
          setMessages(currentMessages => {
            const updatedMessages = [...currentMessages];
            const lastMessage = updatedMessages[updatedMessages.length - 1];
            
            if (lastMessage.role === "assistant" && lastMessage.isGenerating) {
              lastMessage.content += token;
            }
            
            return updatedMessages;
          });
        },
        onProcessing: (processingData) => {
          // Update internal thoughts and processing state
          setMessages(currentMessages => {
            const updatedMessages = [...currentMessages];
            const lastMessage = updatedMessages[updatedMessages.length - 1];
            
            if (lastMessage.role === "assistant" && lastMessage.isGenerating) {
              lastMessage.processingState = processingData.state || "Processing your query...";
              
              if (processingData.internal_thoughts) {
                lastMessage.metadata.internal_thoughts = processingData.internal_thoughts;
              }
            }
            
            return updatedMessages;
          });
        },
        onComplete: (completeData) => {
          // Finalize the message when streaming is complete
          setMessages(currentMessages => {
            const updatedMessages = [...currentMessages];
            const lastMessage = updatedMessages[updatedMessages.length - 1];
            
            if (lastMessage.role === "assistant" && lastMessage.isGenerating) {
              // Update with final data
              lastMessage.isGenerating = false;
              lastMessage.model = completeData.model;
              lastMessage.timestamp = new Date().toISOString();
              
              // Update metadata
              lastMessage.metadata = {
                sources: completeData.sources || [],
                context_quality: completeData.context_quality,
                hallucination_risk: completeData.hallucination_risk,
                context_used: completeData.context_used,
                internal_thoughts: completeData.internal_thoughts || lastMessage.metadata.internal_thoughts
              };
            }
            
            return updatedMessages;
          });
          
          // Update the session with new messages
          if (activeSessionId) {
            const finalMessages = messages.map(msg => {
              if (msg.isGenerating) {
                return {
                  ...msg,
                  isGenerating: false
                };
              }
              return msg;
            });
            updateSessionWithMessages(activeSessionId, finalMessages);
          }
          
          setIsLoading(false);
        }
      });
      
      // If we got here without streaming working, just use the regular method
      if (!response) {
        throw new Error('Streaming failed, falling back to regular method');
      }
    } catch (err) {
      console.error('Failed to send streaming message, falling back to regular method:', err);
      
      // Fall back to non-streaming method
      try {
        // Remove the temporary message if it exists
        setMessages(currentMessages => {
          const lastMessage = currentMessages[currentMessages.length - 1];
          if (lastMessage.role === "assistant" && lastMessage.isGenerating) {
            return currentMessages.slice(0, -1);
          }
          return currentMessages;
        });
        
        // Create updated messages without the temp message
        const cleanMessages = messages.filter(msg => !msg.isGenerating);
        
        // Prepare request payload without streaming
        const payload = {
          messages: cleanMessages.map(msg => ({
            role: msg.role,
            content: msg.content
          })),
          use_rag: true,
          stream: false,
          session_id: activeSessionId
        };
        
        // Send request to API
        const response = await chatService.sendChatMessage(payload);
        
        // Handle response
        if (response && response.message) {
          // Add assistant's response to messages
          const newMessages = [...cleanMessages, {
            role: "assistant",
            content: response.message,
            timestamp: new Date().toISOString(),
            model: response.model,
            metadata: {
              sources: response.sources || [],
              context_quality: response.context_quality,
              hallucination_risk: response.hallucination_risk,
              context_used: response.context_used,
              internal_thoughts: response.internal_thoughts
            }
          }];
          
          setMessages(newMessages);
          
          // Update the session
          if (activeSessionId) {
            updateSessionWithMessages(activeSessionId, newMessages);
          }
        } else {
          throw new Error('Unexpected response format');
        }
      } catch (fallbackErr) {
        console.error('Both streaming and fallback methods failed:', fallbackErr);
        setError(`Failed to send message: ${fallbackErr.message || 'Unknown error'}`);
        
        toast({
          title: 'Error',
          description: `Failed to send message: ${fallbackErr.message || 'Unknown error'}`,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleSelectSession = async (sessionId) => {
    if (sessionId === activeSessionId) return;
    
    setError(null);
    setIsLoading(true);
    
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
    <Flex h="100vh" flexDirection="column">
      {/* Header */}
      <Box 
        py={3} 
        px={6} 
        borderBottomWidth="1px" 
        borderColor={borderColor}
        bg={headerBg}
        zIndex="1"
      >
        <Flex justify="space-between" align="center">
          <Heading 
            size="md" 
            fontWeight="600"
            color={accentColor}
            display="flex"
            alignItems="center"
          >
            <Shield size={20} style={{ marginRight: '8px' }} />
            RegulaIte
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
          {/* Error notification */}
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
                  leftIcon={<RefreshCw size={12} />} 
                  onClick={() => setError(null)}
                  colorScheme="red"
                  variant="outline"
                >
                  Dismiss
                </Button>
              </Flex>
            </Box>
          )}
          
          {/* Message area */}
          <VStack 
            spacing={4} 
            flex="1" 
            overflowY="auto" 
            p={4} 
            align="stretch"
          >
            {/* Welcome header for new chats */}
            {messages.length <= 1 && (
              <Box textAlign="center" my={8}>
                <Heading as="h1" size="lg" mb={4} color={textColor}>
                  How can I help with GRC today?
                </Heading>
                <Text color={secondaryTextColor} mb={6}>
                  Ask me anything about governance, risk, and compliance
                </Text>
                
                {/* Suggested questions */}
                <HStack justify="center" spacing={2} wrap="wrap" mb={6}>
                  {suggestedQuestions.map((question, index) => (
                    <Button
                      key={index}
                      size="sm"
                      borderColor={questionButtonBorder}
                      bg={questionButtonBg}
                      variant="outline"
                      px={4}
                      onClick={() => handleSuggestedQuestion(question)}
                      _hover={{
                        bg: buttonHoverBg,
                        borderColor: buttonHoverBorderColor
                      }}
                      m={1}
                    >
                      {question}
                    </Button>
                  ))}
                </HStack>
              </Box>
            )}
            
            {/* Messages */}
            {messages.map((message, index) => (
              <ChatMessage 
                key={index} 
                message={message} 
                isLast={index === messages.length - 1}
              />
            ))}
            
            {/* Loading indicator */}
            {isLoading && (
              <Flex justify="center" pt={4} pb={2}>
                <Spinner 
                  thickness="3px"
                  speed="0.65s"
                  emptyColor="gray.200"
                  color={accentColor}
                  size="md"
                />
              </Flex>
            )}
            
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
    </Flex>
  );
};

export default ChatPage; 