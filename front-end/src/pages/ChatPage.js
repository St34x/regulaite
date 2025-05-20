import React, { useState, useEffect, useRef } from 'react';
import { Shield, PanelLeft, X, RefreshCw } from 'lucide-react';
import ChatMessage from '../components/chat/ChatMessage';
import ChatHistory from '../components/chat/ChatHistory';
import useMediaQuery from '../hooks/useMediaQuery';
import chatService from '../services/chatService';
import configService from '../services/configService';
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
  const [advancedSettings, setAdvancedSettings] = useState({
    agent: {
      use_agent: false,
      agent_type: null,
      use_tree_reasoning: false,
      tree_template: 'default_understanding'
    },
    llm: {
      model: 'gpt-4',
      temperature: 0.7,
      max_tokens: 2048,
      top_p: 1.0,
      frequency_penalty: 0.0,
      presence_penalty: 0.0
    }
  });
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

  // Fetch initial LLM settings
  useEffect(() => {
    const fetchInitialSettings = async () => {
      try {
        const fetchedLlmSettings = await configService.getLlmConfig();
        if (fetchedLlmSettings) {
          setAdvancedSettings(prevSettings => ({
            ...prevSettings,
            llm: { ...prevSettings.llm, ...fetchedLlmSettings }
          }));
        }
      } catch (error) {
        console.error("Failed to fetch initial LLM settings:", error);
        toast({
          title: 'Error',
          description: 'Could not load model settings. Using defaults.',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    };
    fetchInitialSettings();
  }, [toast]);

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
    if (!content || content.trim() === "") return;

    try {
      setIsLoading(true);
      setError("");

      // Create a new user message
      const userMessage = {
        role: "user",
        content: content,
        timestamp: new Date().toISOString(),
      };

      // Get current messages
      const currentMessages = [...messages, userMessage];
      setMessages(currentMessages);

      // Ensure we're not in loading state before scrolling
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      }, 0);

      // Create options for the chat service
      const options = {
        // LLM parameters
        model: advancedSettings.llm.model,
        temperature: advancedSettings.llm.temperature,
        max_tokens: advancedSettings.llm.max_tokens,
        top_p: advancedSettings.llm.top_p,
        frequency_penalty: advancedSettings.llm.frequency_penalty,
        presence_penalty: advancedSettings.llm.presence_penalty,
        
        // Context settings
        includeContext: true,
        contextQuery: null,
      };

      // Add agent options if agent is enabled
      if (advancedSettings.agent.use_agent) {
        options.agent = {
          agent_type: advancedSettings.agent.agent_type,
          use_tree_reasoning: advancedSettings.agent.use_tree_reasoning,
          tree_template: advancedSettings.agent.tree_template,
        };
        
        options.agent_params = {
          show_reasoning: true
        };
      }

      // Format all conversation messages for context
      const contextMessages = currentMessages.map(m => ({
        role: m.role,
        content: m.content
      }));

      // Send the message to the chat service
      const response = await chatService.sendMessage(
        activeSessionId,
        content,
        options,
        contextMessages
      );
      
      // Create the assistant response object
      const assistantMessage = {
        role: "assistant",
        content: response.message || response.response || "",
        timestamp: new Date().toISOString(),
        model: response.model || advancedSettings.llm.model,
        agent_type: response.agent_type || (advancedSettings.agent.use_agent ? advancedSettings.agent.agent_type : null),
        agent_used: response.agent_used || advancedSettings.agent.use_agent,
        tree_reasoning_used: response.tree_reasoning_used || advancedSettings.agent.use_tree_reasoning,
        source_documents: response.source_documents || response.sources || [],
        execution_id: response.execution_id || response.agent_execution_id,
        traversal_path: response.traversal_path,
      };

      // Update messages state with the assistant's response
      setMessages([...currentMessages, assistantMessage]);

      // Update agent progress state if execution_id is provided
      if (response.execution_id || response.agent_execution_id) {
        setAgentProgress({
          execution_id: response.execution_id || response.agent_execution_id,
          status: response.agent_status || 'completed',
          progress_percent: 100,
        });
      }

      // Update the sessions list with the new message (for preview)
      updateSessionWithMessages(activeSessionId, [...currentMessages, assistantMessage]);

      // Scroll to the bottom of the messages
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      }, 0);
    } catch (error) {
      console.error("Error sending message:", error);
      setError(
        `An error occurred while sending your message: ${error.message || "Unknown error"}`
      );
      // Try to scroll to show the error
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      }, 0);
    } finally {
      setIsLoading(false);
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
      
      // First, remove from local state to update UI immediately
      const updatedSessions = sessions.filter(session => session.id !== sessionId);
      setSessions(updatedSessions);
      
      // If the session was a fallback session, we don't need to delete from server
      const isFallbackSession = sessionToDelete.is_fallback === true || 
                                sessionId.startsWith('fallback-') || 
                                !sessionId.includes('-');
      
      if (isFallbackSession) {
        console.log('Skipping server delete for fallback session:', sessionId);
        
        // If we deleted the active session, switch to another one
        if (sessionId === activeSessionId) {
          if (updatedSessions.length > 0) {
            handleSelectSession(updatedSessions[0].id);
          } else {
            handleNewSession();
          }
        }
        
        toast({
          title: 'Conversation Deleted',
          description: 'The conversation has been removed.',
          status: 'success',
          duration: 3000,
          isClosable: true,
        });
        
        return;
      }
      
      try {
        // For regular sessions, attempt to delete from server
        console.log('Deleting session from server:', sessionId);
        await chatService.deleteSession(sessionId);
        console.log('Session successfully deleted from server');
      } catch (serverError) {
        console.error('Server deletion failed but continuing:', serverError);
        // Continue with UI updates even if server deletion fails
        // The session is already removed from local state
      }
      
      // If we deleted the active session, switch to another one
      if (sessionId === activeSessionId) {
        if (updatedSessions.length > 0) {
          handleSelectSession(updatedSessions[0].id);
        } else {
          handleNewSession();
        }
      }
      
      toast({
        title: 'Conversation Deleted',
        description: 'The conversation has been deleted.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      
    } catch (error) {
      console.error('Error in handleDeleteSession:', error);
      
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

  const handleSuggestedQuestion = (question) => {
    handleSendMessage(question);
  };

  // New handler for settings changes
  const handleSettingsChange = async (newSettings) => {
    console.log('Settings changed:', newSettings);
    // Make sure we have all the required properties to avoid issues
    const validatedSettings = {
      agent: {
        use_agent: newSettings.agent?.use_agent || false,
        agent_type: newSettings.agent?.agent_type || null,
        use_tree_reasoning: newSettings.agent?.use_tree_reasoning || false,
        tree_template: newSettings.agent?.tree_template || 'default_understanding'
      },
      llm: {
        model: newSettings.llm?.model || 'gpt-4',
        temperature: newSettings.llm?.temperature ?? 0.7,
        max_tokens: newSettings.llm?.max_tokens || 2048,
        top_p: newSettings.llm?.top_p ?? 1.0,
        frequency_penalty: newSettings.llm?.frequency_penalty ?? 0.0,
        presence_penalty: newSettings.llm?.presence_penalty ?? 0.0
      }
    };
    // Optimistically update the UI for responsiveness, but we'll confirm with backend response
    setAdvancedSettings(validatedSettings); 

    try {
      const savedLlmSettings = await configService.updateLlmConfig(validatedSettings.llm);
      // Update the state with the authoritative response from the backend
      setAdvancedSettings(prevSettings => ({
        ...prevSettings, // Keep current agent settings
        llm: { ...prevSettings.llm, ...savedLlmSettings } // Overwrite llm settings with response
      }));

      toast({
        title: 'Settings Saved',
        description: 'Model parameters have been updated.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      console.error("Failed to save LLM settings:", error);
      toast({
        title: 'Error Saving Settings',
        description: error.message || 'Could not save model parameters. Please try again.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      // Optionally, if save fails, you might want to revert the optimistic update
      // by re-fetching the original settings or rolling back to a previous state.
      // For now, the optimistic update remains, and an error is shown.
    }
  };

  return (
    <Box h="100vh" display="flex" flexDir="column">
      {/* Main layout */}
      <Flex flex="1" overflow="hidden">
        {/* Sidebar */}
        <Box
          bg={sidebarBg}
          borderRightWidth="1px"
          borderRightColor={borderColor}
          display="flex"
          flexDir="column"
          w="64"
          transition="all 0.3s"
          transform={isSidebarOpen ? "translateX(0)" : "translateX(-100%)"}
          position={isMobile ? "absolute" : "relative"}
          zIndex={isMobile ? "10" : "auto"}
          h={isMobile ? "full" : "auto"}
        >
          <Box flex="1" overflowY="auto">
            <ChatHistory
              sessions={sessions}
              activeSessionId={activeSessionId}
              onSelectSession={handleSelectSession}
              onNewSession={handleNewSession}
              onDeleteSession={handleDeleteSession}
            />
          </Box>
        </Box>

        {/* Main content */}
        <Box flex="1" display="flex" flexDir="column" overflow="hidden">
          {/* Chat Messages */}
          <Box flex="1" overflowY="auto" bg={chatBg} p="4">
            {error && error.trim() !== '' && (
              <Box bg={errorBg} borderWidth="1px" borderColor={errorBorderColor} color={errorColor} px="4" py="2" borderRadius="md" mb="4">
                <Text>{error}</Text>
              </Box>
            )}

            {messages.length === 0 ? (
              <Box textAlign="center" color={secondaryTextColor} mt="8">
                <Text mb="4">No messages yet. Start a conversation!</Text>
                <Box display="grid" gridTemplateColumns={{base: "1fr", md: "1fr 1fr"}} gap="2" maxW="2xl" mx="auto">
                  {suggestedQuestions.map((question, index) => (
                    <Button
                      key={index}
                      bg={questionButtonBg}
                      p="3"
                      textAlign="left"
                      borderWidth="1px"
                      borderColor={questionButtonBorder}
                      borderRadius="lg"
                      _hover={{ borderColor: buttonHoverBorderColor, bg: buttonHoverBg }}
                      transition="colors 0.2s"
                      h="auto"
                      onClick={() => handleSuggestedQuestion(question)}
                    >
                      {question}
                    </Button>
                  ))}
                </Box>
              </Box>
            ) : (
              <VStack spacing="4" align="stretch" maxW="3xl" mx="auto">
                {messages.map((message, index) => (
                  <ChatMessage
                    key={index}
                    message={message}
                    isLoading={isLoading && index === messages.length - 1}
                    previousMessage={index > 0 ? messages[index - 1] : null}
                    agentInfo={message.role === 'assistant' && advancedSettings.agent.use_agent ? {
                      agent_type: advancedSettings.agent.agent_type,
                      reasoning_path: message.reasoning_path || null,
                      source_documents: message.source_documents || null,
                    } : null}
                  />
                ))}
                <Box ref={messagesEndRef} />
              </VStack>
            )}
          </Box>

          {/* Input area */}
          <ChatControls
            onSendMessage={handleSendMessage}
            disabled={isLoading}
            onSettingsChange={handleSettingsChange}
            initialSettings={advancedSettings}
            reasoningNodeId={reasoningNodeId}
          />
        </Box>
      </Flex>
    </Box>
  );
};

export default ChatPage; 