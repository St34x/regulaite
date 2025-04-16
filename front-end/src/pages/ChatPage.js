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
  const [advancedSettings, setAdvancedSettings] = useState({
    agent: {
      use_agent: false,
      agent_type: null,
      use_tree_reasoning: false,
      tree_template: 'default'
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
        await handleNewSession();
      }
    } catch (err) {
      console.error('Failed to fetch chat sessions:', err);
      let errorMessage = 'Failed to load chat sessions. Creating a new session.';
      
      if (err.response) {
        const status = err.response.status;
        if (status === 401) {
          errorMessage = 'Authentication required to access chat sessions.';
          // Redirect to login page if unauthorized
          navigate('/login');
          return;
        } else if (status === 500) {
          errorMessage = 'Server error while loading sessions. Creating a new session.';
        }
      } else if (err.request) {
        errorMessage = 'Network error. Unable to connect to the chat server.';
      }
      
      setError(errorMessage);
      
      // Only create fallback sessions if not redirecting due to auth error
      const fallbackSessionId = Date.now().toString();
      setSessions([{
        id: fallbackSessionId,
        title: "New Conversation",
        date: "Just now",
        preview: "",
        messages: [initialMessage],
      }]);
      
      setActiveSessionId(fallbackSessionId);
      setMessages([initialMessage]);
    }
  };

  const handleSendMessage = async (content) => {
    // Don't proceed if already loading
    if (isLoading) return;
    
    try {
      setError(null);
      setIsLoading(true);
      setReasoningNodeId(null);
      setAgentProgress(null);

      // Create a session if none exists
      if (!activeSessionId) {
        const sessionId = await handleNewSession();
        setActiveSessionId(sessionId);
      }

      // Add user message immediately to UI
      const userMessage = { role: "user", content };
      const updatedMessages = [...messages, userMessage];
      setMessages(updatedMessages);

      // Prepare options with agent and LLM settings
      const options = {
        ...advancedSettings.llm,
        includeContext: true,
        agent: advancedSettings.agent.use_agent ? {
          agent_type: advancedSettings.agent.agent_type,
          use_tree_reasoning: advancedSettings.agent.use_tree_reasoning,
          tree_template: advancedSettings.agent.use_tree_reasoning ? advancedSettings.agent.tree_template : null
        } : null
      };

      // Stream the assistant's response
      let assistantContent = '';
      const assistantMessage = { role: "assistant", content: '' };
      
      // Add placeholder message that will be updated
      const messagesWithPlaceholder = [...updatedMessages, assistantMessage];
      setMessages(messagesWithPlaceholder);

      try {
        // Pass the full conversation history to maintain context
        const response = await chatService.sendMessageStreaming(
          activeSessionId,
          content,
          (chunk) => {
            assistantContent += chunk;
            // Create a new message object each time to ensure React detects the change
            const updatedAssistantMessage = { 
              role: "assistant", 
              content: assistantContent 
            };
            // Create a new array to ensure React state update
            setMessages([...updatedMessages, updatedAssistantMessage]);
          },
          options,
          updatedMessages  // Pass all messages to maintain conversation context
        );
        
        // Handle agent progress information if returned
        if (response && response.agent_execution_id) {
          setAgentProgress({
            execution_id: response.agent_execution_id,
            status: 'running'
          });
        }

        // Final update to make sure we have the complete message
        const finalAssistantMessage = { 
          role: "assistant", 
          content: assistantContent 
        };
        
        // Set with complete message to ensure we have the full response
        setMessages([...updatedMessages, finalAssistantMessage]);

        // Update session in state with the new messages
        updateSessionWithMessages(activeSessionId, [...updatedMessages, finalAssistantMessage]);
      } catch (streamError) {
        console.error('Streaming error:', streamError);
        
        // Fall back to non-streaming API if streaming fails
        try {
          // Pass the full conversation history to maintain context
          const response = await chatService.sendMessage(
            activeSessionId, 
            content, 
            options,
            updatedMessages  // Pass all messages to maintain conversation context
          );
          const fallbackMessage = { role: "assistant", content: response.message };
          setMessages([...updatedMessages, fallbackMessage]);
          updateSessionWithMessages(activeSessionId, [...updatedMessages, fallbackMessage]);
        } catch (fallbackError) {
          console.error('Fallback error:', fallbackError);
          setError('Failed to send message. Please try again.');
          // Remove the placeholder message
          setMessages(updatedMessages);
        }
      }
    } catch (err) {
      console.error('Chat error:', err);
      let errorMessage = 'An error occurred while sending your message. Please try again.';
      
      if (err.response) {
        const status = err.response.status;
        if (status === 401) {
          errorMessage = 'Your session has expired. Please log in again.';
          navigate('/login');
          return;
        } else if (status === 429) {
          errorMessage = 'Rate limit exceeded. Please wait a moment and try again.';
        } else if (err.response.data && err.response.data.detail) {
          errorMessage = err.response.data.detail;
        }
      } else if (err.request) {
        errorMessage = 'Network error. Please check your connection and try again.';
      }
      
      setError(errorMessage);
      toast({
        title: 'Chat Error',
        description: errorMessage,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
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
      // Create a new session
      const response = await chatService.createSession();
      const newSessionId = response.session_id;
      
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
      
      if (err.response) {
        const status = err.response.status;
        if (status === 401) {
          errorMessage = 'Authentication required to create a new session.';
          // Redirect to login page if unauthorized
          navigate('/login');
          return;
        } else if (status === 500) {
          errorMessage = 'Server error while creating a new session.';
        }
      } else if (err.request) {
        errorMessage = 'Network error. Unable to connect to the chat server.';
      }
      
      setError(errorMessage);
      
      // Create a fallback session ID
      const fallbackSessionId = Date.now().toString();
      setSessions([{
        id: fallbackSessionId,
        title: "New Conversation",
        date: "Just now",
        preview: "",
        messages: [initialMessage],
      }]);
      
      setActiveSessionId(fallbackSessionId);
      setMessages([initialMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteSession = async (sessionId) => {
    setError(null);
    
    try {
      // Delete the session
      await chatService.deleteSession(sessionId);
      
      // Remove from the sessions list
      const updatedSessions = sessions.filter(session => session.id !== sessionId);
      setSessions(updatedSessions);
      
      // If the active session was deleted, select another one or create a new one
      if (sessionId === activeSessionId) {
        if (updatedSessions.length > 0) {
          handleSelectSession(updatedSessions[0].id);
        } else {
          await handleNewSession();
        }
      }
    } catch (err) {
      console.error('Failed to delete session:', err);
      let errorMessage = 'Failed to delete the conversation.';
      
      if (err.response) {
        const status = err.response.status;
        if (status === 401) {
          errorMessage = 'Authentication required to delete this session.';
          // Redirect to login page if unauthorized
          navigate('/login');
          return;
        } else if (status === 403) {
          errorMessage = 'You do not have permission to delete this conversation.';
        } else if (status === 404) {
          errorMessage = 'Chat session not found. It may have already been deleted.';
          // Remove from the local list anyway
          setSessions(sessions.filter(session => session.id !== sessionId));
        }
      } else if (err.request) {
        errorMessage = 'Network error. Unable to connect to the chat server.';
      }
      
      setError(errorMessage);
      toast({
        title: 'Delete Error',
        description: errorMessage,
        status: 'error',
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
  const handleSettingsChange = (newSettings) => {
    console.log('Settings changed:', newSettings);
    // Make sure we have all the required properties to avoid issues
    const validatedSettings = {
      agent: {
        use_agent: newSettings.agent?.use_agent || false,
        agent_type: newSettings.agent?.agent_type || null,
        use_tree_reasoning: newSettings.agent?.use_tree_reasoning || false,
        tree_template: newSettings.agent?.tree_template || 'default'
      },
      llm: {
        model: newSettings.llm?.model || 'gpt-4',
        temperature: newSettings.llm?.temperature || 0.7,
        max_tokens: newSettings.llm?.max_tokens || 2048,
        top_p: newSettings.llm?.top_p || 1.0,
        frequency_penalty: newSettings.llm?.frequency_penalty || 0.0,
        presence_penalty: newSettings.llm?.presence_penalty || 0.0
      }
    };
    setAdvancedSettings(validatedSettings);
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
          <Flex p="3" borderBottomWidth="1px" borderBottomColor={borderColor} justify="space-between" align="center">
            <Heading size="md" display="flex" alignItems="center" color={accentColor}>
              <Shield size={20} style={{ marginRight: '8px' }} />
              RegulAIte
            </Heading>
            {isMobile && (
              <IconButton
                icon={<X size={20} />}
                variant="ghost"
                size="sm"
                aria-label="Close sidebar"
                onClick={() => setIsSidebarOpen(false)}
              />
            )}
          </Flex>

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
          {/* Header */}
          <Flex 
            bg={headerBg}
            borderBottomWidth="1px"
            borderBottomColor={borderColor}
            p="3"
            align="center"
            justify="space-between"
          >
            <Flex align="center">
              {!isSidebarOpen && (
                <IconButton
                  icon={<PanelLeft size={20} />}
                  variant="ghost"
                  size="sm"
                  mr="2"
                  aria-label="Open sidebar"
                  onClick={() => setIsSidebarOpen(true)}
                />
              )}
              <Heading size="md" color={textColor}>Chat</Heading>
            </Flex>
            <Flex align="center" gap="2">
              {isLoading && (
                <Flex align="center" fontSize="sm" color={secondaryTextColor}>
                  <Spinner size="sm" mr="2" color={accentColor} />
                  <Text>Processing...</Text>
                </Flex>
              )}
              
              <IconButton
                icon={<RefreshCw size={20} />}
                variant="ghost"
                size="sm"
                aria-label="New Chat"
                onClick={() => handleNewSession(true)}
              />
            </Flex>
          </Flex>

          {/* Chat Messages */}
          <Box flex="1" overflowY="auto" bg={chatBg} p="4">
            {error && (
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