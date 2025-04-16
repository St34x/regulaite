import React from 'react';
import { User, Bot, Info } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Box, Flex, Text, Badge, Spinner, useColorModeValue, Icon } from '@chakra-ui/react';

/**
 * Renders a single chat message
 * @param {Object} props
 * @param {Object} props.message - Message object with role and content
 * @param {boolean} props.isLoading - Whether this message is still loading
 * @param {Object} props.agentInfo - Optional agent information
 */
const ChatMessage = ({ message, isLoading = false, agentInfo = null }) => {
  const isUser = message.role === 'user';
  
  // Theme colors
  const accentColor = "#4415b6";
  const userBgColor = useColorModeValue('blue.50', 'blue.900');
  const assistantBgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const userIconBg = useColorModeValue('blue.100', 'blue.800');
  const botIconBg = useColorModeValue(accentColor + '15', 'rgba(68, 21, 182, 0.3)');
  const textColor = useColorModeValue('gray.800', 'gray.100');
  const mutedTextColor = useColorModeValue('gray.600', 'gray.400');
  const dividerColor = useColorModeValue('gray.200', 'gray.600');
  const userMessageShadow = "0 2px 4px rgba(0, 0, 0, 0.05)";
  const assistantMessageShadow = "0 2px 6px rgba(68, 21, 182, 0.08)";
  const agentInfoBg = useColorModeValue('gray.50', 'gray.700');

  return (
    <Box 
      display="flex"
      width="full"
      alignItems="flex-start"
      gap={4}
      p={4}
      rounded="lg"
      mb={4}
      bg={isUser ? userBgColor : assistantBgColor}
      borderWidth={1}
      borderColor={isUser ? 'blue.200' : borderColor}
      boxShadow={isUser ? userMessageShadow : assistantMessageShadow}
      transition="all 0.2s ease"
      _hover={{
        boxShadow: isUser ? "0 3px 6px rgba(0, 0, 0, 0.08)" : "0 3px 8px rgba(68, 21, 182, 0.12)",
      }}
    >
      <Flex
        h="36px"
        w="36px"
        alignItems="center"
        justifyContent="center"
        rounded="full"
        bg={isUser ? userIconBg : botIconBg}
        color={isUser ? 'blue.600' : accentColor}
        boxShadow="0 2px 4px rgba(0, 0, 0, 0.1)"
      >
        {isUser ? (
          <Icon as={User} boxSize={4.5} />
        ) : (
          <Icon as={Bot} boxSize={4.5} />
        )}
      </Flex>
      
      <Box flex="1" mt={0.5}>
        {/* Message Header */}
        <Flex alignItems="center" gap={2} mb={2}>
          <Text fontSize="sm" fontWeight="medium" color={isUser ? 'blue.600' : accentColor}>
            {isUser ? 'You' : 'RegulAIte Assistant'}
          </Text>
          
          {!isUser && agentInfo && agentInfo.agent_type && (
            <Badge 
              bg={accentColor} 
              color="white" 
              variant="solid" 
              fontSize="xs"
              borderRadius="full"
              px={2}
            >
              {agentInfo.agent_type} Agent
            </Badge>
          )}
          
          {isLoading && (
            <Spinner size="xs" color={accentColor} ml={2} thickness="2px" speed="0.8s" />
          )}
        </Flex>
        
        {/* Message Content */}
        <Box 
          className="prose-sm prose max-w-none" 
          color={textColor}
          px={1}
        >
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </Box>
        
        {/* Agent Information */}
        {!isUser && agentInfo && (
          <Box 
            mt={3} 
            pt={2} 
            borderTop="1px solid" 
            borderColor={dividerColor} 
            fontSize="xs" 
            color={mutedTextColor}
            bg={agentInfoBg}
            p={2}
            borderRadius="md"
          >
            {agentInfo.reasoning_path && (
              <Flex align="center" mb={1}>
                <Icon as={Info} boxSize={3} mr={1} color={accentColor} />
                <Text>Reasoning: {agentInfo.reasoning_path.join(' > ')}</Text>
              </Flex>
            )}
            
            {agentInfo.source_documents && agentInfo.source_documents.length > 0 && (
              <Flex direction="column">
                <Text fontWeight="medium" mb={1}>Sources:</Text>
                {agentInfo.source_documents.map((doc, idx) => (
                  <Text key={idx} ml={2}>• {doc.title || doc.source}</Text>
                ))}
              </Flex>
            )}
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default ChatMessage; 