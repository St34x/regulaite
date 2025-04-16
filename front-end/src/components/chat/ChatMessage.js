import React from 'react';
import { User, Bot, Info, Cpu, ArrowDownRight } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Box, Flex, Text, Badge, Spinner, useColorModeValue, Icon, Tooltip } from '@chakra-ui/react';

/**
 * Renders a single chat message
 * @param {Object} props
 * @param {Object} props.message - Message object with role and content
 * @param {boolean} props.isLoading - Whether this message is still loading
 * @param {Object} props.agentInfo - Optional agent information
 * @param {Object} props.previousMessage - Previous message in the conversation (helps with context)
 */
const ChatMessage = ({ message, isLoading = false, agentInfo = null, previousMessage = null }) => {
  const isUser = message.role === 'user';
  const isShortUserMessage = isUser && message.content.trim().length <= 20;
  const showContextIndicator = isShortUserMessage && previousMessage && previousMessage.role === 'user';
  
  // Theme colors
  const accentColor = useColorModeValue("#4415b6", "#6c45e7");
  const accentColorLight = useColorModeValue("rgba(68, 21, 182, 0.1)", "rgba(108, 69, 231, 0.15)");
  const accentColorLighter = useColorModeValue("rgba(68, 21, 182, 0.05)", "rgba(108, 69, 231, 0.08)");
  const userBgColor = useColorModeValue('blue.50', 'blue.900');
  const assistantBgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const userIconBg = useColorModeValue('blue.100', 'blue.800');
  const botIconBg = useColorModeValue(accentColorLight, 'rgba(108, 69, 231, 0.3)');
  const textColor = useColorModeValue('gray.800', 'gray.100');
  const mutedTextColor = useColorModeValue('gray.600', 'gray.400');
  const dividerColor = useColorModeValue('gray.200', 'gray.600');
  const userMessageShadow = "0 2px 4px rgba(0, 0, 0, 0.05)";
  const assistantMessageShadow = "0 2px 6px rgba(68, 21, 182, 0.08)";
  const agentInfoBg = useColorModeValue('gray.50', 'gray.700');
  const contextIndicatorColor = useColorModeValue('gray.400', 'gray.500');

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
        boxShadow: isUser ? "0 3px 6px rgba(0, 0, 0, 0.08)" : "0 3px 8px rgba(108, 69, 231, 0.12)",
        borderColor: isUser ? 'blue.300' : accentColorLighter,
      }}
    >
      <Flex
        h="40px"
        w="40px"
        alignItems="center"
        justifyContent="center"
        rounded="full"
        bg={isUser ? userIconBg : botIconBg}
        color={isUser ? 'blue.600' : accentColor}
        boxShadow="0 2px 4px rgba(0, 0, 0, 0.1)"
        transition="all 0.2s ease"
        _hover={{
          transform: "scale(1.05)",
          boxShadow: "0 3px 5px rgba(0, 0, 0, 0.15)"
        }}
      >
        {isUser ? (
          <Icon as={User} boxSize={5} />
        ) : (
          <Icon as={Bot} boxSize={5} />
        )}
      </Flex>
      
      <Box flex="1" mt={0.5}>
        {/* Message Header */}
        <Flex alignItems="center" gap={2} mb={2} flexWrap="wrap">
          <Text fontSize="sm" fontWeight="semibold" color={isUser ? 'blue.600' : accentColor}>
            {isUser ? 'You' : 'RegulAIte Assistant'}
          </Text>
          
          {showContextIndicator && (
            <Tooltip 
              label={`In context of: "${previousMessage.content}"`} 
              placement="top" 
              hasArrow
            >
              <Flex 
                alignItems="center" 
                fontSize="xs" 
                color={contextIndicatorColor}
                cursor="help"
              >
                <Icon as={ArrowDownRight} boxSize={3} mr={1} />
                <Text>context</Text>
              </Flex>
            </Tooltip>
          )}
          
          {!isUser && agentInfo && agentInfo.agent_type && (
            <Badge 
              bg={accentColor} 
              color="white" 
              variant="solid" 
              fontSize="xs"
              borderRadius="full"
              px={2}
              boxShadow="0 1px 2px rgba(68, 21, 182, 0.3)"
            >
              {agentInfo.agent_type} Agent
            </Badge>
          )}
          
          {/* Model Badge */}
          {!isUser && (message.model || (agentInfo && agentInfo.model)) && (
            <Badge
              bg="gray.100"
              color="gray.700"
              variant="subtle"
              fontSize="xs"
              borderRadius="full"
              px={2}
              display="flex"
              alignItems="center"
              gap={1}
            >
              <Icon as={Cpu} boxSize={3} />
              <Text>{message.model || (agentInfo && agentInfo.model)}</Text>
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
          lineHeight="1.6"
          fontSize="sm"
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
            p={3}
            borderRadius="md"
            boxShadow="inset 0 1px 3px rgba(0, 0, 0, 0.05)"
          >
            {agentInfo.reasoning_path && (
              <Flex align="center" mb={1}>
                <Icon as={Info} boxSize={3} mr={1} color={accentColor} />
                <Text fontWeight="medium">Reasoning:</Text>
                <Text ml={1}>{agentInfo.reasoning_path.join(' > ')}</Text>
              </Flex>
            )}
            
            {agentInfo.source_documents && agentInfo.source_documents.length > 0 && (
              <Flex direction="column">
                <Text fontWeight="medium" mb={1} color={accentColor}>Sources:</Text>
                {agentInfo.source_documents.map((doc, idx) => (
                  <Text key={idx} ml={2} mb={0.5}>• {doc.title || doc.source}</Text>
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