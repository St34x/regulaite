import React, { useState, useEffect } from 'react';
import { User, Bot, FileText, ChevronDown, ChevronUp } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Box, Flex, Text, Badge, Spinner, useColorModeValue, Icon, Button, Collapse, VStack } from '@chakra-ui/react';
import ProcessingStatus from './ProcessingStatus';

/**
 * Renders a single chat message with clean, professional styling
 */
const ChatMessage = ({ message, isLoading = false, agentInfo = null, previousMessage = null }) => {
  const [showSources, setShowSources] = useState(false);
  const [showAllSources, setShowAllSources] = useState(false);
  
  const isUser = message.role === 'user';
  
  // Check if message has sources
  const hasSources = !isUser && message.metadata && message.metadata.sources && message.metadata.sources.length > 0;
  const hasInternalThoughts = !isUser && message.metadata && message.metadata.internal_thoughts;
  const isProcessing = isLoading && !isUser && message.processingState;
  
  // Theme colors - simplified and cleaner
  const accentColor = '#4415b6';
  const userBg = useColorModeValue('blue.50', 'blue.900');
  const assistantBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const textColor = useColorModeValue('gray.800', 'gray.100');
  const mutedTextColor = useColorModeValue('gray.600', 'gray.400');
  const iconBg = useColorModeValue('gray.100', 'gray.700');
  const codeBg = useColorModeValue('gray.100', 'gray.700');
  const sourceBg = useColorModeValue('gray.50', 'gray.700');
  
  // Clean message content
  const cleanMessageContent = () => {
    if (isUser || !message.content) return message.content;
    
    let cleanedContent = message.content;
    
    // Remove any internal thoughts tags that might have leaked through
    cleanedContent = cleanedContent.replace(/<internal_thoughts>[\s\S]*?<\/internal_thoughts>/g, '');
    
    // Remove malformed internal thoughts patterns
    cleanedContent = cleanedContent.replace(/<internal[^>]*thoughts[^>]*>/g, '');
    cleanedContent = cleanedContent.replace(/<\/internal[^>]*thoughts[^>]*>/g, '');
    
    // Remove any orphaned internal_thoughts text
    cleanedContent = cleanedContent.replace(/internal_thoughts/g, '');
    
    // Clean up duplication patterns that can occur during streaming
    
    // Pattern 1: Immediate word duplication "word word" -> "word"
    cleanedContent = cleanedContent.replace(/(\b\w+)\s+\1\b/g, '$1');
    
    // Pattern 2: Character-level duplication within words "D'D'après" -> "D'après"  
    cleanedContent = cleanedContent.replace(/(\w+)('\w+)\1\2/g, '$1$2');
    
    // Pattern 3: Partial word duplication "aprèsaprès" -> "après"
    cleanedContent = cleanedContent.replace(/(\w{3,})\1/g, '$1');
    
    // Pattern 4: Complex pattern like "Les risLes risques" -> "Les risques"
    cleanedContent = cleanedContent.replace(/(\w{3,})\s+\1(\w+)/g, '$1$2');
    
    // Pattern 5: Syllable duplication like "sontques sont" -> "sont"
    cleanedContent = cleanedContent.replace(/(\w+)(\w{3,})\s+\1\s+\2/g, '$1 $2');
    
    // Pattern 6: Number duplication like "15 à15 à 25 25" -> "15 à 25"
    cleanedContent = cleanedContent.replace(/(\d+)\s+à\1\s+à\s+(\d+)\s+\2/g, '$1 à $2');
    
    // Pattern 7: Phrase duplication "dans la dans la" -> "dans la"
    cleanedContent = cleanedContent.replace(/(\w+\s+\w+)\s+\1/g, '$1');
    
    // Clean up excessive whitespace
    cleanedContent = cleanedContent.replace(/\s+/g, ' ').trim();
    
    return cleanedContent;
  };

  return (
    <Box 
      display="flex"
      width="full"
      alignItems="flex-start"
      gap={3}
      p={4}
      mb={4}
      bg={isUser ? userBg : assistantBg}
      borderWidth={1}
      borderColor={borderColor}
      borderRadius="lg"
      boxShadow="sm"
    >
      {/* Avatar */}
      <Flex
        h="32px"
        w="32px"
        alignItems="center"
        justifyContent="center"
        rounded="full"
        bg={iconBg}
        color={isUser ? 'blue.600' : accentColor}
        flexShrink={0}
      >
        <Icon as={isUser ? User : Bot} boxSize={4} />
      </Flex>
      
      <Box flex="1" minW={0}>
        {/* Header */}
        <Flex alignItems="center" gap={2} mb={2}>
          <Text fontSize="sm" fontWeight="medium" color={isUser ? 'blue.600' : accentColor}>
            {isUser ? 'You' : 'RegulAIte'}
          </Text>
          
          {!isUser && agentInfo && agentInfo.agent_used && (
            <Badge 
              bg={accentColor} 
              color="white" 
              size="sm"
              borderRadius="full"
            >
              AI Agent
            </Badge>
          )}
          
          {isLoading && (
            <Spinner size="xs" color={accentColor} />
          )}
        </Flex>
        
        {/* Message Content */}
        <Box 
          color={textColor}
          fontSize="sm"
          lineHeight="1.6"
          sx={{
            '& p': { mb: 2 },
            '& p:last-child': { mb: 0 },
            '& ul, & ol': { pl: 4, mb: 2 },
            '& li': { mb: 1 },
            '& h1, & h2, & h3': { fontWeight: 'semibold', mb: 2, mt: 3 },
            '& h1:first-child, & h2:first-child, & h3:first-child': { mt: 0 },
            '& code': { 
              bg: codeBg, 
              px: 1, 
              py: 0.5, 
              borderRadius: 'sm',
              fontSize: 'xs'
            },
            '& pre': { 
              bg: codeBg, 
              p: 3, 
              borderRadius: 'md',
              overflow: 'auto',
              fontSize: 'xs'
            }
          }}
        >
          <ReactMarkdown>{cleanMessageContent()}</ReactMarkdown>
        </Box>
        
        {/* Processing Status - Simplified */}
        {isProcessing && (
          <ProcessingStatus
            processingState={message.processingState}
            isProcessing={isProcessing}
            startTime={message.metadata?.startTime}
          />
        )}
        
        {/* Sources - Simplified */}
        {hasSources && (
          <Box mt={3} pt={3} borderTop="1px solid" borderColor={borderColor}>
            <Button
              size="sm"
              variant="ghost"
              leftIcon={<Icon as={FileText} />}
              rightIcon={<Icon as={showSources ? ChevronUp : ChevronDown} />}
              onClick={() => setShowSources(!showSources)}
              color={accentColor}
              fontSize="xs"
            >
              {message.metadata.sources.length} source{message.metadata.sources.length !== 1 ? 's' : ''}
            </Button>
            
            <Collapse in={showSources}>
              <VStack align="stretch" spacing={2} mt={2}>
                {(showAllSources ? message.metadata.sources : message.metadata.sources.slice(0, 3)).map((source, idx) => (
                  <Box 
                    key={idx}
                    p={2}
                    bg={sourceBg}
                    borderRadius="md"
                    fontSize="xs"
                  >
                    <Text fontWeight="medium" mb={1} color={textColor}>
                      {source.title || 'Document'}
                    </Text>
                    {source.content && (
                      <Text 
                        color={mutedTextColor} 
                        noOfLines={2}
                        fontSize="xs"
                      >
                        {source.content}
                      </Text>
                    )}
                  </Box>
                ))}
                {message.metadata.sources.length > 3 && !showAllSources && (
                  <Button
                    size="xs"
                    variant="ghost"
                    color={accentColor}
                    fontSize="xs"
                    onClick={() => setShowAllSources(true)}
                    _hover={{ bg: 'transparent', textDecoration: 'underline' }}
                  >
                    +{message.metadata.sources.length - 3} more sources
                  </Button>
                )}
                {showAllSources && message.metadata.sources.length > 3 && (
                  <Button
                    size="xs"
                    variant="ghost"
                    color={accentColor}
                    fontSize="xs"
                    onClick={() => setShowAllSources(false)}
                    _hover={{ bg: 'transparent', textDecoration: 'underline' }}
                  >
                    Show fewer sources
                  </Button>
                )}
              </VStack>
            </Collapse>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default ChatMessage; 