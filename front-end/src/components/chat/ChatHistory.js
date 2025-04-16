import React, { useState } from 'react';
import { MessageSquare, Plus, Search, Trash2 } from 'lucide-react';
import { 
  Box, 
  Button, 
  Flex, 
  Heading, 
  Input, 
  InputGroup, 
  InputLeftElement,
  Text, 
  VStack, 
  HStack,
  IconButton,
  Divider,
  useColorModeValue 
} from '@chakra-ui/react';
import { SearchIcon, AddIcon, DeleteIcon, ChatIcon } from '@chakra-ui/icons';

/**
 * Chat history sidebar component
 * @param {Object} props
 * @param {Array} props.sessions - Array of chat sessions
 * @param {string} props.activeSessionId - ID of the currently active session
 * @param {Function} props.onSelectSession - Function to call when a session is selected
 * @param {Function} props.onNewSession - Function to call when a new session is created
 * @param {Function} props.onDeleteSession - Function to call when a session is deleted
 */
const ChatHistory = ({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  
  // Theme colors
  const accentColor = useColorModeValue('#4415b6', '#6c45e7');
  const accentLight = useColorModeValue('#4415b615', 'rgba(108, 69, 231, 0.15)');
  const accentLighter = useColorModeValue('#4415b608', 'rgba(108, 69, 231, 0.08)');
  const accentMedium = useColorModeValue('#4415b630', 'rgba(108, 69, 231, 0.3)');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const bgHover = useColorModeValue(accentLighter, 'rgba(108, 69, 231, 0.1)');
  const bgActive = useColorModeValue(accentLight, 'rgba(108, 69, 231, 0.15)');
  const textColor = useColorModeValue('gray.800', 'gray.200');
  const secondaryTextColor = useColorModeValue('gray.500', 'gray.400');
  const tertiaryTextColor = useColorModeValue('gray.400', 'gray.500');
  const inputBg = useColorModeValue('white', 'gray.700');
  const newButtonBg = useColorModeValue('white', 'gray.800');
  const hoverButtonBgLight = '#3a1296';
  const hoverButtonBgDark = '#7d5df5';
  const buttonHoverBg = useColorModeValue(hoverButtonBgLight, hoverButtonBgDark);

  const filteredSessions = sessions.filter((session) => 
    session.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <Flex direction="column" h="100%" w="100%" borderRightWidth="1px" borderColor={borderColor}>
      <Flex 
        align="center" 
        justify="space-between" 
        p={4} 
        boxShadow="0 1px 2px rgba(0, 0, 0, 0.05)"
        bg={useColorModeValue('white', 'gray.800')}
        position="relative"
        zIndex="1"
      >
        <Heading size="sm" color={accentColor}>Chat History</Heading>
        <IconButton
          icon={<AddIcon />}
          onClick={onNewSession}
          aria-label="New Chat"
          size="sm"
          variant="ghost"
          borderRadius="full"
          color={accentColor}
          _hover={{ bg: accentLight }}
        />
      </Flex>
      
      <Box px={4} py={3}>
        <InputGroup size="sm">
          <InputLeftElement pointerEvents="none">
            <SearchIcon color={secondaryTextColor} />
          </InputLeftElement>
          <Input
            placeholder="Search conversations"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            borderRadius="md"
            bg={inputBg}
            _focus={{
              borderColor: accentColor,
              boxShadow: `0 0 0 1px ${accentColor}`
            }}
          />
        </InputGroup>
      </Box>
      
      <Box flex="1" overflowY="auto" p={3}>
        <Button
          leftIcon={<AddIcon />}
          bg={newButtonBg}
          color={accentColor}
          justifyContent="flex-start"
          width="full"
          mb={3}
          onClick={onNewSession}
          size="sm"
          boxShadow="0 1px 3px rgba(0, 0, 0, 0.05)"
          _hover={{
            bg: accentLight,
            transform: 'translateY(-1px)',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)'
          }}
          transition="all 0.2s ease"
        >
          New Chat
        </Button>
        
        {filteredSessions.length > 0 ? (
          <VStack spacing={2} align="stretch">
            {filteredSessions.map((session) => (
              <Box
                key={session.id}
                p={3}
                borderRadius="md"
                cursor="pointer"
                bg={activeSessionId === session.id ? bgActive : 'transparent'}
                _hover={{ bg: bgHover }}
                onClick={() => onSelectSession(session.id)}
                position="relative"
                role="group"
                transition="all 0.2s ease"
                borderLeft={activeSessionId === session.id ? `2px solid ${accentColor}` : '2px solid transparent'}
                boxShadow={activeSessionId === session.id ? '0 1px 3px rgba(0, 0, 0, 0.08)' : 'none'}
              >
                <Flex justify="space-between" align="center">
                  <HStack spacing={2}>
                    <ChatIcon 
                      fontSize="xs" 
                      color={activeSessionId === session.id ? accentColor : secondaryTextColor} 
                    />
                    <Text 
                      fontWeight={activeSessionId === session.id ? "semibold" : "medium"} 
                      fontSize="sm" 
                      color={activeSessionId === session.id ? accentColor : textColor}
                      maxW="170px"
                      overflow="hidden"
                      textOverflow="ellipsis"
                      whiteSpace="nowrap"
                    >
                      {session.title}
                    </Text>
                  </HStack>
                  
                  <IconButton
                    icon={<DeleteIcon />}
                    size="xs"
                    variant="ghost"
                    aria-label="Delete"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteSession(session.id);
                    }}
                    opacity="0"
                    _groupHover={{ opacity: 1 }}
                    borderRadius="full"
                    color="red.500"
                    _hover={{ bg: 'red.50', color: 'red.600' }}
                    transition="all 0.2s ease"
                  />
                </Flex>
                
                <Text noOfLines={2} fontSize="xs" color={secondaryTextColor} mt={1}>
                  {session.preview}
                </Text>
                
                <Text 
                  fontSize="xs" 
                  color={tertiaryTextColor} 
                  mt={2}
                  maxW="100%"
                  overflow="hidden"
                  textOverflow="ellipsis"
                  whiteSpace="nowrap"
                >
                  {session.date}
                </Text>
              </Box>
            ))}
          </VStack>
        ) : (
          <Flex 
            direction="column" 
            align="center" 
            justify="center" 
            h="full" 
            p={6} 
            textAlign="center" 
            bg={accentLighter}
            borderRadius="md"
            mt={4}
          >
            <ChatIcon boxSize={8} color={accentColor} opacity={0.7} />
            <Text mt={3} fontSize="sm" fontWeight="medium" color={textColor}>
              No conversations found
            </Text>
            <Text mt={1} fontSize="xs" color={secondaryTextColor}>
              Start a new chat or try a different search.
            </Text>
            <Button
              mt={4}
              size="sm"
              leftIcon={<AddIcon />}
              onClick={onNewSession}
              color="white"
              bg={accentColor}
              _hover={{ bg: buttonHoverBg }}
              borderRadius="md"
            >
              New Chat
            </Button>
          </Flex>
        )}
      </Box>
    </Flex>
  );
};

export default ChatHistory; 