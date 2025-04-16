import React, { useState } from 'react';
import { Box, HStack, Text, VStack, Button, useColorModeValue, useDisclosure } from '@chakra-ui/react';
import { SettingsIcon } from '@chakra-ui/icons';
import ChatInput from './ChatInput';
import AdvancedSettings from './AdvancedSettings';

/**
 * Chat controls component that combines input and advanced settings
 */
const ChatControls = ({ 
  onSendMessage, 
  disabled = false, 
  onSettingsChange,
  initialSettings = {},
  reasoningNodeId = null
}) => {
  // Drawer controls using Chakra's useDisclosure
  const { isOpen, onOpen, onClose } = useDisclosure();
  
  // Theme colors
  const accentColor = '#4415b6'; 
  const accentLight = '#4415b615';
  const buttonBg = useColorModeValue('white', 'gray.700');
  const buttonHoverBg = useColorModeValue(accentLight, 'gray.600');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const textColor = useColorModeValue('purple.700', 'purple.200');
  const bgColor = useColorModeValue('purple.50', 'purple.900');
  const bgBase = useColorModeValue('white', 'gray.800');
  
  // Simple handler for settings changes - only passes up to parent
  const handleSettingsChange = (newSettings) => {
    if (onSettingsChange) {
      onSettingsChange(newSettings);
    }
  };
  
  // Shows the tree reasoning node info if it's active
  const showReasoningInfo = initialSettings.agent && 
    initialSettings.agent.use_agent && 
    initialSettings.agent.use_tree_reasoning && 
    reasoningNodeId;

  return (
    <Box position="relative" bg={bgBase}>
      {/* Settings button positioned absolutely in the top-right corner */}
      <Box position="absolute" top="0" right="0" p={2}>
        <Button
          leftIcon={<SettingsIcon />}
          size="sm"
          variant="outline"
          onClick={onOpen}
          bg={buttonBg}
          color={accentColor}
          _hover={{ bg: buttonHoverBg }}
          borderColor={accentColor}
        >
          Settings
        </Button>
      </Box>
      
      <Box borderTop="1px solid" borderTopColor={borderColor} bg={bgBase} p={3} pt={10}>
        <VStack spacing={3} maxW="100%" mx="auto">
          <ChatInput onSendMessage={onSendMessage} disabled={disabled} />
          
          {/* Tree reasoning info */}
          {showReasoningInfo && (
            <HStack 
              w="100%" 
              px={3} 
              py={2} 
              bg={bgColor} 
              borderRadius="md" 
              fontSize="xs" 
              color={textColor}
              borderWidth="1px"
              borderColor="purple.200"
            >
              <Text fontWeight="medium">Current Reasoning:</Text>
              <Text>{reasoningNodeId}</Text>
            </HStack>
          )}
        </VStack>
      </Box>
      
      {/* Only render the drawer when it's open to avoid any unnecessary updates */}
      {isOpen && (
        <AdvancedSettings 
          isOpen={isOpen} 
          onClose={onClose}
          initialSettings={initialSettings}
          onSettingsChange={handleSettingsChange}
        />
      )}
    </Box>
  );
};

export default ChatControls; 