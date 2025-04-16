import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Box, HStack, Text, VStack, Flex, IconButton, useColorModeValue, Button, useDisclosure } from '@chakra-ui/react';
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
  // Use useRef to track previous settings to prevent unnecessary updates
  const prevSettingsRef = useRef(null);
  
  // Drawer controls
  const { isOpen, onOpen, onClose } = useDisclosure();
  
  // Theme colors
  const accentColor = '#4415b6'; 
  const accentLight = '#4415b615';
  const buttonBg = useColorModeValue('white', 'gray.700');
  const buttonHoverBg = useColorModeValue(accentLight, 'gray.600');
  const buttonColor = useColorModeValue(accentColor, 'white');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  
  const textColor = useColorModeValue('purple.700', 'purple.200');
  const bgColor = useColorModeValue('purple.50', 'purple.900');
  const bgBase = useColorModeValue('white', 'gray.800');
  
  // Initialize the settings ref when component mounts
  useEffect(() => {
    prevSettingsRef.current = JSON.stringify(initialSettings);
  }, []);
  
  // Function to handle settings change from AdvancedSettings
  const handleSettingsChange = useCallback((newSettings) => {
    try {
      // Only trigger the parent callback if settings actually changed
      const newSettingsStr = JSON.stringify(newSettings);
      
      if (newSettingsStr !== prevSettingsRef.current) {
        prevSettingsRef.current = newSettingsStr;
        onSettingsChange(newSettings);
      }
    } catch (err) {
      console.error("Error handling settings change:", err);
    }
  }, [onSettingsChange]);

  // Custom drawer close handler to ensure clean state
  const handleDrawerClose = useCallback(() => {
    onClose();
  }, [onClose]);

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
          {initialSettings.agent && 
            initialSettings.agent.use_agent && 
            initialSettings.agent.use_tree_reasoning && 
            reasoningNodeId && (
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
      
      {/* Advanced Settings Drawer */}
      {isOpen && (
        <AdvancedSettings 
          isOpen={isOpen} 
          onClose={handleDrawerClose}
          initialSettings={initialSettings}
          onSettingsChange={handleSettingsChange}
        />
      )}
    </Box>
  );
};

export default ChatControls; 