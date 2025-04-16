import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  VStack, 
  HStack, 
  Text, 
  Button, 
  Drawer,
  DrawerBody,
  DrawerFooter,
  DrawerHeader,
  DrawerOverlay,
  DrawerContent,
  DrawerCloseButton,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Badge,
  useColorModeValue
} from '@chakra-ui/react';

import AgentSelector from './AgentSelector';
import ModelParamsSelector from './ModelParamsSelector';
import DecisionTreeVisualizer from './DecisionTreeVisualizer';

/**
 * Advanced settings component that combines agent settings, LLM parameters,
 * and decision tree visualization into a drawer
 */
const AdvancedSettings = ({ isOpen, onClose, onSettingsChange, initialSettings = {} }) => {
  // Default settings
  const defaultSettings = {
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
  };
  
  // Use ref to prevent unnecessary updates
  const settingsRef = useRef(null);
  const previousOpenState = useRef(isOpen);
  
  // Local draft settings that will be applied on drawer close
  const [draftSettings, setDraftSettings] = useState({
    agent: { ...defaultSettings.agent },
    llm: { ...defaultSettings.llm }
  });
  
  // Initialize settings when the drawer opens
  useEffect(() => {
    // Only initialize settings when drawer opens to avoid unnecessary updates
    if (isOpen && !previousOpenState.current) {
      const mergedSettings = {
        agent: { ...defaultSettings.agent, ...(initialSettings.agent || {}) },
        llm: { ...defaultSettings.llm, ...(initialSettings.llm || {}) }
      };
      setDraftSettings(mergedSettings);
      settingsRef.current = JSON.stringify(mergedSettings);
    }
    
    previousOpenState.current = isOpen;
  }, [isOpen, initialSettings]);
  
  // Colors
  const accentColor = '#4415b6';
  const badgeBg = useColorModeValue('purple.100', 'purple.800');
  const badgeColor = useColorModeValue('purple.800', 'purple.100');
  const secondaryText = useColorModeValue('gray.600', 'gray.400');
  
  // Handle agent settings change (only updates draft)
  const handleAgentChange = (agentSettings) => {
    setDraftSettings(prev => ({
      ...prev,
      agent: agentSettings
    }));
  };
  
  // Handle LLM parameters change (only updates draft)
  const handleParamsChange = (llmParams) => {
    setDraftSettings(prev => ({
      ...prev,
      llm: llmParams
    }));
  };
  
  // Apply changes when clicking Apply button
  const handleApply = () => {
    try {
      const currentSettings = JSON.stringify(draftSettings);
      // Only update if settings have changed
      if (currentSettings !== settingsRef.current) {
        onSettingsChange(draftSettings);
        settingsRef.current = currentSettings;
      }
      onClose();
    } catch (err) {
      console.error("Error applying settings:", err);
      onClose();
    }
  };

  // Handle drawer close event - apply current draft settings
  const handleClose = () => {
    handleApply();
  };

  // Reset to defaults
  const handleReset = () => {
    try {
      const resetSettings = { ...defaultSettings };
      setDraftSettings(resetSettings);
      onSettingsChange(resetSettings);
      settingsRef.current = JSON.stringify(resetSettings);
      onClose();
    } catch (err) {
      console.error("Error resetting settings:", err);
      onClose();
    }
  };

  return (
    <Drawer 
      isOpen={isOpen} 
      placement="right" 
      onClose={handleClose} 
      size="md"
    >
      <DrawerOverlay />
      <DrawerContent>
        <DrawerCloseButton />
        <DrawerHeader borderBottomWidth="1px">
          <Text color={accentColor}>Advanced Settings</Text>
          <Text fontSize="sm" fontWeight="normal" color={secondaryText}>
            Configure AI agents, model parameters, and decision trees
          </Text>
        </DrawerHeader>

        <DrawerBody>
          <Tabs colorScheme="purple" variant="enclosed">
            <TabList>
              <Tab>AI Agent</Tab>
              <Tab>Model Parameters</Tab>
              {draftSettings.agent.use_tree_reasoning && draftSettings.agent.tree_template && (
                <Tab>Decision Tree</Tab>
              )}
            </TabList>
            
            <TabPanels>
              {/* Agent Settings Tab */}
              <TabPanel p={3}>
                <AgentSelector 
                  onAgentChange={handleAgentChange} 
                  initialAgent={draftSettings.agent} 
                />
              </TabPanel>
              
              {/* Model Parameters Tab */}
              <TabPanel p={3}>
                <ModelParamsSelector 
                  onParamsChange={handleParamsChange} 
                  initialParams={draftSettings.llm} 
                />
              </TabPanel>
              
              {/* Decision Tree Tab */}
              {draftSettings.agent.use_tree_reasoning && draftSettings.agent.tree_template && (
                <TabPanel p={3}>
                  <DecisionTreeVisualizer 
                    treeId={draftSettings.agent.tree_template} 
                  />
                </TabPanel>
              )}
            </TabPanels>
          </Tabs>
        </DrawerBody>

        <DrawerFooter borderTopWidth="1px">
          <HStack spacing={4}>
            <Button 
              variant="outline" 
              mr={3} 
              onClick={handleReset}
              size="sm"
            >
              Reset to Default
            </Button>
            <Button 
              colorScheme="purple" 
              onClick={handleApply}
              size="sm"
              bg={accentColor}
              _hover={{ bg: '#3a1296' }}
            >
              Apply
            </Button>
          </HStack>
        </DrawerFooter>
      </DrawerContent>
    </Drawer>
  );
};

export default AdvancedSettings; 