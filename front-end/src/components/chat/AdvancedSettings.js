import React, { useState, useEffect } from 'react';
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
  // Default settings - never changes
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
  
  // Local state
  const [agentSettings, setAgentSettings] = useState({...defaultSettings.agent});
  const [llmSettings, setLlmSettings] = useState({...defaultSettings.llm});
  const [showTreeTab, setShowTreeTab] = useState(false);
  
  // Colors
  const accentColor = '#4415b6';
  const secondaryText = useColorModeValue('gray.600', 'gray.400');

  // Initialize settings when drawer opens
  useEffect(() => {
    if (isOpen) {
      // Initialize from props, with fallbacks
      setAgentSettings({
        ...defaultSettings.agent,
        ...(initialSettings.agent || {})
      });
      
      setLlmSettings({
        ...defaultSettings.llm,
        ...(initialSettings.llm || {})
      });
      
      // Check if tree tab should be shown
      setShowTreeTab(
        initialSettings.agent?.use_tree_reasoning && 
        initialSettings.agent?.tree_template
      );
    }
  }, [isOpen, initialSettings]);
  
  // Handle Apply button click
  const handleApply = () => {
    // Create a new combined settings object
    const newSettings = {
      agent: agentSettings,
      llm: llmSettings
    };
    
    // Send it to the parent component
    onSettingsChange(newSettings);
    
    // Close the drawer
    onClose();
  };

  // Handle Reset button click
  const handleReset = () => {
    // Reset to defaults
    setAgentSettings({...defaultSettings.agent});
    setLlmSettings({...defaultSettings.llm});
    
    // Apply the defaults
    onSettingsChange({...defaultSettings});
    
    // Close the drawer
    onClose();
  };
  
  // Agent selector only updates local state
  const handleAgentChange = (newAgentSettings) => {
    setAgentSettings(newAgentSettings);
    setShowTreeTab(newAgentSettings.use_tree_reasoning && newAgentSettings.tree_template);
  };
  
  // Model parameters only update local state
  const handleParamsChange = (newLlmParams) => {
    setLlmSettings(newLlmParams);
  };

  return (
    <Drawer 
      isOpen={isOpen} 
      placement="right" 
      onClose={onClose} 
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
              {showTreeTab && (
                <Tab>Decision Tree</Tab>
              )}
            </TabList>
            
            <TabPanels>
              {/* Agent Settings Tab */}
              <TabPanel p={3}>
                <AgentSelector 
                  onAgentChange={handleAgentChange} 
                  initialAgent={agentSettings} 
                />
              </TabPanel>
              
              {/* Model Parameters Tab */}
              <TabPanel p={3}>
                <ModelParamsSelector 
                  onParamsChange={handleParamsChange} 
                  initialParams={llmSettings} 
                />
              </TabPanel>
              
              {/* Decision Tree Tab */}
              {showTreeTab && (
                <TabPanel p={3}>
                  <DecisionTreeVisualizer 
                    treeId={agentSettings.tree_template} 
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