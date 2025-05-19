import React, { useState, useEffect } from 'react';
import { Box, FormControl, FormLabel, Select, Switch, Tooltip, Badge, Flex, Text, Spinner, useColorModeValue } from '@chakra-ui/react';
import { InfoIcon } from '@chakra-ui/icons';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Default agent settings - pure constant
const DEFAULT_AGENT_SETTINGS = { 
  use_agent: false, 
  agent_type: null, 
  use_tree_reasoning: false,
  tree_template: 'default_understanding'
};

/**
 * Component for selecting an AI agent and its parameters for chat
 * Only triggers onAgentChange on explicit user interactions
 */
const AgentSelector = ({ onAgentChange, initialAgent = null }) => {
  // API data state
  const [agentTypes, setAgentTypes] = useState({});
  const [agentMetadata, setAgentMetadata] = useState([]);
  const [availableTrees, setAvailableTrees] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Form state - completely local
  const [settings, setSettings] = useState({
    ...DEFAULT_AGENT_SETTINGS,
    ...(typeof initialAgent === 'object' ? initialAgent : {})
  });
  
  // Theme colors
  const accentColor = '#4415b6';
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const bgColor = useColorModeValue('white', 'gray.800');
  const textColor = useColorModeValue('gray.800', 'gray.200');
  const mutedColor = useColorModeValue('gray.600', 'gray.400');
  const hoverBgColor = useColorModeValue('#4415b610', 'gray.700');

  // Only update local state from props when initialAgent changes
  useEffect(() => {
    if (initialAgent && typeof initialAgent === 'object') {
      setSettings(prevSettings => ({
        ...prevSettings,
        ...initialAgent
      }));
    }
  }, [initialAgent]);

  // Fetch agent types and metadata on component mount
  useEffect(() => {
    const fetchAgentData = async () => {
      setIsLoading(true);
      try {
        // Fetch agent types
        const typesResponse = await axios.get(`${API_URL}/agents/types`);
        setAgentTypes(typesResponse.data);
        
        // Fetch agent metadata
        const metadataResponse = await axios.get(`${API_URL}/agents/metadata`);
        setAgentMetadata(metadataResponse.data);
        
        // Fetch decision trees
        const treesResponse = await axios.get(`${API_URL}/agents/trees`);
        setAvailableTrees(treesResponse.data);
        
        setError(null);
      } catch (err) {
        console.error('Error fetching agent data:', err);
        setError('Failed to load agent data. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchAgentData();
  }, []);
  
  // Event handlers - update local state and notify parent with callback
  const handleAgentToggle = () => {
    const newSettings = {
      ...settings,
      use_agent: !settings.use_agent
    };
    
    setSettings(newSettings);
    onAgentChange(newSettings);
  };

  const handleAgentTypeChange = (event) => {
    const newSettings = {
      ...settings,
      agent_type: event.target.value
    };
    
    setSettings(newSettings);
    onAgentChange(newSettings);
  };

  const handleTreeToggle = () => {
    const newSettings = {
      ...settings,
      use_tree_reasoning: !settings.use_tree_reasoning
    };
    
    setSettings(newSettings);
    onAgentChange(newSettings);
  };

  const handleTreeChange = (event) => {
    const newSettings = {
      ...settings,
      tree_template: event.target.value
    };
    
    setSettings(newSettings);
    onAgentChange(newSettings);
  };

  // Find the current agent metadata
  const currentAgentMeta = agentMetadata.find(a => a.id === settings.agent_type) || null;

  if (isLoading) {
    return (
      <Box p={2} textAlign="center">
        <Spinner size="sm" color={accentColor} mr={2} />
        <Text display="inline" fontSize="sm">Loading agents...</Text>
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={2} color="red.500" fontSize="sm">
        {error}
      </Box>
    );
  }

  return (
    <Box 
      borderWidth="1px" 
      borderRadius="md" 
      p={3} 
      bg={bgColor} 
      shadow="sm" 
      borderColor={borderColor}
      transition="all 0.2s ease"
      _hover={{ boxShadow: "0 2px 6px rgba(0,0,0,0.05)" }}
    >
      <Flex justifyContent="space-between" mb={2} alignItems="center">
        <FormControl display="flex" alignItems="center">
          <FormLabel htmlFor="agent-toggle" mb={0} fontSize="sm" fontWeight="medium" color={textColor}>
            Use AI Agent
          </FormLabel>
          <Switch 
            id="agent-toggle" 
            isChecked={settings.use_agent} 
            onChange={handleAgentToggle} 
            colorScheme="purple"
          />
        </FormControl>
        
        {currentAgentMeta && (
          <Tooltip 
            label={currentAgentMeta.description} 
            fontSize="sm" 
            placement="top"
          >
            <Badge 
              bg={accentColor} 
              color="white" 
              variant="solid" 
              p={1} 
              borderRadius="md"
            >
              {currentAgentMeta.name} <InfoIcon ml={1} boxSize={3} />
            </Badge>
          </Tooltip>
        )}
      </Flex>

      {settings.use_agent && (
        <>
          <FormControl mb={3}>
            <FormLabel htmlFor="agent-select" fontSize="sm" mb={1} color={textColor}>
              Agent Type
            </FormLabel>
            <Select 
              id="agent-select" 
              value={settings.agent_type || ''} 
              onChange={handleAgentTypeChange}
              placeholder="Select an agent"
              size="sm"
              focusBorderColor={accentColor}
            >
              {Object.entries(agentTypes).map(([id, description]) => (
                <option key={id} value={id}>
                  {id.charAt(0).toUpperCase() + id.slice(1)} Agent
                </option>
              ))}
            </Select>
          </FormControl>

          <FormControl display="flex" alignItems="center" mb={2}>
            <FormLabel htmlFor="tree-toggle" mb={0} fontSize="sm" fontWeight="medium" color={textColor}>
              Use Tree Reasoning
            </FormLabel>
            <Switch 
              id="tree-toggle" 
              isChecked={settings.use_tree_reasoning} 
              onChange={handleTreeToggle} 
              colorScheme="purple"
            />
          </FormControl>

          {settings.use_tree_reasoning && (
            <FormControl mb={3}>
              <FormLabel htmlFor="tree-select" fontSize="sm" mb={1} color={textColor}>
                Decision Tree Template
              </FormLabel>
              <Select 
                id="tree-select" 
                value={settings.tree_template || 'default_understanding'} 
                onChange={handleTreeChange}
                size="sm"
                focusBorderColor={accentColor}
              >
                <option value="default_understanding">Default Tree</option>
                {Object.entries(availableTrees).map(([id, tree]) => (
                  <option key={id} value={id}>
                    {tree.name || id}
                  </option>
                ))}
              </Select>
            </FormControl>
          )}

          {currentAgentMeta && currentAgentMeta.capabilities && currentAgentMeta.capabilities.length > 0 && (
            <Box 
              fontSize="xs" 
              color={mutedColor} 
              mt={2} 
              p={2} 
              bg={hoverBgColor} 
              borderRadius="md"
            >
              <Text fontWeight="medium" mb={1}>Capabilities:</Text>
              <Flex flexWrap="wrap" gap={1}>
                {currentAgentMeta.capabilities.map((cap, index) => (
                  <Badge 
                    key={index} 
                    colorScheme="purple" 
                    variant="subtle" 
                    fontSize="11px"
                  >
                    {cap.name}
                  </Badge>
                ))}
              </Flex>
            </Box>
          )}
        </>
      )}
    </Box>
  );
};

export default AgentSelector; 