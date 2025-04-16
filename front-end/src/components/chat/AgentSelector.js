import React, { useState, useEffect, useRef } from 'react';
import { Box, FormControl, FormLabel, Select, Switch, Tooltip, Badge, Flex, Text, Spinner, Button, useToast, useColorModeValue } from '@chakra-ui/react';
import { InfoIcon } from '@chakra-ui/icons';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Component for selecting an AI agent and its parameters for chat
 */
const AgentSelector = ({ onAgentChange, initialAgent = null }) => {
  const [agentTypes, setAgentTypes] = useState({});
  const [agentMetadata, setAgentMetadata] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // To prevent redundant updates when parent re-renders with same props
  const initialAgentRef = useRef(null);
  const isInitialRender = useRef(true);
  
  // Default agent settings
  const defaultAgentSettings = { 
    use_agent: false, 
    agent_type: null, 
    use_tree_reasoning: false,
    tree_template: 'default'
  };
  
  // Initialize from initialAgent prop
  const initialAgentObj = typeof initialAgent === 'object' ? initialAgent : defaultAgentSettings;
  
  const [selectedAgent, setSelectedAgent] = useState(initialAgentObj.agent_type);
  const [useAgent, setUseAgent] = useState(initialAgentObj.use_agent);
  const [useTreeReasoning, setUseTreeReasoning] = useState(initialAgentObj.use_tree_reasoning);
  const [availableTrees, setAvailableTrees] = useState({});
  const [selectedTree, setSelectedTree] = useState(initialAgentObj.tree_template || 'default');
  
  const toast = useToast();
  
  // Theme colors
  const accentColor = '#4415b6';
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const bgColor = useColorModeValue('white', 'gray.800');
  const textColor = useColorModeValue('gray.800', 'gray.200');
  const mutedColor = useColorModeValue('gray.600', 'gray.400');
  const hoverBgColor = useColorModeValue('#4415b610', 'gray.700');

  // Track initialAgent changes without triggering onAgentChange
  useEffect(() => {
    // Skip the first render since we've already initialized in useState
    if (isInitialRender.current) {
      isInitialRender.current = false;
      initialAgentRef.current = initialAgent;
      return;
    }
    
    // Only update state if we get a new different agent configuration
    if (initialAgent) {
      const currentValue = JSON.stringify(initialAgent);
      const prevValue = initialAgentRef.current ? JSON.stringify(initialAgentRef.current) : null;
      
      if (currentValue !== prevValue) {
        initialAgentRef.current = initialAgent;
        
        if (typeof initialAgent === 'object') {
          setSelectedAgent(initialAgent.agent_type);
          setUseAgent(initialAgent.use_agent);
          setUseTreeReasoning(initialAgent.use_tree_reasoning);
          setSelectedTree(initialAgent.tree_template || 'default');
        }
      }
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
        toast({
          title: 'Error loading agents',
          description: 'Could not load available AI agents. Using standard chat only.',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchAgentData();
  }, [toast]);
  
  // Creates an object with current agent settings
  const getCurrentAgentSettings = () => ({
    use_agent: useAgent,
    agent_type: selectedAgent,
    use_tree_reasoning: useTreeReasoning,
    tree_template: selectedTree,
  });

  // Handle agent selection change
  const handleAgentChange = (event) => {
    const agentType = event.target.value;
    setSelectedAgent(agentType);
    
    // Notify parent component with updated settings
    const updatedSettings = {
      ...getCurrentAgentSettings(),
      agent_type: agentType,
    };
    
    // Only call onAgentChange if it's from a user interaction
    onAgentChange(updatedSettings);
  };

  // Handle agent toggle
  const handleAgentToggle = () => {
    const newUseAgent = !useAgent;
    setUseAgent(newUseAgent);
    
    // Update the parent component
    const updatedSettings = {
      ...getCurrentAgentSettings(),
      use_agent: newUseAgent,
    };
    
    onAgentChange(updatedSettings);
  };

  // Handle tree reasoning toggle
  const handleTreeReasoningToggle = () => {
    const newUseTree = !useTreeReasoning;
    setUseTreeReasoning(newUseTree);
    
    // Update the parent component
    const updatedSettings = {
      ...getCurrentAgentSettings(),
      use_tree_reasoning: newUseTree,
    };
    
    onAgentChange(updatedSettings);
  };

  // Handle tree template selection
  const handleTreeChange = (event) => {
    const treeId = event.target.value;
    setSelectedTree(treeId);
    
    // Update the parent component
    const updatedSettings = {
      ...getCurrentAgentSettings(),
      tree_template: treeId,
    };
    
    onAgentChange(updatedSettings);
  };

  // Find the current agent metadata
  const currentAgentMeta = agentMetadata.find(a => a.id === selectedAgent) || null;

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
            isChecked={useAgent} 
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

      {useAgent && (
        <>
          <FormControl mb={3}>
            <FormLabel htmlFor="agent-select" fontSize="sm" mb={1} color={textColor}>
              Agent Type
            </FormLabel>
            <Select 
              id="agent-select" 
              value={selectedAgent || ''} 
              onChange={handleAgentChange}
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
              isChecked={useTreeReasoning} 
              onChange={handleTreeReasoningToggle} 
              colorScheme="purple"
            />
          </FormControl>

          {useTreeReasoning && (
            <FormControl mb={3}>
              <FormLabel htmlFor="tree-select" fontSize="sm" mb={1} color={textColor}>
                Decision Tree Template
              </FormLabel>
              <Select 
                id="tree-select" 
                value={selectedTree || 'default'} 
                onChange={handleTreeChange}
                size="sm"
                focusBorderColor={accentColor}
              >
                <option value="default">Default Tree</option>
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