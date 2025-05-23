import React from 'react';
import { Box, FormControl, FormLabel, Switch, Badge, Flex, Text, useColorModeValue } from '@chakra-ui/react';
import { InfoIcon } from '@chakra-ui/icons';

/**
 * Component for displaying autonomous agent status
 * The agent automatically determines the best approach for each query
 */
const AgentSelector = ({ onAgentChange, initialAgent = null }) => {
  // Simple state for autonomous agent
  const [useAgent, setUseAgent] = React.useState(true);
  
  // Theme colors
  const accentColor = '#4415b6';
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const bgColor = useColorModeValue('white', 'gray.800');
  const textColor = useColorModeValue('gray.800', 'gray.200');
  const hoverBgColor = useColorModeValue('#4415b610', 'gray.700');

  // Update from initial agent settings
  React.useEffect(() => {
    if (initialAgent && typeof initialAgent === 'object') {
      setUseAgent(initialAgent.use_agent !== false);
    }
  }, [initialAgent]);

  // Handle agent toggle
  const handleAgentToggle = () => {
    const newValue = !useAgent;
    setUseAgent(newValue);
    onAgentChange({ use_agent: newValue });
  };

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
            Autonomous AI Agent
          </FormLabel>
          <Switch 
            id="agent-toggle" 
            isChecked={useAgent} 
            onChange={handleAgentToggle} 
            colorScheme="purple"
          />
        </FormControl>
        
        {useAgent && (
          <Badge 
            bg={accentColor} 
            color="white" 
            variant="solid" 
            p={1} 
            borderRadius="md"
          >
            Active <InfoIcon ml={1} boxSize={3} />
          </Badge>
        )}
      </Flex>

      {useAgent && (
        <Box 
          fontSize="xs" 
          color={textColor} 
          mt={2} 
          p={2} 
          bg={hoverBgColor} 
          borderRadius="md"
        >
          <Text fontWeight="medium" mb={1}>Autonomous Agent Features:</Text>
          <Text>• Automatically selects the best approach for your query</Text>
          <Text>• Intelligent context retrieval and reasoning</Text>
          <Text>• Adaptive response generation</Text>
        </Box>
      )}
    </Box>
  );
};

export default AgentSelector; 