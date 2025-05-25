import React from 'react';
import { Box, Badge, Flex, Text, useColorModeValue } from '@chakra-ui/react';
import { InfoIcon } from '@chakra-ui/icons';

/**
 * Component for displaying autonomous agent status
 * Agents are always enabled and automatically determine the best approach for each query
 */
const AgentSelector = () => {
  // Theme colors
  const accentColor = '#4415b6';
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const bgColor = useColorModeValue('white', 'gray.800');
  const textColor = useColorModeValue('gray.800', 'gray.200');
  const hoverBgColor = useColorModeValue('#4415b610', 'gray.700');

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
        <Text fontSize="sm" fontWeight="medium" color={textColor}>
          Autonomous AI Agent
        </Text>
        
        <Badge 
          bg={accentColor} 
          color="white" 
          variant="solid" 
          p={1} 
          borderRadius="md"
        >
          Always Active <InfoIcon ml={1} boxSize={3} />
        </Badge>
      </Flex>

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
        <Text>• Always enabled for optimal performance</Text>
      </Box>
    </Box>
  );
};

export default AgentSelector; 