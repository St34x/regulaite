import React from 'react';
import { useTheme } from '../../contexts/ThemeContext';
import { Box, Text, VStack, HStack, useColorModeValue } from '@chakra-ui/react';

const ThemeTest = () => {
  const { theme } = useTheme();
  
  // Colors using Chakra UI's color mode
  const chakraBackground = useColorModeValue('white', 'gray.800');
  const chakraText = useColorModeValue('gray.800', 'white');
  
  return (
    <VStack spacing={6} p={6} width="100%">
      <Text fontSize="2xl" fontWeight="bold">Theme Test Component</Text>
      
      <Box p={4} borderRadius="md" width="100%" maxW="600px">
        <Text mb={2} fontWeight="bold">Current Theme State:</Text>
        <Text>Current theme: {theme}</Text>
        <Text>Dark mode class: {document.documentElement.classList.contains('dark') ? 'Yes' : 'No'}</Text>
      </Box>
      
      <HStack spacing={4} width="100%" maxW="600px">
        <Box 
          p={4} 
          borderRadius="md" 
          bg={chakraBackground} 
          color={chakraText}
          boxShadow="md"
          flex="1"
        >
          <Text fontWeight="bold">Chakra UI Themed</Text>
          <Text>This box uses Chakra's useColorModeValue</Text>
        </Box>
        
        <Box 
          p={4} 
          borderRadius="md"
          className="bg-background text-foreground"
          boxShadow="md"
          flex="1"
        >
          <Text fontWeight="bold">Tailwind CSS Themed</Text>
          <Text>This box uses Tailwind CSS classes</Text>
        </Box>
      </HStack>
      
      {/* Additional theme test elements */}
      <HStack spacing={4} width="100%" maxW="600px">
        <Box 
          p={4} 
          borderRadius="md"
          className="bg-card text-card-foreground"
          boxShadow="md"
          flex="1"
        >
          <Text fontWeight="bold">Card Styling</Text>
          <Text>Using bg-card and text-card-foreground</Text>
        </Box>
        
        <Box 
          p={4} 
          borderRadius="md"
          className="bg-muted text-muted-foreground"
          boxShadow="md"
          flex="1"
        >
          <Text fontWeight="bold">Muted Styling</Text>
          <Text>Using bg-muted and text-muted-foreground</Text>
        </Box>
      </HStack>
      
      <Box
        p={4}
        borderRadius="md"
        className="bg-primary text-primary-foreground"
        width="100%"
        maxW="600px"
      >
        <Text fontWeight="bold">Primary Color</Text>
        <Text>Using bg-primary and text-primary-foreground</Text>
      </Box>
    </VStack>
  );
};

export default ThemeTest; 