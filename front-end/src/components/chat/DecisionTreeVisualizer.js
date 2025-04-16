import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  Text, 
  VStack, 
  HStack, 
  Badge, 
  Button, 
  Spinner,
  useToast,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon
} from '@chakra-ui/react';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Component to visualize decision trees
 */
const DecisionTreeVisualizer = ({ treeId, highlightedNodeId = null }) => {
  const [treeData, setTreeData] = useState(null);
  const [svgContent, setSvgContent] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('graph'); // 'graph' or 'json'
  const svgContainerRef = useRef(null);
  
  const toast = useToast();

  // Fetch tree data and visualization
  useEffect(() => {
    const fetchTreeData = async () => {
      if (!treeId) return;
      
      setIsLoading(true);
      try {
        // Fetch tree data
        const treeResponse = await axios.get(`${API_URL}/agents/trees/${treeId}`);
        setTreeData(treeResponse.data);
        
        // Fetch SVG visualization
        const svgResponse = await axios.get(`${API_URL}/agents/visualize-tree/${treeId}?format=svg`);
        setSvgContent(svgResponse.data);
        
        setError(null);
      } catch (err) {
        console.error('Error fetching tree data:', err);
        setError('Failed to load decision tree visualization.');
        toast({
          title: 'Error loading tree',
          description: 'Could not load the decision tree visualization.',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchTreeData();
  }, [treeId, toast]);

  // Apply highlighting to the highlighted node when SVG content changes or highlighted node changes
  useEffect(() => {
    if (svgContent && highlightedNodeId && viewMode === 'graph' && svgContainerRef.current) {
      // Insert SVG content into container
      svgContainerRef.current.innerHTML = svgContent;
      
      // Find the node element to highlight
      const nodeElement = svgContainerRef.current.querySelector(`[id="node_${highlightedNodeId}"]`);
      if (nodeElement) {
        // Add highlight styling
        nodeElement.style.stroke = '#4415b6';
        nodeElement.style.strokeWidth = '3px';
      }
    } else if (svgContent && viewMode === 'graph' && svgContainerRef.current) {
      // Just insert SVG without highlighting
      svgContainerRef.current.innerHTML = svgContent;
    }
  }, [svgContent, highlightedNodeId, viewMode]);

  // Render tree nodes recursively as a JSON tree
  const renderTreeNode = (node, nodeId, depth = 0) => {
    if (!node) return null;
    
    const isHighlighted = nodeId === highlightedNodeId;
    
    return (
      <Box 
        key={nodeId} 
        ml={depth * 4} 
        my={1} 
        p={2} 
        borderLeft="2px solid" 
        borderLeftColor={isHighlighted ? "#4415b6" : "gray.200"}
        bg={isHighlighted ? "purple.50" : "transparent"}
      >
        <HStack spacing={2} mb={1}>
          <Text fontWeight={isHighlighted ? "bold" : "medium"} fontSize="sm">
            {node.type === 'decision' ? '🔍' : node.type === 'action' ? '⚡' : '❓'} {node.name || nodeId}
          </Text>
          
          <Badge colorScheme={
            node.type === 'decision' ? 'blue' : 
            node.type === 'action' ? 'green' : 'yellow'
          } fontSize="xs">
            {node.type}
          </Badge>
        </HStack>
        
        {node.description && (
          <Text fontSize="xs" color="gray.600" mb={1}>
            {node.description}
          </Text>
        )}
        
        {node.children && (
          <VStack align="stretch" spacing={0}>
            {Object.entries(node.children).map(([childId, childNode]) => 
              renderTreeNode(childNode, childId, depth + 1)
            )}
          </VStack>
        )}
      </Box>
    );
  };

  if (isLoading) {
    return (
      <Box textAlign="center" p={4}>
        <Spinner size="md" color="#4415b6" />
        <Text mt={2} fontSize="sm">Loading decision tree...</Text>
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={4} borderRadius="md" bg="red.50" color="red.800">
        <Text fontSize="sm">{error}</Text>
      </Box>
    );
  }

  if (!treeData) {
    return (
      <Box p={4} borderRadius="md" bg="gray.50">
        <Text fontSize="sm">No tree data available. Select a valid decision tree.</Text>
      </Box>
    );
  }

  return (
    <Box borderWidth="1px" borderRadius="md" p={3} bg="white" shadow="sm">
      <Accordion allowToggle defaultIndex={[0]}>
        <AccordionItem border="none">
          <AccordionButton px={0} _hover={{ bg: 'transparent' }}>
            <Box flex="1" textAlign="left">
              <Text fontWeight="medium" fontSize="sm">Decision Tree: {treeData.name}</Text>
            </Box>
            <AccordionIcon />
          </AccordionButton>
          
          <AccordionPanel pb={4} px={0}>
            <VStack align="stretch" spacing={4}>
              {/* Tree Metadata */}
              <Box>
                <Text fontSize="xs" color="gray.600" mb={1}>
                  {treeData.description}
                </Text>
                <HStack spacing={2}>
                  <Badge colorScheme="purple" fontSize="xs">ID: {treeData.id}</Badge>
                  <Badge colorScheme="gray" fontSize="xs">v{treeData.version}</Badge>
                  {treeData.is_default && (
                    <Badge colorScheme="green" fontSize="xs">Default</Badge>
                  )}
                </HStack>
              </Box>
              
              {/* View Mode Toggle */}
              <HStack justifyContent="center" spacing={2}>
                <Button 
                  size="xs" 
                  variant={viewMode === 'graph' ? 'solid' : 'outline'} 
                  colorScheme="purple"
                  onClick={() => setViewMode('graph')}
                >
                  Graph View
                </Button>
                <Button 
                  size="xs" 
                  variant={viewMode === 'json' ? 'solid' : 'outline'} 
                  colorScheme="purple"
                  onClick={() => setViewMode('json')}
                >
                  JSON View
                </Button>
              </HStack>
              
              {/* Tree Visualization */}
              {viewMode === 'graph' ? (
                <Box 
                  ref={svgContainerRef} 
                  overflowX="auto" 
                  maxW="100%" 
                  h="300px"
                  borderWidth="1px"
                  borderRadius="md"
                  borderColor="gray.200"
                  p={2}
                />
              ) : (
                <Box 
                  maxH="400px" 
                  overflowY="auto"
                  borderWidth="1px"
                  borderRadius="md"
                  borderColor="gray.200"
                  p={2}
                >
                  {treeData.nodes && Object.entries(treeData.nodes).map(([nodeId, node]) => {
                    // Only render root nodes (nodes at the top level)
                    if (!Object.values(treeData.nodes).some(n => 
                      n.children && Object.values(n.children).some(child => child && child.id === nodeId)
                    )) {
                      return renderTreeNode(node, nodeId);
                    }
                    return null;
                  })}
                </Box>
              )}
            </VStack>
          </AccordionPanel>
        </AccordionItem>
      </Accordion>
    </Box>
  );
};

export default DecisionTreeVisualizer; 