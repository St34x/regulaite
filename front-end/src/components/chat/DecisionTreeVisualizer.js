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
  const [viewMode, setViewMode] = useState('json'); // Default to 'json' since graph might not be available
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
        
        // Try to fetch SVG visualization if available
        try {
          const svgResponse = await axios.get(`${API_URL}/agents/visualize-tree/${treeId}?format=svg`);
          setSvgContent(svgResponse.data);
          setViewMode('graph'); // Switch to graph mode if SVG is available
        } catch (svgError) {
          console.warn('SVG visualization not available:', svgError);
          setSvgContent(null);
          setViewMode('json'); // Fallback to JSON view
        }
        
        setError(null);
      } catch (err) {
        console.error('Error fetching tree data:', err);
        setError('Failed to load decision tree data.');
        toast({
          title: 'Error loading tree',
          description: 'Could not load the decision tree data.',
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

  // Helper function to build a tree structure from nodes object
  const buildTreeStructure = (nodes, rootNodeId) => {
    if (!nodes || !rootNodeId || !nodes[rootNodeId]) return null;
    
    const visitedNodes = new Set();
    
    const buildNode = (nodeId) => {
      if (visitedNodes.has(nodeId) || !nodes[nodeId]) return null;
      visitedNodes.add(nodeId);
      
      const node = nodes[nodeId];
      const children = {};
      
      // For decision nodes, process options
      if (node.type === 'decision' && node.options && Array.isArray(node.options)) {
        node.options.forEach(option => {
          if (option.next && nodes[option.next]) {
            children[option.next] = buildNode(option.next);
          }
        });
      } 
      // For other node types, check next property
      else if (node.next && nodes[node.next]) {
        children[node.next] = buildNode(node.next);
      }
      
      return {
        ...node,
        name: node.id,
        description: node.query || node.action || node.response_template || '',
        children: Object.keys(children).length > 0 ? children : null
      };
    };
    
    return buildNode(rootNodeId);
  };

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
            {node.type === 'decision' ? '🔍' : node.type === 'action' ? '⚡' : '💬'} {nodeId}
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
        
        {/* For decision nodes, show options */}
        {node.type === 'decision' && node.options && Array.isArray(node.options) && (
          <Box ml={2} mt={1} fontSize="xs">
            <Text fontWeight="medium" mb={1}>Options:</Text>
            {node.options.map((option, idx) => (
              <HStack key={idx} spacing={2} mb={0.5}>
                <Badge colorScheme="gray" fontSize="xs">{option.value}</Badge>
                <Text>{option.label}</Text>
                {option.next && <Text color="gray.500">→ {option.next}</Text>}
              </HStack>
            ))}
          </Box>
        )}
        
        {/* For action nodes, show action and next */}
        {node.type === 'action' && (
          <Box ml={2} mt={1} fontSize="xs">
            <Text fontWeight="medium" mb={1}>Action: {node.action}</Text>
            {node.next && <Text color="gray.500">Next: {node.next}</Text>}
          </Box>
        )}
        
        {node.children && (
          <VStack align="stretch" spacing={0} mt={2}>
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

  // Build tree structure from nodes
  const treeStructure = treeData.nodes ? 
    buildTreeStructure(treeData.nodes, treeData.root_node) : null;

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
                  {treeData.version && <Badge colorScheme="gray" fontSize="xs">v{treeData.version}</Badge>}
                  {treeData.is_default && (
                    <Badge colorScheme="green" fontSize="xs">Default</Badge>
                  )}
                </HStack>
              </Box>
              
              {/* View Mode Toggle - Only show if SVG content is available */}
              {svgContent && (
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
                    Tree View
                  </Button>
                </HStack>
              )}
              
              {/* Tree Visualization */}
              <Box 
                borderWidth="1px" 
                borderRadius="md" 
                p={3} 
                bg="white" 
                shadow="inner"
                overflowX="auto"
                maxHeight="500px"
                overflowY="auto"
              >
                {viewMode === 'graph' && svgContent ? (
                  <Box 
                    ref={svgContainerRef} 
                    width="100%" 
                    minHeight="300px"
                    display="flex"
                    justifyContent="center"
                  />
                ) : (
                  <Box>
                    {treeStructure ? (
                      renderTreeNode(treeStructure, treeData.root_node)
                    ) : (
                      <Text fontSize="sm" color="gray.500" textAlign="center" py={10}>
                        Cannot visualize this tree structure.
                      </Text>
                    )}
                  </Box>
                )}
              </Box>
            </VStack>
          </AccordionPanel>
        </AccordionItem>
      </Accordion>
    </Box>
  );
};

export default DecisionTreeVisualizer; 