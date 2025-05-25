import React, { useState } from 'react';
import { 
  Box, 
  Text, 
  Button, 
  Collapse, 
  VStack, 
  HStack,
  Badge,
  Divider,
  useColorModeValue 
} from '@chakra-ui/react';
import { FileText, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * SourcesSection - Displays RAG sources with document information
 * Features: Hidden by default, expandable, shows document name, text chunk, and relevance
 */
const SourcesSection = ({ 
  sources = [], 
  isStreaming = false 
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showAllSources, setShowAllSources] = useState(false);

  // Theme colors - moved to top level to fix hooks rule
  const accentColor = '#4415b6';
  const sourcesBg = useColorModeValue('gray.50', 'gray.800');
  const sourcesBorderColor = useColorModeValue('gray.200', 'gray.600');
  const sourcesTextColor = useColorModeValue('gray.800', 'gray.200');
  const sourcesSecondaryColor = useColorModeValue('gray.600', 'gray.400');
  const sourceItemBg = useColorModeValue('white', 'gray.700');
  const sourceItemBorderColor = useColorModeValue('gray.100', 'gray.600');
  const sourcesHoverBg = useColorModeValue('gray.100', 'gray.700');
  const chunkTextBg = useColorModeValue('gray.50', 'gray.800');

  // Don't render if no sources or still streaming
  if (!sources || sources.length === 0 || isStreaming) {
    return null;
  }

  const displayedSources = showAllSources ? sources : sources.slice(0, 3);
  const hasMoreSources = sources.length > 3;

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
  };

  const toggleShowAll = () => {
    setShowAllSources(!showAllSources);
  };

  // Calculate relevance color
  const getRelevanceColor = (score) => {
    if (score >= 0.8) return 'green';
    if (score >= 0.6) return 'yellow';
    if (score >= 0.4) return 'orange';
    return 'red';
  };

  // Format relevance score
  const formatRelevance = (score) => {
    if (typeof score === 'number') {
      return `${Math.round(score * 100)}%`;
    }
    return 'N/A';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.4 }}
    >
      <Box
        bg={sourcesBg}
        borderWidth="1px"
        borderColor={sourcesBorderColor}
        borderRadius="lg"
        overflow="hidden"
        boxShadow="sm"
      >
        {/* Header */}
        <HStack
          p={3}
          justify="space-between"
          align="center"
          cursor="pointer"
          onClick={toggleExpanded}
          _hover={{ bg: sourcesHoverBg }}
          transition="background-color 0.2s"
        >
          <HStack spacing={2}>
            <FileText size={16} color={accentColor} />
            <Text
              fontSize="sm"
              fontWeight="medium"
              color={sourcesTextColor}
            >
              Sources
            </Text>
            <Badge
              colorScheme="purple"
              size="sm"
              borderRadius="full"
            >
              {sources.length}
            </Badge>
          </HStack>

          <HStack spacing={2}>
            <Text fontSize="xs" color={sourcesSecondaryColor}>
              {isExpanded ? 'Hide' : 'Show'} references
            </Text>
            {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </HStack>
        </HStack>

        {/* Collapsible content */}
        <Collapse in={isExpanded} animateOpacity>
          <Box p={3} pt={0}>
            <VStack spacing={3} align="stretch">
              {displayedSources.map((source, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                >
                  <Box
                    bg={sourceItemBg}
                    borderWidth="1px"
                    borderColor={sourceItemBorderColor}
                    borderRadius="md"
                    p={3}
                    boxShadow="xs"
                  >
                    {/* Source header */}
                    <HStack justify="space-between" align="flex-start" mb={2}>
                      <VStack align="stretch" spacing={1} flex={1}>
                        {/* File name */}
                        <Text
                          fontSize="sm"
                          fontWeight="medium"
                          color={sourcesTextColor}
                          noOfLines={1}
                        >
                          📄 {source.filename || source.original_filename || source.title || `Document ${index + 1}`}
                        </Text>
                      </VStack>

                      {/* Match percentage */}
                      {(source.match_percentage !== undefined || source.relevance_score !== undefined) && (
                        <Badge
                          colorScheme={getRelevanceColor(source.relevance_score || source.match_percentage / 100)}
                          size="sm"
                          borderRadius="full"
                        >
                          {source.match_percentage !== undefined 
                            ? `${source.match_percentage}%` 
                            : formatRelevance(source.relevance_score)
                          }
                        </Badge>
                      )}
                    </HStack>

                    {/* Chunk text content */}
                    {(source.chunk_text || source.content) && (
                      <Box
                        bg={chunkTextBg}
                        borderRadius="md"
                        p={3}
                        mt={2}
                      >
                        <Text
                          fontSize="xs"
                          color={sourcesSecondaryColor}
                          lineHeight="1.5"
                          noOfLines={4}
                          fontFamily="system-ui"
                        >
                          {source.chunk_text || source.content}
                        </Text>
                      </Box>
                    )}

                    {/* External link if available */}
                    {source.url && (
                      <HStack mt={2}>
                        <Button
                          size="xs"
                          variant="ghost"
                          leftIcon={<ExternalLink size={12} />}
                          color={accentColor}
                          onClick={() => window.open(source.url, '_blank')}
                        >
                          View Source
                        </Button>
                      </HStack>
                    )}
                  </Box>
                </motion.div>
              ))}

              {/* Show more/less button */}
              {hasMoreSources && (
                <Box textAlign="center" pt={2}>
                  <Button
                    size="sm"
                    variant="ghost"
                    color={accentColor}
                    onClick={toggleShowAll}
                    fontSize="xs"
                  >
                    {showAllSources 
                      ? `Show Less` 
                      : `Show ${sources.length - 3} More Sources`
                    }
                  </Button>
                </Box>
              )}
            </VStack>
          </Box>
        </Collapse>
      </Box>
    </motion.div>
  );
};

export default SourcesSection; 