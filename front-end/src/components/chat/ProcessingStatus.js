import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Flex, 
  Text, 
  VStack, 
  HStack, 
  Spinner, 
  Progress, 
  Icon, 
  Badge,
  Collapse,
  useColorModeValue,
  keyframes,
  Tooltip,
  Divider
} from '@chakra-ui/react';
import { 
  BrainCircuit, 
  Search, 
  Cpu, 
  CheckCircle, 
  Clock, 
  Zap,
  Database,
  MessageSquare,
  BookOpen,
  Target,
  ChevronDown,
  ChevronUp,
  Shield,
  Filter,
  BarChart3,
  Settings,
  TrendingUp,
  Activity,
  AlertCircle,
  Wifi,
  WifiOff
} from 'lucide-react';

const pulseAnimation = keyframes`
  0% { opacity: 0.6; }
  50% { opacity: 1; }
  100% { opacity: 0.6; }
`;

// New animation for step completion
const completionAnimation = keyframes`
  0% { transform: scale(1); }
  50% { transform: scale(1.1); }
  100% { transform: scale(1); }
`;

// New animation for confidence meter
const confidenceAnimation = keyframes`
  0% { width: 0%; }
  100% { width: var(--confidence-width); }
`;

/**
 * Enhanced processing status component showing AI internal decisions
 */
const ProcessingStatus = ({ 
  processingState, 
  internalThoughts, 
  isProcessing = true,
  expanded = true,
  processingSteps = [], // New prop for actual backend steps
  currentStep = 0,
  totalSteps = 7, // Updated for new step count
  startTime = null, // New prop to track processing start time
  contextMetadata = null, // New prop for context insights
  isConnected = true, // New prop to show connection status
  requestId = null // New prop to track request ID
}) => {
  const [isExpanded, setIsExpanded] = useState(expanded);
  const [processingTime, setProcessingTime] = useState(0);
  const [confidenceLevel, setConfidenceLevel] = useState(0);
  const [lastActivity, setLastActivity] = useState(Date.now());

  // Track processing time
  useEffect(() => {
    let interval = null;
    if (isProcessing && startTime) {
      interval = setInterval(() => {
        setProcessingTime(Math.floor((Date.now() - startTime) / 1000));
      }, 1000);
    } else {
      setProcessingTime(0);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isProcessing, startTime]);

  // Update last activity when processing steps change
  useEffect(() => {
    if (processingSteps.length > 0) {
      setLastActivity(Date.now());
    }
  }, [processingSteps]);

  // Calculate confidence level based on processing progress and context
  useEffect(() => {
    const completedSteps = displaySteps.filter(step => step.status === 'completed').length;
    const baseConfidence = (completedSteps / displaySteps.length) * 0.7; // 70% from completion
    
    // Add context quality bonus
    let contextBonus = 0;
    if (contextMetadata) {
      if (contextMetadata.context_quality === 'high') contextBonus = 0.2;
      else if (contextMetadata.context_quality === 'medium') contextBonus = 0.1;
      
      // Reduce confidence if high hallucination risk
      if (contextMetadata.hallucination_risk > 0.5) {
        contextBonus -= 0.1;
      }
    }
    
    setConfidenceLevel(Math.min(1, baseConfidence + contextBonus + 0.1)); // +0.1 base confidence
  }, [processingSteps, contextMetadata]);

  // Theme colors - ALL HOOKS MUST BE AT THE TOP
  const accentColor = useColorModeValue('#4415b6', '#6c45e7');
  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const textColor = useColorModeValue('gray.800', 'gray.100');
  const mutedTextColor = useColorModeValue('gray.600', 'gray.400');
  const processingBg = useColorModeValue('blue.50', 'blue.900');
  const processingColor = useColorModeValue('blue.700', 'blue.200');
  const successColor = useColorModeValue('green.500', 'green.300');
  const progressBarBg = useColorModeValue('gray.100', 'gray.700');
  const hoverBg = useColorModeValue('blue.100', 'blue.800');
  const currentStatusBg = useColorModeValue('gray.50', 'gray.700');
  const internalThoughtsBg = useColorModeValue('purple.50', 'purple.900');
  const internalThoughtsTextColor = useColorModeValue('purple.800', 'purple.200');
  const internalThoughtsContentColor = useColorModeValue('purple.700', 'purple.300');
  const contextInsightsBg = useColorModeValue('teal.50', 'teal.900');
  const contextInsightsColor = useColorModeValue('teal.700', 'teal.200');
  const confidenceBg = useColorModeValue('green.50', 'green.900');
  const confidenceColor = useColorModeValue('green.700', 'green.200');
  const warningColor = useColorModeValue('orange.500', 'orange.300');
  const errorColor = useColorModeValue('red.500', 'red.300');

  // Early return after all hooks
  if (!isProcessing && !internalThoughts && !contextMetadata) {
    return null;
  }

  // Define enhanced fallback steps with detailed descriptions (used when no backend steps available)
  const fallbackSteps = [
    {
      id: 'query_analysis',
      label: 'Query Analysis',
      description: 'Parsing natural language and identifying key concepts',
      details: 'Understanding intent, extracting entities, and determining complexity',
      icon: Target,
      status: 'pending'
    },
    {
      id: 'intent_classification',
      label: 'Intent Classification',
      description: 'Classifying query domain and type',
      details: 'Determining if this is compliance, policy, or guidance related',
      icon: Filter,
      status: 'pending'
    },
    {
      id: 'knowledge_search',
      label: 'Knowledge Search',
      description: 'Performing hybrid search across knowledge base',
      details: 'Using vector + semantic search across regulatory documents',
      icon: Search,
      status: 'pending'
    },
    {
      id: 'context_retrieval',
      label: 'Context Retrieval',
      description: 'Ranking and filtering relevant documents',
      details: 'Evaluating relevance scores and selecting best matches',
      icon: Database,
      status: 'pending'
    },
    {
      id: 'context_evaluation',
      label: 'Context Evaluation',
      description: 'Assessing information quality and relevance',
      details: 'Analyzing content accuracy and query alignment',
      icon: BarChart3,
      status: 'pending'
    },
    {
      id: 'reasoning_preparation',
      label: 'Reasoning Setup',
      description: 'Organizing information and reasoning framework',
      details: 'Establishing logical paths and information hierarchy',
      icon: BrainCircuit,
      status: 'pending'
    },
    {
      id: 'response_generation',
      label: 'Response Generation',
      description: 'Synthesizing comprehensive answer',
      details: 'Combining domain expertise with retrieved information',
      icon: MessageSquare,
      status: 'pending'
    }
  ];

  // Map backend step names to icons
  const getIconForStep = (stepName) => {
    const iconMap = {
      'query_analysis': Target,
      'intent_classification': Filter,
      'knowledge_search': Search,
      'context_retrieval': Database,
      'context_evaluation': BarChart3,
      'reasoning_preparation': BrainCircuit,
      'reasoning': BrainCircuit,
      'response_generation': MessageSquare
    };
    return iconMap[stepName] || Cpu;
  };

  // Use backend steps if available, otherwise use fallback
  const displaySteps = processingSteps.length > 0 
    ? processingSteps.map(step => ({
        id: step.step,
        label: step.step.split('_').map(word => 
          word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' '),
        description: step.message,
        details: step.details || 'Processing step in progress...',
        icon: getIconForStep(step.step),
        status: step.status || 'pending',
        stepNumber: step.stepNumber,
        contextMetadata: step.contextMetadata
      }))
    : fallbackSteps.map((step, index) => ({
        ...step,
        status: index < currentStep ? 'completed' : 
                index === currentStep ? 'in_progress' : 'pending'
      }));

  // Calculate overall progress
  const completedSteps = displaySteps.filter(step => step.status === 'completed').length;
  const actualTotalSteps = displaySteps.length;
  const progressPercentage = actualTotalSteps > 0 ? (completedSteps / actualTotalSteps) * 100 : 0;

  // Check if processing seems stalled
  const isStalled = isProcessing && processingTime > 30 && (Date.now() - lastActivity) > 15000; // 15 seconds without activity

  return (
    <Box
      bg={bgColor}
      borderWidth="1px"
      borderColor={borderColor}
      borderRadius="lg"
      overflow="hidden"
      boxShadow="0 2px 4px rgba(68, 21, 182, 0.05)"
      mb={4}
    >
      {/* Header */}
      <Flex
        align="center"
        justify="space-between"
        p={3}
        bg={processingBg}
        cursor="pointer"
        onClick={() => setIsExpanded(!isExpanded)}
        _hover={{ bg: hoverBg }}
        transition="all 0.2s ease"
      >
        <HStack spacing={3}>
          <Flex align="center" gap={2}>
            {isProcessing ? (
              <Spinner 
                size="sm" 
                color={accentColor}
                animation={`${pulseAnimation} 2s infinite`}
              />
            ) : (
              <Icon as={CheckCircle} color={successColor} boxSize={4} />
            )}
            <Text fontWeight="semibold" color={processingColor} fontSize="sm">
              {isProcessing ? 'Processing Your Query' : 'Processing Complete'}
            </Text>
          </Flex>
          
          {/* Connection Status */}
          <Tooltip label={isConnected ? 'Connected to AI backend' : 'Connection issues detected'}>
            <Icon 
              as={isConnected ? Wifi : WifiOff} 
              color={isConnected ? successColor : errorColor} 
              boxSize={3} 
            />
          </Tooltip>
          
          {/* Request ID for debugging */}
          {requestId && (
            <Tooltip label={`Request ID: ${requestId}`}>
              <Badge size="xs" colorScheme="gray" borderRadius="full">
                {requestId.slice(-8)}
              </Badge>
            </Tooltip>
          )}
        </HStack>
        
        <HStack spacing={2}>
          {/* Processing Time */}
          {isProcessing && processingTime > 0 && (
            <HStack spacing={1}>
              <Icon as={Clock} boxSize={3} color={mutedTextColor} />
              <Text fontSize="xs" color={mutedTextColor}>
                {processingTime}s
              </Text>
            </HStack>
          )}
          
          {/* Stalled Warning */}
          {isStalled && (
            <Tooltip label="Processing is taking longer than expected">
              <Icon as={AlertCircle} color={warningColor} boxSize={4} />
            </Tooltip>
          )}
          
          {/* Progress Indicator */}
          <Text fontSize="xs" color={mutedTextColor}>
            {completedSteps}/{actualTotalSteps}
          </Text>
          
          <Icon 
            as={isExpanded ? ChevronUp : ChevronDown} 
            color={mutedTextColor} 
            boxSize={4}
            transition="transform 0.2s"
          />
        </HStack>
      </Flex>

      {/* Progress Bar */}
      {isProcessing && (
        <Box px={3} pb={2} bg={processingBg}>
          <Progress
            value={progressPercentage}
            size="sm"
            colorScheme="purple"
            borderRadius="full"
            bg={progressBarBg}
          />
        </Box>
      )}

      {/* Content */}
      <Collapse in={isExpanded}>
        <Box p={4}>
          {/* Processing Steps */}
          {displaySteps.length > 0 && (
            <VStack spacing={3} align="stretch" mb={4}>
              {displaySteps.map((step, index) => (
                <StepItem
                  key={step.id}
                  step={step}
                  isActive={index === currentStep}
                  isCompleted={step.status === 'completed'}
                  isProcessing={step.status === 'in_progress'}
                />
              ))}
            </VStack>
          )}

          {/* Current Processing State */}
          {processingState && (
            <Box
              bg={currentStatusBg}
              p={3}
              borderRadius="md"
              borderLeft="3px solid"
              borderLeftColor={accentColor}
              mb={3}
            >
              <HStack spacing={2}>
                <Icon as={Clock} boxSize={3} color={accentColor} />
                <Text fontSize="xs" fontWeight="medium" color={textColor}>
                  Current Status:
                </Text>
              </HStack>
              <Text fontSize="xs" color={mutedTextColor} mt={1}>
                {processingState}
              </Text>
            </Box>
          )}

          {/* Internal Thoughts */}
          {internalThoughts && (
            <Box
              bg={internalThoughtsBg}
              p={3}
              borderRadius="md"
              borderLeft="3px solid"
              borderLeftColor={accentColor}
            >
              <HStack spacing={2} mb={2}>
                <Icon as={BrainCircuit} boxSize={3} color={accentColor} />
                <Text fontSize="xs" fontWeight="medium" color={internalThoughtsTextColor}>
                  Internal Reasoning:
                </Text>
                {isProcessing && (
                  <Spinner size="xs" color={accentColor} thickness="2px" />
                )}
              </HStack>
              <Text 
                fontSize="xs" 
                color={internalThoughtsContentColor}
                whiteSpace="pre-wrap"
                fontFamily="monospace"
                lineHeight="1.4"
              >
                {internalThoughts.replace(/<\/?internal_thoughts>/g, '')}
              </Text>
            </Box>
          )}

          {/* Context Metadata */}
          {contextMetadata && (
            <Box
              bg={contextInsightsBg}
              p={3}
              borderRadius="md"
              borderLeft="3px solid"
              borderLeftColor={accentColor}
            >
              <HStack spacing={2} mb={2}>
                <Icon as={Shield} boxSize={3} color={accentColor} />
                <Text fontSize="xs" fontWeight="medium" color={contextInsightsColor}>
                  Context Metadata:
                </Text>
              </HStack>
              <Text 
                fontSize="xs" 
                color={contextInsightsColor}
                whiteSpace="pre-wrap"
                fontFamily="monospace"
                lineHeight="1.4"
              >
                {contextMetadata.replace(/<\/?context_metadata>/g, '')}
              </Text>
            </Box>
          )}
        </Box>
      </Collapse>
    </Box>
  );
};

/**
 * Individual step item component with enhanced details
 */
const StepItem = ({ step, isActive, isCompleted, isProcessing }) => {
  // ALL HOOKS MUST BE AT THE TOP
  const textColor = useColorModeValue('gray.800', 'gray.100');
  const mutedTextColor = useColorModeValue('gray.600', 'gray.400');
  const detailTextColor = useColorModeValue('gray.500', 'gray.500');
  const accentColor = useColorModeValue('#4415b6', '#6c45e7');
  const successColor = useColorModeValue('green.500', 'green.300');
  const stepBg = useColorModeValue('gray.50', 'gray.700');
  const activeBg = useColorModeValue('blue.50', 'blue.900');

  const getStepColor = () => {
    if (isCompleted) return successColor;
    if (isProcessing || isActive) return accentColor;
    return mutedTextColor;
  };

  const getStepIcon = () => {
    if (isCompleted) return CheckCircle;
    if (isProcessing || isActive) return step.icon;
    return step.icon;
  };

  const getStepBg = () => {
    if (isProcessing || isActive) return activeBg;
    return 'transparent';
  };

  return (
    <Box
      p={3}
      borderRadius="md"
      bg={getStepBg()}
      border="1px solid"
      borderColor={isActive ? accentColor : 'transparent'}
      transition="all 0.2s"
    >
      <HStack spacing={3} align="start">
        <Box
          mt={0.5}
          as="span"
          animation={isProcessing ? `${pulseAnimation} 2s infinite` : 'none'}
        >
          <Icon
            as={getStepIcon()}
            boxSize={4}
            color={getStepColor()}
          />
        </Box>
        
        <Box flex="1">
          <HStack spacing={2} align="center" mb={1}>
            <Text
              fontSize="sm"
              fontWeight={isActive ? "semibold" : "medium"}
              color={getStepColor()}
            >
              {step.label}
            </Text>
            
            {isProcessing && (
              <Spinner size="xs" color={accentColor} thickness="2px" />
            )}
            
            {isCompleted && (
              <Badge colorScheme="green" size="sm" borderRadius="full">
                ✓
              </Badge>
            )}
          </HStack>
          
          <Text fontSize="xs" color={mutedTextColor} mb={1}>
            {step.description}
          </Text>
          
          {/* Show detailed step information when active or processing */}
          {(isActive || isProcessing) && step.details && (
            <Text fontSize="xs" color={detailTextColor} fontStyle="italic">
              {step.details}
            </Text>
          )}
          
          {/* Show context metadata for specific steps */}
          {step.contextMetadata && (
            <Box mt={2} p={2} bg={stepBg} borderRadius="sm">
              <Text fontSize="xs" color={detailTextColor}>
                📊 Sources: {step.contextMetadata.source_count} | 
                Quality: {step.contextMetadata.context_quality} |
                {step.contextMetadata.hallucination_risk && 
                  ` Risk: ${Math.round(step.contextMetadata.hallucination_risk * 100)}%`
                }
              </Text>
            </Box>
          )}
        </Box>
      </HStack>
    </Box>
  );
};

export default ProcessingStatus; 