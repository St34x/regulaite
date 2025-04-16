import React, { useState, useEffect } from 'react';
import { 
  Box, 
  FormControl, 
  FormLabel, 
  Select, 
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberDecrementStepper,
  NumberIncrementStepper,
  Tooltip,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  HStack,
  Text,
  VStack
} from '@chakra-ui/react';
import { InfoIcon } from '@chakra-ui/icons';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Component for selecting and configuring LLM parameters
 */
const ModelParamsSelector = ({ onParamsChange, initialParams = {} }) => {
  const [config, setConfig] = useState(null);
  const [params, setParams] = useState({
    model: initialParams.model || 'gpt-4',
    temperature: initialParams.temperature !== undefined ? initialParams.temperature : 0.7,
    max_tokens: initialParams.max_tokens || 2048,
    top_p: initialParams.top_p !== undefined ? initialParams.top_p : 1.0,
    frequency_penalty: initialParams.frequency_penalty !== undefined ? initialParams.frequency_penalty : 0.0,
    presence_penalty: initialParams.presence_penalty !== undefined ? initialParams.presence_penalty : 0.0
  });
  
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch available models and default parameters
  useEffect(() => {
    const fetchConfig = async () => {
      setIsLoading(true);
      try {
        const response = await axios.get(`${API_URL}/config`);
        setConfig(response.data);
        
        // Update with defaults from config if available
        if (response.data && response.data.llm) {
          setParams(prev => ({
            model: initialParams.model || response.data.llm.model,
            temperature: initialParams.temperature !== undefined ? initialParams.temperature : response.data.llm.temperature,
            max_tokens: initialParams.max_tokens || response.data.llm.max_tokens,
            top_p: initialParams.top_p !== undefined ? initialParams.top_p : response.data.llm.top_p,
            frequency_penalty: initialParams.frequency_penalty !== undefined ? initialParams.frequency_penalty : response.data.llm.frequency_penalty,
            presence_penalty: initialParams.presence_penalty !== undefined ? initialParams.presence_penalty : response.data.llm.presence_penalty
          }));
        }
        
        setError(null);
      } catch (err) {
        console.error('Error fetching config:', err);
        setError('Failed to load model configuration.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchConfig();
  }, [initialParams]);

  // Notify parent when params change
  useEffect(() => {
    onParamsChange(params);
  }, [params, onParamsChange]);

  // Handle parameter change
  const handleParamChange = (param, value) => {
    setParams(prev => ({
      ...prev,
      [param]: value
    }));
  };

  // Available models - fallback if config not available
  const availableModels = [
    { id: 'gpt-4', name: 'GPT-4' },
    { id: 'gpt-4-turbo', name: 'GPT-4 Turbo' },
    { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo' },
    { id: 'claude-3-opus', name: 'Claude 3 Opus' },
    { id: 'claude-3-sonnet', name: 'Claude 3 Sonnet' },
    { id: 'claude-3-haiku', name: 'Claude 3 Haiku' }
  ];

  if (isLoading) {
    return (
      <Box p={2} opacity={0.7}>
        <Text fontSize="sm">Loading model parameters...</Text>
      </Box>
    );
  }

  return (
    <Box borderWidth="1px" borderRadius="md" p={3} bg="white" shadow="sm">
      <Accordion allowToggle defaultIndex={[0]}>
        <AccordionItem border="none">
          <AccordionButton px={0} _hover={{ bg: 'transparent' }}>
            <Box flex="1" textAlign="left">
              <Text fontWeight="medium" fontSize="sm">LLM Parameters</Text>
            </Box>
            <AccordionIcon />
          </AccordionButton>
          
          <AccordionPanel pb={4} px={0}>
            <VStack spacing={4} align="stretch">
              {/* Model Selection */}
              <FormControl>
                <FormLabel htmlFor="model-select" fontSize="sm" mb={1}>
                  Model
                </FormLabel>
                <Select
                  id="model-select"
                  value={params.model}
                  onChange={(e) => handleParamChange('model', e.target.value)}
                  size="sm"
                >
                  {availableModels.map(model => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))}
                </Select>
              </FormControl>
              
              {/* Temperature */}
              <FormControl>
                <HStack justify="space-between" mb={1}>
                  <FormLabel htmlFor="temperature-slider" fontSize="sm" mb={0}>
                    Temperature
                  </FormLabel>
                  <Tooltip label="Controls randomness: lower values are more focused, higher values are more creative" placement="top">
                    <InfoIcon boxSize={3} color="gray.500" />
                  </Tooltip>
                </HStack>
                <HStack spacing={4}>
                  <Slider
                    id="temperature-slider"
                    min={0}
                    max={2}
                    step={0.1}
                    value={params.temperature}
                    onChange={(value) => handleParamChange('temperature', value)}
                    colorScheme="purple"
                    flex="1"
                  >
                    <SliderTrack>
                      <SliderFilledTrack bg="#4415b6" />
                    </SliderTrack>
                    <SliderThumb />
                  </Slider>
                  <NumberInput
                    value={params.temperature}
                    onChange={(valueString) => {
                      const value = parseFloat(valueString);
                      if (!isNaN(value)) {
                        handleParamChange('temperature', Math.max(0, Math.min(2, value)));
                      }
                    }}
                    step={0.1}
                    min={0}
                    max={2}
                    size="xs"
                    maxW="60px"
                  >
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </HStack>
              </FormControl>
              
              {/* Max Tokens */}
              <FormControl>
                <HStack justify="space-between" mb={1}>
                  <FormLabel htmlFor="max-tokens-input" fontSize="sm" mb={0}>
                    Max Tokens
                  </FormLabel>
                  <Tooltip label="Maximum number of tokens (words/characters) in the response" placement="top">
                    <InfoIcon boxSize={3} color="gray.500" />
                  </Tooltip>
                </HStack>
                <NumberInput
                  id="max-tokens-input"
                  value={params.max_tokens}
                  onChange={(valueString) => {
                    const value = parseInt(valueString);
                    if (!isNaN(value)) {
                      handleParamChange('max_tokens', Math.max(1, Math.min(8192, value)));
                    }
                  }}
                  min={1}
                  max={8192}
                  step={1}
                  size="sm"
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </FormControl>
              
              {/* Top P */}
              <FormControl>
                <HStack justify="space-between" mb={1}>
                  <FormLabel htmlFor="top-p-slider" fontSize="sm" mb={0}>
                    Top P
                  </FormLabel>
                  <Tooltip label="Controls diversity: 0.1 means only consider tokens with the top 10% probability" placement="top">
                    <InfoIcon boxSize={3} color="gray.500" />
                  </Tooltip>
                </HStack>
                <HStack spacing={4}>
                  <Slider
                    id="top-p-slider"
                    min={0.1}
                    max={1}
                    step={0.05}
                    value={params.top_p}
                    onChange={(value) => handleParamChange('top_p', value)}
                    colorScheme="purple"
                    flex="1"
                  >
                    <SliderTrack>
                      <SliderFilledTrack bg="#4415b6" />
                    </SliderTrack>
                    <SliderThumb />
                  </Slider>
                  <NumberInput
                    value={params.top_p}
                    onChange={(valueString) => {
                      const value = parseFloat(valueString);
                      if (!isNaN(value)) {
                        handleParamChange('top_p', Math.max(0.1, Math.min(1, value)));
                      }
                    }}
                    step={0.05}
                    min={0.1}
                    max={1}
                    size="xs"
                    maxW="60px"
                  >
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </HStack>
              </FormControl>
              
              {/* Frequency Penalty */}
              <FormControl>
                <HStack justify="space-between" mb={1}>
                  <FormLabel htmlFor="freq-penalty-slider" fontSize="sm" mb={0}>
                    Frequency Penalty
                  </FormLabel>
                  <Tooltip label="Reduces repetition by penalizing tokens that have already appeared" placement="top">
                    <InfoIcon boxSize={3} color="gray.500" />
                  </Tooltip>
                </HStack>
                <HStack spacing={4}>
                  <Slider
                    id="freq-penalty-slider"
                    min={0}
                    max={2}
                    step={0.1}
                    value={params.frequency_penalty}
                    onChange={(value) => handleParamChange('frequency_penalty', value)}
                    colorScheme="purple"
                    flex="1"
                  >
                    <SliderTrack>
                      <SliderFilledTrack bg="#4415b6" />
                    </SliderTrack>
                    <SliderThumb />
                  </Slider>
                  <NumberInput
                    value={params.frequency_penalty}
                    onChange={(valueString) => {
                      const value = parseFloat(valueString);
                      if (!isNaN(value)) {
                        handleParamChange('frequency_penalty', Math.max(0, Math.min(2, value)));
                      }
                    }}
                    step={0.1}
                    min={0}
                    max={2}
                    size="xs"
                    maxW="60px"
                  >
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </HStack>
              </FormControl>
              
              {/* Presence Penalty */}
              <FormControl>
                <HStack justify="space-between" mb={1}>
                  <FormLabel htmlFor="presence-penalty-slider" fontSize="sm" mb={0}>
                    Presence Penalty
                  </FormLabel>
                  <Tooltip label="Encourages the model to talk about new topics" placement="top">
                    <InfoIcon boxSize={3} color="gray.500" />
                  </Tooltip>
                </HStack>
                <HStack spacing={4}>
                  <Slider
                    id="presence-penalty-slider"
                    min={0}
                    max={2}
                    step={0.1}
                    value={params.presence_penalty}
                    onChange={(value) => handleParamChange('presence_penalty', value)}
                    colorScheme="purple"
                    flex="1"
                  >
                    <SliderTrack>
                      <SliderFilledTrack bg="#4415b6" />
                    </SliderTrack>
                    <SliderThumb />
                  </Slider>
                  <NumberInput
                    value={params.presence_penalty}
                    onChange={(valueString) => {
                      const value = parseFloat(valueString);
                      if (!isNaN(value)) {
                        handleParamChange('presence_penalty', Math.max(0, Math.min(2, value)));
                      }
                    }}
                    step={0.1}
                    min={0}
                    max={2}
                    size="xs"
                    maxW="60px"
                  >
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </HStack>
              </FormControl>
            </VStack>
          </AccordionPanel>
        </AccordionItem>
      </Accordion>
    </Box>
  );
};

export default ModelParamsSelector; 