import React, { useEffect, useState } from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { 
  Box, 
  Flex, 
  Heading, 
  Text, 
  Button, 
  Grid, 
  GridItem,
  Card, 
  CardHeader, 
  CardBody, 
  CardFooter,
  Spinner,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  useColorModeValue
} from '@chakra-ui/react';
import { useAuth } from '../contexts/AuthContext';
import UserProfile from '../components/auth/UserProfile';

const DashboardPage = () => {
  const { currentUser } = useAuth();
  const [loading, setLoading] = useState(true);
  const [dashboardData, setDashboardData] = useState(null);
  
  // Chakra UI colors
  const accentColor = '#4415b6';
  const cardBg = useColorModeValue('white', 'gray.700');

  useEffect(() => {
    // Simulate fetching dashboard data
    const fetchData = async () => {
      try {
        // This would be replaced with a real API call
        setTimeout(() => {
          setDashboardData({
            documentCount: 5,
            recentDocuments: [],
            recentChats: []
          });
          setLoading(false);
        }, 800);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <Flex justify="center" align="center" height="100vh">
        <Spinner size="xl" color={accentColor} />
        <Text ml={4}>Loading dashboard...</Text>
      </Flex>
    );
  }

  return (
    <Box p={6} minH="100vh">
      <Box maxW="7xl" mx="auto">
        <Flex 
          direction={{ base: 'column', md: 'row' }} 
          justify="space-between" 
          align={{ base: 'flex-start', md: 'center' }}
          mb={8}
          gap={4}
        >
          <Box>
            <Heading as="h1" size="xl">Dashboard</Heading>
            <Text color="gray.500">
              Welcome back, {currentUser?.full_name || 'User'}!
            </Text>
          </Box>
          <Flex gap={2}>
            <Button as={RouterLink} to="/chat" variant="outline">
              Chat
            </Button>
            <Button 
              as={RouterLink} 
              to="/documents" 
              bg={accentColor}
              color="white"
              _hover={{ bg: '#3a1296' }}
            >
              View Documents
            </Button>
          </Flex>
        </Flex>

        <Grid 
          templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(3, 1fr)' }}
          gap={6}
        >
          <Card bg={cardBg} boxShadow="md" borderRadius="lg">
            <CardHeader>
              <Heading size="md">Documents</Heading>
              <Text fontSize="sm" color="gray.500">View your regulatory documents</Text>
            </CardHeader>
            <CardBody>
              <Stat>
                <StatNumber fontSize="3xl">{dashboardData?.documentCount || 0}</StatNumber>
                <StatHelpText>Total documents in your repository</StatHelpText>
              </Stat>
            </CardBody>
            <CardFooter>
              <Button 
                as={RouterLink} 
                to="/documents"
                variant="outline"
                width="full"
              >
                Browse Documents
              </Button>
            </CardFooter>
          </Card>

          <Card bg={cardBg} boxShadow="md" borderRadius="lg">
            <CardHeader>
              <Heading size="md">AI Chat</Heading>
              <Text fontSize="sm" color="gray.500">Chat with your regulatory AI assistant</Text>
            </CardHeader>
            <CardBody>
              <Text fontSize="sm" color="gray.500">
                Ask questions about your documents and get instant answers
              </Text>
            </CardBody>
            <CardFooter>
              <Button 
                as={RouterLink} 
                to="/chat"
                variant="outline"
                width="full"
              >
                Start Chatting
              </Button>
            </CardFooter>
          </Card>

          <Card bg={cardBg} boxShadow="md" borderRadius="lg">
            <CardHeader>
              <Heading size="md">User Profile</Heading>
              <Text fontSize="sm" color="gray.500">Your account information</Text>
            </CardHeader>
            <CardBody>
              <UserProfile />
            </CardBody>
          </Card>
        </Grid>
      </Box>
    </Box>
  );
};

export default DashboardPage; 