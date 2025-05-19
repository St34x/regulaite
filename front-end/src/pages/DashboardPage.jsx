import React, { useEffect, useState, useCallback } from 'react';
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
import { getDocumentStats } from '../services/documentService';

// Import our custom dashboard widgets
import ComplianceDonutChart from '../components/dashboard/ComplianceDonutChart';
import RiskTrendChart from '../components/dashboard/RiskTrendChart';
import FrameworkComplianceChart from '../components/dashboard/FrameworkComplianceChart';
import RiskRadarChart from '../components/dashboard/RiskRadarChart';
import RecentAlertsWidget from '../components/dashboard/RecentAlertsWidget';

const DashboardPage = () => {
  const { currentUser } = useAuth();
  const [loading, setLoading] = useState(true);
  const [dashboardData, setDashboardData] = useState(null);
  const [error, setError] = useState(null);
  
  // Chakra UI colors
  const accentColor = '#4415b6';
  const cardBg = useColorModeValue('white', 'gray.700');

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      // Get real document statistics from API
      const docStats = await getDocumentStats();
      
      setDashboardData({
        documentCount: docStats.total_documents,
        totalChunks: docStats.total_chunks,
        documentsByType: docStats.documents_by_type,
        documentsByLanguage: docStats.documents_by_language,
        recentUploads: docStats.recent_uploads,
        totalStorageMb: docStats.total_storage_mb
      });
      setError(null); // Clear any previous errors
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setError('Failed to load dashboard data. Please try again later.');
    } finally {
      setLoading(false);
    }
  }, []); // Empty dependency array because getDocumentStats doesn't depend on props/state here

  useEffect(() => {
    fetchData();

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        fetchData();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    // Cleanup listener on component unmount
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [fetchData]);

  if (loading && !dashboardData) { // Show loading only on initial load or if data is null
    return (
      <Flex justify="center" align="center" height="100vh">
        <Spinner size="xl" color={accentColor} />
        <Text ml={4}>Loading dashboard...</Text>
      </Flex>
    );
  }

  if (error) {
    return (
      <Flex justify="center" align="center" height="100vh" direction="column">
        <Text color="red.500" fontSize="lg" mb={4}>{error}</Text>
        <Button onClick={() => window.location.reload()} colorScheme="purple">
          Retry
        </Button>
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

        {/* Original widgets */}
        <Grid 
          templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(3, 1fr)' }}
          gap={6}
          mb={8}
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
              {dashboardData?.totalStorageMb > 0 && (
                <Stat mt={2}>
                  <StatNumber fontSize="xl">{dashboardData.totalStorageMb.toFixed(2)} MB</StatNumber>
                  <StatHelpText>Total storage used</StatHelpText>
                </Stat>
              )}
              {dashboardData?.totalChunks > 0 && (
                <Stat mt={2}>
                  <StatNumber fontSize="xl">{dashboardData.totalChunks}</StatNumber>
                  <StatHelpText>Total document chunks</StatHelpText>
                </Stat>
              )}
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

        {/* New visualization widgets */}
        <Heading as="h2" size="lg" mb={4}>Compliance & Risk Analytics</Heading>
        <Grid 
          templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(3, 1fr)' }}
          gap={6}
          mb={8}
        >
          <GridItem colSpan={{ base: 1, lg: 1 }}>
            <ComplianceDonutChart />
          </GridItem>
          <GridItem colSpan={{ base: 1, lg: 2 }}>
            <RiskTrendChart />
          </GridItem>
          <GridItem colSpan={{ base: 1, lg: 2 }}>
            <FrameworkComplianceChart />
          </GridItem>
          <GridItem colSpan={{ base: 1, lg: 1 }}>
            <RecentAlertsWidget />
          </GridItem>
        </Grid>
        
        {/* Additional widget section */}
        <Heading as="h2" size="lg" mb={4}>Risk Assessment</Heading>
        <Grid 
          templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(3, 1fr)' }}
          gap={6}
        >
          <GridItem colSpan={{ base: 1, lg: 3 }}>
            <RiskRadarChart />
          </GridItem>
        </Grid>
      </Box>
    </Box>
  );
};

export default DashboardPage; 