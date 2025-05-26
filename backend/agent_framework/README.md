# Système d'Agents RegulAIte

## Vue d'ensemble

Le système d'agents RegulAIte implémente une architecture multi-agents pour les analyses GRC (Gouvernance, Risque, Conformité) en français. Le système utilise un agent orchestrateur principal qui coordonne des agents spécialisés selon les besoins de chaque requête.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Interface Utilisateur                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              API Endpoints (/agents/*)                 │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                   Couche d'Orchestration                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               OrchestratorAgent                         │    │
│  │  • Analyse de requêtes  • Planification  • Synthèse    │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                     Agents Spécialisés                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │    Risk     │  │  Compliance  │  │    Governance          │  │
│  │ Assessment  │  │   Analysis   │  │    Analysis            │  │
│  │   Agent     │  │    Agent     │  │     Agent              │  │
│  └─────────────┘  └──────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                    Outils Universels                            │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  Document   │  │   Entity     │  │   Cross-Reference      │  │
│  │   Finder    │  │  Extractor   │  │       Tool             │  │
│  └─────────────┘  └──────────────┘  └────────────────────────┘  │
│                         ┌──────────────┐                        │
│                         │   Temporal   │                        │
│                         │   Analyzer   │                        │
│                         └──────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
                                 │
┌─────────────────────────────────────────────────────────────────┐
│                  Couche d'Intégration                           │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  LLM Client │  │  RAG System  │  │     Base de            │  │
│  │  (GPT-4.1)  │  │  (HyPERAG)   │  │   Connaissances        │  │
│  └─────────────┘  └──────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Composants Intégrés

### 1. Agent Orchestrateur (`OrchestratorAgent`)

**Fichier**: `orchestrator.py`

L'orchestrateur principal qui :
- Analyse les requêtes utilisateur en français
- Détermine quels agents spécialisés mobiliser
- Coordonne l'exécution séquentielle ou parallèle
- Synthétise les résultats finaux

**Capacités** :
- Analyse intelligente des requêtes GRC
- Planification d'exécution multi-agents
- Gestion de la collaboration entre agents
- Synthèse de résultats cohérents

### 2. Client LLM (`LLMClient`)

**Fichier**: `llm_integration.py`

Client optimisé pour GPT-4.1 avec :
- Prompts système spécialisés par domaine GRC
- Support natif du français
- Méthodes d'analyse structurée
- Fonctions spécialisées (évaluation risque, analyse conformité)

**Agents supportés** :
- `risk_assessment` : Expert EBIOS/MEHARI
- `compliance_analysis` : Expert RGPD/ISO27001/DORA  
- `governance_analysis` : Expert gouvernance IT
- `document_finder` : Expert recherche documentaire GRC

### 3. Document Finder (`DocumentFinder`)

**Fichier**: `tools/document_finder.py`

Outil de recherche intelligente qui :
- Classifie automatiquement les documents GRC
- Recherche sémantique + métadonnées
- Détection de frameworks (ISO27001, RGPD, DORA)
- Relations entre documents

**Types de documents supportés** :
- Politiques, Procédures, Instructions
- Audits, Évaluations de risque
- Rapports de conformité, Pentests
- Cartographies, Plans d'action

## Intégration dans RegulAIte

### Endpoints API

#### 1. Orchestration Principal
```
POST /agents/orchestrator/execute
{
    "query": "Analyser les risques dans les nouvelles politiques IT",
    "session_id": "optional",
    "include_details": true
}
```

#### 2. Exécution Agent Spécifique
```
POST /agents/execute
{
    "agent_type": "orchestrator|risk_assessment|compliance_analysis|governance_analysis",
    "query": "...",
    "parameters": {...}
}
```

#### 3. Test Document Finder
```
GET /agents/tools/document-finder?query=politique%20sécurité&frameworks=ISO27001,RGPD
```

#### 4. Test Système
```
GET /api/agents/test
```

### Initialisation

Le système s'initialise automatiquement au démarrage de l'application :

1. **Startup Event** (`main.py`) :
   ```python
   init_agent_system()  # Prépare les composants
   ```

2. **Initialisation Lazy** :
   - L'orchestrateur se crée au premier appel
   - Les agents spécialisés s'enregistrent automatiquement
   - Le Document Finder s'intègre au système RAG

### Configuration

Variables d'environnement requises :
```bash
OPENAI_API_KEY=your_api_key_here
QDRANT_URL=http://qdrant:6333
```

## Cas d'Usage Typiques

### 1. Analyse de Risque
```
Requête : "Identifier les risques cybersécurité dans notre nouvelle architecture cloud"

Flux :
1. Orchestrateur → analyse la requête
2. Mobilise risk_assessment + document_finder
3. Document Finder → trouve documents pertinents
4. Risk Assessment → analyse selon EBIOS
5. Orchestrateur → synthèse finale
```

### 2. Vérification de Conformité
```
Requête : "Vérifier notre conformité RGPD pour le traitement des données clients"

Flux :
1. Orchestrateur → détecte framework RGPD
2. Mobilise compliance_analysis + document_finder
3. Document Finder → politiques/procédures RGPD
4. Compliance Analysis → gap analysis
5. Orchestrateur → rapport de conformité
```

### 3. Analyse Multi-Framework
```
Requête : "Préparer un rapport ISO 27001 en tenant compte de DORA"

Flux :
1. Orchestrateur → détecte multi-framework
2. Mobilise compliance_analysis + governance_analysis
3. Agents collaborent pour mapping croisé
4. Orchestrateur → rapport consolidé
```

## Tests d'Intégration

### Script de Test
```bash
cd backend
python test_agent_integration.py
```

### Tests Inclus
- ✅ Client LLM (GPT-4.1)
- ✅ Document Finder (classification + recherche)
- ✅ Orchestrateur (analyse + planification)
- ✅ Système complet (end-to-end)

## Prochaines Étapes

### Agents Spécialisés à Implémenter
1. **RiskAssessmentAgent** - Méthodologies EBIOS/MEHARI
2. **ComplianceAnalysisAgent** - RGPD/ISO27001/DORA
3. **GovernanceAnalysisAgent** - Gouvernance IT

### Outils Universels à Compléter
1. **EntityExtractor** - Extraction entités GRC
2. **CrossReferenceTool** - Relations documents/contrôles
3. **TemporalAnalyzer** - Analyse temporelle/tendances

### Améliorations
- Cache intelligent des résultats d'agents
- Métriques de performance par agent
- Interface de debugging/monitoring
- Support multi-sessions pour continuité

## Support et Maintenance

- **Logs** : Tous les composants loggent dans le système centralisé
- **Monitoring** : Endpoint `/api/status` inclut statut agents
- **Debugging** : Endpoint `/api/agents/test` pour validation
- **Performance** : Métriques d'exécution par agent 

## Language Support

The agent framework provides multilingual support with enhanced language detection for French, Spanish, and English queries.

### Language Detection

The system uses an improved keyword-based language detection algorithm that:

- **Enhanced French Detection**: Includes common French words like "bonjour", "peux", "tu", "me", "tes", "présenter", etc.
- **Comprehensive Coverage**: Covers articles, pronouns, verbs, and domain-specific terms
- **Accurate Scoring**: Uses weighted scoring to determine the most likely language
- **Fallback Support**: Defaults to English when language cannot be determined

### Example Language Detection

```python
from agent_framework.integrations.llm_integration import detect_language

# French detection
detect_language("Bonjour, peux-tu me présenter tes capacités ?")  # Returns: 'fr'
detect_language("Comment analyser les risques de conformité ?")   # Returns: 'fr'

# English detection  
detect_language("What are your capabilities?")                    # Returns: 'en'
detect_language("Show me the compliance analysis")                # Returns: 'en'

# Spanish detection
detect_language("Hola, ¿cómo estás?")                            # Returns: 'es'
```

### Language-Specific Instructions

The LLM integration automatically adds appropriate language instructions based on detected language:

- **French**: Instructions to always respond in French
- **Spanish**: Instructions to always respond in Spanish  
- **English**: Instructions to respond in English

This ensures consistent language usage throughout the conversation regardless of source document language. 