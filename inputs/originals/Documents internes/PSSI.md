# Politique de Sécurité des Systèmes d'Information (PSSI)
# NEO FINANCIA

## Document final pour mise en œuvre

---

## Table des matières

1. [Introduction et contexte](#1-introduction-et-contexte)
2. [Cadre réglementaire et conformité](#2-cadre-réglementaire-et-conformité)
3. [Gouvernance et organisation de la sécurité](#3-gouvernance-et-organisation-de-la-sécurité)
4. [Gestion des risques cyber](#4-gestion-des-risques-cyber)
5. [Protection des données](#5-protection-des-données)
6. [Sécurité des infrastructures cloud](#6-sécurité-des-infrastructures-cloud)
7. [Architecture de sécurité zero-trust](#7-architecture-de-sécurité-zero-trust)
8. [Sécurité applicative et DevSecOps](#8-sécurité-applicative-et-devsecops)
9. [Sécurité des canaux digitaux](#9-sécurité-des-canaux-digitaux)
10. [Gestion des tiers et de la chaîne d'approvisionnement](#10-gestion-des-tiers-et-de-la-chaîne-dapprovisionnement)
11. [Détection et réponse aux incidents](#11-détection-et-réponse-aux-incidents)
12. [Continuité d'activité et résilience opérationnelle](#12-continuité-dactivité-et-résilience-opérationnelle)
13. [Sensibilisation et formation](#13-sensibilisation-et-formation)
14. [Assurance et vérification](#14-assurance-et-vérification)
15. [Innovation et technologies émergentes](#15-innovation-et-technologies-émergentes)
16. [Annexes](#16-annexes)

---

## 1. Introduction et contexte

### 1.1 Présentation de Neo Financia

Neo Financia est une néobanque européenne offrant des services bancaires innovants à travers les canaux numériques. Fondée sur un modèle digital-first, l'institution sert 2 millions de clients avec une équipe de 1000 collaborateurs répartis entre son siège social à Paris et ses bureaux de Lyon et Londres.

La banque propose une gamme complète de services financiers incluant:
- Crédits consommateurs et immobiliers
- Services aux professionnels
- Comptes courants personnels et professionnels
- Produits d'épargne (Livret A, PEA, PEA-PME)
- Services bancaires digitaux via web et mobile

### 1.2 Vision et stratégie de cybersécurité

Neo Financia place la sécurité et la confiance au cœur de sa proposition de valeur. La vision de cybersécurité s'articule autour de quatre piliers:

1. **Protection proactive**: Anticiper et contrer les menaces avant qu'elles n'affectent nos services
2. **Sécurité by design**: Intégrer la sécurité dès la conception de nos produits et services
3. **Confiance digitale**: Garantir à nos clients la protection de leurs données et actifs
4. **Résilience opérationnelle**: Maintenir la continuité de service même en cas d'incidents

### 1.3 Enjeux spécifiques au modèle néobanque européenne

Le modèle de néobanque européenne expose Neo Financia à des défis particuliers:

- Dépendance vitale aux infrastructures technologiques et digitales
- Complexité réglementaire liée à la présence multi-juridictionnelle (France, UE, Royaume-Uni post-Brexit)
- Attentes élevées des clients en matière de disponibilité (24/7) et d'expérience utilisateur fluide
- Surface d'attaque élargie par la multiplication des canaux d'accès et APIs
- Rivalité avec les acteurs traditionnels nécessitant d'innover sans compromettre la sécurité
- Gestion des risques transfrontaliers, particulièrement entre l'UE et le Royaume-Uni

### 1.4 Approche d'amélioration continue de la maturité digitale

Avec une maturité digitale actuelle de 3.3/5, Neo Financia vise à atteindre un niveau 4.5/5 d'ici 2027. Cette progression s'appuiera sur:

- L'adoption progressive des technologies cloud-natives sécurisées
- Le renforcement continu des compétences cyber de l'ensemble des collaborateurs
- L'automatisation croissante des processus de sécurité
- L'enrichissement des capacités d'analyse de données pour la détection des menaces
- L'alignement sur les standards internationaux de sécurité (ISO 27001, PCI-DSS)

---

## 2. Cadre réglementaire et conformité

### 2.1 Cartographie des exigences légales et réglementaires

#### 2.1.1 Cadre européen

- **RGPD (2016/679)**: Protection des données personnelles des clients et salariés
- **Directive NIS2**: Tests d'intrusion obligatoires tous les six mois, partage des IoC avec l'ENISA
- **DORA (Digital Operational Resilience Act)**: Gestion des risques liés aux tiers, tests de résilience
- **PSD2**: Authentification forte (SCA), communication sécurisée, gestion des fraudes
- **eIDAS2**: Identité numérique et signatures électroniques
- **MiFID II**: Transparence pré-négociation, séparation des coûts de recherche, enregistrement des algorithmes
- **Bâle III révisé**: Ratio de fonds propres CET1 à 4,5% minimum, exigences de liquidité

#### 2.1.2 Cadre français

- **Code monétaire et financier**: Obligations spécifiques des établissements de crédit
- **LPM (Loi de Programmation Militaire)**: Protection des infrastructures critiques
- **Loi Informatique et Libertés**: Compléments nationaux au RGPD
- **Exigences ACPR**: Audits de liquidité trimestriels pour les banques d'importance systémique
- **ANSSI**: Certification SecNumCloud pour les infrastructures cloud hébergeant des données sensibles
- **Régime OIV**: Obligation de recourir à des sociétés de cybersécurité agréées pour les audits

#### 2.1.3 Royaume-Uni post-Brexit

- **UK GDPR**: Cadre similaire au RGPD avec assouplissements spécifiques
- **UK NIS Regulations**: Reporting d'incidents raccourci à 48h (contre 72h dans l'UE)
- **FCA/PRA Requirements**: Ratio de levier à 2,5% pour stimuler le crédit aux PME
- **Programme de cyber-résilience certifiée**: Avantages fiscaux liés à la certification

### 2.2 Matrice de correspondance entre contrôles PSSI et exigences réglementaires

| Domaine PSSI | RGPD | NIS2 | DORA | PSD2 | MiFID II | Bâle III | ANSSI | UK GDPR |
|--------------|------|------|------|------|----------|----------|-------|---------|
| Protection des données | ● | ◐ | ◐ | ● | ○ | ○ | ◐ | ● |
| Continuité d'activité | ○ | ● | ● | ◐ | ○ | ◐ | ● | ○ |
| Sécurité infrastructure | ◐ | ● | ● | ◐ | ○ | ○ | ● | ◐ |
| Gestion des accès | ● | ◐ | ◐ | ● | ◐ | ○ | ◐ | ● |
| Sécurité applicative | ◐ | ◐ | ● | ● | ◐ | ○ | ● | ◐ |
| Gestion des incidents | ● | ● | ● | ● | ◐ | ○ | ● | ● |
| Gestion des tiers | ● | ◐ | ● | ● | ◐ | ○ | ◐ | ● |

Légende: ● Exigence forte | ◐ Exigence moyenne | ○ Exigence faible ou indirecte

### 2.3 Processus de veille réglementaire et mise à jour

Neo Financia maintiendra une veille réglementaire continue par:
- Une équipe dédiée à la conformité réglementaire
- L'abonnement à des services spécialisés de veille juridique
- La participation active aux groupes de travail sectoriels (FBF, AFNOR, UK Finance)
- Des revues trimestrielles des évolutions réglementaires impactant la sécurité
- Une mise à jour annuelle de la PSSI pour intégrer les nouvelles exigences

### 2.4 Gestion de la conformité multi-juridictionnelle

Pour gérer les exigences multiples et parfois divergentes:
- Application d'un "principe du plus contraignant" entre les juridictions UE et UK
- Documentation des exceptions juridiques nécessaires par territoire
- Cartographie des flux de données transfrontaliers
- Processus d'évaluation d'impact pour les nouveaux projets multi-juridictionnels
- Mécanismes de cloisonnement pour la ségrégation des données selon les exigences territoriales

---

## 3. Gouvernance et organisation de la sécurité

### 3.1 Structure organisationnelle et comitologie

#### 3.1.1 Organisation de la sécurité

```
Conseil d'Administration
        ↓
    Comité Risques
        ↓
Comité Exécutif (COMEX)
        ↓
Comité de Sécurité (COSEC)
    ↙       ↓       ↘
RSSI    DSI     Conformité
```

Le RSSI est rattaché directement au Directeur des Risques, membre du COMEX.

#### 3.1.2 Comitologie de sécurité

| Instance | Fréquence | Participants | Objectifs |
|----------|-----------|--------------|-----------|
| Conseil d'Administration | Annuelle | CA, DG, DR, RSSI | Validation stratégie cyber, appétence au risque |
| Comité Risques | Trimestrielle | Membres CA, DG, DR, RSSI | Supervision risques cyber, validation investissements majeurs |
| COMEX Cyber | Trimestrielle | COMEX, RSSI | Pilotage exécutif, arbitrages stratégiques |
| COSEC | Mensuelle | RSSI, DSI, métiers, conformité | Suivi opérationnel, incidents, projets |
| Comité Crise Cyber | Ad hoc | Cellule de crise prédéfinie | Gestion des incidents majeurs |

#### 3.1.3 Équipe cybersécurité

Équipe centralisée de 30 collaborateurs (ratio 3% des effectifs) organisée en pôles:
- Gouvernance, risques et conformité (GRC)
- Architecture et conseil sécurité
- Sécurité opérationnelle (SecOps)
- SOC et réponse aux incidents
- Tests et assurance sécurité

### 3.2 Rôles et responsabilités (RACI)

| Fonction | Définition PSSI | Architecture sécurité | Gestion incidents | Conformité réglementaire | Contrôles sécurité |
|----------|----------------|---------------------|-------------------|------------------------|------------------|
| Conseil Administration | A | I | I | A | I |
| Direction Générale | A | A | I | A | A |
| RSSI | R | A | A | R | A |
| DSI | C | R | R | C | R |
| Direction Conformité | C | I | C | R | C |
| Responsables métiers | C | C | C | C | R |
| Équipe cybersécurité | R | R | R | R | R |
| Collaborateurs | I | I | C | I | R |

Légende: R = Responsible, A = Accountable, C = Consulted, I = Informed

### 3.3 Séparation des responsabilités et délégations d'autorité

- Principe de séparation des tâches pour toutes les opérations sensibles
- Double validation obligatoire pour les opérations à haut risque
- Procédure de délégation d'autorité formalisée et auditée
- Revue semestrielle des droits administratifs
- Limitation des accès privilégiés selon le principe du moindre privilège

### 3.4 Interface entre équipe sécurité et autres départements

- Nomination de correspondants sécurité dans chaque direction
- Participation du RSSI aux comités de projets stratégiques
- Processus d'escalade direct RSSI → DG pour les risques critiques
- Intégration de KPIs sécurité dans les objectifs des dirigeants
- Revue sécurité obligatoire dans les processus de transformation

### 3.5 Modèle d'organisation pour les entités internationales

- Fonction cybersécurité centralisée avec harmonisation des processus
- Relais locaux au Royaume-Uni pour assurer la conformité post-Brexit
- Autonomie encadrée pour les adaptations nécessaires aux contextes locaux
- Reporting consolidé et centralisation des incidents majeurs
- Audits croisés entre les entités pour garantir la cohérence de l'approche sécurité

---

## 4. Gestion des risques cyber

### 4.1 Méthodologie d'évaluation des risques

Neo Financia adopte la méthodologie EBIOS Risk Manager 2023, enrichie de composantes spécifiques au secteur bancaire digital:

1. **Identification des actifs numériques critiques**:
   - Infrastructure de compte et paiement
   - Plateformes d'API et d'Open Banking
   - Systèmes d'authentification client
   - Applications mobiles et web
   - Infrastructure cloud

2. **Analyse des menaces et vulnérabilités**:
   - Veille sur les cybermenaces ciblant le secteur financier
   - Analyse de l'exposition spécifique aux néobanques
   - Évaluation des vulnérabilités techniques et organisationnelles

3. **Évaluation de l'impact et de la vraisemblance**:
   - Échelle d'impact 1-5 (financier, réputationnel, opérationnel, réglementaire)
   - Échelle de vraisemblance 1-5 basée sur l'historique sectoriel et les capacités des attaquants
   - Matrice de criticité résultante

4. **Stratégies de traitement des risques**:
   - Éviter: renoncer à certaines activités trop risquées
   - Réduire: déployer des contrôles de sécurité
   - Transférer: assurance cyber, externalisation contrôlée
   - Accepter: documentation formelle et validation par la gouvernance

### 4.2 Cartographie des risques prioritaires

| ID | Scénario de risque | Impact | Vraisemblance | Criticité | Traitement |
|----|-------------------|--------|--------------|-----------|------------|
| R1 | Compromission des API Open Banking | 5 | 4 | 20 (Critique) | Réduire |
| R2 | Exfiltration massive de données client | 5 | 3 | 15 (Élevé) | Réduire |
| R3 | Fraude par détournement des systèmes de paiement | 4 | 4 | 16 (Élevé) | Réduire |
| R4 | Indisponibilité majeure des services bancaires en ligne | 4 | 3 | 12 (Élevé) | Réduire + Transférer |
| R5 | Attaque de la chaîne d'approvisionnement via fournisseurs | 4 | 3 | 12 (Élevé) | Réduire |
| R6 | Non-conformité réglementaire (DORA, NIS2, RGPD) | 5 | 2 | 10 (Modéré) | Réduire |
| R7 | Compromission des postes de travail des collaborateurs | 3 | 4 | 12 (Élevé) | Réduire |
| R8 | Attaque ciblée par phishing sur les dirigeants | 4 | 4 | 16 (Élevé) | Réduire |
| R9 | Vol d'identifiants d'authentification des clients | 4 | 4 | 16 (Élevé) | Réduire |
| R10 | Ransomware touchant l'infrastructure critique | 5 | 3 | 15 (Élevé) | Réduire + Transférer |

### 4.3 Critères d'acceptation des risques et appétence

Neo Financia définit son appétence au risque selon les principes suivants:

- **Risques critiques (16-25)**: Aucune acceptation sans validation du Conseil d'Administration et mesures compensatoires robustes
- **Risques élevés (10-15)**: Acceptation possible par le COMEX avec plan de traitement documenté
- **Risques modérés (5-9)**: Acceptation possible par le COSEC avec surveillance continue
- **Risques faibles (1-4)**: Acceptation possible par le RSSI avec revue annuelle

La tolérance au risque résiduel est conditionnée par:
- L'absence d'impact direct sur la sécurité des fonds des clients
- Le maintien des objectifs de RTO/RPO (<15 minutes)
- La conformité réglementaire
- La protection des données sensibles des clients

### 4.4 Processus de revue et mise à jour

Le cycle de gestion des risques s'articule comme suit:
- Revue complète annuelle de la cartographie des risques
- Mise à jour trimestrielle des scénarios prioritaires
- Évaluation des risques pour tout nouveau projet significatif
- Réévaluation systématique après chaque incident majeur
- Alignement avec les rapports d'audit et de tests de sécurité

---

## 5. Protection des données

### 5.1 Gouvernance des données personnelles et financières

#### 5.1.1 Organisation et responsabilités

- DPO désigné et rattaché à la Direction Juridique, avec lien fonctionnel au RSSI
- Comité de protection des données trimestriel
- Registre centralisé des traitements de données
- Propriétaires de données désignés pour chaque catégorie d'information
- Programme de Privacy by Design intégré au cycle de développement

#### 5.1.2 Classification des données

| Niveau | Description | Exemples | Contrôles requis |
|--------|-------------|----------|-----------------|
| C3 - Critique | Données hautement sensibles | Données d'authentification, clés cryptographiques | Chiffrement fort, accès très restreint, traçabilité complète |
| C2 - Confidentiel | Données confidentielles | Transactions financières, données personnelles sensibles | Chiffrement, accès restreint, traçabilité |
| C1 - Interne | Usage interne uniquement | Procédures, communications internes | Contrôle d'accès, pas de diffusion externe |
| C0 - Public | Information publique | Brochures, conditions générales | Aucune restriction particulière |

### 5.2 Mesures techniques de protection

#### 5.2.1 Chiffrement des données

- **Données au repos**: Chiffrement AES-256 pour toutes les données C2 et C3
- **Données en transit**: TLS 1.3 obligatoire, certificats validés par autorité reconnue
- **Bases de données**: Chiffrement transparent, gestion des clés séparée
- **Terminaux et supports**: Chiffrement intégral des postes de travail et appareils mobiles
- **Clés cryptographiques**: Gestion des clés via HSM et solution de key management dédiée

#### 5.2.2 Tokenisation et anonymisation

- Tokenisation des données de paiement selon PCI-DSS
- Pseudonymisation systématique des données clients pour les environnements de test
- Anonymisation irréversible pour les analyses statistiques
- Techniques d'anonymisation différentielle pour les cas d'usage analytiques
- Validation périodique des techniques d'anonymisation

### 5.3 Contrôles de prévention des fuites de données (DLP)

- Solution DLP déployée sur l'ensemble du périmètre Neo Financia:
  - Points terminaux (endpoints): contrôle des transferts locaux
  - Réseau: analyse des flux sortants
  - Cloud: contrôle des données partagées via SaaS
  - Email: filtrage des pièces jointes et contenus sensibles
- Classification automatique des documents
- Blogage des transferts non autorisés de données sensibles
- Alertes en temps réel pour les tentatives d'exfiltration
- Analyse comportementale pour détecter les anomalies

### 5.4 Gestion des droits des personnes concernées

Processus formalisés pour répondre aux demandes d'exercice des droits:
- Droit d'accès: 30 jours maximum, vérification d'identité renforcée
- Droit de rectification: processus de validation et traçabilité des modifications
- Droit à l'effacement: procédure d'effacement sécurisé, avec preuve d'exécution
- Droit à la portabilité: format structuré, transmissible de façon sécurisée
- Droit d'opposition au profilage: mécanisme de désactivation immédiate

### 5.5 Transferts transfrontaliers France-UK post-Brexit

- Cartographie précise des flux de données entre l'UE et le Royaume-Uni
- Clauses contractuelles types (CCT) mises à jour selon les recommandations de la CNIL
- Évaluation d'impact pour chaque catégorie de transfert
- Mesures techniques supplémentaires (chiffrement, anonymisation) pour les données sensibles
- Processus d'autorisation préalable pour tout nouveau transfert
- Suivi des évolutions réglementaires impactant l'adéquation UE-UK

### 5.6 Gestion des incidents impliquant des données personnelles

- Procédure spécifique pour les violations de données personnelles
- Critères d'évaluation de la gravité et de la notification
- Délai de notification à la CNIL (<72h) et à l'ICO UK (<48h)
- Modèles de communication aux personnes concernées
- Processus de documentation et retour d'expérience
- Simulations régulières d'incidents de données

---

## 6. Sécurité des infrastructures cloud

### 6.1 Stratégie multi-cloud et cloud souverain

Neo Financia adopte une stratégie hybride combinant:
- **Cloud public**: AWS et Microsoft Azure pour la flexibilité et l'innovation
- **Cloud souverain**: OVHcloud qualifié SecNumCloud pour les données les plus sensibles
- **Infrastructure privée**: Datacenters propres pour certains systèmes critiques

La répartition des charges suit les principes:
- Données critiques et réglementées: prioritairement sur cloud souverain
- Services exposés externes: architecture multi-cloud avec répartition des risques
- Applications internes: selon classification de sensibilité et besoins techniques

### 6.2 Exigences pour les environnements PaaS/IaaS/SaaS

#### 6.2.1 Infrastructure as a Service (IaaS)

- Isolation réseau complète (VPC/VNET) avec chiffrement des communications
- Hardening systématique des instances selon CIS Benchmarks niveau 2
- Gestion des accès privilégiés via bastion hosts et session recording
- Isolation stricte des environnements de production, préproduction et développement
- Automatisation complète du déploiement (Infrastructure as Code)

#### 6.2.2 Platform as a Service (PaaS)

- Sécurisation des connexions aux services managés
- Vérification des mécanismes d'authentification et d'autorisation
- Chiffrement des données au repos sur tous les services PaaS
- Validation des paramètres de sécurité par défaut
- Monitoring renforcé des accès et opérations

#### 6.2.3 Software as a Service (SaaS)

- Due diligence sécurité et conformité préalable à l'adoption
- Intégration avec le SSO d'entreprise et MFA obligatoire
- Vérification de la localisation des données
- Limitation des droits administrateurs et séparation des rôles
- Contrôle des partages externes et prévention de fuites de données

### 6.3 Sécurité des conteneurs et OpenShift

- Application du principe "security by default" pour tous les conteneurs
- Scan automatisé des images avant déploiement
- Isolation réseau au niveau des pods et namespaces
- Gestion sécurisée des secrets via Vault intégré
- Configuration renforcée du plan de contrôle OpenShift
- Surveillance des comportements anormaux des conteneurs
- Mise à jour rapide (<72h) des composants vulnérables

### 6.4 Gestion des identités et des accès dans le cloud

- Solution IAM centralisée avec fédération d'identité
- Authentification multifacteur obligatoire pour tous les accès
- Attribution des privilèges selon le principe du moindre privilège
- Rotation automatique des clés d'accès et credentials
- Revue mensuelle des droits d'accès aux environnements cloud
- Gestion du cycle de vie des comptes de service
- Isolation des droits entre environnements et projets

### 6.5 Cloud Security Posture Management (CSPM)

Neo Financia déploie une solution CSPM complète pour:
- Scanner continuellement les configurations cloud
- Détecter les écarts par rapport aux bonnes pratiques
- Vérifier la conformité réglementaire spécifique au secteur financier
- Assurer l'homogénéité des contrôles entre fournisseurs
- Générer des tableaux de bord de conformité pour le COSEC
- Remédier automatiquement aux écarts mineurs de configuration

### 6.6 Surveillance et détection des menaces cloud

- Centralisation des logs cloud dans un SIEM
- Détection des comportements anormaux (accès inhabituels, élévation de privilèges)
- Surveillance des API et configurations exposées
- Détection des tentatives d'exfiltration de données
- Alertes en temps réel pour les événements critiques
- Corrélation entre signaux provenant de différents clouds
- Tests d'intrusion semestriels des environnements cloud

---

## 7. Architecture de sécurité zero-trust

### 7.1 Principes d'architecture sécurisée pour une néobanque

Neo Financia adopte une approche Zero Trust basée sur les principes:
- "Ne jamais faire confiance, toujours vérifier"
- Vérification systématique de chaque accès
- Restriction des accès au strict nécessaire
- Moindre privilège par défaut
- Inspection et journalisation exhaustives
- Segmentation fine des ressources

L'architecture de sécurité s'articule autour de trois axes:
1. **Sécurisation de l'identité**: MFA, gestion des accès privilégiés, authentification contextuelle
2. **Sécurisation des données**: chiffrement, contrôle d'accès granulaire, DLP
3. **Sécurisation des applications**: micro-segmentation, sécurité applicative, API Gateway sécurisée

### 7.2 Segmentation réseau et micro-segmentation

#### 7.2.1 Zones de sécurité

| Zone | Description | Niveau de protection |
|------|-------------|----------------------|
| Z0 | Services critiques financiers | Isolation maximale, contrôles renforcés |
| Z1 | Données clients et applications métier | Haute sécurité, accès contrôlé |
| Z2 | Services internes | Sécurité standard, accès authentifié |
| Z3 | DMZ et services exposés | Protection périmétrique avancée |
| Z4 | Postes de travail | Protection endpoint, isolation du réseau interne |

#### 7.2.2 Principes de micro-segmentation

- Segmentation basée sur les identités et contextes plutôt que sur les adresses IP
- Politique par défaut de refus ("default deny")
- Règles de trafic explicites pour chaque application
- Inspection du trafic Est-Ouest entre micro-segments
- Adaptation dynamique des contrôles selon le niveau de risque
- Visibilité totale sur les communications entre services

### 7.3 Authentification et autorisation continues

- Évaluation de risque en temps réel pour chaque requête
- Authentification adaptative selon le contexte (localisation, appareil, comportement)
- MFA renforcé pour les opérations sensibles
- Révocation immédiate des sessions suspectées compromises
- Token à courte durée de vie (15 minutes maximum)
- Validation continue des attributs durant toute la session
- Contrôles comportementaux (UEBA) intégrés au processus d'autorisation

### 7.4 Contrôles de sécurité immuables et vérifiables

- Infrastructure as Code avec validation automatisée des configurations
- Tests de conformité intégrés au pipeline CI/CD
- Chiffrement des configurations sensibles
- Rollback automatique en cas de détection d'anomalies
- Processus de validation en quatre yeux pour les changements critiques
- Vérification de l'intégrité des composants de l'infrastructure
- Traçabilité complète des modifications via journalisation immuable

### 7.5 Gestion des identités et des accès pour 1000 collaborateurs

- Système IAM centralisé avec provisioning automatisé
- Authentification multifacteur obligatoire pour tous les collaborateurs
- Gestion des droits basée sur les rôles (RBAC) et attributs (ABAC)
- Processus formel de revue des accès trimestrielle
- Déprovisionnement automatique lors des départs
- Contrôle des élévations de privilèges via workflows d'approbation
- Monitoring des comportements anormaux d'authentification

### 7.6 Sécurité des endpoints et postes de travail

- Solution EDR (Endpoint Detection and Response) sur tous les postes
- Chiffrement intégral des disques
- Gestion centralisée des configurations et mises à jour
- Contrôle applicatif (whitelisting)
- Protection contre les menaces avancées (sandboxing)
- Segmentation locale sur les postes (micro-virtualisation)
- Détection des comportements anormaux
- Protection contre les vulnérabilités non patchées (virtual patching)

---

## 8. Sécurité applicative et DevSecOps

### 8.1 Cycle de vie de développement sécurisé

Neo Financia intègre la sécurité à toutes les étapes du développement selon la méthodologie Secure SDLC:

#### 8.1.1 Phase de conception
- Analyse de risque sécurité préliminaire
- Modélisation des menaces (STRIDE/DREAD)
- Définition des exigences de sécurité
- Revue d'architecture sécurité

#### 8.1.2 Phase de développement
- Formation continue des développeurs à la sécurité
- Utilisation de bibliothèques sécurisées pré-approuvées
- Peer-reviews de code avec focus sécurité
- Tests unitaires incluant des scénarios d'attaque

#### 8.1.3 Phase de test
- Tests d'intrusion manuels pour applications critiques
- Tests de charge et fuzzing
- Vérification des recommandations OWASP Top 10
- Simulation d'attaques sur les environnements de préproduction

#### 8.1.4 Phase de déploiement
- Validation des configurations de sécurité
- Déploiement via pipeline sécurisé
- Vérification d'intégrité des composants
- Surveillance renforcée post-déploiement

#### 8.1.5 Phase de maintenance
- Gestion des vulnérabilités
- Patch management accéléré
- Révision périodique de la sécurité
- Amélioration continue basée sur les retours du SOC

### 8.2 Sécurité dans le pipeline CI/CD

Neo Financia intègre les outils de sécurité suivants dans son pipeline DevOps:

| Étape | Outils | Contrôles |
|-------|--------|-----------|
| Commit | Pre-commit hooks, Code linters | Détection de secrets, conformité au style de code sécurisé |
| Build | SAST, SCA | Analyse de code statique, scan des dépendances |
| Test | DAST, IAST | Tests de sécurité dynamiques, fuzzing |
| Deploy | CSPM, Secret scanning | Validation des configurations cloud, détection des secrets exposés |
| Runtime | RASP, WAF | Protection applicative en temps réel, blocage des attaques |

Exigences spécifiques:
- Bloquage du pipeline en cas de vulnérabilité critique (CVSS ≥ 9.0)
- Approbation requise pour les vulnérabilités hautes (CVSS 7.0-8.9)
- Documentation obligatoire pour les exceptions temporaires
- Scan complet de l'image finale avant déploiement
- Traçabilité des validations de sécurité

### 8.3 Tests de sécurité automatisés

- Tests unitaires de sécurité pour les fonctions critiques
- Tests d'intégration sécurité automatisés
- Tests de pénétration automatisés hebdomadaires
- Fuzzing des API et interfaces utilisateur
- Scans de vulnérabilités quotidiens
- Vérification automatique des configurations sécurisées
- Tests de résilience par injection de chaos

### 8.4 Gestion des vulnérabilités applicatives

Neo Financia applique une politique stricte de gestion des vulnérabilités:

| Criticité | Délai de correction | Mesures |
|-----------|---------------------|---------|
| Critique (CVSS 9.0-10.0) | 24h | Correction d'urgence, mesures d'atténuation immédiates |
| Haute (CVSS 7.0-8.9) | 7 jours | Planification prioritaire, correction au prochain sprint |
| Moyenne (CVSS 4.0-6.9) | 30 jours | Intégration au backlog, suivi régulier |
| Basse (CVSS 0.1-3.9) | 90 jours | Traitement par lot, surveillance |

Processus de gestion:
- Scan continu des environnements
- Priorisation basée sur l'exploitabilité et l'exposition
- Validation des corrections par re-test automatisé
- Reporting hebdomadaire au COSEC sur les vulnérabilités non corrigées
- Analyse trimestrielle des tendances et causes racines

### 8.5 Sécurité des APIs (internes et Open Banking)

#### 8.5.1 Contrôles généraux API

- Implémentation OAuth 2.0 avec OpenID Connect
- Validation stricte des entrées et sorties
- Rate limiting et protection contre les abus
- Journalisation complète des accès et opérations
- Surveillance comportementale pour détecter les anomalies
- Chiffrement TLS 1.3 obligatoire

#### 8.5.2 Exigences spécifiques Open Banking

- Conformité aux standards API PSD2
- Authentification renforcée pour les opérations sensibles
- Tokenisation des identifiants de compte
- Limitation de la portée (scope) des autorisations
- Contrôle granulaire des consentements
- Révocation instantanée des accès compromis
- Audit trail complet pour les accès tiers

### 8.6 Revue de code et principes de codage sécurisé

- Guidelines de codage sécurisé par langage et framework
- Revue de code obligatoire pour les modifications de composants critiques
- Utilisation d'outils d'analyse statique dans les IDE
- Formation continue aux techniques de codage sécurisé
- Bibliothèque de composants sécurisés pré-approuvés
- Veille sur les vulnérabilités spécifiques aux frameworks utilisés
- Champions sécurité désignés dans chaque équipe de développement

---

## 9. Sécurité des canaux digitaux

### 9.1 Protection des applications web et mobiles

#### 9.1.1 Applications web

- WAF (Web Application Firewall) pour toutes les applications exposées
- Protection contre les attaques OWASP Top 10
- Détection et prévention des injections (SQL, XSS, CSRF)
- En-têtes de sécurité HTTP renforcés (CSP, HSTS, X-Frame-Options)
- Protection contre les attaques de session (cookie sécurisé, anti-CSRF)
- Vérification des bibliothèques JavaScript tierces
- Audits de sécurité trimestriels

#### 9.1.2 Applications mobiles

- Obfuscation du code et protection contre le reverse engineering
- Détection des environnements compromis (jailbreak/root)
- Stockage sécurisé des données sensibles (keychain/keystore)
- Protection contre les attaques par overlay
- Certificat SSL pinning
- Détection des applications frauduleuses (brand protection)
- Communications app-serveur chiffrées et authentifiées

### 9.2 Sécurité des API exposées

- API Gateway centralisée avec contrôles de sécurité uniformes
- Authentification et autorisation pour chaque appel
- Validation des schémas et payloads
- Contrôle granulaire des permissions (scopes)
- Rate limiting adaptatif selon le profil d'usage
- Détection des anomalies comportementales
- Monitoring en temps réel des appels API
- Traçabilité complète pour audit et détection d'incidents

### 9.3 Authentification forte des clients

Neo Financia met en œuvre une stratégie d'authentification multi-niveaux pour ses 2 millions de clients:

| Niveau de risque | Méthode d'authentification | Cas d'usage |
|------------------|----------------------------|-------------|
| Standard | Identifiant + mot de passe + 2FA | Connexion initiale, consultation |
| Élevé | Authentification biométrique ou code généré | Transactions < 1000€, modifications profil |
| Critique | Validation multiple (app + SMS ou notification) | Transactions > 1000€, ajout bénéficiaire |

Caractéristiques du système:
- Authentification adaptative basée sur l'analyse de risque
- Biométrie conforme aux standards FIDO2
- Gestion sécurisée des identités côté serveur
- Options d'authentification accessibles (handicap, contextes variés)
- Surveillance des tentatives suspectes et blocage préventif

### 9.4 Prévention de la fraude en ligne

Neo Financia déploie un système multi-couches de détection de fraude:

- Analyse comportementale des utilisateurs (profils de navigation, habitudes de transaction)
- Détection d'anomalies basée sur le machine learning
- Vérification de la cohérence des informations (localisation, appareil, comportement)
- Scoring de risque en temps réel pour chaque transaction
- Contrôles supplémentaires pour les transactions à haut risque
- Détection des tentatives d'ingénierie sociale
- Blocage automatique des tentatives multiples échouées

### 9.5 Surveillance des transactions suspectes

- Moteur de règles avancé pour identifier les patterns suspects
- Analyse comportementale pour détecter les déviations
- Alertes en temps réel pour les opérations anormales
- Capacités de blocage et de mise en attente des transactions douteuses
- Processus de validation humaine pour les cas complexes
- Rétroaction des investigations pour améliorer les modèles
- Reporting régulier sur les tendances de fraude détectées

### 9.6 Protection contre les attaques ciblant les banques digitales

- Défense contre les attaques DDoS via protection multi-couches
- Détection des tentatives d'automatisation et de bots
- Protection contre le credential stuffing et brute force
- Monitoring des menaces ciblant les néobanques (TTPs spécifiques)
- Vérification de l'intégrité des applications mobiles
- Détection des tentatives de SIM swapping
- Surveillance des marketplaces d'identifiants volés
- Protection contre les attaques d'account takeover

---

## 10. Gestion des tiers et de la chaîne d'approvisionnement

### 10.1 Évaluation des risques fournisseurs

Neo Financia applique une méthodologie structurée d'évaluation des fournisseurs:

#### 10.1.1 Catégorisation des fournisseurs

| Niveau | Criticité | Exemples | Niveau d'évaluation |
|--------|-----------|----------|---------------------|
| T1 | Critique | Fournisseurs cloud, processeurs de paiement | Évaluation approfondie + audit sur site |
| T2 | Élevé | Fournisseurs SaaS avec accès aux données clients | Évaluation détaillée + contrôles documentaires |
| T3 | Modéré | Services IT sans accès aux données critiques | Questionnaire complet + certifications |
| T4 | Faible | Services auxiliaires | Questionnaire simplifié |

#### 10.1.2 Critères d'évaluation

- Posture de sécurité technique et organisationnelle
- Conformité réglementaire (RGPD, DORA, NIS2)
- Capacités de réponse aux incidents
- Continuité d'activité et résilience
- Pratiques de développement sécurisé
- Gestion des sous-traitants
- Localisation des données et transferts transfrontaliers

### 10.2 Exigences de sécurité pour les partenaires fintech

Les partenariats fintech font l'objet d'exigences spécifiques:

- Certification ISO 27001 recommandée ou plan d'action documenté
- Architecture API sécurisée et documentée
- Chiffrement de bout en bout des données sensibles
- Tests d'intrusion indépendants annuels
- Authentification forte pour toutes les intégrations
- Surveillance continue des accès et opérations
- Plan de réponse aux incidents coordonné
- Capacité à respecter les RTO/RPO de Neo Financia (<15min)

### 10.3 Surveillance continue des risques tiers

- Monitoring en temps réel des services critiques
- Dashboards de performance et disponibilité
- Alertes automatiques sur les incidents de sécurité
- Veille sur les vulnérabilités affectant les technologies utilisées
- Évaluation périodique des changements organisationnels ou techniques
- Surveillance des évolutions réglementaires impactant les fournisseurs
- Tests de résilience incluant les prestataires critiques

### 10.4 Contrôles contractuels et audits

Tous les contrats avec les fournisseurs critiques (T1, T2) incluent:

- Clauses de sécurité et confidentialité renforcées
- Engagements de niveaux de service (SLA) avec pénalités
- Droit d'audit pour Neo Financia et les régulateurs
- Obligation de notification des incidents sous 24h
- Exigences spécifiques DORA pour les fournisseurs TIC
- Garanties de conformité réglementaire multi-juridictions
- Clauses de réversibilité et portabilité des données

Programme d'audit:
- Audits complets annuels pour les fournisseurs T1
- Revues documentaires semestrielles pour les T2
- Tests de sécurité techniques pour les fournisseurs d'infrastructure critique
- Exercices conjoints de continuité d'activité

### 10.5 Gestion des incidents impliquant des tiers

- Processus coordonné de gestion des incidents
- Canaux de communication dédiés et sécurisés
- Playbooks spécifiques par typologie de prestataire
- Responsabilités clairement définies (RACI)
- Obligations de reporting interne et externe
- Garanties contractuelles d'assistance et coopération
- Retours d'expérience obligatoires après incident
- Amélioration continue des mécanismes de détection

### 10.6 Plan de sortie et de transition

Pour chaque fournisseur critique:
- Plan de sortie documenté et testé annuellement
- Estimation des impacts métier et techniques
- Identification des compétences et ressources nécessaires
- Gestion des données et procédures d'extraction
- Alternatives pré-identifiées et évaluées
- Mécanismes de transfert sécurisé des actifs
- Périodes de transition et de coexistence
- Indicateurs de suivi de la transition

---

## 11. Détection et réponse aux incidents

### 11.1 Capacités de SOC adaptées à une néobanque

Neo Financia opère un SOC interne avec couverture étendue:

- Fonctionnement 24/7/365
- Équipe de 10 analystes spécialisés
- Triage et analyse de niveau 1 et 2
- Expertise en forensique et remédiation
- Couverture complète des environnements (on-premise, cloud, SaaS)
- Focalisation sur les menaces spécifiques au secteur financier
- Capacités d'investigation avancées

Architecture technique:
- SIEM centralisé avec collecte de logs de toutes les sources
- EDR déployé sur tous les systèmes critiques
- NDR pour l'analyse des flux réseau
- SOAR pour l'automatisation et l'orchestration
- Threat intelligence platform intégrée
- Sandboxing pour l'analyse des fichiers suspects
- Honeypots stratégiquement déployés

### 11.2 Surveillance des menaces et intelligence

- Abonnement à des flux de Threat Intelligence spécialisés finance
- Programme de chasse aux menaces (threat hunting) proactif
- Veille sur les forums et marketplaces underground
- Participation aux CERT sectoriels (financier, national)
- Surveillance des IOCs spécifiques au secteur bancaire
- Corrélation des événements internes avec les tendances globales
- Alerting personnalisé selon le profil de risque Neo Financia

### 11.3 Playbooks de réponse aux incidents

Neo Financia maintient des playbooks détaillés pour les scénarios prioritaires:

| Type d'incident | Actions clés | Temps de réponse cible |
|-----------------|--------------|------------------------|
| Compromission de compte utilisateur | Isolation, investigation, réinitialisation sécurisée | 30 minutes |
| Attaque DDoS | Activation protection, filtrage, communication | 15 minutes |
| Exfiltration de données | Blocage des flux, analyse d'impact, confinement | 20 minutes |
| Malware/Ransomware | Isolation, analyse, containment, restauration | 45 minutes |
| Fraude financière | Suspension transaction, enquête, remédiation | 10 minutes |
| Intrusion système | Isolation, forensique, éradication, restauration | 60 minutes |

Chaque playbook inclut:
- Critères de détection et confirmation
- Chaîne d'escalade et matrice de responsabilités
- Procédures techniques détaillées
- Modèles de communication interne et externe
- Liste des parties prenantes à impliquer
- Indicateurs de suivi et résolution

### 11.4 Processus d'escalade et de prise de décision

Neo Financia définit 4 niveaux d'incidents avec processus d'escalade adaptés:

| Niveau | Définition | Escalade | Décisionnaire | Délai notification |
|--------|------------|----------|---------------|-------------------|
| P1 | Impact critique, risque systémique | Immédiate DG, CA | Cellule de crise | < 15 minutes |
| P2 | Impact majeur sur activité critique | RSSI, DSI, Métier | COSEC d'urgence | < 30 minutes |
| P3 | Impact limité sur service | Équipe SOC, RSSI | RSSI | < 2 heures |
| P4 | Impact mineur, pas d'effet client | Analyste SOC | Responsable SOC | Rapport quotidien |

Processus de décision:
- Évaluation rapide basée sur critères objectifs
- Activation automatique des niveaux d'escalade selon seuils
- Communication structurée via canaux sécurisés
- Documentation exhaustive des décisions et justifications
- Revue post-incident avec analyse des délais de réaction

### 11.5 Communication de crise et notification

- Plan de communication multi-niveaux (interne, clients, régulateurs, public)
- Modèles pré-approuvés par type d'incident
- Canaux de communication redondants et sécurisés
- Désignation des porte-paroles autorisés
- Coordination avec équipes juridiques et communication
- Procédures de notification aux autorités:
  - ANSSI: incidents significatifs (<24h)
  - CNIL: violations de données (<72h)
  - ACPR: incidents opérationnels significatifs (<4h)
  - ICO UK: violations de données (<48h)
  - Banque de France: incidents de paiement (<2h)

### 11.6 Analyse forensique et lessons learned

- Capacités forensiques internes pour première analyse
- Partenariats avec experts externes pour investigations complexes
- Procédures de collecte de preuves juridiquement valides
- Analyse de la chaîne d'attaque complète (kill chain)
- Identification des défaillances techniques et organisationnelles
- Documentation structurée des incidents dans une base de connaissances
- Processus formel de lessons learned avec plan d'action
- Suivi des améliorations issues des retours d'expérience

---

## 12. Continuité d'activité et résilience opérationnelle

### 12.1 Stratégie de continuité adaptée au modèle néobanque

Neo Financia adopte une approche de continuité prioritisant les services digitaux critiques:

#### 12.1.1 Principes directeurs

- Disponibilité 24/7 des services clients essentiels
- Architecture hautement résiliente pour les fonctions critiques
- Redondance systématique des composants stratégiques
- Approche multi-site et multi-cloud
- Automatisation maximale des procédures de bascule
- Tests fréquents des mécanismes de reprise

#### 12.1.2 Hiérarchisation des services

| Niveau | Criticité | Services | RTO | RPO |
|--------|-----------|----------|-----|-----|
| S1 | Critique | Authentification, paiements, consultation comptes | <15 min | <15 min |
| S2 | Haute | Virements programmés, gestion des bénéficiaires | <60 min | <30 min |
| S3 | Moyenne | Services de crédit, épargne, assurance | <4 heures | <60 min |
| S4 | Faible | Fonctionnalités secondaires, reporting | <24 heures | <24 heures |

### 12.2 Plan de reprise d'activité pour les services critiques

Neo Financia maintient un plan de reprise détaillé couvrant:

- Procédures de bascule technique détaillées
- Mécanismes de synchronisation des données
- Reprise en mode dégradé avec fonctionnalités minimales
- Rollback et retour à la normale
- Chaîne de communication et d'escalade
- Liste des responsables et suppléants
- Localisation des ressources et accès d'urgence
- Documentation technique actualisée

Architecture de reprise:
- Infrastructure active-active pour les services S1
- Redondance géographique entre datacenters et clouds
- Réplication synchrone des données critiques
- Bascule automatisée pour les services critiques
- Capacité de reprise multi-niveaux (application, serveur, datacenter)

### 12.3 Tests et exercices de continuité

Programme complet de tests incluant:

| Type de test | Fréquence | Périmètre | Participants |
|--------------|-----------|-----------|--------------|
| Test techniques unitaires | Mensuel | Composants individuels | Équipes techniques |
| Test de basculement | Trimestriel | Services S1 par rotation | IT, Sécurité, Métiers |
| Exercice de simulation | Semestriel | Scénario de crise complet | COSEC élargi |
| Test complet PRA | Annuel | Ensemble du SI critique | Toute l'organisation |
| Test surprise | Annuel | Défini par le RSSI | Équipes concernées |

Chaque test fait l'objet:
- D'un plan de test documenté et validé
- De critères de succès mesurables
- D'un rapport détaillé des résultats
- D'un plan d'action pour les anomalies détectées
- D'un suivi des KPIs de performance (temps de bascule, perte de données)

### 12.4 Gestion de crise cyber

Neo Financia dispose d'une organisation de crise structurée:

- Cellule de crise cyber préétablie (compositions, suppléants)
- Salle de crise physique et virtuelle sécurisée
- Procédures d'activation et de fonctionnement
- Outils de communication de crise sécurisés
- Tableaux de bord temps réel de suivi de crise
- Coordination avec les autorités et partenaires
- Simulation d'incidents majeurs biannuelle
- Formation spécifique des membres de la cellule de crise

### 12.5 Résilience par conception

Neo Financia intègre les principes de résilience dans sa conception:

- Architecture sans point unique de défaillance
- Capacité de dégradation gracieuse des services
- Limitation de propagation des incidents (bulkheads)
- Circuit breakers pour prévenir les effets cascade
- Auto-scaling pour absorber les pics de charge
- Throttling intelligent pour préserver les fonctions essentielles
- Conception pour la reprise (design for recovery)
- Chaos engineering pour tester la robustesse

### 12.6 Conformité avec les exigences DORA

Neo Financia se conforme aux exigences DORA par:

- Cartographie complète des dépendances critiques
- Tests de résilience opérationnelle réguliers
- Gestion renforcée des risques liés aux tiers critiques
- Reporting standardisé des incidents significatifs
- Plan de communication structuré avec les autorités
- Tests d'intrusion réguliers sur les systèmes critiques
- Participation aux exercices sectoriels de résilience
- Documentation des procédures de gestion de crise conformes aux exigences

---

## 13. Sensibilisation et formation

### 13.1 Programme de sensibilisation pour 1000 employés

Neo Financia déploie un programme complet de sensibilisation:

#### 13.1.1 Parcours de base pour tous les collaborateurs

| Module | Format | Durée | Fréquence | Validation |
|--------|--------|-------|-----------|------------|
| Fondamentaux cybersécurité | E-learning | 1h | Annuelle | Quiz 80% |
| Phishing et ingénierie sociale | E-learning + simulations | 30 min | Trimestrielle | Tests pratiques |
| Protection des données | E-learning | 45 min | Annuelle | Cas pratiques |
| Sécurité mobile et télétravail | Webinar | 30 min | Semestrielle | Quiz |
| Gestion des incidents | Guide interactif | 20 min | Annuelle | Simulation |

#### 13.1.2 Actions complémentaires

- Campagnes de communication multicanal (intranet, emails, affiches)
- Newsletter cybersécurité mensuelle avec actualités et conseils
- Évènements spéciaux (Cybersecurity Awareness Month)
- Relais par les managers (talking points sécurité)
- Communauté d'ambassadeurs cyber dans les départements
- Plateforme d'e-learning accessible à la demande
- Concours et challenges de sécurité

### 13.2 Formation spécialisée par rôle

Neo Financia adapte la formation selon les profils:

| Rôle | Formation spécifique | Certification visée |
|------|----------------------|---------------------|
| Développeurs | Secure coding, OWASP Top 10, DevSecOps | CSSLP ou équivalent |
| Administrateurs IT | Hardening, détection d'intrusion, réponse | CISSP, GIAC |
| Équipe sécurité | Formation avancée par spécialité | SANS, OSCP, CISM |
| Management | Leadership cyber, gestion de crise | CISM |
| Finance/RH | Protection données sensibles, fraude | Internal certs |
| Service client | Détection fraude, ingénierie sociale | Internal certs |

Approche:
- Learning paths personnalisés par fonction
- Alternance théorie/pratique avec labs et exercices
- Certifications externes financées pour rôles critiques
- Communautés de pratique par domaine d'expertise
- Partage de connaissances interne (lunch & learn)

### 13.3 Mesure de l'efficacité et KPIs

Neo Financia évalue l'efficacité du programme par:

#### 13.3.1 Indicateurs quantitatifs

- Taux de participation aux formations (objectif >95%)
- Scores moyens aux évaluations (objectif >85%)
- Taux de clic aux campagnes de phishing simulé (objectif <5%)
- Taux de signalement des emails suspects (objectif >80%)
- Nombre d'incidents causés par erreur utilisateur (réduction annuelle 20%)
- Délai moyen de signalement des incidents (objectif <30 min)

#### 13.3.2 Évaluation qualitative

- Enquêtes de perception de la sécurité
- Entretiens ciblés avec échantillon représentatif
- Observation des comportements en situation réelle
- Feedback des responsables sur les pratiques d'équipe
- Analyse des tendances en matière d'incidents
- Tests de situation à l'improviste

### 13.4 Promotion de la culture de sécurité

Neo Financia développe une culture sécurité par:

- Leadership visible de la direction (tone at the top)
- Intégration de critères sécurité dans l'évaluation des performances
- Reconnaissance des comportements exemplaires (champions sécurité)
- Politique de non-blâme pour encourager le signalement
- Communication transparente sur les incidents et enseignements
- Inclusion de la sécurité dans les valeurs d'entreprise
- Prise en compte du feedback des collaborateurs pour améliorer les processus

### 13.5 Gestion du shadow IT

Approche pragmatique de gestion du shadow IT:

- Campagnes de sensibilisation aux risques spécifiques
- Processus simplifié d'évaluation et approbation des outils
- Catalogue d'alternatives sécurisées pré-approuvées
- Détection technique des solutions non autorisées
- Période d'amnistie pour déclaration sans conséquence
- Accompagnement pour la transition vers des solutions validées
- Focus sur la compréhension des besoins métier sous-jacents

### 13.6 Exercices de simulation

Programme d'exercices pratiques:

- Simulations de phishing personnalisées par département
- Exercices de table (tabletop) pour le management
- Simulations de crise cyber pour l'équipe de direction
- Tests de social engineering (téléphone, physique) avec consentement
- Exercices d'intervention technique pour les équipes IT
- Simulations de fuite de données pour les équipes juridiques et communication
- Debriefings détaillés et plans d'action après chaque exercice

---

## 14. Assurance et vérification

### 14.1 Programme d'audit interne

Neo Financia maintient un programme d'audit interne rigoureux:

#### 14.1.1 Plan d'audit pluriannuel

| Année | Thème | Périmètre | Approche |
|-------|-------|-----------|----------|
| 2025 | Gestion des accès | IAM, PAM, MFA | Contrôles techniques + process |
| 2025 | Sécurité cloud | Multicloud, SecNumCloud | Configuration + gouvernance |
| 2026 | DevSecOps | Pipeline, SDLC | Maturité + implémentation |
| 2026 | Détection & réponse | SOC, SIEM, EDR | Capacités réelles + tests |
| 2027 | Protection des données | DLP, chiffrement | Contrôles + flux |
| 2027 | Résilience | PCA/PRA, BCDR | Tests + documentation |

#### 14.1.2 Méthodologie d'audit

- Référentiels utilisés: NIST CSF, ISO 27001, COBIT, FFIEC
- Équipe d'audit indépendante du RSSI
- Combinaison d'interviews, revues documentaires et tests techniques
- Évaluation de la maturité et de l'efficacité réelle
- Notation standardisée des constats
- Suivi formalisé des plans d'action
- Rapport au Comité d'Audit et des Risques

### 14.2 Tests de pénétration et red teaming

Neo Financia conduit un programme complet de tests offensifs:

#### 14.2.1 Tests de pénétration

| Type | Fréquence | Portée | Méthodologie |
|------|-----------|--------|--------------|
| Applications critiques | Semestrielle | Authentification, apps bancaires | OWASP Testing Guide |
| Infrastructure externe | Trimestrielle | Périmètre Internet, DMZ | PTES, NIST SP800-115 |
| Infrastructure interne | Annuelle | Systèmes internes, AD | PTES, NIST SP800-115 |
| Applications mobiles | À chaque version majeure | iOS, Android | OWASP MSTG |
| Environnement cloud | Trimestrielle | AWS, Azure, OVH | CSA, CIS Benchmarks |

#### 14.2.2 Exercices de Red Team

- Campagnes de simulation avancée d'attaques (APT-like)
- Scénarios complets basés sur les menaces réelles du secteur
- Approche "purple team" avec observation par les équipes défensives
- Tests sans connaissance préalable des équipes de défense
- Objectifs basés sur les actifs critiques (crown jewels)
- Rapport détaillé avec évaluation de l'efficacité des défenses
- Recommandations d'amélioration priorisées

### 14.3 Surveillance continue de la conformité

- Monitoring automatisé des configurations contre les baselines
- Scans de vulnérabilité hebdomadaires sur l'ensemble du périmètre
- Évaluation continue de la conformité réglementaire
- Tableaux de bord de conformité en temps réel
- Alertes automatiques sur les écarts significatifs
- Revues trimestrielles de la posture de sécurité globale
- Contrôles continus via agents et scanners spécialisés

### 14.4 Revues de sécurité périodiques

Neo Financia conduit des revues périodiques de ses composants critiques:

- Revue mensuelle des configurations et droits d'accès
- Revue trimestrielle des exceptions et dérogations
- Analyse semestrielle de l'architecture de sécurité
- Évaluation annuelle complète de la posture de sécurité
- Benchmark sectoriel annuel (comparaison avec pairs)
- Revue des technologies de sécurité et roadmap d'évolution
- Vérification de l'efficacité des contrôles compensatoires

### 14.5 Reporting et tableaux de bord

Neo Financia maintient un système de reporting à plusieurs niveaux:

#### 14.5.1 Tableaux de bord opérationnels (quotidien/hebdomadaire)

- Incidents et alertes de sécurité
- Vulnérabilités détectées et statut de correction
- Performance des contrôles de sécurité
- Métriques de détection et réponse (MTTD, MTTR)
- Tendances des attaques et tentatives

#### 14.5.2 Reporting tactique (mensuel)

- Synthèse des incidents et problèmes récurrents
- Statut des plans d'action en cours
- Métriques de conformité et écarts
- Risques émergents et évolutions de la menace
- Performance des équipes sécurité

#### 14.5.3 Reporting stratégique (trimestriel)

- Indicateurs de maturité et évolution
- Risques majeurs et tendances
- Investissements et ROI sécurité
- Benchmarks sectoriels
- Initiatives stratégiques et roadmap

### 14.6 Processus d'amélioration continue

Neo Financia applique un cycle PDCA (Plan-Do-Check-Act) à sa sécurité:

- Collecte continue du feedback des parties prenantes
- Analyse des incidents et presque-incidents
- Veille sur les évolutions des menaces et technologies
- Intégration des enseignements des tests et audits
- Revue périodique des politiques et procédures
- Programme formel d'amélioration avec KPIs
- Partage des bonnes pratiques au sein de l'organisation
- Benchmark avec les standards internationaux

---

## 15. Innovation et technologies émergentes

### 15.1 Sécurité de l'IA et du machine learning

Neo Financia encadre l'utilisation de l'IA par:

#### 15.1.1 Gouvernance de l'IA

- Comité d'éthique IA avec participation du RSSI
- Validation obligatoire des cas d'usage sensibles
- Supervision humaine pour les décisions critiques (human-in-the-loop)
- Évaluation d'impact pour toute nouvelle implémentation d'IA

#### 15.1.2 Contrôles techniques

- Protection contre l'empoisonnement des modèles
- Détection des tentatives d'évasion (adversarial ML)
- Surveillance des biais et dérives dans les algorithmes
- Chiffrement des modèles et données d'entraînement
- Tests de robustesse des systèmes d'IA
- Explicabilité des décisions algorithmiques

### 15.2 Blockchain et crypto-actifs

En préparation de l'étude sur l'intégration des cryptomonnaies, Neo Financia établit:

- Framework d'évaluation sécurité des technologies blockchain
- Exigences pour la gestion sécurisée des clés cryptographiques
- Procédures de validation des smart contracts
- Contrôles pour la détection des transactions suspectes
- Standards de sécurité pour les portefeuilles numériques
- Analyse de risque spécifique aux crypto-actifs
- Conformité avec les exigences MiCA (Markets in Crypto-Assets)

### 15.3 Biométrie et identification avancée

Neo Financia déploie des solutions d'identification de pointe:

- Biométrie multimodale (combinaison de facteurs)
- Détection du vivant (liveness detection)
- Stockage sécurisé des gabarits biométriques
- Alternatives inclusives pour tous les utilisateurs
- Conformité avec les exigences eIDAS2
- Protection contre les attaques de présentation
- Évaluation continue des taux de faux positifs/négatifs

### 15.4 Analyse comportementale et détection des fraudes

- Modèles de profilage comportemental des utilisateurs
- Détection des anomalies basée sur le machine learning
- Analyse contextuelle multi-facteurs en temps réel
- Ajustement dynamique des seuils d'alerte
- Réduction des faux positifs par apprentissage continu
- Corrélation des comportements à travers les canaux
- Détection précoce des signaux faibles de fraude

### 15.5 Automatisation de la sécurité

Neo Financia investit dans l'automatisation:

- Orchestration de la réponse aux incidents (SOAR)
- Remédiation automatique des vulnérabilités standard
- Security-as-Code pour l'infrastructure cloud
- Provisionnement et déprovisionnement automatisés des accès
- Auto-healing pour les configurations dérivantes
- Validation continue de la conformité
- Déploiement automatisé des contrôles de sécurité

### 15.6 Veille technologique

Neo Financia maintient un processus de veille structuré:
- Participation aux forums sectoriels (FS-ISAC, ANSSI, ECB)
- Collaboration avec des laboratoires de recherche
- Programme d'innovation sécurité en interne
- Évaluation des technologies émergentes
- Proof of Concepts sur technologies prometteuses
- Partenariats avec des startups cybersécurité
- Analyse prospective des futures menaces

---

## 16. Annexes

### 16.1 Glossaire et définitions

| Terme | Définition |
|-------|------------|
| ACPR | Autorité de Contrôle Prudentiel et de Résolution |
| API | Application Programming Interface |
| DORA | Digital Operational Resilience Act |
| DLP | Data Loss Prevention |
| EDR | Endpoint Detection and Response |
| IAM | Identity and Access Management |
| MFA | Multi-Factor Authentication |
| NIS2 | Network and Information Security Directive 2 |
| RPO | Recovery Point Objective |
| RTO | Recovery Time Objective |
| SIEM | Security Information and Event Management |
| SOC | Security Operations Center |
| SOAR | Security Orchestration, Automation and Response |
| Zero Trust | Modèle de sécurité qui ne fait confiance à aucun utilisateur ou système par défaut |

### 16.2 Références normatives et réglementaires

- ISO/IEC 27001:2022 - Systèmes de management de la sécurité de l'information
- ISO/IEC 27002:2022 - Code de bonnes pratiques pour la sécurité de l'information
- RGPD - Règlement Général sur la Protection des Données (2016/679)
- DORA - Digital Operational Resilience Act (2022/2554)
- NIS2 - Directive Network and Information Security 2 (2022/0383)
- PCI-DSS v4.0 - Payment Card Industry Data Security Standard
- Règlement eIDAS 2 - Electronic Identification, Authentication and Trust Services
- Guides ANSSI (SecNumCloud, PASSI, PAMS)
- FCA Handbook (UK) - Financial Conduct Authority
- NIST Cybersecurity Framework 2.0
- NIST SP 800-53 Rev. 5

### 16.3 Schémas d'architecture sécurisée

#### 16.3.1 Architecture réseau globale

```
Internet
   │
   ▼
┌────────────┐    ┌────────────┐
│ WAF + CDN  │───►│  API GW    │
└────────────┘    └─────┬──────┘
                        │
┌────────────┐    ┌─────▼──────┐
│  DDoS      │───►│ App Layer  │
│ Protection │    │ Firewalls  │
└────────────┘    └─────┬──────┘
                        │
                  ┌─────▼──────┐
                  │ Zero Trust │
                  │ Segmentation│
                  └──┬───────┬─┘
                     │       │
              ┌──────▼─┐ ┌───▼─────┐
              │ Zone S1 │ │ Zone S2 │
              │Critical │ │Business │
              └──────┬─┘ └───┬─────┘
                     │       │
              ┌──────▼───────▼─────┐
              │ Security Monitoring │
              │ SIEM / SOC / SOAR  │
              └────────────────────┘
```

#### 16.3.2 Architecture d'authentification

```
┌──────────────┐      ┌────────────┐      ┌────────────┐
│ Application  │─────►│  IAM/IDP   │◄─────┤  MFA       │
│ Mobile/Web   │      │  Service   │      │  Service   │
└──────────────┘      └─────┬──────┘      └────────────┘
                            │
                      ┌─────▼──────┐      ┌────────────┐
                      │ Risk-based │◄─────┤ Behavior   │
                      │ Auth Engine│      │ Analytics  │
                      └─────┬──────┘      └────────────┘
                            │
                      ┌─────▼──────┐
                      │ Directory  │
                      │ Services   │
                      └────────────┘
```

#### 16.3.3 Architecture cloud sécurisée

```
┌───────────────────────────────────────────────────────┐
│                  Cloud Souverain                      │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────┐   │
│  │ Données     │    │ Services    │    │ Payment  │   │
│  │ Critiques   │    │ Core Banking│    │ Services │   │
│  └─────────────┘    └─────────────┘    └──────────┘   │
└───────────────────────────────────────────────────────┘

┌───────────────────┐    ┌───────────────────┐
│    AWS Cloud      │    │   Azure Cloud     │
│  ┌─────────────┐  │    │  ┌─────────────┐  │
│  │ Services    │  │    │  │ Services    │  │
│  │ Non-critiques│  │    │  │ Non-critiques│  │
│  └─────────────┘  │    │  └─────────────┘  │
└───────────────────┘    └───────────────────┘

┌───────────────────────────────────────────────────────┐
│                  CSPM / Cloud IAM                     │
└───────────────────────────────────────────────────────┘
```

### 16.4 Modèles de documents

- Formulaire d'analyse de risque simplifié
- Template d'analyse d'impact RGPD
- Matrice de conformité réglementaire
- Checklist de sécurité pour l'évaluation des fournisseurs
- Modèle de plan de test d'intrusion
- Template de rapport d'incident
- Guide d'évaluation sécurité cloud
- Processus de demande de dérogation aux exigences PSSI

### 16.5 Procédures et guides opérationnels

- Procédure de gestion des accès privilégiés
- Guide de durcissement des serveurs Linux/Windows
- Procédure de réponse aux incidents
- Guide de sécurisation des API
- Procédures de sauvegarde et restauration
- Procédure de gestion des vulnérabilités
- Méthodologie d'analyse de risque détaillée
- Guide des revues de sécurité applicatives
- Procédure d'urgence en cas d'incident majeur

---

*Document approuvé par le Conseil d'Administration de Neo Financia le 15 avril 2025*

*Référence: PSSI-NEO-2025-V1.0*

*Classification: INTERNE*

*Propriétaire: RSSI*

*Révision: Annuelle*