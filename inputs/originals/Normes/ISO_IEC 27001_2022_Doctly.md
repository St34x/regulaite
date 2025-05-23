# INTERNATIONAL STANDARD

# ISO/IEC 27001

Third edition
2022-10

## Information security, cybersecurity and privacy protection — Information security management systems — Requirements

Sécurité de l'information, cybersécurité et protection de la vie privée — Systèmes de management de la sécurité de l'information — Exigences

![Image of ISO and IEC logos]

Reference number
ISO/IEC 27001:2022(E)

© ISO/IEC 2022


-------------------------------------------------------------------------------- Page 2

ISO/IEC 27001:2022(E)

![Image of a copyright logo]

COPYRIGHT PROTECTED DOCUMENT

© ISO/IEC 2022

All rights reserved. Unless otherwise specified, or required in the context of its implementation, no part of this publication may be reproduced or utilized otherwise in any form or by any means, electronic or mechanical, including photocopying, or posting on the internet or an intranet, without prior written permission. Permission can be requested from either ISO at the address below or ISO's member body in the country of the requester.

ISO copyright office
CP 401 • Ch. de Blandonnet 8
CH-1214 Vernier, Geneva
Phone: +41 22 749 01 11
Email: copyright@iso.org
Website: www.iso.org
Published in Switzerland

ii

© ISO/IEC 2022 - All rights reserved


-------------------------------------------------------------------------------- Page 3

ISO/IEC 27001:2022(E)

Contents

Page

Foreword iv

Introduction v

1 Scope 1

2 Normative references 1

3 Terms and definitions 1

4 Context of the organization 1

4.1 Understanding the organization and its context 1

4.2 Understanding the needs and expectations of interested parties 1

4.3 Determining the scope of the information security management system 2

4.4 Information security management system 2

5 Leadership 2

5.1 Leadership and commitment 2

5.2 Policy 3

5.3 Organizational roles, responsibilities and authorities 3

6 Planning 3

6.1 Actions to address risks and opportunities 3

6.1.1 General 3

6.1.2 Information security risk assessment 4

6.1.3 Information security risk treatment 4

6.2 Information security objectives and planning to achieve them 5

7 Support 6

7.1 Resources 6

7.2 Competence 6

7.3 Awareness 6

7.4 Communication 6

7.5 Documented information 6

7.5.1 General 6

7.5.2 Creating and updating 7

7.5.3 Control of documented information 7

8 Operation 7

8.1 Operational planning and control 7

8.2 Information security risk assessment 8

8.3 Information security risk treatment 8

9 Performance evaluation 8

9.1 Monitoring, measurement, analysis and evaluation 8

9.2 Internal audit 8

9.2.1 General 8

9.2.2 Internal audit programme 9

9.3 Management review 9

9.3.1 General 9

9.3.2 Management review inputs 9

9.3.3 Management review results 9

10 Improvement 10

10.1 Continual improvement 10

10.2 Nonconformity and corrective action 10

Annex A (normative) Information security controls reference 11

Bibliography 19

© ISO/IEC 2022 - All rights reserved

iii


-------------------------------------------------------------------------------- Page 4

# ISO/IEC 27001:2022(E)

## Foreword

ISO (the International Organization for Standardization) and IEC (the International Electrotechnical Commission) form the specialized system for worldwide standardization. National bodies that are members of ISO or IEC participate in the development of International Standards through technical committees established by the respective organization to deal with particular fields of technical activity. ISO and IEC technical committees collaborate in fields of mutual interest. Other international organizations, governmental and non-governmental, in liaison with ISO and IEC, also take part in the work.

The procedures used to develop this document and those intended for its further maintenance are described in the ISO/IEC Directives, Part 1. In particular, the different approval criteria needed for the different types of document should be noted. This document was drafted in accordance with the editorial rules of the ISO/IEC Directives, Part 2 (see www.iso.org/directives or www.iec.ch/members_experts/refdocs).

Attention is drawn to the possibility that some of the elements of this document may be the subject of patent rights. ISO and IEC shall not be held responsible for identifying any or all such patent rights. Details of any patent rights identified during the development of the document will be in the Introduction and/or on the ISO list of patent declarations received (see www.iso.org/patents) or the IEC list of patent declarations received (see https://patents.iec.ch).

Any trade name used in this document is information given for the convenience of users and does not constitute an endorsement.

For an explanation of the voluntary nature of standards, the meaning of ISO specific terms and expressions related to conformity assessment, as well as information about ISO's adherence to the World Trade Organization (WTO) principles in the Technical Barriers to Trade (TBT) see www.iso.org/iso/foreword.html. In the IEC, see www.iec.ch/understanding-standards.

This document was prepared by Joint Technical Committee ISO/IEC JTC 1, Information Technology, Subcommittee SC 27, Information security, cybersecurity and privacy protection.

This third edition cancels and replaces the second edition (ISO/IEC 27001:2013), which has been technically revised. It also incorporates the Technical Corrigenda ISO/IEC 27001:2013/Cor 1:2014 and ISO/IEC 27001:2013/Cor 2:2015.

The main changes are as follows:

- the text has been aligned with the harmonized structure for management system standards and ISO/IEC 27002:2022.

Any feedback or questions on this document should be directed to the user's national standards body. A complete listing of these bodies can be found at www.iso.org/members.html and www.iec.ch/national-committees.


-------------------------------------------------------------------------------- Page 5

# Introduction

0.1 General

This document has been prepared to provide requirements for establishing, implementing, maintaining and continually improving an information security management system. The adoption of an information security management system is a strategic decision for an organization. The establishment and implementation of an organization's information security management system is influenced by the organization's needs and objectives, security requirements, the organizational processes used and the size and structure of the organization. All of these influencing factors are expected to change over time.

The information security management system preserves the confidentiality, integrity and availability of information by applying a risk management process and gives confidence to interested parties that risks are adequately managed.

It is important that the information security management system is part of and integrated with the organization's processes and overall management structure and that information security is considered in the design of processes, information systems, and controls. It is expected that an information security management system implementation will be scaled in accordance with the needs of the organization.

This document can be used by internal and external parties to assess the organization's ability to meet the organization's own information security requirements.

The order in which requirements are presented in this document does not reflect their importance or imply the order in which they are to be implemented. The list items are enumerated for reference purpose only.

ISO/IEC 27000 describes the overview and the vocabulary of information security management systems, referencing the information security management system family of standards (including ISO/IEC 27003[^2], ISO/IEC 27004[^3] and ISO/IEC 27005[^4]), with related terms and definitions.

0.2 Compatibility with other management system standards

This document applies the high-level structure, identical sub-clause titles, identical text, common terms, and core definitions defined in Annex SL of ISO/IEC Directives, Part 1, Consolidated ISO Supplement, and therefore maintains compatibility with other management system standards that have adopted the Annex SL.

This common approach defined in the Annex SL will be useful for those organizations that choose to operate a single management system that meets the requirements of two or more management system standards.


-------------------------------------------------------------------------------- Page 6



-------------------------------------------------------------------------------- Page 7

INTERNATIONAL STANDARD ISO/IEC 27001:2022(E)

# Information security, cybersecurity and privacy protection – Information security management systems – Requirements

1 Scope

This document specifies the requirements for establishing, implementing, maintaining and continually improving an information security management system within the context of the organization. This document also includes requirements for the assessment and treatment of information security risks tailored to the needs of the organization. The requirements set out in this document are generic and are intended to be applicable to all organizations, regardless of type, size or nature. Excluding any of the requirements specified in Clauses 4 to 10 is not acceptable when an organization claims conformity to this document.

2 Normative references

The following documents are referred to in the text in such a way that some or all of their content constitutes requirements of this document. For dated references, only the edition cited applies. For undated references, the latest edition of the referenced document (including any amendments) applies.

ISO/IEC 27000, *Information technology — Security techniques — Information security management systems — Overview and vocabulary*

3 Terms and definitions

For the purposes of this document, the terms and definitions given in ISO/IEC 27000 apply.

ISO and IEC maintain terminology databases for use in standardization at the following addresses:

*   ISO Online browsing platform: available at https://www.iso.org/obp
*   IEC Electropedia: available at https://www.electropedia.org/

4 Context of the organization

4.1 Understanding the organization and its context

The organization shall determine external and internal issues that are relevant to its purpose and that affect its ability to achieve the intended outcome(s) of its information security management system.

NOTE Determining these issues refers to establishing the external and internal context of the organization considered in Clause 5.4.1 of ISO 31000:2018[5].

4.2 Understanding the needs and expectations of interested parties

The organization shall determine:

a) interested parties that are relevant to the information security management system;

b) the relevant requirements of these interested parties;

c) which of these requirements will be addressed through the information security management system.


-------------------------------------------------------------------------------- Page 8

ISO/IEC 27001:2022(E)

NOTE The requirements of interested parties can include legal and regulatory requirements and contractual obligations.

## 4.3 Determining the scope of the information security management system

The organization shall determine the boundaries and applicability of the information security management system to establish its scope.

When determining this scope, the organization shall consider:

a) the external and internal issues referred to in 4.1;

b) the requirements referred to in 4.2;

c) interfaces and dependencies between activities performed by the organization, and those that are performed by other organizations.

The scope shall be available as documented information.

## 4.4 Information security management system

The organization shall establish, implement, maintain and continually improve an information security management system, including the processes needed and their interactions, in accordance with the requirements of this document.

# 5 Leadership

## 5.1 Leadership and commitment

Top management shall demonstrate leadership and commitment with respect to the information security management system by:

a) ensuring the information security policy and the information security objectives are established and are compatible with the strategic direction of the organization;

b) ensuring the integration of the information security management system requirements into the organization's processes;

c) ensuring that the resources needed for the information security management system are available;

d) communicating the importance of effective information security management and of conforming to the information security management system requirements;

e) ensuring that the information security management system achieves its intended outcome(s);

f) directing and supporting persons to contribute to the effectiveness of the information security management system;

g) promoting continual improvement; and

h) supporting other relevant management roles to demonstrate their leadership as it applies to their areas of responsibility.

NOTE Reference to "business" in this document can be interpreted broadly to mean those activities that are core to the purposes of the organization's existence.


-------------------------------------------------------------------------------- Page 9

ISO/IEC 27001:2022(E)

## 5.2 Policy

Top management shall establish an information security policy that:

a) is appropriate to the purpose of the organization;

b) includes information security objectives (see 6.2) or provides the framework for setting information security objectives;

c) includes a commitment to satisfy applicable requirements related to information security;

d) includes a commitment to continual improvement of the information security management system.

The information security policy shall:

e) be available as documented information;

f) be communicated within the organization;

g) be available to interested parties, as appropriate.

## 5.3 Organizational roles, responsibilities and authorities

Top management shall ensure that the responsibilities and authorities for roles relevant to information security are assigned and communicated within the organization.

Top management shall assign the responsibility and authority for:

a) ensuring that the information security management system conforms to the requirements of this document;

b) reporting on the performance of the information security management system to top management.

NOTE Top management can also assign responsibilities and authorities for reporting performance of the information security management system within the organization.

## 6 Planning

### 6.1 Actions to address risks and opportunities

#### 6.1.1 General

When planning for the information security management system, the organization shall consider the issues referred to in 4.1 and the requirements referred to in 4.2 and determine the risks and opportunities that need to be addressed to:

a) ensure the information security management system can achieve its intended outcome(s);

b) prevent, or reduce, undesired effects;

c) achieve continual improvement.

The organization shall plan:

d) actions to address these risks and opportunities; and

e) how to
1) integrate and implement the actions into its information security management system processes; and
2) evaluate the effectiveness of these actions.

© ISO/IEC 2022 - All rights reserved

3


-------------------------------------------------------------------------------- Page 10

ISO/IEC 27001:2022(E)

## 6.1.2 Information security risk assessment

The organization shall define and apply an information security risk assessment process that:

a) establishes and maintains information security risk criteria that include:

1) the risk acceptance criteria; and

2) criteria for performing information security risk assessments;

b) ensures that repeated information security risk assessments produce consistent, valid and comparable results;

c) identifies the information security risks:

1) apply the information security risk assessment process to identify risks associated with the loss of confidentiality, integrity and availability for information within the scope of the information security management system; and

2) identify the risk owners;

d) analyses the information security risks:

1) assess the potential consequences that would result if the risks identified in 6.1.2 c) 1) were to materialize;

2) assess the realistic likelihood of the occurrence of the risks identified in 6.1.2 c) 1); and

3) determine the levels of risk;

e) evaluates the information security risks:

1) compare the results of risk analysis with the risk criteria established in 6.1.2 a); and

2) prioritize the analysed risks for risk treatment.

The organization shall retain documented information about the information security risk assessment process.

## 6.1.3 Information security risk treatment

The organization shall define and apply an information security risk treatment process to:

a) select appropriate information security risk treatment options, taking account of the risk assessment results;

b) determine all controls that are necessary to implement the information security risk treatment option(s) chosen;

NOTE 1 Organizations can design controls as required, or identify them from any source.

c) compare the controls determined in 6.1.3 b) above with those in Annex A and verify that no necessary controls have been omitted;

NOTE 2 Annex A contains a list of possible information security controls. Users of this document are directed to Annex A to ensure that no necessary information security controls are overlooked.

NOTE 3 The information security controls listed in Annex A are not exhaustive and additional information security controls can be included if needed.

d) produce a Statement of Applicability that contains:

— the necessary controls (see 6.1.3 b) and c));


-------------------------------------------------------------------------------- Page 11

ISO/IEC 27001:2022(E)

— justification for their inclusion;

— whether the necessary controls are implemented or not; and

— the justification for excluding any of the Annex A controls.

e) formulate an information security risk treatment plan; and

f) obtain risk owners' approval of the information security risk treatment plan and acceptance of the residual information security risks.

The organization shall retain documented information about the information security risk treatment process.

NOTE 4 The information security risk assessment and treatment process in this document aligns with the principles and generic guidelines provided in ISO 31000[5].

## 6.2 Information security objectives and planning to achieve them

The organization shall establish information security objectives at relevant functions and levels.

The information security objectives shall:

a) be consistent with the information security policy;

b) be measurable (if practicable);

c) take into account applicable information security requirements, and results from risk assessment and risk treatment;

d) be monitored;

e) be communicated;

f) be updated as appropriate;

g) be available as documented information.

The organization shall retain documented information on the information security objectives.

When planning how to achieve its information security objectives, the organization shall determine:

h) what will be done;

i) what resources will be required;

j) who will be responsible;

k) when it will be completed; and

l) how the results will be evaluated.

## 6.3 Planning of changes

When the organization determines the need for changes to the information security management system, the changes shall be carried out in a planned manner.

© ISO/IEC 2022 - All rights reserved

5


-------------------------------------------------------------------------------- Page 12

ISO/IEC 27001:2022(E)

# 7 Support

## 7.1 Resources

The organization shall determine and provide the resources needed for the establishment, implementation, maintenance and continual improvement of the information security management system.

## 7.2 Competence

The organization shall:

a) determine the necessary competence of person(s) doing work under its control that affects its information security performance;

b) ensure that these persons are competent on the basis of appropriate education, training, or experience;

c) where applicable, take actions to acquire the necessary competence, and evaluate the effectiveness of the actions taken; and

d) retain appropriate documented information as evidence of competence.

NOTE Applicable actions can include, for example: the provision of training to, the mentoring of, or the re-assignment of current employees; or the hiring or contracting of competent persons.

## 7.3 Awareness

Persons doing work under the organization's control shall be aware of:

a) the information security policy;

b) their contribution to the effectiveness of the information security management system, including the benefits of improved information security performance; and

c) the implications of not conforming with the information security management system requirements.

## 7.4 Communication

The organization shall determine the need for internal and external communications relevant to the information security management system including:

a) on what to communicate;

b) when to communicate;

c) with whom to communicate;

d) how to communicate.

## 7.5 Documented information

### 7.5.1 General

The organization's information security management system shall include:

a) documented information required by this document; and


-------------------------------------------------------------------------------- Page 13

ISO/IEC 27001:2022(E)

b) documented information determined by the organization as being necessary for the effectiveness
of the information security management system.

NOTE The extent of documented information for an information security management system can differ
from one organization to another due to:

1) the size of organization and its type of activities, processes, products and services;

2) the complexity of processes and their interactions; and

3) the competence of persons.

7.5.2 Creating and updating

When creating and updating documented information the organization shall ensure appropriate:

a) identification and description (e.g. a title, date, author, or reference number);

b) format (e.g. language, software version, graphics) and media (e.g. paper, electronic); and

c) review and approval for suitability and adequacy.

7.5.3 Control of documented information

Documented information required by the information security management system and by this
document shall be controlled to ensure:

a) it is available and suitable for use, where and when it is needed; and

b) it is adequately protected (e.g. from loss of confidentiality, improper use, or loss of integrity).

For the control of documented information, the organization shall address the following activities, as
applicable:

c) distribution, access, retrieval and use;

d) storage and preservation, including the preservation of legibility;

e) control of changes (e.g. version control); and

f) retention and disposition.

Documented information of external origin, determined by the organization to be necessary for
the planning and operation of the information security management system, shall be identified as
appropriate, and controlled.

NOTE Access can imply a decision regarding the permission to view the documented information only, or
the permission and authority to view and change the documented information, etc.

8 Operation

8.1 Operational planning and control

The organization shall plan, implement and control the processes needed to meet requirements, and to
implement the actions determined in Clause 6, by:

- establishing criteria for the processes;
- implementing control of the processes in accordance with the criteria.

Documented information shall be available to the extent necessary to have confidence that the
processes have been carried out as planned.

© ISO/IEC 2022 - All rights reserved
7


-------------------------------------------------------------------------------- Page 14

ISO/IEC 27001:2022(E)

The organization shall control planned changes and review the consequences of unintended changes, taking action to mitigate any adverse effects, as necessary.

The organization shall ensure that externally provided processes, products or services that are relevant to the information security management system are controlled.

## 8.2 Information security risk assessment

The organization shall perform information security risk assessments at planned intervals or when significant changes are proposed or occur, taking account of the criteria established in 6.1.2 a).

The organization shall retain documented information of the results of the information security risk assessments.

## 8.3 Information security risk treatment

The organization shall implement the information security risk treatment plan.

The organization shall retain documented information of the results of the information security risk treatment.

# 9 Performance evaluation

## 9.1 Monitoring, measurement, analysis and evaluation

The organization shall determine:

a) what needs to be monitored and measured, including information security processes and controls;

b) the methods for monitoring, measurement, analysis and evaluation, as applicable, to ensure valid results. The methods selected should produce comparable and reproducible results to be considered valid;

c) when the monitoring and measuring shall be performed;

d) who shall monitor and measure;

e) when the results from monitoring and measurement shall be analysed and evaluated;

f) who shall analyse and evaluate these results.

Documented information shall be available as evidence of the results.

The organization shall evaluate the information security performance and the effectiveness of the information security management system.

## 9.2 Internal audit

### 9.2.1 General

The organization shall conduct internal audits at planned intervals to provide information on whether the information security management system:

a) conforms to

1) the organization's own requirements for its information security management system;


-------------------------------------------------------------------------------- Page 15

ISO/IEC 27001:2022(E)

2) the requirements of this document;

b) is effectively implemented and maintained.

## 9.2.2 Internal audit programme

The organization shall plan, establish, implement and maintain an audit programme(s), including the frequency, methods, responsibilities, planning requirements and reporting.

When establishing the internal audit programme(s), the organization shall consider the importance of the processes concerned and the results of previous audits.

The organization shall:

a) define the audit criteria and scope for each audit;

b) select auditors and conduct audits that ensure objectivity and the impartiality of the audit process;

c) ensure that the results of the audits are reported to relevant management;

Documented information shall be available as evidence of the implementation of the audit programme(s) and the audit results.

## 9.3 Management review

### 9.3.1 General

Top management shall review the organization's information security management system at planned intervals to ensure its continuing suitability, adequacy and effectiveness.

### 9.3.2 Management review inputs

The management review shall include consideration of:

a) the status of actions from previous management reviews;

b) changes in external and internal issues that are relevant to the information security management system;

c) changes in needs and expectations of interested parties that are relevant to the information security management system;

d) feedback on the information security performance, including trends in:

1) nonconformities and corrective actions;

2) monitoring and measurement results;

3) audit results;

4) fulfilment of information security objectives;

e) feedback from interested parties;

f) results of risk assessment and status of risk treatment plan;

g) opportunities for continual improvement.

### 9.3.3 Management review results

The results of the management review shall include decisions related to continual improvement opportunities and any needs for changes to the information security management system.

© ISO/IEC 2022 - All rights reserved

9


-------------------------------------------------------------------------------- Page 16

ISO/IEC 27001:2022(E)

Documented information shall be available as evidence of the results of management reviews.

# 10 Improvement

## 10.1 Continual improvement

The organization shall continually improve the suitability, adequacy and effectiveness of the information security management system.

## 10.2 Nonconformity and corrective action

When a nonconformity occurs, the organization shall:
a) react to the nonconformity, and as applicable:
1) take action to control and correct it;
2) deal with the consequences;
b) evaluate the need for action to eliminate the causes of nonconformity, in order that it does not recur or occur elsewhere, by:
1) reviewing the nonconformity;
2) determining the causes of the nonconformity; and
3) determining if similar nonconformities exist, or could potentially occur;
c) implement any action needed;
d) review the effectiveness of any corrective action taken; and
e) make changes to the information security management system, if necessary.

Corrective actions shall be appropriate to the effects of the nonconformities encountered.

Documented information shall be available as evidence of:
f) the nature of the nonconformities and any subsequent actions taken,
g) the results of any corrective action.


-------------------------------------------------------------------------------- Page 17

ISO/IEC 27001:2022(E)

## Annex A
(normative)

## Information security controls reference

The information security controls listed in Table A.1 are directly derived from and aligned with those listed in ISO/IEC 27002:2022[^1], Clauses 5 to 8, and shall be used in context with 6.1.3.

Table A.1 — Information security controls

| 5    | Organizational controls                                   |                                                                                                                                                                                                                                                                                  |
| ---- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5.1  | Policies for information secu-rity                        | Control Information security policy and topic-specific policies shall be defined, approved by management, published, communicated to and acknowledged by relevant personnel and relevant interested parties, and reviewed at planned intervals and if significant changes occur. |
| 5.2  | Information security roles and responsibilities           | Control Information security roles and responsibilities shall be defined and allocated according to the organization needs.                                                                                                                                                      |
| 5.3  | Segregation of duties                                     | Control Conflicting duties and conflicting areas of responsibility shall be segregated.                                                                                                                                                                                          |
| 5.4  | Management responsibilities                               | Control Management shall require all personnel to apply information security in accordance with the established information security policy, topic-specific policies and procedures of the organization.                                                                         |
| 5.5  | Contact with authorities                                  | Control The organization shall establish and maintain contact with relevant authorities.                                                                                                                                                                                         |
| 5.6  | Contact with special interest groups                      | Control The organization shall establish and maintain contact with special interest groups or other specialist security forums and professional associations.                                                                                                                    |
| 5.7  | Threat intelligence                                       | Control Information relating to information security threats shall be collected and analysed to produce threat intelligence.                                                                                                                                                     |
| 5.8  | Information security in project management                | Control Information security shall be integrated into project management.                                                                                                                                                                                                        |
| 5.9  | Inventory of information and other associated assets      | Control An inventory of information and other associated assets, including owners, shall be developed and maintained.                                                                                                                                                            |
| 5.10 | Acceptable use of information and other associated assets | Control Rules for the acceptable use and procedures for handling information and other associated assets shall be identified, documented and implemented.                                                                                                                        |
| 5.11 | Return of assets                                          | Control Personnel and other interested parties as appropriate shall return all the organization's assets in their possession upon change or termination of their employment, contract or agreement.                                                                              |


[^1]:


-------------------------------------------------------------------------------- Page 18

Table A.1 (continued)

|      |                                                                                                  |                                                                                                                                                                                                                         |
| ---- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5.12 | Classification of information                                                                    | Control Information shall be classified according to the information security needs of the organization based on confidentiality, integrity, availability and relevant interested party requirements.                   |
| 5.13 | Labelling of information                                                                         | Control An appropriate set of procedures for information labelling shall be developed and implemented in accordance with the information classification scheme adopted by the organization.                             |
| 5.14 | Information transfer                                                                             | Control Information transfer rules, procedures, or agreements shall be in place for all types of transfer facilities within the organization and between the organization and other parties.                            |
| 5.15 | Access control                                                                                   | Control Rules to control physical and logical access to information and other associated assets shall be established and implemented based on business and information security requirements.                           |
| 5.16 | Identity management                                                                              | Control The full life cycle of identities shall be managed.                                                                                                                                                             |
| 5.17 | Authentication information                                                                       | Control Allocation and management of authentication information shall be controlled by a management process, including advising personnel on appropriate handling of authentication information.                        |
| 5.18 | Access rights                                                                                    | Control Access rights to information and other associated assets shall be provisioned, reviewed, modified and removed in accordance with the organization's topic-specific policy on and rules for access control.      |
| 5.19 | Information security in supplier relationships                                                   | Control Processes and procedures shall be defined and implemented to manage the information security risks associated with the use of supplier's products or services.                                                  |
| 5.20 | Addressing information security within supplier agreements                                       | Control Relevant information security requirements shall be established and agreed with each supplier based on the type of supplier relationship.                                                                       |
| 5.21 | Managing information security in the information and communication technology (ICT) supply chain | Control Processes and procedures shall be defined and implemented to manage the information security risks associated with the ICT products and services supply chain.                                                  |
| 5.22 | Monitoring, review and change management of supplier services                                    | Control The organization shall regularly monitor, review, evaluate and manage change in supplier information security practices and service delivery.                                                                   |
| 5.23 | Information security for use of cloud services                                                   | Control Processes for acquisition, use, management and exit from cloud services shall be established in accordance with the organization's information security requirements.                                           |
| 5.24 | Information security incident management planning and preparation                                | Control The organization shall plan and prepare for managing information security incidents by defining, establishing and communicating information security incident management processes, roles and responsibilities. |


-------------------------------------------------------------------------------- Page 19

ISO/IEC 27001:2022(E)

Table A.1 (continued)

|      |                                                                        |                                                                                                                                                                                                                                   |
| ---- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5.25 | Assessment and decision on in-formation security events                | Control The organization shall assess information security events and decide if they are to be categorized as information security incidents.                                                                                     |
| 5.26 | Response to information security incidents                             | Control Information security incidents shall be responded to in accordance with the documented procedures.                                                                                                                        |
| 5.27 | Learning from information security incidents                           | Control Knowledge gained from information security incidents shall be used to strengthen and improve the information security controls.                                                                                           |
| 5.28 | Collection of evidence                                                 | Control The organization shall establish and implement procedures for the iden-tification, collection, acquisition and preservation of evidence related to information security events.                                           |
| 5.29 | Information security during disruption                                 | Control The organization shall plan how to maintain information security at an appropriate level during disruption.                                                                                                               |
| 5.30 | ICT readiness for business con-tinuity                                 | Control ICT readiness shall be planned, implemented, maintained and tested based on business continuity objectives and ICT continuity requirements.                                                                               |
| 5.31 | Legal, statutory, regulatory and contractual requirements              | Control Legal, statutory, regulatory and contractual requirements relevant to information security and the organization's approach to meet these requirements shall be identified, documented and kept up to date.                |
| 5.32 | Intellectual property rights                                           | Control The organization shall implement appropriate procedures to protect intellectual property rights.                                                                                                                          |
| 5.33 | Protection of records                                                  | Control Records shall be protected from loss, destruction, falsification, unau-thorized access and unauthorized release.                                                                                                          |
| 5.34 | Privacy and protection of person-al identifiable information (PII)     | Control The organization shall identify and meet the requirements regarding the preservation of privacy and protection of PII according to applicable laws and regulations and contractual requirements.                          |
| 5.35 | Independent review of informa-tion security                            | Control The organization's approach to managing information security and its implementation including people, processes and technologies shall be reviewed independently at planned intervals, or when significant changes occur. |
| 5.36 | Compliance with policies, rules and standards for information security | Control Compliance with the organization's information security policy, top-ic-specific policies, rules and standards shall be regularly reviewed.                                                                                |
| 5.37 | Documented operating proce-dures                                       | Control Operating procedures for information processing facilities shall be documented and made available to personnel who need them.                                                                                             |


-------------------------------------------------------------------------------- Page 20

ISO/IEC 27001:2022(E)

Table A.1 (continued)

|     |                                                            |                                                                                                                                                                                                                                                                                                                                                                   |
| --- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6   | People controls                                            |                                                                                                                                                                                                                                                                                                                                                                   |
| 6.1 | Screening                                                  | **Control** Background verification checks on all candidates to become personnel shall be carried out prior to joining the organization and on an ongoing basis taking into consideration applicable laws, regulations and ethics and be proportional to the business requirements, the classification of the information to be accessed and the perceived risks. |
| 6.2 | Terms and conditions of employment                         | **Control** The employment contractual agreements shall state the personnel's and the organization's responsibilities for information security.                                                                                                                                                                                                                   |
| 6.3 | Information security awareness, education and training     | **Control** Personnel of the organization and relevant interested parties shall receive appropriate information security awareness, education and training and regular updates of the organization's information security policy, topic-specific policies and procedures, as relevant for their job function.                                                     |
| 6.4 | Disciplinary process                                       | **Control** A disciplinary process shall be formalized and communicated to take actions against personnel and other relevant interested parties who have committed an information security policy violation.                                                                                                                                                      |
| 6.5 | Responsibilities after termination or change of employment | **Control** Information security responsibilities and duties that remain valid after termination or change of employment shall be defined, enforced and communicated to relevant personnel and other interested parties.                                                                                                                                          |
| 6.6 | Confidentiality or non-disclosure agreements               | **Control** Confidentiality or non-disclosure agreements reflecting the organization's needs for the protection of information shall be identified, documented, regularly reviewed and signed by personnel and other relevant interested parties.                                                                                                                 |
| 6.7 | Remote working                                             | **Control** Security measures shall be implemented when personnel are working remotely to protect information accessed, processed or stored outside the organization's premises.                                                                                                                                                                                  |
| 6.8 | Information security event reporting                       | **Control** The organization shall provide a mechanism for personnel to report observed or suspected information security events through appropriate channels in a timely manner.                                                                                                                                                                                 |
| 7   | Physical controls                                          |                                                                                                                                                                                                                                                                                                                                                                   |
| 7.1 | Physical security perimeters                               | **Control** Security perimeters shall be defined and used to protect areas that contain information and other associated assets.                                                                                                                                                                                                                                  |
| 7.2 | Physical entry                                             | **Control** Secure areas shall be protected by appropriate entry controls and access points.                                                                                                                                                                                                                                                                      |
| 7.3 | Securing offices, rooms and facilities                     | **Control** Physical security for offices, rooms and facilities shall be designed and implemented.                                                                                                                                                                                                                                                                |
| 7.4 | Physical security monitoring                               | **Control** Premises shall be continuously monitored for unauthorized physical access.                                                                                                                                                                                                                                                                            |


-------------------------------------------------------------------------------- Page 21

Table A.1 (continued)

|      |                                                       |                                                                                                                                                                                                             |
| ---- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 7.5  | Protecting against physical and environmental threats | **Control** Protection against physical and environmental threats, such as natural disasters and other intentional or unintentional physical threats to infrastructure shall be designed and implemented.   |
| 7.6  | Working in secure areas                               | **Control** Security measures for working in secure areas shall be designed and implemented.                                                                                                                |
| 7.7  | Clear desk and clear screen                           | **Control** Clear desk rules for papers and removable storage media and clear screen rules for information processing facilities shall be defined and appropriately enforced.                               |
| 7.8  | Equipment siting and protection                       | **Control** Equipment shall be sited securely and protected.                                                                                                                                                |
| 7.9  | Security of assets off-premises                       | **Control** Off-site assets shall be protected.                                                                                                                                                             |
| 7.10 | Storage media                                         | **Control** Storage media shall be managed through their life cycle of acquisition, use, transportation and disposal in accordance with the organization's classification scheme and handling requirements. |
| 7.11 | Supporting utilities                                  | **Control** Information processing facilities shall be protected from power failures and other disruptions caused by failures in supporting utilities.                                                      |
| 7.12 | Cabling security                                      | **Control** Cables carrying power, data or supporting information services shall be protected from interception, interference or damage.                                                                    |
| 7.13 | Equipment maintenance                                 | **Control** Equipment shall be maintained correctly to ensure availability, integrity and confidentiality of information.                                                                                   |
| 7.14 | Secure disposal or re-use of equipment                | **Control** Items of equipment containing storage media shall be verified to ensure that any sensitive data and licensed software has been removed or securely overwritten prior to disposal or re-use.     |

8 Technological controls

|     |                                |                                                                                                                                                               |
| --- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 8.1 | User end point devices         | **Control** Information stored on, processed by or accessible via user end point devices shall be protected.                                                  |
| 8.2 | Privileged access rights       | **Control** The allocation and use of privileged access rights shall be restricted and managed.                                                               |
| 8.3 | Information access restriction | **Control** Access to information and other associated assets shall be restricted in accordance with the established topic-specific policy on access control. |
| 8.4 | Access to source code          | **Control** Read and write access to source code, development tools and software libraries shall be appropriately managed.                                    |


-------------------------------------------------------------------------------- Page 22

Table A.1 (continued)

|      |                                                 |                                                                                                                                                                                                                                              |
| ---- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 8.5  | Secure authentication                           | **Control** Secure authentication technologies and procedures shall be implemented based on information access restrictions and the topic-specific policy on access control.                                                                 |
| 8.6  | Capacity management                             | **Control** The use of resources shall be monitored and adjusted in line with current and expected capacity requirements.                                                                                                                    |
| 8.7  | Protection against malware                      | **Control** Protection against malware shall be implemented and supported by appropriate user awareness.                                                                                                                                     |
| 8.8  | Management of technical vulnerabilities         | **Control** Information about technical vulnerabilities of information systems in use shall be obtained, the organization's exposure to such vulnerabilities shall be evaluated and appropriate measures shall be taken.                     |
| 8.9  | Configuration management                        | **Control** Configurations, including security configurations, of hardware, software, services and networks shall be established, documented, implemented, monitored and reviewed.                                                           |
| 8.10 | Information deletion                            | **Control** Information stored in information systems, devices or in any other storage media shall be deleted when no longer required.                                                                                                       |
| 8.11 | Data masking                                    | **Control** Data masking shall be used in accordance with the organization's topic-specific policy on access control and other related topic-specific policies, and business requirements, taking applicable legislation into consideration. |
| 8.12 | Data leakage prevention                         | **Control** Data leakage prevention measures shall be applied to systems, networks and any other devices that process, store or transmit sensitive information.                                                                              |
| 8.13 | Information backup                              | **Control** Backup copies of information, software and systems shall be maintained and regularly tested in accordance with the agreed topic-specific policy on backup.                                                                       |
| 8.14 | Redundancy of information processing facilities | **Control** Information processing facilities shall be implemented with redundancy sufficient to meet availability requirements.                                                                                                             |
| 8.15 | Logging                                         | **Control** Logs that record activities, exceptions, faults and other relevant events shall be produced, stored, protected and analysed.                                                                                                     |
| 8.16 | Monitoring activities                           | **Control** Networks, systems and applications shall be monitored for anomalous behaviour and appropriate actions taken to evaluate potential information security incidents.                                                                |
| 8.17 | Clock synchronization                           | **Control** The clocks of information processing systems used by the organization shall be synchronized to approved time sources.                                                                                                            |


-------------------------------------------------------------------------------- Page 23

ISO/IEC 27001:2022(E)

Table A.1 (continued)

|      |                                    |                                                                                                                                                   |
| ---- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 8.18 | Use of privileged utility programs | Control The use of utility programs that can be capable of overriding system and application controls shall be restricted and tightly controlled. |
| 8.19 | Installation of software on op-    | Control Procedures and measures shall be implemented to securely manage software installation on operational systems.                             |
|      | erational systems                  |                                                                                                                                                   |
| 8.20 | Networks security                  | Control Networks and network devices shall be secured, managed and controlled to protect information in systems and applications.                 |
| 8.21 | Security of network services       | Control Security mechanisms, service levels and service requirements of network services shall be identified, implemented and monitored.          |
| 8.22 | Segregation of networks            | Control Groups of information services, users and information systems shall be segregated in the organization's networks.                         |
| 8.23 | Web filtering                      | Control Access to external websites shall be managed to reduce exposure to malicious content.                                                     |
| 8.24 | Use of cryptography                | Control Rules for the effective use of cryptography, including cryptographic key management, shall be defined and implemented.                    |
| 8.25 | Secure development life cycle      | Control Rules for the secure development of software and systems shall be established and applied.                                                |
| 8.26 | Application security require-      | Control Information security requirements shall be identified, specified and approved when developing or acquiring applications.                  |
|      | ments                              |                                                                                                                                                   |
| 8.27 | Secure system architecture and     | Control Principles for engineering secure systems shall be established, documented, maintained and applied to any information system development  |
|      | engineering principles             | activities.                                                                                                                                       |
| 8.28 | Secure coding                      | Control Secure coding principles shall be applied to software development.                                                                        |
| 8.29 | Security testing in development    | Control Security testing processes shall be defined and implemented in the development life cycle.                                                |
|      | and acceptance                     |                                                                                                                                                   |
| 8.30 | Outsourced development             | Control The organization shall direct, monitor and review the activities related to outsourced system development.                                |
| 8.31 | Separation of development, test    | Control Development, testing and production environments shall be separated and secured.                                                          |
|      | and production environments        |                                                                                                                                                   |
| 8.32 | Change management                  | Control Changes to information processing facilities and information systems shall be subject to change management procedures.                    |
| 8.33 | Test information                   | Control Test information shall be appropriately selected, protected and managed.                                                                  |

© ISO/IEC 2022 - All rights reserved 17


-------------------------------------------------------------------------------- Page 24

Table A.1 (continued)

| 8.34 | Protection of information systems during audit testing | Control Audit tests and other assurance activities involving assessment of operational systems shall be planned and agreed between the tester and appropriate management. |


-------------------------------------------------------------------------------- Page 25

ISO/IEC 27001:2022(E)

# Bibliography

[1] ISO/IEC 27002:2022, Information security, cybersecurity and privacy protection — Information security controls

[2] ISO/IEC 27003, Information technology — Security techniques — Information security management systems — Guidance

[3] ISO/IEC 27004, Information technology — Security techniques — Information security management — Monitoring, measurement, analysis and evaluation

[4] ISO/IEC 27005, Information security, cybersecurity and privacy protection — Guidance on managing information security risks

[5] ISO 31000:2018, Risk management — Guidelines


-------------------------------------------------------------------------------- Page 26

ISO/IEC 27001:2022(E)

ICS 03.100.70; 35.030

Price based on 19 pages

© ISO/IEC 2022 - All rights reserved
