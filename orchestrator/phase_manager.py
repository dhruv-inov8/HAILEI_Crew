"""
HAILEI Phase Management System

Handles phase-specific workflow logic for conversational course design.
Each phase has specific requirements and agent coordination patterns.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .conversation_state import PhaseStatus, AgentStatus


class PhaseManager:
    """
    Manages phase-specific workflows and agent coordination.
    
    Each phase has:
    - Introduction message for user
    - Agent activation sequence
    - Success criteria
    - Transition logic to next phase
    """
    
    def __init__(self, orchestrator):
        """Initialize with reference to main orchestrator"""
        self.orchestrator = orchestrator
        self.logger = logging.getLogger('hailei_phase_manager')
        
        # Phase-specific configurations
        self.phase_configs = {
            'course_overview': {
                'intro_message': self._get_course_overview_intro,
                'agents': ['hailei4t_coordinator_agent'],
                'requires_approval': True,
                'parallel_execution': False,
                'success_criteria': 'user_approval'
            },
            'foundation_design': {
                'intro_message': self._get_foundation_design_intro,
                'agents': ['ipdai_agent'],
                'requires_approval': True,
                'parallel_execution': False,
                'success_criteria': 'agent_output_approved'
            },
            'content_creation': {
                'intro_message': self._get_content_creation_intro,
                'agents': ['cauthai_agent'],
                'requires_approval': True,
                'parallel_execution': False,
                'success_criteria': 'agent_output_approved'
            },
            'technical_design': {
                'intro_message': self._get_technical_design_intro,
                'agents': ['tfdai_agent'],
                'requires_approval': True,
                'parallel_execution': False,
                'success_criteria': 'agent_output_approved'
            },
            'quality_review': {
                'intro_message': self._get_quality_review_intro,
                'agents': ['editorai_agent'],
                'requires_approval': True,
                'parallel_execution': False,
                'success_criteria': 'agent_output_approved'
            },
            'ethical_audit': {
                'intro_message': self._get_ethical_audit_intro,
                'agents': ['ethosai_agent'],
                'requires_approval': True,
                'parallel_execution': False,
                'success_criteria': 'agent_output_approved'
            },
            'final_integration': {
                'intro_message': self._get_final_integration_intro,
                'agents': ['hailei4t_coordinator_agent'],
                'requires_approval': True,
                'parallel_execution': False,
                'success_criteria': 'final_deliverable_approved'
            }
        }
    
    async def execute_phase(self, phase_id: str) -> bool:
        """
        Execute a complete phase workflow.
        
        Args:
            phase_id: Phase identifier
            
        Returns:
            bool: Success status
        """
        if phase_id not in self.phase_configs:
            raise ValueError(f"Unknown phase configuration: {phase_id}")
        
        config = self.phase_configs[phase_id]
        phase_state = self.orchestrator.conversation_state.phases[phase_id]
        
        self.logger.info(f"Executing phase: {phase_id}")
        
        try:
            # 1. Execute agents for this phase (they will present their output and wait for approval)
            if config['parallel_execution'] and len(config['agents']) > 1:
                # Parallel execution for multiple agents
                success = await self._execute_agents_parallel(config['agents'], phase_id)
            else:
                # Sequential execution
                success = await self._execute_agents_sequential(config['agents'], phase_id)
            
            if not success:
                self.logger.error(f"Agent execution failed for phase: {phase_id}")
                return False
            
            # 2. Handle iterative refinement if needed (agents will wait for user approval)
            await self._handle_phase_refinement(phase_id)
            
            # 4. Complete phase
            return await self.orchestrator.complete_phase(phase_id)
            
        except Exception as e:
            self.logger.error(f"Phase execution failed for {phase_id}: {e}")
            phase_state.status = PhaseStatus.PENDING  # Reset for retry
            raise
    
    async def _execute_agents_sequential(self, agent_ids: List[str], phase_id: str) -> bool:
        """Execute agents sequentially with user approval between each"""
        for agent_id in agent_ids:
            if agent_id not in self.orchestrator.agents:
                self.logger.warning(f"Agent {agent_id} not found, skipping")
                continue
            
            # Get task description for this agent in this phase
            task_description = self._get_agent_task_description(agent_id, phase_id)
            
            # Activate agent
            output = await self.orchestrator.activate_agent(agent_id, task_description)
            
            # Present output to user for review
            review_message = self._format_agent_output_for_review(output)
            
            # Enter refinement loop
            approved = await self._agent_refinement_loop(agent_id, output, review_message)
            
            if not approved:
                self.logger.warning(f"Agent {agent_id} output not approved for phase {phase_id}")
                return False
        
        return True
    
    async def _execute_agents_parallel(self, agent_ids: List[str], phase_id: str) -> bool:
        """Execute multiple agents in parallel (future enhancement)"""
        # For now, fall back to sequential
        # This will be enhanced in Phase 3: Advanced Features
        return await self._execute_agents_sequential(agent_ids, phase_id)
    
    async def _agent_refinement_loop(self, agent_id: str, initial_output, review_message: str) -> bool:
        """
        Handle iterative refinement loop for agent output.
        
        Args:
            agent_id: Agent identifier
            initial_output: Initial agent output
            review_message: Formatted review message
            
        Returns:
            bool: True if approved, False if rejected
        """
        current_output = initial_output
        max_refinements = 5  # Prevent infinite loops
        refinement_count = 0
        
        while refinement_count < max_refinements:
            # Present current output to user
            user_response = await self.orchestrator.request_user_input(
                f"{review_message}\\n\\nOptions:\\n1. Approve this output\\n2. Request modifications\\n3. Reject and skip",
                input_type="choice",
                options=["approve", "modify", "reject"]
            )
            
            if user_response == "WAITING_FOR_USER_INPUT":
                # Agent is waiting for user input via WebSocket
                # Don't auto-approve - let the WebSocket handler manage approval
                self.logger.info(f"Agent {agent_id} is waiting for user approval via WebSocket")
                return False  # Don't complete phase yet, wait for user
            elif user_response.lower() in ['approve', 'approved', '1']:
                # User approved - mark output as approved
                await self.orchestrator.approve_output(agent_id)
                return True
            
            elif user_response.lower() in ['modify', 'modifications', '2']:
                # User wants modifications
                feedback = await self.orchestrator.request_user_input(
                    "Please provide specific feedback for improvements:",
                    input_type="feedback"
                )
                
                # Process feedback and get refined output
                current_output = await self.orchestrator.process_user_feedback(
                    agent_id, feedback
                )
                
                # Update review message with refined output
                review_message = self._format_agent_output_for_review(current_output)
                refinement_count += 1
            
            elif user_response.lower() in ['reject', 'skip', '3']:
                # User rejected - do not approve
                return False
            
            else:
                # Invalid response, ask again
                continue
        
        # Max refinements reached
        self.logger.warning(f"Max refinements reached for {agent_id}")
        return False
    
    async def _handle_phase_refinement(self, phase_id: str):
        """Handle any phase-level refinement needs"""
        # Check if all outputs in phase are approved
        phase_state = self.orchestrator.conversation_state.phases[phase_id]
        unapproved_outputs = []
        
        for agent_id in phase_state.assigned_agents:
            output = self.orchestrator.conversation_state.get_latest_output(agent_id, phase_id)
            if output and not output.is_approved:
                unapproved_outputs.append(agent_id)
        
        if unapproved_outputs:
            self.logger.warning(f"Phase {phase_id} has unapproved outputs: {unapproved_outputs}")
            # Could implement phase-level refinement logic here
    
    def _get_agent_task_description(self, agent_id: str, phase_id: str) -> str:
        """Get specific task description for agent in phase"""
        task_templates = {
            'coordinator': {
                'course_overview': """
                Present the course overview to the user and get approval to proceed.
                
                Course Request: {course_title}
                
                Provide a clear summary of:
                1. Course title and description
                2. Target audience and level
                3. Duration and structure
                4. Learning objectives overview
                5. Framework approach (KDKA + PRRR)
                
                Ask for user confirmation to begin the design process.
                """,
                'final_integration': """
                Compile all approved specialist outputs into a comprehensive course design package.
                
                Create final deliverable including:
                1. Complete course foundation (from IPDAi)
                2. Instructional content (from CAuthAi)
                3. Technical implementation plan (from TFDAi)
                4. Quality-reviewed materials (from EditorAi)
                5. Ethical compliance certification (from EthosAi)
                6. Implementation roadmap
                7. Next steps recommendations
                
                Present as a professional course design package ready for deployment.
                """
            },
            'ipdai': {
                'foundation_design': """
                Create comprehensive course foundation using HAILEI frameworks.
                
                Using the provided KDKA and PRRR frameworks, develop:
                
                1. **Learning Objectives Structure**:
                   - Terminal Learning Objectives (TLOs) using Bloom's Taxonomy
                   - Enabling Learning Objectives (ELOs) mapped to TLOs
                   - Cognitive complexity validation
                
                2. **KDKA Framework Application**:
                   - Knowledge: Core concepts and skills alignment
                   - Delivery: Modality recommendations
                   - Context: Authentic scenario integration
                   - Assessment: Formative and summative alignment
                
                3. **Course Structure**:
                   - Weekly module breakdown for {duration} weeks
                   - Module learning outcomes
                   - Progressive skill building sequence
                
                4. **PRRR Integration Strategy**:
                   - Personal connection opportunities
                   - Relatable examples and analogies
                   - Relative comparisons and choices
                   - Real-world application scenarios
                
                5. **Draft Syllabus**:
                   - Course policies and expectations
                   - Assessment framework
                   - Resource requirements
                   - Technology integration plan
                
                Ensure all outputs align with {course_level} level and {course_credits} credit expectations.
                """
            },
            'cauthai': {
                'content_creation': """
                Develop comprehensive instructional content based on approved course foundation.
                
                Using IPDAi's approved foundation and PRRR methodology, create:
                
                1. **Learning Activities** (PRRR-based):
                   - Personal: Activities that connect to learner experiences
                   - Relatable: Cross-disciplinary analogies and examples
                   - Relative: Comparison exercises and decision scenarios
                   - Real-world: Authentic stakeholder-centered tasks
                
                2. **Content Development**:
                   - Detailed lecture outlines with key concepts
                   - Interactive exercises and hands-on activities
                   - Discussion prompts for collaborative learning
                   - Case studies and scenario-based learning
                
                3. **Assessment Design**:
                   - Rubrics aligned with learning objectives
                   - Formative assessment checkpoints
                   - Summative evaluation instruments
                   - Peer review and self-assessment tools
                
                4. **Resource Curation**:
                   - Required and supplemental readings
                   - Multimedia resources (videos, simulations)
                   - Tools and software recommendations
                   - External learning platforms integration
                
                5. **Engagement Strategies**:
                   - Active learning techniques
                   - Collaborative project frameworks
                   - Student choice and personalization options
                   - Inclusive participation methods
                
                Ensure content supports diverse learning styles and maintains PRRR engagement throughout.
                """
            },
            'tfdai': {
                'technical_design': """
                Create comprehensive LMS implementation plan for {lms_platform} platform.
                
                Based on CAuthAi's instructional content, design technical specifications:
                
                1. **LMS Feature Mapping**:
                   - Content delivery mechanisms
                   - Assessment and quiz integration
                   - Discussion forum structure
                   - Gradebook configuration
                   - Media and resource organization
                
                2. **Navigation Design**:
                   - Course structure and module layout
                   - User experience flow optimization
                   - Accessibility navigation features
                   - Mobile responsiveness considerations
                
                3. **Integration Requirements**:
                   - SCORM package specifications
                   - LTI tool integrations
                   - External platform connections
                   - API integration needs
                   - Single sign-on requirements
                
                4. **Technical Specifications**:
                   - File format requirements
                   - Storage and bandwidth needs
                   - Security and privacy settings
                   - Backup and recovery protocols
                
                5. **Implementation Plan**:
                   - Phase-by-phase deployment schedule
                   - Resource requirements
                   - Testing and quality assurance protocols
                   - Training needs for instructors
                   - Student onboarding process
                
                6. **Accessibility Compliance**:
                   - WCAG 2.1 AA standard adherence
                   - Universal Design for Learning integration
                   - Assistive technology compatibility
                   - Alternative format provisions
                
                Provide detailed technical roadmap for seamless course deployment.
                """
            },
            'editorai': {
                'quality_review': """
                Conduct comprehensive quality review and enhancement of all course materials.
                
                Review outputs from IPDAi, CAuthAi, and TFDAi for:
                
                1. **Content Quality**:
                   - Grammar, spelling, and clarity
                   - Academic tone and professionalism
                   - Consistency across all materials
                   - Logical flow and organization
                
                2. **Framework Compliance**:
                   - KDKA framework marker validation
                   - PRRR principle integration verification
                   - Bloom's taxonomy alignment check
                   - Learning objective-content alignment
                
                3. **Accessibility Review**:
                   - Universal Design for Learning compliance
                   - Inclusive language and representation
                   - Alternative format readiness
                   - Barrier identification and removal
                
                4. **Educational Standards**:
                   - Pedagogical best practices adherence
                   - Evidence-based learning principles
                   - Engagement strategy effectiveness
                   - Assessment validity and reliability
                
                5. **Enhancement Recommendations**:
                   - Content improvement suggestions
                   - Engagement optimization opportunities
                   - Technology integration enhancements
                   - Student success factor improvements
                
                6. **Compliance Validation**:
                   - Institutional policy alignment
                   - Accreditation standard compliance
                   - Legal and ethical requirement adherence
                
                Provide detailed editor report with all enhancements and validations completed.
                """
            },
            'ethosai': {
                'ethical_audit': """
                Conduct comprehensive ethical compliance audit of complete course package.
                
                Examine all materials for ethical integrity and compliance:
                
                1. **AI in Education Ethics**:
                   - Responsible AI tool integration
                   - Transparency in AI-assisted learning
                   - Student data privacy protection
                   - Algorithmic bias prevention
                
                2. **Universal Design for Learning**:
                   - Inclusive design principles validation
                   - Accessibility standard compliance
                   - Diverse learner accommodation
                   - Equity in learning opportunities
                
                3. **Privacy and Data Protection**:
                   - Student information security
                   - FERPA compliance verification
                   - Data collection transparency
                   - Consent and opt-out provisions
                
                4. **Bias and Fairness Assessment**:
                   - Content bias identification
                   - Cultural sensitivity review
                   - Representation and inclusion audit
                   - Stereotype avoidance verification
                
                5. **Academic Integrity**:
                   - Plagiarism prevention measures
                   - Citation and attribution standards
                   - Original content verification
                   - Intellectual property compliance
                
                6. **Inclusivity Validation**:
                   - Diverse perspective representation
                   - Multicultural sensitivity
                   - Socioeconomic accessibility
                   - Gender and identity inclusion
                
                Provide final ethical compliance certification with any required modifications.
                """
            }
        }
        
        # Get template and format with current context
        template = task_templates.get(agent_id, {}).get(phase_id, "Complete assigned task for this phase.")
        
        # Format with conversation context
        context = self.orchestrator.conversation_state
        course_request = context.course_request
        
        formatted_task = template.format(
            course_title=course_request.get('course_title', 'the course'),
            duration=course_request.get('course_duration_weeks', 16),
            course_level=course_request.get('course_level', 'undergraduate'),
            course_credits=course_request.get('course_credits', 3),
            lms_platform=course_request.get('lms_platform', 'Canvas')
        )
        
        return formatted_task.strip()
    
    def _format_agent_output_for_review(self, output) -> str:
        """Format agent output for user review"""
        return f"""
**{output.agent_name} Output for {output.phase}**

{output.content}

---
*Generated at: {output.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*
*Version: {output.version}*

Please review this output carefully. You can:
- Approve it to proceed to the next step
- Request specific modifications
- Reject it if it doesn't meet your requirements
"""
    
    # Phase introduction message generators
    def _get_course_overview_intro(self) -> str:
        course = self.orchestrator.conversation_state.course_request
        return f"""
🎓 **Welcome to HAILEI Course Design**

Thank you for submitting your course request. I'll be your main coordinator throughout this process, working with our specialized educational AI agents to create a comprehensive course design.

**Course Overview:**
📚 **Title**: {course.get('course_title', 'Your Course')}
🎯 **Level**: {course.get('course_level', 'Not specified')}
⏱️ **Duration**: {course.get('course_duration_weeks', 'Not specified')} weeks
💳 **Credits**: {course.get('course_credits', 'Not specified')}

**Our Process:**
We'll work through 5 specialized phases:
1. **Foundation Design** - Instructional planning with KDKA & PRRR frameworks
2. **Content Creation** - Learning activities and materials development
3. **Technical Design** - LMS implementation planning
4. **Quality Review** - Enhancement and accessibility compliance
5. **Ethical Audit** - Final compliance and inclusivity verification

Each phase will involve you directly - you'll review each specialist's work and can request modifications until you're completely satisfied.

Ready to begin creating your course with our specialized agents?
"""
    
    def _get_foundation_design_intro(self) -> str:
        return f"""
🏗️ **Phase 1: Course Foundation Design**

Now I'll connect you with **IPDAi** (Instructional Planning & Design Agent), our specialist in educational foundations using HAILEI's proprietary frameworks.

**What IPDAi will create:**
- Learning objectives using Bloom's Taxonomy
- Course structure with weekly module breakdown  
- KDKA framework integration (Knowledge, Delivery, Context, Assessment)
- PRRR engagement strategy (Personal, Relatable, Relative, Real-world)
- Draft syllabus and assessment framework

IPDAi will present their work for your review, and you can request any modifications until the foundation perfectly matches your vision.

Ready to proceed with IPDAi for course foundation design?
"""
    
    def _get_content_creation_intro(self) -> str:
        return f"""
✍️ **Phase 2: Instructional Content Creation**

Excellent work on the course foundation! Now I'll connect you with **CAuthAi** (Course Authoring Agent), our specialist in creating engaging instructional content.

**What CAuthAi will develop:**
- PRRR-based learning activities and exercises
- Detailed content for each module
- Assessment rubrics and evaluation tools
- Interactive elements and engagement strategies
- Curated educational resources and readings

CAuthAi will build upon IPDAi's approved foundation to create rich, engaging learning experiences that maintain the PRRR methodology throughout.

Ready to proceed with CAuthAi for content creation?
"""
    
    def _get_technical_design_intro(self) -> str:
        lms = self.orchestrator.conversation_state.course_request.get('lms_platform', 'Canvas')
        return f"""
⚙️ **Phase 3: Technical Implementation Design**

Great progress on content creation! Now I'll connect you with **TFDAi** (Technical & Functional Design Agent), our specialist in LMS implementation.

**What TFDAi will design:**
- {lms} platform integration specifications
- Navigation structure and user experience
- Technical requirements and feature mapping
- SCORM/LTI integration planning
- Accessibility and mobile responsiveness
- Implementation timeline and resource requirements

TFDAi will translate your educational content into a practical, deployable technical plan for seamless course delivery.

Ready to proceed with TFDAi for technical design?
"""
    
    def _get_quality_review_intro(self) -> str:
        return f"""
🔍 **Phase 4: Quality Review & Enhancement**

Excellent technical planning! Now I'll connect you with **EditorAi** (Content Review & Enhancement Agent), our quality assurance specialist.

**What EditorAi will accomplish:**
- Comprehensive content review and editing
- Framework compliance validation (KDKA & PRRR)
- Accessibility and inclusivity enhancement
- Educational standards verification
- Consistency and professional polish
- Enhancement recommendations

EditorAi will ensure all materials meet the highest educational standards and maintain framework integrity throughout.

Ready to proceed with EditorAi for quality review?
"""
    
    def _get_ethical_audit_intro(self) -> str:
        return f"""
🛡️ **Phase 5: Ethical Compliance Audit**

Excellent quality improvements! Now for our final specialist review with **EthosAi** (Ethical Oversight Agent), our ethics and compliance expert.

**What EthosAi will verify:**
- Ethical AI integration and transparency
- Universal Design for Learning compliance
- Privacy and data protection standards
- Bias detection and inclusivity validation
- Academic integrity safeguards
- Cultural sensitivity and representation

EthosAi will provide final certification that your course meets all ethical and compliance standards for responsible education.

Ready to proceed with EthosAi for the final ethical audit?
"""
    
    def _get_final_integration_intro(self) -> str:
        return f"""
🎉 **Final Integration & Delivery**

Congratulations! All specialists have completed their work and received your approval. Now I'll compile everything into your comprehensive course design package.

**Your complete deliverable will include:**
- ✅ Approved course foundation with learning objectives
- ✅ Comprehensive instructional content and activities
- ✅ Technical implementation plan for deployment
- ✅ Quality-reviewed and polished materials
- ✅ Ethical compliance certification
- 📋 Implementation roadmap and next steps
- 🚀 Professional course package ready for deployment

This final integration will create a cohesive, deployment-ready course design that maintains consistency across all frameworks and requirements.

Ready for me to compile your final course design package?
"""